"""Start, stop, adopt, and health-check a single model-server subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil

from backend.config import AppConfig, ModelConfig, ROOT, WILDCARD_HOSTS
from backend.metrics import parse_tokens_per_sec, stats_tokens_per_sec

Status = Literal["idle", "starting", "running", "stopping", "unhealthy", "error"]
LogListener = Callable[[dict[str, Any]], None]
StatusListener = Callable[[], None]

EXTRA_BIN_DIRS = (
    Path.home() / ".local" / "bin",
    Path.home() / ".cargo" / "bin",
    Path("/opt/cuda/bin"),
    Path("/usr/local/cuda/bin"),
    Path("/usr/local/bin"),
    Path("/usr/bin"),
)

LOG_HISTORY = 500
READY_HINTS = (
    "api server is ready",
    "http server listening",
    "server is listening",
    "ready to serve",
    "listening at http",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ERROR_LINE_RE = re.compile(
    r"(?:cudaMalloc failed|out of memory|unable to allocate CUDA|"
    r"KeyError:|failed to open GGUF|failed to load model|"
    r"\bERROR\b).{0,220}",
    re.IGNORECASE,
)


def extra_path() -> str:
    parts = [str(p) for p in EXTRA_BIN_DIRS if p.is_dir()]
    current = os.environ.get("PATH", "")
    return os.pathsep.join(parts + ([current] if current else []))


def which_executable(name: str) -> str | None:
    if os.path.sep in name:
        path = Path(name).expanduser()
        return str(path) if path.is_file() else None
    found = shutil.which(name, path=extra_path())
    if found:
        return found
    for directory in EXTRA_BIN_DIRS:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def path_exists(path: str | None) -> bool | None:
    if not path:
        return None
    return Path(path).expanduser().exists()


def inspect_weights(path: str | None) -> dict[str, Any]:
    """Return whether a GGUF file or HF shard directory looks complete."""
    if not path:
        return {"exists": False, "complete": False, "detail": "no model_path configured"}
    target = Path(path).expanduser()
    if target.is_file():
        size = target.stat().st_size
        if size < 1024:
            return {"exists": True, "complete": False, "detail": f"{target.name} is empty"}
        return {"exists": True, "complete": True, "detail": None, "bytes": size}
    if not target.is_dir():
        return {"exists": False, "complete": False, "detail": f"weights not on disk yet: {target}"}
    index = target / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text())
        except json.JSONDecodeError:
            return {"exists": True, "complete": False, "detail": "checkpoint index is unreadable"}
        shards = {str(name) for name in (payload.get("weight_map") or {}).values()}
        missing = sorted(name for name in shards if not (target / name).is_file())
        if missing:
            preview = ", ".join(missing[:5])
            extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            return {
                "exists": True,
                "complete": False,
                "detail": f"still downloading ({len(missing)} files missing: {preview}{extra})",
                "missing": missing,
            }
        return {"exists": True, "complete": True, "detail": None}
    if any(target.glob("*.safetensors")) or any(target.glob("*.gguf")):
        return {"exists": True, "complete": True, "detail": None}
    return {"exists": True, "complete": False, "detail": "checkpoint directory is incomplete"}


def detect_cuda_home() -> str | None:
    for candidate in (os.environ.get("CUDA_HOME"), "/opt/cuda", "/usr/local/cuda"):
        if candidate and Path(candidate, "bin", "nvcc").is_file():
            return candidate
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LogLine:
    ts: str
    stream: str
    line: str
    model_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "stream": self.stream,
            "line": self.line,
            "model_id": self.model_id,
        }


@dataclass
class Runtime:
    model_id: str | None = None
    pid: int | None = None
    started_at: float | None = None
    status: Status = "idle"
    last_error: str | None = None
    served_model: str | None = None
    tokens_per_sec: float | None = None
    healthy: bool = False
    adopted: bool = False
    log_path: str | None = None
    last_health_at: float | None = None
    exit_code: int | None = None
    saw_ready: bool = False


@dataclass
class ProcessManager:
    config: AppConfig
    on_log: LogListener | None = None
    on_status: StatusListener | None = None
    state: Runtime = field(default_factory=Runtime)
    logs: deque[LogLine] = field(default_factory=lambda: deque(maxlen=LOG_HISTORY))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _http: httpx.AsyncClient | None = None
    _tail_stop: asyncio.Event | None = None
    _tail_task: asyncio.Task[None] | None = None
    _health_task: asyncio.Task[None] | None = None

    def __post_init__(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        path = Path(self.config.panel.state_dir)
        if not path.is_absolute():
            path = ROOT / path
        return path

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    def bind_http(self, client: httpx.AsyncClient) -> None:
        self._http = client

    def _notify_status(self) -> None:
        if self.on_status:
            self.on_status()

    def note(self, message: str) -> None:
        self._emit(message, stream="panel")

    def _emit(self, line: str, model_id: str | None = None, stream: str = "stdout") -> None:
        cleaned = ANSI_RE.sub("", line).rstrip("\n")
        entry = LogLine(
            ts=_utc_now(),
            stream=stream,
            line=cleaned,
            model_id=model_id if model_id is not None else self.state.model_id,
        )
        self.logs.append(entry)
        self._observe_line(cleaned)
        tok = parse_tokens_per_sec(cleaned)
        if tok is not None:
            self.state.tokens_per_sec = tok
        if self.on_log:
            self.on_log(entry.as_dict())

    def _observe_line(self, line: str) -> None:
        lowered = line.lower()
        if any(hint in lowered for hint in READY_HINTS):
            self.state.saw_ready = True
        err = ERROR_LINE_RE.search(line)
        if err and "futurewarning" not in lowered:
            candidate = err.group(0).strip()[:300]
            if candidate.lower() in {"error", "err"}:
                pass
            elif not self.state.last_error or len(candidate) >= len(self.state.last_error):
                self.state.last_error = candidate

    def _waits_for_ready_log(self, model: ModelConfig) -> bool:
        backend = (model.backend or "").lower()
        binary = Path(model.launch[0]).name.lower()
        return backend in {"freetoken", "ft"} or binary == "ft"

    def _engine_ready(self, model: ModelConfig) -> bool:
        if self.state.saw_ready or self.state.adopted:
            return True
        if not self._waits_for_ready_log(model):
            return True
        # Panel restart often happens after the "ready" log line has scrolled away.
        if self.state.started_at and (time.time() - self.state.started_at) > 20:
            return True
        return False

    def recent_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        items = list(self.logs)[-limit:]
        return [item.as_dict() for item in items]

    def snapshot(self) -> dict[str, Any]:
        model = None
        if self.state.model_id:
            try:
                model = self.config.model(self.state.model_id)
            except KeyError:
                model = None
        uptime = None
        if self.state.started_at:
            uptime = max(0, int(time.time() - self.state.started_at))
        pid_alive = self._pid_alive(self.state.pid)
        advertised = self.config.public_host
        public_origin = model.public_origin(advertised) if model else None
        public_base = model.public_base_url(advertised) if model else None
        return {
            "status": self.state.status,
            "model_id": self.state.model_id,
            "display_name": model.display_name if model else None,
            "backend": model.backend if model else None,
            "port": model.port if model else None,
            "host": model.client_host(advertised) if model else advertised,
            "bind_host": model.host if model else None,
            "base_url": public_base,
            "origin": public_origin,
            "pid": self.state.pid if pid_alive else None,
            "uptime_sec": uptime if pid_alive else None,
            "healthy": bool(self.state.healthy and pid_alive),
            "served_model": self.state.served_model,
            "tokens_per_sec": self.state.tokens_per_sec,
            "adopted": self.state.adopted,
            "last_error": self.state.last_error,
            "loading": self.state.status == "starting",
            "log_path": self.state.log_path,
            "started_at": (
                datetime.fromtimestamp(self.state.started_at, tz=timezone.utc).isoformat()
                if self.state.started_at
                else None
            ),
            "exclusive": True,
        }

    def inventory(self) -> list[dict[str, Any]]:
        active = self.state.model_id if self._pid_alive(self.state.pid) else None
        rows = []
        for model in self.config.models:
            exe = which_executable(model.launch[0])
            advertised = self.config.public_host
            weights = inspect_weights(model.model_path)
            rows.append(
                {
                    "id": model.id,
                    "display_name": model.display_name,
                    "backend": model.backend,
                    "port": model.port,
                    "host": model.client_host(advertised),
                    "bind_host": model.host,
                    "base_url": model.public_base_url(advertised),
                    "origin": model.public_origin(advertised),
                    "launch": model.launch,
                    "model_path": model.model_path,
                    "path_exists": weights["exists"],
                    "weights_complete": weights["complete"],
                    "weights_detail": weights.get("detail"),
                    "binary": model.launch[0],
                    "binary_path": exe,
                    "binary_found": bool(exe),
                    "quant": model.quant,
                    "approx_disk_gb": model.approx_disk_gb,
                    "approx_vram_gb": model.approx_vram_gb,
                    "approx_ram_gb": model.approx_ram_gb,
                    "notes": model.notes,
                    "active": model.id == active,
                }
            )
        return rows

    def snippets(self) -> dict[str, Any] | None:
        if not self.state.model_id:
            return None
        try:
            model = self.config.model(self.state.model_id)
        except KeyError:
            return None
        served = self.state.served_model or model.served_model_name or model.id
        return build_snippets(model, served, self.config.public_host)

    def panel_info(self) -> dict[str, Any]:
        host = self.config.public_host
        bind = self.config.panel.host
        port = self.config.panel.port
        return {
            "bind": bind,
            "port": port,
            "advertised_host": host,
            "url": f"http://{host}:{port}",
            "lan": bind in WILDCARD_HOSTS,
        }

    async def recover(self) -> None:
        saved = self._load_state_file()
        if saved:
            model_id = saved.get("model_id")
            pid = saved.get("pid")
            if model_id and pid and self._pid_alive(int(pid)):
                try:
                    model = self.config.model(str(model_id))
                except KeyError:
                    model = None
                if model and self._cmdline_matches(int(pid), model):
                    await self._adopt(model, int(pid), adopted=True)
                    started = saved.get("started_at")
                    if isinstance(started, (int, float)):
                        self.state.started_at = float(started)
                    self.state.saw_ready = True
                    self._emit(f"re-attached to pid {pid} after panel restart", model.id)
                    return
        for model in self.config.models:
            pid = pid_on_port(model.port)
            if pid and self._pid_alive(pid):
                await self._adopt(model, pid, adopted=True)
                self._emit(
                    f"detected existing listener on :{model.port} (pid {pid})",
                    model.id,
                )
                return

    async def start(self, model_id: str) -> dict[str, Any]:
        async with self._lock:
            model = self.config.model(model_id)
            if self.state.model_id == model.id and self._pid_alive(self.state.pid):
                return self.snapshot()
            exe = which_executable(model.launch[0])
            if not exe:
                raise RuntimeError(f"executable not found: {model.launch[0]}")
            weights = inspect_weights(model.model_path)
            if not weights["complete"]:
                raise RuntimeError(weights.get("detail") or "weights are not ready yet")
            if self.state.model_id and self._pid_alive(self.state.pid):
                await self._stop_locked()
            occupant = pid_on_port(model.port)
            if occupant and self._pid_alive(occupant):
                if self._cmdline_matches(occupant, model):
                    await self._adopt(model, occupant, adopted=True)
                    self._emit(f"adopted already-running server pid {occupant}", model.id)
                    return self.snapshot()
                cmd = _cmdline(occupant)
                raise RuntimeError(
                    f"port {model.port} is already in use by pid {occupant}"
                    + (f" ({cmd})" if cmd else "")
                )
            await self._spawn(model, exe)
            return self.snapshot()

    async def stop(self, model_id: str | None = None) -> dict[str, Any]:
        async with self._lock:
            if model_id and self.state.model_id and model_id != self.state.model_id:
                raise RuntimeError(
                    f"{model_id} is not the running model ({self.state.model_id})"
                )
            await self._stop_locked()
            return self.snapshot()

    async def switch(self, model_id: str) -> dict[str, Any]:
        return await self.start(model_id)

    async def health_once(self) -> dict[str, Any]:
        await self._poll_health()
        return self.snapshot()

    async def test_completion(
        self, prompt: str = "Reply with the single word: pong", max_tokens: int = 32
    ) -> dict[str, Any]:
        snap = self.snapshot()
        if not snap["model_id"] or not snap["healthy"]:
            raise RuntimeError("no healthy model is running")
        model = self.config.model(snap["model_id"])
        served = await self._served_model_name(model) or model.served_model_name or model.id
        payload: dict[str, Any] = {
            "model": served,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        # Qwen3.6 otherwise spends the whole budget inside <think> and returns empty content.
        payload["reasoning_effort"] = "none"
        payload["enable_thinking"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.perf_counter()
        response = await self._client().post(
            f"{model.base_url}/chat/completions",
            json=payload,
            timeout=120.0,
            headers={"Authorization": "Bearer local"},
        )
        latency_ms = (time.perf_counter() - started) * 1000
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        if response.status_code >= 400:
            raise RuntimeError(
                f"chat completion failed ({response.status_code}): "
                f"{response.text[:500]}"
            )
        content = ""
        reasoning = ""
        usage = {}
        if isinstance(body, dict):
            usage = body.get("usage") or {}
            choices = body.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = (message.get("content") or choices[0].get("text") or "").strip()
                reasoning = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                )
                if isinstance(reasoning, str):
                    reasoning = reasoning.strip()
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        tps = None
        if completion_tokens and latency_ms > 0:
            tps = round(completion_tokens / (latency_ms / 1000), 2)
            self.state.tokens_per_sec = tps
        return {
            "ok": True,
            "latency_ms": round(latency_ms, 1),
            "tokens_per_sec": tps,
            "model": served,
            "content": content,
            "reasoning": reasoning,
            "usage": usage,
            "prompt": prompt,
        }

    async def run_bandwidth_bench(self) -> dict[str, Any]:
        exe = which_executable("ft")
        if not exe:
            raise RuntimeError("ft not found on PATH — install FreeToken first")
        self._emit("running `ft bench bw` (once per GPU; writes ~/.cache/freetoken/benchbw/)")
        env = os.environ.copy()
        env["PATH"] = extra_path()
        cuda_home = detect_cuda_home()
        if cuda_home:
            env.setdefault("CUDA_HOME", cuda_home)
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env["PATH"]
        proc = await asyncio.create_subprocess_exec(
            exe,
            "bench",
            "bw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            self._emit(raw.decode("utf-8", errors="replace").rstrip("\n"))
        code = await proc.wait()
        if code != 0:
            raise RuntimeError(f"ft bench bw exited {code}")
        return {"ok": True, "exit_code": code}

    async def start_background_tasks(self) -> None:
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop(), name="health-loop")

    async def aclose(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        await self._stop_tail()
        self._persist()

    async def _spawn(self, model: ModelConfig, exe: str) -> None:
        argv = [exe, *model.launch[1:]]
        log_path = self.state_dir / "logs" / f"{model.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = extra_path()
        cuda_home = detect_cuda_home()
        if cuda_home:
            env.setdefault("CUDA_HOME", cuda_home)
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env["PATH"]
        env.update(model.env)
        banner = (
            f"\n======== {_utc_now()} start {model.id} "
            f"{' '.join(argv)} ========\n"
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(banner)
        self._emit(f"starting: {' '.join(argv)}", model.id)
        log_fh = log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                argv,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=model.cwd,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to spawn {exe}: {exc}") from exc
        finally:
            log_fh.close()
        self.state = Runtime(
            model_id=model.id,
            pid=proc.pid,
            started_at=time.time(),
            status="starting",
            last_error=None,
            served_model=model.served_model_name,
            tokens_per_sec=None,
            healthy=False,
            adopted=False,
            log_path=str(log_path),
            saw_ready=False,
        )
        self._persist()
        await self._start_tail(log_path)
        self._notify_status()

    async def _adopt(self, model: ModelConfig, pid: int, adopted: bool) -> None:
        await self._stop_tail()
        log_path = Path(self.state.log_path) if self.state.log_path else (
            self.state_dir / "logs" / f"{model.id}.log"
        )
        started = None
        try:
            started = psutil.Process(pid).create_time()
        except psutil.Error:
            started = time.time()
        self.state = Runtime(
            model_id=model.id,
            pid=pid,
            started_at=started,
            status="starting",
            last_error=None,
            served_model=model.served_model_name,
            tokens_per_sec=None,
            healthy=False,
            adopted=adopted,
            log_path=str(log_path),
        )
        if log_path.is_file():
            self._preload_logs(log_path)
        await self._start_tail(log_path)
        self._persist()
        await self._poll_health()
        self._notify_status()

    async def _stop_locked(self) -> None:
        pid = self.state.pid
        model_id = self.state.model_id
        if not pid or not self._pid_alive(pid):
            await self._stop_tail()
            self.state = Runtime()
            self._persist()
            self._notify_status()
            return
        self.state.status = "stopping"
        self.state.healthy = False
        self._notify_status()
        self._emit(f"stopping pid {pid} (SIGTERM, then SIGKILL after timeout)", model_id)
        timeout = self.config.panel.stop_timeout_sec
        await asyncio.to_thread(kill_tree, pid, timeout)
        await self._stop_tail()
        self._emit("process stopped", model_id)
        self.state = Runtime()
        self._persist()
        self._notify_status()

    async def _health_loop(self) -> None:
        interval = self.config.panel.health_interval_sec
        while True:
            try:
                await self._poll_health()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.note(f"health loop error: {exc}")
            await asyncio.sleep(interval)

    async def _poll_health(self) -> None:
        if not self.state.model_id:
            return
        if self.state.pid and not self._pid_alive(self.state.pid):
            if self.state.status not in {"idle", "stopping"}:
                self.state.status = "error"
                self.state.healthy = False
                if not self.state.last_error:
                    self.state.last_error = "process exited"
                self._emit(
                    f"process exited: {self.state.last_error}",
                    self.state.model_id,
                    stream="panel",
                )
                self.state.pid = None
                self._persist()
                self._notify_status()
            return
        if not self.state.pid:
            return
        try:
            model = self.config.model(self.state.model_id)
        except KeyError:
            return
        http_up, served, tps = await self._probe(model)
        self.state.last_health_at = time.time()
        changed = False
        if served and served != self.state.served_model:
            self.state.served_model = served
            changed = True
        if tps is not None:
            self.state.tokens_per_sec = tps
        ready = self._engine_ready(model)
        if http_up and ready and self.state.status != "running":
            self.state.healthy = True
            self.state.status = "running"
            self.state.last_error = None
            changed = True
            self._emit(f"endpoint healthy at {model.public_base_url(self.config.public_host)}", model.id)
        elif http_up and not ready:
            self.state.healthy = False
            if self.state.status != "starting":
                self.state.status = "starting"
                changed = True
        elif not http_up and self.state.status == "running":
            self.state.healthy = False
            self.state.status = "unhealthy"
            changed = True
        if ready and http_up != self.state.healthy:
            self.state.healthy = http_up
            changed = True
        if changed:
            self._persist()
            self._notify_status()

    async def _probe(self, model: ModelConfig) -> tuple[bool, str | None, float | None]:
        timeout = self.config.panel.health_timeout_sec
        client = self._client()
        served = None
        tps = None
        try:
            response = await client.get(
                f"{model.probe_origin}{model.health_path}",
                timeout=timeout,
            )
            if response.status_code >= 400:
                return False, None, None
            try:
                body = response.json()
            except ValueError:
                body = None
            served = _model_id_from_models_payload(body)
        except httpx.HTTPError:
            return False, None, None
        if model.backend == "freetoken":
            try:
                stats = await client.get(f"{model.probe_origin}/v1/stats", timeout=timeout)
                if stats.status_code < 400:
                    tps = stats_tokens_per_sec(stats.json())
            except (httpx.HTTPError, ValueError):
                pass
        return True, served, tps

    async def _served_model_name(self, model: ModelConfig) -> str | None:
        if self.state.served_model:
            return self.state.served_model
        _, served, _ = await self._probe(model)
        if served:
            self.state.served_model = served
        return served

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    async def _start_tail(self, log_path: Path) -> None:
        await self._stop_tail()
        self._tail_stop = asyncio.Event()
        offset = log_path.stat().st_size if log_path.exists() else 0
        self._tail_task = asyncio.create_task(
            self._tail_file(log_path, offset, self._tail_stop),
            name="log-tail",
        )

    def _preload_logs(self, path: Path, limit: int = 200) -> None:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        except OSError:
            return
        for line in lines:
            cleaned = ANSI_RE.sub("", line)
            self._observe_line(cleaned)
            self.logs.append(
                LogLine(
                    ts=_utc_now(),
                    stream="stdout",
                    line=cleaned,
                    model_id=self.state.model_id,
                )
            )

    async def _stop_tail(self) -> None:
        if self._tail_stop:
            self._tail_stop.set()
        if self._tail_task:
            self._tail_task.cancel()
            try:
                await self._tail_task
            except asyncio.CancelledError:
                pass
        self._tail_task = None
        self._tail_stop = None

    async def _tail_file(self, path: Path, offset: int, stop: asyncio.Event) -> None:
        try:
            while not path.exists() and not stop.is_set():
                await asyncio.sleep(0.2)
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                while not stop.is_set():
                    line = fh.readline()
                    if line:
                        self._emit(line.rstrip("\n"))
                        if any(hint in line.lower() for hint in READY_HINTS):
                            self._notify_status()
                        continue
                    await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            self._emit(f"log tail failed: {exc}", stream="panel")

    def _persist(self) -> None:
        payload = {
            "model_id": self.state.model_id,
            "pid": self.state.pid,
            "started_at": self.state.started_at,
            "log_path": self.state.log_path,
            "adopted": self.state.adopted,
            "status": self.state.status,
            "last_error": self.state.last_error,
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.state_file)

    def _load_state_file(self) -> dict[str, Any] | None:
        if not self.state_file.is_file():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    @staticmethod
    def _cmdline_matches(pid: int, model: ModelConfig) -> bool:
        cmd = _cmdline(pid)
        if not cmd:
            return False
        haystack = cmd.lower()
        if str(model.port) in haystack:
            return True
        if model.model_path and Path(model.model_path).name.lower() in haystack:
            return True
        binary = Path(model.launch[0]).name.lower()
        return binary in haystack


def _cmdline(pid: int) -> str:
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except psutil.Error:
        return ""


def pid_on_port(port: int) -> int | None:
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.Error, OSError):
        connections = []
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or not conn.laddr:
            continue
        if conn.laddr.port == port and conn.pid:
            return conn.pid
    return _pid_on_port_procfs(port)


def _pid_on_port_procfs(port: int) -> int | None:
    """Fallback that does not need net_connections privileges."""
    hex_port = f"{port:04X}"
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            state = parts[3]
            inode = parts[9]
            if state != "0A":
                continue
            if local.rsplit(":", 1)[-1].upper() == hex_port:
                inodes.add(inode)
    if not inodes:
        return None
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return int(entry.name)
        except OSError:
            continue
    return None


def kill_tree(pid: int, timeout: float) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = None
    if pgid == pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        try:
            parent.terminate()
        except psutil.Error:
            pass
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
    _, alive = psutil.wait_procs([parent, *children], timeout=timeout)
    if not alive:
        return
    if pgid == pid:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=3)


def _model_id_from_models_payload(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    if body.get("id"):
        return str(body["id"])
    if body.get("model"):
        return str(body["model"])
    return None


def build_snippets(model: ModelConfig, served: str, public_host: str) -> dict[str, Any]:
    host = model.client_host(public_host)
    origin = model.public_origin(public_host)
    base = model.public_base_url(public_host)
    curl = (
        f"curl http://{host}:{model.port}/v1/chat/completions \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -H 'Authorization: Bearer local' \\\n"
        f"  -d '{{\n"
        f'    "model": "{served}",\n'
        f'    "messages": [{{"role": "user", "content": "hello"}}],\n'
        f'    "max_tokens": 256\n'
        f"  }}'"
    )
    continue_block = (
        "{\n"
        f'  "title": "{model.display_name}",\n'
        '  "provider": "openai",\n'
        f'  "model": "{served}",\n'
        f'  "apiBase": "{base}",\n'
        '  "apiKey": "local"\n'
        "}"
    )
    aider = (
        f"set -x OPENAI_API_BASE {base}\n"
        "set -x OPENAI_API_KEY local\n"
        f"aider --model openai/{served}"
    )
    cline = (
        f"Base URL: {base}\n"
        "API Provider: OpenAI Compatible\n"
        f"Model ID: {served}\n"
        "API Key: local"
    )
    ft_launch = f"ft launch claude --server {origin}"
    anthropic = origin if model.backend == "freetoken" else None
    return {
        "base_url": base,
        "origin": origin,
        "openai_base_url": base,
        "anthropic_base_url": anthropic,
        "model": served,
        "api_key": "local",
        "curl": curl,
        "continue": continue_block,
        "aider_fish": aider,
        "cline": cline,
        "ft_launch": ft_launch if model.backend == "freetoken" else None,
    }
