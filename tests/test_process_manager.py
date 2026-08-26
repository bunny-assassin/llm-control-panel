from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from backend.config import load_config
from backend.process_manager import ProcessManager, inspect_weights

DUMMY = Path(__file__).resolve().parent / "dummy_openai_server.py"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _write_config(tmp_path: Path, ports: list[int]):
    models = []
    for i, port in enumerate(ports, start=1):
        models.append(
            {
                "id": f"dummy-{i}",
                "display_name": f"Dummy {i}",
                "backend": "custom",
                "port": port,
                "host": "127.0.0.1",
                "launch": [sys.executable, str(DUMMY), str(port)],
                "model_path": str(DUMMY),
            }
        )
    doc = {
        "panel": {
            "host": "127.0.0.1",
            "port": 8765,
            "stop_timeout_sec": 5,
            "health_timeout_sec": 1,
            "health_interval_sec": 0.2,
            "state_dir": str(tmp_path / "run"),
        },
        "models": models,
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_config(path)


async def _wait_healthy(manager: ProcessManager, timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    last = manager.snapshot()
    while time.time() < deadline:
        last = await manager.health_once()
        if last.get("healthy"):
            return last
        if last.get("status") == "error":
            break
        await asyncio.sleep(0.1)
    raise AssertionError(f"never became healthy: {last}")


@pytest.fixture
async def http():
    client = httpx.AsyncClient()
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_start_health_test_stop(tmp_path: Path, http: httpx.AsyncClient):
    cfg = _write_config(tmp_path, [_free_port()])
    manager = ProcessManager(cfg)
    manager.bind_http(http)
    try:
        snap = await manager.start("dummy-1")
        assert snap["status"] in {"starting", "running"}
        healthy = await _wait_healthy(manager)
        assert healthy["healthy"] is True
        assert healthy["port"] == cfg.models[0].port
        result = await manager.test_completion()
        assert result["ok"] is True
        assert "pong" in result["content"]
        assert result["latency_ms"] >= 0
        stopped = await manager.stop("dummy-1")
        assert stopped["status"] == "idle"
        assert stopped["model_id"] is None
    finally:
        await manager.aclose()


@pytest.mark.asyncio
async def test_switch_enforces_single_process(tmp_path: Path, http: httpx.AsyncClient):
    cfg = _write_config(tmp_path, [_free_port(), _free_port()])
    manager = ProcessManager(cfg)
    manager.bind_http(http)
    try:
        await manager.start("dummy-1")
        first = await _wait_healthy(manager)
        pid1 = first["pid"]
        await manager.start("dummy-2")
        second = await _wait_healthy(manager)
        assert second["model_id"] == "dummy-2"
        assert second["pid"] != pid1
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                response = await http.get(
                    f"http://127.0.0.1:{cfg.models[0].port}/v1/models",
                    timeout=0.5,
                )
            except httpx.HTTPError:
                break
            if response.status_code >= 400:
                break
        else:
            raise AssertionError("dummy-1 still serving after switch")
    finally:
        await manager.stop()
        await manager.aclose()


@pytest.mark.asyncio
async def test_missing_binary_does_not_mark_error(tmp_path: Path, http: httpx.AsyncClient):
    cfg = _write_config(tmp_path, [_free_port()])
    cfg.models[0].launch[0] = "definitely-not-installed-llm-backend"
    manager = ProcessManager(cfg)
    manager.bind_http(http)
    try:
        with pytest.raises(RuntimeError, match="executable not found"):
            await manager.start("dummy-1")
        snap = manager.snapshot()
        assert snap["status"] == "idle"
        assert snap["model_id"] is None
    finally:
        await manager.aclose()


@pytest.mark.asyncio
async def test_recover_reattaches_to_running_server(tmp_path: Path, http: httpx.AsyncClient):
    cfg = _write_config(tmp_path, [_free_port()])
    first = ProcessManager(cfg)
    first.bind_http(http)
    try:
        await first.start("dummy-1")
        await _wait_healthy(first)
        pid = first.snapshot()["pid"]
        await first.aclose()
        second = ProcessManager(cfg)
        second.bind_http(http)
        await second.recover()
        snap = await _wait_healthy(second)
        assert snap["pid"] == pid
        assert snap["model_id"] == "dummy-1"
        await second.stop()
        await second.aclose()
    except Exception:
        await first.stop()
        await first.aclose()
        raise


def test_inspect_weights_missing_and_incomplete(tmp_path: Path):
    missing = inspect_weights(str(tmp_path / "nope.gguf"))
    assert missing["exists"] is False
    assert missing["complete"] is False

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 2048)
    ok = inspect_weights(str(gguf))
    assert ok["exists"] is True
    assert ok["complete"] is True

    ckpt = tmp_path / "hf"
    ckpt.mkdir()
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}})
    )
    (ckpt / "a.safetensors").write_bytes(b"x")
    partial = inspect_weights(str(ckpt))
    assert partial["exists"] is True
    assert partial["complete"] is False
    assert "still downloading" in (partial["detail"] or "")
    (ckpt / "b.safetensors").write_bytes(b"x")
    assert inspect_weights(str(ckpt))["complete"] is True


@pytest.mark.asyncio
async def test_start_rejects_incomplete_weights(tmp_path: Path, http: httpx.AsyncClient):
    cfg = _write_config(tmp_path, [_free_port()])
    cfg.models[0].model_path = str(tmp_path / "not-downloaded.gguf")
    manager = ProcessManager(cfg)
    manager.bind_http(http)
    try:
        with pytest.raises(RuntimeError, match="weights not on disk"):
            await manager.start("dummy-1")
        snap = manager.snapshot()
        assert snap["status"] == "idle"
        assert snap["pid"] is None
    finally:
        await manager.aclose()
