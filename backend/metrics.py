"""Host RAM and NVIDIA VRAM / util polling."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

import psutil

NVIDIA_SMI_QUERY = (
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
)


def _bytes_to_gb(n: int | float) -> float:
    return round(n / (1024**3), 2)


def ram_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    used = vm.total - vm.available
    return {
        "total_gb": _bytes_to_gb(vm.total),
        "used_gb": _bytes_to_gb(used),
        "available_gb": _bytes_to_gb(vm.available),
        "percent": round((used / vm.total) * 100, 1) if vm.total else 0.0,
    }


def _parse_smi_line(line: str) -> dict[str, Any] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    def num(value: str) -> float | None:
        if not value or value in {"[N/A]", "N/A"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    used = num(parts[2])
    total = num(parts[3])
    util = num(parts[4]) if len(parts) > 4 else None
    temp = num(parts[5]) if len(parts) > 5 else None
    return {
        "index": int(num(parts[0]) or 0),
        "name": parts[1],
        "used_mb": used,
        "total_mb": total,
        "used_gb": round(used / 1024, 2) if used is not None else None,
        "total_gb": round(total / 1024, 2) if total is not None else None,
        "percent": round((used / total) * 100, 1) if used is not None and total else 0.0,
        "util_percent": util,
        "temperature_c": temp,
    }


def gpu_snapshot() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    try:
        proc = subprocess.run(
            [smi, NVIDIA_SMI_QUERY, "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
        return {"available": False, "error": err, "gpus": []}
    gpus = []
    for line in proc.stdout.splitlines():
        parsed = _parse_smi_line(line)
        if parsed:
            gpus.append(parsed)
    return {"available": bool(gpus), "error": None if gpus else "no GPUs reported", "gpus": gpus}


def collect_metrics() -> dict[str, Any]:
    gpu = gpu_snapshot()
    primary = gpu["gpus"][0] if gpu.get("gpus") else None
    return {"ram": ram_snapshot(), "gpu": gpu, "vram": primary}


TOKENS_PER_SEC_RE = None


def parse_tokens_per_sec(line: str) -> float | None:
    """Pull a tok/s figure out of llama.cpp / FreeToken log lines."""
    import re

    global TOKENS_PER_SEC_RE
    if TOKENS_PER_SEC_RE is None:
        TOKENS_PER_SEC_RE = re.compile(
            r"(?:"
            r"(\d+(?:\.\d+)?)\s*(?:tok(?:ens)?(?:/|\s+per\s+)s(?:ec(?:ond)?)?|t/s)\b"
            r"|"
            r"(\d+(?:\.\d+)?)\s*tokens per second"
            r")",
            re.IGNORECASE,
        )
    match = TOKENS_PER_SEC_RE.search(line)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value > 1_000_000:
        return None
    return value


def stats_tokens_per_sec(payload: Any) -> float | None:
    """Best-effort tok/s from FreeToken GET /v1/stats (schema varies by version)."""
    if not isinstance(payload, dict):
        return None
    keys = (
        "tokens_per_second",
        "tokens_per_sec",
        "generation_throughput",
        "decode_throughput",
        "throughput",
        "tok_s",
        "tps",
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    for nested_key in ("performance", "metrics", "decode", "generation"):
        nested = payload.get(nested_key)
        found = stats_tokens_per_sec(nested) if isinstance(nested, dict) else None
        if found:
            return found
    return None
