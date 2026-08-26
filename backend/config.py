"""Load and validate the YAML model registry."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "models.yaml"
LOCAL_CONFIG_PATH = ROOT / "config" / "models.local.yaml"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "*"})


def detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("1.1.1.1", 80))
            ip = sock.getsockname()[0]
    except OSError:
        return None
    if not ip or ip.startswith("127."):
        return None
    return ip


def advertised_host(bind: str, configured: str | None) -> str:
    if configured and configured not in {"auto", ""}:
        return configured
    if bind in WILDCARD_HOSTS or configured == "auto":
        return detect_lan_ip() or "127.0.0.1"
    return bind


def probe_host(bind: str) -> str:
    return "127.0.0.1" if bind in WILDCARD_HOSTS else bind


@dataclass
class PanelConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    stop_timeout_sec: float = 30.0
    health_timeout_sec: float = 2.0
    metrics_interval_sec: float = 2.0
    health_interval_sec: float = 2.0
    autostart_model: str | None = None
    state_dir: str = ".run"
    advertised_host: str | None = None


@dataclass
class ModelConfig:
    id: str
    display_name: str
    backend: str
    port: int
    launch: list[str]
    host: str = "0.0.0.0"
    model_path: str | None = None
    quant: str | None = None
    approx_disk_gb: float | None = None
    approx_vram_gb: float | None = None
    approx_ram_gb: float | None = None
    served_model_name: str | None = None
    health_path: str = "/v1/models"
    notes: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    def probe_host(self) -> str:
        return probe_host(self.host)

    def client_host(self, public_host: str) -> str:
        return public_host if self.host in WILDCARD_HOSTS else self.host

    @property
    def probe_origin(self) -> str:
        return f"http://{self.probe_host()}:{self.port}"

    def public_origin(self, public_host: str) -> str:
        return f"http://{self.client_host(public_host)}:{self.port}"

    def public_base_url(self, public_host: str) -> str:
        return f"{self.public_origin(public_host)}/v1"

    @property
    def base_url(self) -> str:
        return f"http://{self.probe_host()}:{self.port}/v1"

    @property
    def origin(self) -> str:
        return self.probe_origin


@dataclass
class AppConfig:
    panel: PanelConfig
    models: list[ModelConfig]
    path: Path
    public_host: str = "127.0.0.1"

    def model(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def by_port(self, port: int) -> ModelConfig | None:
        for m in self.models:
            if m.port == port:
                return m
        return None


def _expand(value: str) -> str:
    return str(Path(value).expanduser()) if value else value


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _extract_model_path(launch: list[str]) -> str | None:
    flags = {"--model", "--model-path", "-m"}
    for i, arg in enumerate(launch[:-1]):
        if arg in flags:
            return launch[i + 1]
    return None


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    if not raw.get("id"):
        raise ValueError("each model needs an id")
    launch = raw.get("launch")
    if not isinstance(launch, list) or not launch:
        raise ValueError(f"model {raw['id']!r} needs a non-empty launch list")
    launch = [_expand(str(x)) for x in launch]
    model_path = raw.get("model_path")
    model_path = _expand(str(model_path)) if model_path else _extract_model_path(launch)
    notes = raw.get("notes")
    if isinstance(notes, str):
        notes = " ".join(notes.split())
    env = raw.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError(f"model {raw['id']!r} env must be a mapping")
    return ModelConfig(
        id=str(raw["id"]),
        display_name=str(raw.get("display_name") or raw["id"]),
        backend=str(raw.get("backend") or "custom"),
        port=int(raw["port"]),
        launch=launch,
        host=str(raw.get("host") or "0.0.0.0"),
        model_path=model_path,
        quant=str(raw["quant"]) if raw.get("quant") is not None else None,
        approx_disk_gb=_as_float(raw.get("approx_disk_gb")),
        approx_vram_gb=_as_float(raw.get("approx_vram_gb")),
        approx_ram_gb=_as_float(raw.get("approx_ram_gb")),
        served_model_name=(
            str(raw["served_model_name"]) if raw.get("served_model_name") else None
        ),
        health_path=str(raw.get("health_path") or "/v1/models"),
        notes=notes,
        env={str(k): str(v) for k, v in env.items()},
        cwd=_expand(str(raw["cwd"])) if raw.get("cwd") else None,
    )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key == "models" and isinstance(value, list):
            by_id = {
                m["id"]: dict(m)
                for m in out.get("models") or []
                if isinstance(m, dict) and m.get("id")
            }
            for item in value:
                if isinstance(item, dict) and item.get("id"):
                    previous = by_id.get(item["id"], {})
                    by_id[item["id"]] = {**previous, **item}
            out["models"] = list(by_id.values())
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    local_path = LOCAL_CONFIG_PATH if path is None else None
    if local_path and local_path.is_file():
        with local_path.open() as fh:
            overlay = yaml.safe_load(fh) or {}
        if not isinstance(overlay, dict):
            raise ValueError("models.local.yaml must be a mapping")
        raw = _deep_merge(raw, overlay)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    panel_raw = raw.get("panel") or {}
    autostart = panel_raw.get("autostart_model")
    if autostart in ("", "null", "None"):
        autostart = None
    advertised = panel_raw.get("advertised_host")
    if advertised in ("", "null", "None"):
        advertised = None
    panel = PanelConfig(
        host=str(panel_raw.get("host") or "0.0.0.0"),
        port=int(panel_raw.get("port") or 8765),
        stop_timeout_sec=float(panel_raw.get("stop_timeout_sec") or 30),
        health_timeout_sec=float(panel_raw.get("health_timeout_sec") or 2.0),
        metrics_interval_sec=float(panel_raw.get("metrics_interval_sec") or 2.0),
        health_interval_sec=float(panel_raw.get("health_interval_sec") or 2.0),
        autostart_model=str(autostart) if autostart else None,
        state_dir=str(panel_raw.get("state_dir") or ".run"),
        advertised_host=str(advertised) if advertised else None,
    )
    models = [_parse_model(item) for item in (raw.get("models") or [])]
    if not models:
        raise ValueError("config has no models")
    ids = [m.id for m in models]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model ids in config")
    ports = [m.port for m in models]
    if len(ports) != len(set(ports)):
        raise ValueError("duplicate model ports in config")
    if panel.autostart_model and panel.autostart_model not in ids:
        raise ValueError(f"autostart_model {panel.autostart_model!r} is not in the registry")
    public = advertised_host(panel.host, panel.advertised_host)
    return AppConfig(panel=panel, models=models, path=config_path, public_host=public)
