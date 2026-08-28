from __future__ import annotations

from pathlib import Path

import yaml

from backend.config import _deep_merge, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.panel.host == "0.0.0.0"
    assert cfg.public_host
    assert {m.id for m in cfg.models} == {"qwen3.6-35b-moe", "qwen3.6-27b-dense"}
    assert cfg.models[0].port != cfg.models[1].port
    assert cfg.models[0].launch[0] == "ft"
    assert cfg.models[1].launch[0] == "llama-server"
    assert "--fit" in cfg.models[1].launch
    assert "99" not in cfg.models[1].launch
    fit_val = cfg.models[1].launch[cfg.models[1].launch.index("--fit") + 1]
    assert fit_val == "on"
    assert "--reasoning" in cfg.models[1].launch
    assert cfg.models[1].launch[cfg.models[1].launch.index("--reasoning") + 1] == "off"


def test_tilde_in_launch_is_expanded(tmp_path: Path):
    doc = {
        "panel": {"host": "127.0.0.1", "port": 8765},
        "models": [
            {
                "id": "one",
                "display_name": "One",
                "backend": "custom",
                "port": 9999,
                "launch": ["echo", "~/weights/model.gguf"],
                "model_path": "~/weights/model.gguf",
            }
        ],
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    cfg = load_config(path)
    home = str(Path.home())
    assert cfg.models[0].launch[1].startswith(home)
    assert cfg.models[0].model_path.startswith(home)


def test_wildcard_bind_uses_lan_or_loopback(tmp_path: Path):
    doc = {
        "panel": {"host": "0.0.0.0", "port": 8765, "advertised_host": "10.0.0.26"},
        "models": [
            {
                "id": "one",
                "display_name": "One",
                "backend": "custom",
                "port": 9999,
                "host": "0.0.0.0",
                "launch": ["true"],
            }
        ],
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(doc))
    cfg = load_config(path)
    assert cfg.panel.host == "0.0.0.0"
    assert cfg.public_host == "10.0.0.26"
    assert cfg.models[0].probe_host() == "127.0.0.1"
    assert cfg.models[0].client_host(cfg.public_host) == "10.0.0.26"
    assert cfg.models[0].public_base_url(cfg.public_host) == "http://10.0.0.26:9999/v1"


def test_model_overlay_merges_by_id():
    base = {
        "models": [
            {"id": "a", "port": 1, "display_name": "A"},
            {"id": "b", "port": 2, "display_name": "B"},
        ]
    }
    overlay = {"models": [{"id": "a", "port": 11, "quant": "Q8_0"}]}
    merged = _deep_merge(base, overlay)
    by_id = {m["id"]: m for m in merged["models"]}
    assert by_id["a"]["port"] == 11
    assert by_id["a"]["display_name"] == "A"
    assert by_id["a"]["quant"] == "Q8_0"
    assert by_id["b"]["port"] == 2
