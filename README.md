# Local LLM Control Panel

Start, stop, monitor, and switch between locally hosted coding models from a
browser. Only one model is loaded at a time (16 GB VRAM). The panel and model
servers bind **0.0.0.0** so other machines on your LAN can reach them. There is
**no auth** — do not port-forward these ports past your router.

Designed for a desktop with an RTX 5080 (16 GB) running CachyOS: FreeToken for
Qwen3-Coder-Next (MoE, quality) and llama.cpp for Qwen3.6-27B (dense, fast).

## What you get

- YAML model registry — adding a model is a config change, not a code change
- Start / stop / switch (switch always unloads the current model first)
- Health checks against `/v1/models`, not just “process is alive”
- Live VRAM (`nvidia-smi`), system RAM, log tail, and tok/s when the backend prints it
- One-click chat smoke test (response + latency)
- Copy-able OpenAI base URL, curl, Continue.dev, Cline, Aider, and `ft launch` snippets
- Closing the browser tab does **not** unload the model. Restarting the panel
  re-attaches to the still-running server.

## Prerequisites

Install the backends yourself (the panel only launches them):

```fish
# FreeToken (MoE / Qwen3-Coder-Next)
uv pip install "freetoken[accel]"
# once per GPU, then cached under ~/.cache/freetoken/benchbw/
ft bench bw

# llama.cpp with CUDA (AUR)
paru -S llama.cpp-cuda
```

Weights are assumed to already be on disk. Point the paths in
`config/models.yaml` at them (or overlay `config/models.local.yaml` so you can
pull updates without clobbering local paths).

Default placeholders:

| id | backend | port | path |
|---|---|---|---|
| id | backend | port | path |
|---|---|---|---|
| `qwen3.6-35b-moe` | `ft serve` | 1919 | `~/models/Qwen3.6-35B-A3B-FP8` |
| `qwen3.6-27b-dense` | `llama-server` | 8081 | `~/models/Qwen_Qwen3.6-27B-Q8_0.gguf` |

## Run the panel

```fish
cd ~/projects/llm-control-panel
python -m venv .venv
source .venv/bin/activate.fish
pip install -e ".[dev]"
llm-panel
```

Or without the console script:

```fish
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) on this machine, or
`http://<this-pc-lan-ip>:8765` from another computer on the same network.

If you use `uv`:

```fish
uv sync --extra dev
uv run llm-panel
```

## Config

`config/models.yaml` is the registry. Each entry needs `id`, `backend`, `port`,
and `launch`. Optional: `model_path`, `quant`, footprint estimates, `notes`,
`env`, `served_model_name`.

`panel.autostart_model` can be set to a model id if you want that model loaded
whenever the panel starts (for example from a systemd user service at login).

To keep machine-specific paths out of git:

```yaml
# config/models.local.yaml
models:
  - id: qwen3.6-35b-moe
    launch: [ft, serve, --model, /actual/path/Qwen3.6-35B-A3B-FP8, --host, "0.0.0.0", --port, "1919"]
    model_path: /actual/path/Qwen3.6-35B-A3B-FP8
```

## Agent wiring

With a model healthy, the UI copies the OpenAI-compatible base URL:

```
http://127.0.0.1:<port>/v1
```

API key can be any dummy string (`local`). FreeToken also serves Anthropic
`/v1/messages` on the same origin (`http://127.0.0.1:1919`).

```fish
# Aider
set -x OPENAI_API_BASE http://127.0.0.1:1919/v1
set -x OPENAI_API_KEY local
aider --model openai/<served-model-name>
```

## systemd user service (optional)

```fish
mkdir -p ~/.config/systemd/user
cp ~/projects/llm-control-panel/systemd/llm-control-panel.service ~/.config/systemd/user/
# edit WorkingDirectory / ExecStart if the repo is not at ~/projects/llm-control-panel
systemctl --user daemon-reload
systemctl --user enable --now llm-control-panel.service
```

The unit uses `KillMode=process` so stopping the panel does not kill the model
server. Set `panel.autostart_model` in YAML if you also want a model at login.

## Tests

```fish
.venv/bin/pytest
```

## Layout

```
config/models.yaml          # registry
backend/main.py             # FastAPI app (SSE + REST)
backend/process_manager.py  # start / stop / adopt / health
backend/metrics.py          # nvidia-smi + RAM
frontend/index.html         # single-page UI, no build step
systemd/                    # optional user unit
```
