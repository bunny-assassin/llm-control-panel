"""FastAPI entrypoint for the local LLM control panel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import ROOT, load_config
from backend.metrics import collect_metrics
from backend.process_manager import ProcessManager, build_snippets

FRONTEND = ROOT / "frontend"


class EventHub:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[tuple[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[tuple[str, Any]]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[tuple[str, Any]]) -> None:
        self._subs.discard(queue)

    def publish(self, event: str, data: Any) -> None:
        item = (event, data)
        for queue in list(self._subs):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass


class TestRequest(BaseModel):
    prompt: str = "Reply with the single word: pong"
    max_tokens: int = Field(default=32, ge=1, le=256)


def _http_error(exc: Exception, model_id: str | None = None) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=f"unknown model: {model_id or exc}")
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    hub = EventHub()
    http = httpx.AsyncClient()
    manager = ProcessManager(config)

    def on_log(line: dict[str, Any]) -> None:
        hub.publish("log", line)

    def on_status() -> None:
        hub.publish("status", manager.snapshot())

    manager.on_log = on_log
    manager.on_status = on_status
    manager.bind_http(http)

    app.state.config = config
    app.state.hub = hub
    app.state.manager = manager
    app.state.http = http
    app.state.latest_metrics = collect_metrics()

    await manager.recover()
    await manager.start_background_tasks()
    metrics_task = asyncio.create_task(_metrics_loop(app), name="metrics-loop")

    if config.panel.autostart_model and manager.state.status == "idle":
        try:
            manager.note(f"autostart: {config.panel.autostart_model}")
            await manager.start(config.panel.autostart_model)
        except Exception as exc:
            manager.note(f"autostart failed: {exc}")

    yield

    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    await manager.aclose()
    await http.aclose()


async def _metrics_loop(app: FastAPI) -> None:
    while True:
        data = collect_metrics()
        app.state.latest_metrics = data
        app.state.hub.publish("metrics", data)
        await asyncio.sleep(app.state.config.panel.metrics_interval_sec)


app = FastAPI(
    title="Local LLM Control Panel",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/api/health")
async def panel_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def list_models(request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    return {
        "models": manager.inventory(),
        "status": manager.snapshot(),
        "panel": manager.panel_info(),
    }


@app.get("/api/status")
async def status(request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    return {
        "status": manager.snapshot(),
        "metrics": request.app.state.latest_metrics,
        "panel": manager.panel_info(),
    }


@app.get("/api/logs")
async def logs(request: Request, limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    return {"lines": manager.recent_logs(limit)}


@app.get("/api/snippets")
async def running_snippets(request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    snippets = manager.snippets()
    if not snippets:
        raise HTTPException(status_code=409, detail="no model is running")
    return snippets


@app.get("/api/models/{model_id}/snippets")
async def model_snippets(model_id: str, request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        model = manager.config.model(model_id)
    except KeyError as exc:
        raise _http_error(exc, model_id) from exc
    served = manager.state.served_model if manager.state.model_id == model_id else None
    served = served or model.served_model_name or model.id
    return build_snippets(model, served, manager.config.public_host)


@app.post("/api/models/{model_id}/start")
async def start_model(model_id: str, request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        return await manager.start(model_id)
    except (KeyError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            manager.note(str(exc))
        raise _http_error(exc, model_id) from exc


@app.post("/api/models/{model_id}/switch")
async def switch_model(model_id: str, request: Request) -> dict[str, Any]:
    return await start_model(model_id, request)


@app.post("/api/models/{model_id}/stop")
async def stop_model(model_id: str, request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        return await manager.stop(model_id)
    except (KeyError, RuntimeError) as exc:
        raise _http_error(exc, model_id) from exc


@app.post("/api/stop")
async def stop_running(request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        return await manager.stop()
    except RuntimeError as exc:
        raise _http_error(exc) from exc


@app.post("/api/test")
async def test_completion(request: Request, body: TestRequest) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        return await manager.test_completion(body.prompt, body.max_tokens)
    except RuntimeError as exc:
        manager.note(str(exc))
        raise _http_error(exc) from exc


@app.post("/api/bench/bw")
async def bench_bw(request: Request) -> dict[str, Any]:
    manager: ProcessManager = request.app.state.manager
    try:
        return await manager.run_bandwidth_bench()
    except RuntimeError as exc:
        manager.note(str(exc))
        raise _http_error(exc) from exc


@app.post("/api/reload")
async def reload_config(request: Request) -> dict[str, Any]:
    try:
        config = load_config()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid config: {exc}") from exc
    request.app.state.config = config
    manager: ProcessManager = request.app.state.manager
    manager.config = config
    manager.note("reloaded config/models.yaml")
    return {"ok": True, "models": manager.inventory()}


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    hub: EventHub = request.app.state.hub
    manager: ProcessManager = request.app.state.manager

    async def gen() -> AsyncIterator[str]:
        queue = hub.subscribe()
        try:
            yield _sse("hello", {"ok": True, "panel": manager.panel_info()})
            yield _sse("status", manager.snapshot())
            yield _sse("metrics", request.app.state.latest_metrics)
            yield _sse("logs", {"lines": manager.recent_logs()})
            yield _sse("panel", manager.panel_info())
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=20)
                    yield _sse(event, data)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


def run() -> None:
    config = load_config()
    uvicorn.run(
        "backend.main:app",
        host=config.panel.host,
        port=config.panel.port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    run()
