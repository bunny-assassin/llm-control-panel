"""Minimal OpenAI-compatible stub used by process-manager tests."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path in {"/v1/models", "/v1/models/"}:
            self._send(200, {"object": "list", "data": [{"id": "dummy-model", "object": "model"}]})
            return
        if path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if path in {"/v1/chat/completions", "/v1/chat/completions/"}:
            self._send(
                200,
                {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
                },
            )
            return
        self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[dummy] " + (fmt % args) + "\n")
        sys.stdout.flush()


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    server = ThreadingHTTPServer((host, port), Handler)
    bound = server.server_address[1]
    print(f"API server is ready to serve on {host}:{bound}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
