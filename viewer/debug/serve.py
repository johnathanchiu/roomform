"""Tiny dev server for the scene viewer.

  uv run python viewer/debug/serve.py [--artifacts artifacts] [--port 8790]

Serves viewer.html at /, the artifacts directory at /artifacts/, and a
scene listing at /api/scenes (directories containing scene.json).
Scenes appear on reload — no restart needed.
"""

from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

VIEWER = Path(__file__).resolve().parent / "index.html"


def make_handler(artifacts: Path):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = VIEWER.read_bytes()
                self._send(200, "text/html", body)
            elif self.path == "/api/scenes":
                scenes = sorted(
                    p.parent.name for p in artifacts.glob("*/scene.json")
                )
                self._send(
                    200,
                    "application/json",
                    json.dumps(scenes).encode(),
                )
            elif self.path.startswith("/artifacts/"):
                rel = self.path[len("/artifacts/") :]
                target = (artifacts / rel).resolve()
                if not str(target).startswith(str(artifacts.resolve())):
                    self._send(403, "text/plain", b"forbidden")
                    return
                if not target.is_file():
                    self._send(404, "text/plain", b"not found")
                    return
                ctype = (
                    "application/json"
                    if target.suffix == ".json"
                    else "model/gltf-binary"
                    if target.suffix == ".glb"
                    else "application/octet-stream"
                )
                self._send(200, ctype, target.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(exist_ok=True)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(artifacts))
    print(
        f"roomform viewer: http://127.0.0.1:{args.port}  (artifacts: {artifacts.resolve()})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
