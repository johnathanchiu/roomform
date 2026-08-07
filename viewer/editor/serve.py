"""Dev server for the production editor.

  uv run python viewer/editor/serve.py [--artifacts artifacts]
      [--port 8792]

Serves the prebuilt editor (viewer/editor/dist) at /, the artifacts
directory at /artifacts/, a scene listing at /api/scenes, and accepts
POST /api/scenes/<id>/save with a full SceneDocument body, which is
pydantic-validated before overwriting artifacts/<id>/scene.json.
Everything goes out with ``Cache-Control: no-cache`` so freshly
produced scenes appear on reload.
"""

from __future__ import annotations

import argparse
import functools
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from roomform.contracts import SceneDocument

DIST = Path(__file__).resolve().parent / "dist"


class Handler(SimpleHTTPRequestHandler):
    artifacts: Path

    def do_GET(self):
        if self.path.split("?")[0] == "/api/scenes":
            scenes = sorted(
                p.parent.name for p in self.artifacts.glob("*/scene.json")
            )
            self._send(200, "application/json", json.dumps(scenes).encode())
        elif self.path.startswith("/artifacts/"):
            self._send_artifact(head=False)
        else:
            super().do_GET()  # the built app from dist/

    def do_HEAD(self):
        if self.path.startswith("/artifacts/"):
            self._send_artifact(head=True)
        else:
            super().do_HEAD()

    def do_POST(self):
        parts = self.path.split("?")[0].strip("/").split("/")
        # /api/scenes/<id>/save
        if (
            len(parts) != 4
            or parts[:2] != ["api", "scenes"]
            or parts[3] != "save"
        ):
            self._send(404, "text/plain", b"not found")
            return
        target = self._resolve(parts[2] + "/scene.json")
        if target is None or not target.is_file():
            self._send(404, "text/plain", b"unknown scene")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            doc = SceneDocument(**json.loads(body))
        except (ValueError, TypeError) as err:
            self._send(422, "text/plain", str(err).encode())
            return
        target.write_text(doc.model_dump_json(indent=1))
        self._send(200, "application/json", b'{"ok": true}')

    def _resolve(self, rel: str) -> Path | None:
        target = (self.artifacts / rel).resolve()
        if not str(target).startswith(str(self.artifacts.resolve())):
            return None
        return target

    def _send_artifact(self, head: bool) -> None:
        rel = self.path.split("?")[0][len("/artifacts/") :]
        target = self._resolve(rel)
        if target is None:
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
        self._send(200, ctype, target.read_bytes() if not head else None)

    def _send(self, code: int, ctype: str, body: bytes | None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if body is not None:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body is not None:
            self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(exist_ok=True)
    Handler.artifacts = artifacts
    server = HTTPServer(
        ("127.0.0.1", args.port),
        functools.partial(Handler, directory=str(DIST)),
    )
    print(
        f"roomform editor: http://127.0.0.1:{args.port}"
        f"  (artifacts: {artifacts.resolve()})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
