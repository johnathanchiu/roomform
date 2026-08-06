"""Serve the demo editor's prebuilt dist with sane dev caching.

  uv run python viewer/serve_demo.py DIST_DIR [--port 8792]

Plain ``python -m http.server`` sends no cache headers, so browsers
heuristically cache fixtures/index.json and freshly exported scenes
never appear. Everything here is served with ``Cache-Control:
no-cache`` — assets still 304 on revalidation, but updates are seen.
"""

from __future__ import annotations

import argparse
import functools
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

# The prebuilt editor is served as-is except for one cosmetic override:
# the right-side inspector panel is hidden (ruling: viewport-first UX;
# injected here so the upstream build stays untouched).
_INJECT = (
    "<style>"
    "[data-group]>[data-panel]:last-child{display:none!important}"
    '[data-group]>[data-slot="resizable-handle"]:last-of-type'
    "{display:none!important}"
    "</style>"
)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            with open(os.path.join(self.directory, "index.html"), "rb") as fh:
                body = fh.read().replace(
                    b"</head>", _INJECT.encode() + b"</head>"
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dist_dir")
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    server = HTTPServer(
        ("127.0.0.1", args.port),
        functools.partial(Handler, directory=args.dist_dir),
    )
    print(f"demo editor: http://127.0.0.1:{args.port} ({args.dist_dir})")
    server.serve_forever()


if __name__ == "__main__":
    main()
