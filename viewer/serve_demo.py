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
from http.server import HTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
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
