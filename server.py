#!/usr/bin/env python3
"""
server.py — serve ONLY the viewer/ folder on port 4700.

Standard library only. Usage:
    python3 build.py   # generate viewer/graph-data.js first
    python3 server.py  # then open http://localhost:4700
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 4700
VIEWER_DIR = Path(__file__).parent / "viewer"


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serves files from viewer/ only; directory is pinned via `directory`."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[viewer] {self.address_string()} — {fmt % args}")


def main() -> None:
    if not VIEWER_DIR.is_dir():
        raise SystemExit(f"Viewer folder not found: {VIEWER_DIR}")
    if not (VIEWER_DIR / "graph-data.js").exists():
        print("Warning: viewer/graph-data.js missing — run `python3 build.py` first.")

    handler = partial(ViewerHandler, directory=str(VIEWER_DIR))
    with ThreadingHTTPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Knowledge Galaxy at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
