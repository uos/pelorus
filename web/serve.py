#!/usr/bin/env python3
"""Serve the demo folder for development, with browser caching turned off.

Browsers keep module workers and wasm cached, so after a rebuild a plain
static server can mix old and new files. Usage: web/serve.py [port]
"""

import functools
import http.server
import pathlib
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(NoCacheHandler, directory=pathlib.Path(__file__).parent)
print(f"Serving the Pelorus demo on http://localhost:{port}")
http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
