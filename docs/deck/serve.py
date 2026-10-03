#!/usr/bin/env python3
"""Local slide server: serves the deck and auto-saves Google-Docs-style comments.

Usage:
    python3 docs/deck/serve.py           # http://localhost:8780/
    python3 docs/deck/serve.py 9000      # custom port

Select text on a slide -> "+ comment". Every add / resolve / delete is POSTed to
    POST /api/comments/index.html  ->  docs/deck/comments-index.json
and reloads read it back via GET. Each comment stores the quote, ~50 chars of
context, the slide number + title and the highlighted element's text, so Claude
can "address comments" from the JSON alone.
"""
import http.server
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def end_headers(self):
        # Disable caching so edits to comments.js / comments.css show up on refresh
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        # Permissive CORS — local-only server, anyone running this trusts their own machine
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/api/comments/'):
            return self._get_comments()
        if self.path in ('/', ''):
            self.path = '/index.html'
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/comments/'):
            return self._post_comments()
        self.send_response(404)
        self.end_headers()

    # ── API: comments ──────────────────────────────────────────
    def _get_comments(self):
        doc = self.path[len('/api/comments/'):]
        json_path = self._json_path(doc)
        if not json_path:
            self.send_response(400)
            self.end_headers()
            return
        if not json_path.exists():
            data = b'{"comments":[]}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = json_path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _post_comments(self):
        doc = self.path[len('/api/comments/'):]
        json_path = self._json_path(doc)
        if not json_path:
            self.send_response(400)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body)
            assert isinstance(parsed, dict) and 'comments' in parsed
        except (json.JSONDecodeError, AssertionError):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"invalid payload"}')
            return
        # Pretty-print to disk for diff-friendly storage
        json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding='utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _json_path(self, doc):
        # Sanitize against path traversal — only allow a bare filename
        doc = os.path.basename(doc)
        if not doc or '..' in doc or '/' in doc or '\\' in doc:
            return None
        # Strip the .html / .htm suffix to form comments-<base>.json
        base = doc[:-5] if doc.endswith('.html') else (doc[:-4] if doc.endswith('.htm') else doc)
        if not base:
            return None
        return PROJECT_DIR / f'comments-{base}.json'

    def log_message(self, format, *args):
        if '/api/' in (args[0] if args else ''):
            sys.stderr.write(f"[deck] {format % args}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    print(f'deck: serving {PROJECT_DIR}', file=sys.stderr)
    print(f'  http://localhost:{port}/', file=sys.stderr)
    print(f'Comments auto-save to {PROJECT_DIR / "comments-index.json"}. Ctrl-C to stop.', file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down', file=sys.stderr)


if __name__ == '__main__':
    main()
