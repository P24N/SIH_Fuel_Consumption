"""Local equivalent of the static frontend + Python API on Vercel."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.index import handler


class LocalHandler(handler, SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == '/api':
            return handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if urlparse(self.path).path != '/api':
            return self.respond(404, {'error': 'Not found'})
        return handler.do_POST(self)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(LocalHandler, directory=str(ROOT / 'public')))
    print(f'Fleet dashboard: http://127.0.0.1:{port}', flush=True)
    server.serve_forever()
