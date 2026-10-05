"""Vercel Python Function. No Streamlit session or writable filesystem required."""
import json
import logging
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from cloud.service import BusyError, execute


class handler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        action = parse_qs(urlparse(self.path).query).get('action', ['health'])[0]
        if action not in {'health', 'defaults'}:
            self.respond(405, {'error': 'Use POST for computations.'})
            return
        self.run_action(action, {})

    def do_POST(self):
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            self.respond(415, {'error': 'Content-Type must be application/json.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 100000:
                self.respond(413, {'error': 'Request must be 1–100000 bytes.'})
                return
            data = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self.respond(400, {'error': 'Invalid JSON request.'})
            return
        action = parse_qs(urlparse(self.path).query).get('action', [''])[0]
        self.run_action(action, data)

    def run_action(self, action, data):
        try:
            payload = execute(action, data)
            status = 200
        except ValueError as exc:
            status, payload = 400, {'error': str(exc)}
        except BusyError as exc:
            status, payload = 429, {'error': str(exc)}
        except Exception:
            logging.exception('Fleet API failure: %s', action)
            status, payload = 503, {'error': 'Model or solver unavailable. Check deployment logs and model verification.'}
        try:
            self.respond(status, payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A closed browser connection is not a model failure. Never send a
            # second response after headers or body were already written.
            logging.info('Client disconnected during %s response', action)
