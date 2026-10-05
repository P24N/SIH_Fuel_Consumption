import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from cloud import service
from api.index import handler


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.data = copy.deepcopy(service.execute('defaults', {}))
        self.data.update(generations=5, population=5)

    def test_model_cold_start_and_arbitrary_working_directory(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            try:
                os.chdir(path)
                service.model.cache_clear()
                self.assertEqual(service.execute('health', {})['model'], 'verified')
            finally:
                os.chdir(cwd)

    def test_reject_model_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'model_manifest.json').write_bytes((service.ROOT / 'model_manifest.json').read_bytes())
            (path / 'qirf_fuel_predictor.joblib').write_bytes(b'corrupted model')
            service.model.cache_clear()
            with patch.object(service, 'ROOT', path), self.assertRaisesRegex(RuntimeError, 'checksum'):
                service.model()
        service.model.cache_clear()

    def test_all_operations_produce_strict_json(self):
        for action in ['qga', 'milp', 'fuels', 'stress', 'dynamic']:
            with self.subTest(action=action):
                result = service.execute(action, dict(self.data, n_routes=3, n_vessels=4))
                json.dumps(result, allow_nan=False)
                self.assertTrue(result.get('rows') or result.get('qga'))

    def test_invalid_inputs(self):
        for update in [{'fleet': []}, {'weights': [1, 2]}, {'carbon_tax': float('nan')},
                       {'generations': 10000}, {'population': True}, {'routes': [None]}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                service.execute('qga', dict(self.data, **update))
        with self.assertRaises(ValueError):
            service.execute('dynamic', {'n_routes': 100})

    def test_infeasible_results_are_flagged(self):
        for route in self.data['routes']:
            route['max_arrival_hours'] = 1
        result = service.execute('qga', self.data)
        self.assertTrue(result['warnings'])
        self.assertTrue(all(not r['Feasible'] for r in result['rows']))

    def test_no_cross_request_mutation(self):
        before = copy.deepcopy(self.data)
        service.execute('fuels', self.data)
        self.assertEqual(self.data, before)

    def test_busy_worker(self):
        with service.SOLVER_LOCK:
            with self.assertRaises(service.BusyError):
                service.execute('qga', self.data)

    def test_http_contract(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}/api'
        try:
            with urllib.request.urlopen(base + '?action=health') as response:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(json.load(response)['status'], 'ok')
            request = urllib.request.Request(base + '?action=qga', data=json.dumps(self.data).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(len(json.load(response)['rows']), len(self.data['routes']))
            for data, content_type, expected in [(b'[]', 'application/json', 400),
                                                   (b'{', 'application/json', 400),
                                                   (b'{}', 'text/plain', 415)]:
                request = urllib.request.Request(base + '?action=qga', data=data, headers={'Content-Type': content_type})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(request)
                self.assertEqual(cm.exception.code, expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
