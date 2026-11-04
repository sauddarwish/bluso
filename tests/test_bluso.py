import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bluso
from bluso import (GGUFReader, __version__, default_model, health_payload,
                   managed_runtime_status, normalize_logprobs, parse_prometheus,
                   start_llamacpp, stop_llamacpp, summarize_gguf)


class BlusoTests(unittest.TestCase):
    def test_managed_runtime_status_without_process(self):
        with mock.patch.object(bluso, 'RUNTIME_PROCESS', None):
            self.assertEqual(managed_runtime_status(), {'managed': False, 'running': False})

    def test_start_does_not_duplicate_managed_runtime(self):
        process = mock.Mock(pid=4242)
        process.poll.return_value = None
        with mock.patch.object(bluso, 'RUNTIME_PROCESS', process):
            result = start_llamacpp()
        self.assertEqual(result, {'ok': True, 'state': 'already-starting', 'pid': 4242})

    def test_stop_rejects_unmanaged_runtime(self):
        with mock.patch.object(bluso, 'RUNTIME_PROCESS', None):
            result = stop_llamacpp()
        self.assertFalse(result['ok'])
        self.assertEqual(result['state'], 'not-managed')

    def test_stop_managed_runtime_terminates_process_group(self):
        class FakeProcess:
            pid = 4242

            def poll(self):
                return None

            def wait(self, timeout=None):
                self.timeout = timeout
                return 0

        process = FakeProcess()
        with mock.patch.object(bluso, 'RUNTIME_PROCESS', process), mock.patch.object(bluso.os, 'killpg') as killpg:
            result = stop_llamacpp(timeout=0.1)
            killpg.assert_called_once_with(process.pid, bluso.signal.SIGTERM)
            self.assertEqual(result['state'], 'stopped')
            self.assertFalse(result['forced'])
            self.assertIsNone(bluso.RUNTIME_PROCESS)

    def test_stop_force_kills_runtime_after_timeout(self):
        process = mock.Mock(pid=4242)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('llama-server', 0.1), -9]
        with mock.patch.object(bluso, 'RUNTIME_PROCESS', process), mock.patch.object(bluso.os, 'killpg') as killpg:
            result = stop_llamacpp(timeout=0.1)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(process.pid, bluso.signal.SIGTERM), mock.call(process.pid, bluso.signal.SIGKILL)],
        )
        self.assertTrue(result['forced'])
        self.assertEqual(result['exit_code'], -9)

    def test_health_payload_reports_current_version(self):
        payload = health_payload()
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['version'], __version__)
        self.assertIsInstance(payload['time'], int)

    def test_cli_reports_version(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parents[1] / 'bluso.py'), '--version'],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), f'Bluso {__version__}')

    def test_default_model_uses_first_discovered_model(self):
        discovery = {'models': [
            {'name': 'model-a', 'path': '/models/a.gguf'},
            {'name': 'model-b', 'path': '/models/b.gguf'},
        ]}
        self.assertEqual(default_model(discovery), '/models/a.gguf')

    def test_default_model_handles_empty_discovery(self):
        self.assertIsNone(default_model({'models': []}))

    def test_prometheus_parser(self):
        result = parse_prometheus('# HELP x\nllamacpp:predicted_tokens_seconds 12.5\nmetric{slot="0"} 3\n')
        self.assertEqual(result['llamacpp:predicted_tokens_seconds'], 12.5)
        self.assertEqual(result['metric'], 3.0)

    def test_logprob_normalization(self):
        source = {'content': [{'token': ' blue', 'id': 42, 'logprob': math.log(.7), 'top_logprobs': [
            {'token': ' blue', 'id': 42, 'logprob': math.log(.7)},
            {'token': ' red', 'id': 9, 'logprob': math.log(.2)},
        ]}]}
        result = normalize_logprobs(source)
        self.assertAlmostEqual(result['probability'], .7)
        self.assertEqual(result['alternatives'][1]['token'], ' red')

    def test_gguf_tensor_directory(self):
        def text(value):
            data = value.encode('utf-8')
            return struct.pack('<Q', len(data)) + data

        header = b'GGUF' + struct.pack('<IQQ', 3, 1, 2)
        metadata = (
            text('general.architecture') + struct.pack('<I', 8) + text('test') +
            text('general.alignment') + struct.pack('<II', 4, 32)
        )
        tensor = text('weight') + struct.pack('<IQQIQ', 2, 2, 3, 0, 0)
        descriptor_end = len(header + metadata + tensor)
        padding = b'\0' * ((32 - descriptor_end % 32) % 32)
        payload = b'\0' * 24
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            path.write_bytes(header + metadata + tensor + padding + payload)
            parsed = GGUFReader(path).read()
            self.assertEqual(parsed['tensors'][0]['shape'], [2, 3])
            self.assertEqual(parsed['tensors'][0]['elements'], 6)
            self.assertEqual(parsed['tensors'][0]['stored_bytes'], 24)
            summary = summarize_gguf(path)
            self.assertEqual(summary['parameters'], 6)
            self.assertEqual(summary['tensor_types'][0]['type'], 'F32')


if __name__ == '__main__':
    unittest.main()
