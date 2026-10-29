import math
import struct
import tempfile
import unittest
from pathlib import Path

from bluso import GGUFReader, normalize_logprobs, parse_prometheus, summarize_gguf


class BlusoTests(unittest.TestCase):
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
