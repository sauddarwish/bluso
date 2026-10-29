import math
import unittest

from bluso import normalize_logprobs, parse_prometheus


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


if __name__ == '__main__':
    unittest.main()
