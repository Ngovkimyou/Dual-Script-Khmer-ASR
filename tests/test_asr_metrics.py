import unittest

from khmer_transliteration.asr_metrics import (
    calculate_metrics,
    calculate_prediction_diagnostics,
)


class AsrMetricsTests(unittest.TestCase):
    def test_empty_metrics_are_explicitly_empty(self):
        metrics = calculate_metrics([])
        self.assertIsNone(metrics["cer"])
        self.assertIsNone(metrics["wer"])
        self.assertIsNone(metrics["sentence_accuracy"])

    def test_diagnostics_identify_unknown_and_subscript_rows(self):
        predictions = [
            {
                "reference": "ក ច\u17d2",
                "prediction": "ក[UNK]",
                "latency_seconds": 0.1,
                "real_time_factor": 0.01,
            },
            {
                "reference": "ក",
                "prediction": "ក",
                "latency_seconds": 0.1,
                "real_time_factor": 0.01,
            },
        ]
        diagnostics = calculate_prediction_diagnostics(predictions)
        self.assertEqual(diagnostics["prediction_contains_unk_count"], 1)
        self.assertEqual(diagnostics["prediction_trailing_unk_count"], 1)
        self.assertEqual(diagnostics["prediction_without_spaces_count"], 1)
        self.assertEqual(diagnostics["subscript_reference"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
