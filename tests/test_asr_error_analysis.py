import unittest

from scripts.analyze_asr_errors import analyze_report, grapheme_clusters


class AsrErrorAnalysisTests(unittest.TestCase):
    def test_grapheme_clusters_keep_a_khmer_subscript_together(self):
        text = "".join(chr(codepoint) for codepoint in (0x1785, 0x1784, 0x17D2, 0x17A2, 0x17BB, 0x179B))
        clusters = grapheme_clusters(text)
        self.assertEqual(len(clusters), 3)
        self.assertIn("\u17d2", clusters[1])

    def test_report_separates_subscript_heavy_rows(self):
        report = {
            "model": "example",
            "split": "test",
            "normalization_version": "test",
            "metrics": {},
            "predictions": [
                {
                    "source_row_index": 1,
                    "reference": "ក",
                    "prediction": "ក",
                },
                {
                    "source_row_index": 2,
                    "reference": "ក\u17d2ខ\u17d2គ\u17d2ឃ",
                    "prediction": "ក\u17d2ខ",
                },
            ],
        }
        analysis = analyze_report(report, minimum_coeng_count=3)
        self.assertEqual(analysis["all_rows"]["sample_count"], 2)
        self.assertEqual(analysis["subscript_heavy_rows"]["sample_count"], 1)
        self.assertEqual(analysis["all_rows"]["rows_with_coeng_count_mismatch"], 1)


if __name__ == "__main__":
    unittest.main()
