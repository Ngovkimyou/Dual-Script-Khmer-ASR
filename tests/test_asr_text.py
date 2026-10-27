import unittest

from khmer_transliteration.asr_text import normalize_asr_text


class AsrTextTests(unittest.TestCase):
    def test_normalization_trims_and_collapses_whitespace(self):
        self.assertEqual(normalize_asr_text("  hello\t world\n"), "hello world")

    def test_normalization_handles_none(self):
        self.assertEqual(normalize_asr_text(None), "")


if __name__ == "__main__":
    unittest.main()
