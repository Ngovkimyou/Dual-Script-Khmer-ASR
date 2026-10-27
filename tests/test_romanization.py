import unittest

from khmer_transliteration.romanization import romanize_khmer_text


class RomanizationTests(unittest.TestCase):
    def test_empty_text_stays_empty(self):
        self.assertEqual(romanize_khmer_text(""), "")

    def test_unknown_text_is_preserved(self):
        text = "not-in-the-khmer-dictionary"
        self.assertEqual(romanize_khmer_text(text), text)


if __name__ == "__main__":
    unittest.main()
