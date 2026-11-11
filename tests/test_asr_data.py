import json
import tempfile
import unittest
from pathlib import Path

from khmer_transliteration.asr_data import (
    as_number,
    load_manifest_configuration,
    read_manifest_rows,
    stream_selected_audio,
)


class AsrDataTests(unittest.TestCase):
    def test_as_number_rejects_invalid_values(self):
        self.assertEqual(as_number("1.25"), 1.25)
        self.assertIsNone(as_number("not-a-number"))
        self.assertIsNone(as_number("nan"))

    def test_manifest_rows_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_dir = Path(temporary_directory)
            (manifest_dir / "manifest_summary.json").write_text(
                json.dumps(
                    {
                        "dataset": "example/dataset",
                        "source_split": "train",
                        "revision": "default",
                        "seed": 42,
                        "shuffle_buffer": 100,
                    }
                ),
                encoding="utf-8",
            )
            (manifest_dir / "test.csv").write_text(
                "source_row_index,transcript\n7,hello\n",
                encoding="utf-8",
            )

            rows = read_manifest_rows(manifest_dir, "test", include_split=True)
            configuration = load_manifest_configuration(
                manifest_dir,
                dataset="override/dataset",
                seed=9,
                shuffle_buffer=10,
            )

            self.assertEqual(rows[0]["source_row_index"], 7)
            self.assertEqual(rows[0]["manifest_split"], "test")
            self.assertEqual(configuration["dataset"], "override/dataset")
            self.assertEqual(configuration["seed"], 9)
            self.assertEqual(configuration["shuffle_buffer"], 10)

    def test_complete_audio_cache_skips_dataset_loading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "audio_cache"
            cache_dir.mkdir()
            cache_bytes = b"RIFF-cached-audio"
            (cache_dir / "source_000007.wav").write_bytes(cache_bytes)
            rows = [{"source_row_index": 7, "audio_path": "example.wav"}]
            progress_messages = []

            selected = stream_selected_audio(
                {
                    "dataset": "not-loaded/because-cache-is-complete",
                    "source_split": "train",
                    "seed": 42,
                    "shuffle_buffer": 1,
                },
                rows,
                cache_dir=cache_dir,
                progress_callback=progress_messages.append,
            )

            self.assertEqual(selected[0]["audio_bytes"], cache_bytes)
            self.assertIn("from cache", progress_messages[0])


if __name__ == "__main__":
    unittest.main()
