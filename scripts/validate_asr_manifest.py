"""Validate selected ASR manifest rows against their streamed source records."""

from __future__ import annotations

import argparse
import csv
import io
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_text import ASR_NORMALIZATION_VERSION, normalize_asr_text


DEFAULT_MANIFEST_DIR = ROOT_DIR / "data" / "voice"
DEFAULT_OUTPUT = DEFAULT_MANIFEST_DIR / "manifest_validation.json"


def as_number(value: Any) -> float | None:
    """Convert a value to a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def read_manifests(manifest_dir: Path) -> list[dict[str, Any]]:
    """Read all split CSV files from a manifest directory."""
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        path = manifest_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
            for row in csv.DictReader(manifest_file):
                row["manifest_split"] = split_name
                try:
                    row["source_row_index"] = int(row["source_row_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid source_row_index in {path}: {row}") from exc
                rows.append(row)
    return rows


def read_wav_metadata(audio_bytes: bytes | None) -> dict[str, Any]:
    """Read WAV properties without requiring a third-party audio decoder."""
    if not audio_bytes:
        return {"valid": False, "error": "audio bytes are missing"}

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            return {
                "valid": True,
                "channels": wav_file.getnchannels(),
                "sample_width_bytes": wav_file.getsampwidth(),
                "sampling_rate": sample_rate,
                "frames": frames,
                "duration": round(frames / sample_rate, 3) if sample_rate else None,
            }
    except (EOFError, wave.Error) as exc:
        return {"valid": False, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manifest rows by replaying the deterministic shuffled stream."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--dataset", help="Override the dataset ID from manifest_summary.json.")
    parser.add_argument("--split", help="Override the source split from manifest_summary.json.")
    parser.add_argument("--revision", help="Override the dataset revision from manifest_summary.json.")
    parser.add_argument("--seed", type=int, help="Override the shuffle seed from manifest_summary.json.")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        help="Override the shuffle buffer from manifest_summary.json.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements-asr.txt'."
        ) from exc

    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else ROOT_DIR / args.manifest_dir
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_manifests(manifest_dir)
    expected_by_index = {row["source_row_index"]: row for row in rows}
    if len(expected_by_index) != len(rows):
        raise RuntimeError("source_row_index values are duplicated across manifests")

    dataset_id = args.dataset or summary["dataset"]
    source_split = args.split or summary["source_split"]
    revision = args.revision or summary.get("revision")
    seed = args.seed if args.seed is not None else int(summary["seed"])
    shuffle_buffer = (
        args.shuffle_buffer
        if args.shuffle_buffer is not None
        else int(summary["shuffle_buffer"])
    )

    load_kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": source_split,
        "streaming": True,
    }
    if revision and revision != "default":
        load_kwargs["revision"] = revision

    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    shuffled = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    max_index = max(expected_by_index)
    checked_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    found_indices: set[int] = set()
    for stream_index, row in enumerate(shuffled.take(max_index + 1)):
        if stream_index not in expected_by_index:
            continue

        expected = expected_by_index[stream_index]
        found_indices.add(stream_index)
        actual_transcript = str(row.get("transcript") or "")
        actual_audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}
        actual_path = str(actual_audio.get("path") or "")
        actual_duration = as_number(row.get("duration"))
        audio_info = read_wav_metadata(actual_audio.get("bytes"))

        checks = {
            "audio_path_matches": actual_path == expected["audio_path"],
            "transcript_matches": normalize_asr_text(actual_transcript)
            == expected["normalized_transcript"],
            "duration_matches": actual_duration is None
            or as_number(expected.get("duration")) is None
            or abs(actual_duration - as_number(expected["duration"])) <= 0.01,
            "audio_is_valid_wav": audio_info["valid"],
        }
        if not all(checks.values()):
            mismatches.append(
                {
                    "source_row_index": stream_index,
                    "manifest_split": expected["manifest_split"],
                    "checks": checks,
                    "expected_audio_path": expected["audio_path"],
                    "actual_audio_path": actual_path,
                    "audio": audio_info,
                }
            )

        checked_rows.append(
            {
                "source_row_index": stream_index,
                "manifest_split": expected["manifest_split"],
                "audio": audio_info,
                "checks": checks,
            }
        )

    missing_indices = sorted(set(expected_by_index) - found_indices)
    split_counts = Counter(row["manifest_split"] for row in checked_rows)
    sample_rates = Counter(
        str(row["audio"].get("sampling_rate"))
        for row in checked_rows
        if row["audio"].get("sampling_rate") is not None
    )
    return {
        "dataset": dataset_id,
        "revision": revision or "default",
        "source_split": source_split,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "normalization_version": ASR_NORMALIZATION_VERSION,
        "manifest_rows": len(rows),
        "checked_rows": len(checked_rows),
        "missing_source_rows": missing_indices,
        "mismatch_count": len(mismatches),
        "split_counts": dict(split_counts),
        "sampling_rates": dict(sample_rates),
        "valid_wav_count": sum(row["audio"].get("valid", False) for row in checked_rows),
        "mismatches": mismatches,
        "note": "Validation replays the deterministic shuffled stream and decodes selected WAV headers only.",
    }


def main() -> int:
    args = parse_args()
    try:
        report = validate(args)
    except Exception as exc:
        print(f"Manifest validation failed: {exc}")
        return 1

    output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest rows: {report['manifest_rows']}")
    print(f"Checked rows: {report['checked_rows']}")
    print(f"Valid WAV files: {report['valid_wav_count']}")
    print(f"Mismatches: {report['mismatch_count']}")
    print(f"Saved report to: {output_path}")
    return 0 if not report["missing_source_rows"] and not report["mismatch_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
