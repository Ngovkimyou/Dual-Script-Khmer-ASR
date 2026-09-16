"""Inspect a streamed Hugging Face Khmer ASR dataset without downloading it."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_text import (
    ASR_NORMALIZATION_VERSION,
    normalize_asr_text,
)


DEFAULT_OUTPUT = ROOT_DIR / "reports" / "asr" / "dataset_inspection.json"


def as_number(value: Any) -> float | None:
    """Convert a value to a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if number == number and number not in (float("inf"), float("-inf")) else None


def feature_description(features: Any) -> dict[str, Any]:
    """Convert dataset feature objects into JSON-safe descriptions."""
    if not features:
        return {}

    descriptions: dict[str, Any] = {}
    for name, feature in features.items():
        try:
            descriptions[name] = feature.to_dict()
        except AttributeError:
            descriptions[name] = repr(feature)
    return descriptions


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return useful duration statistics for the inspected rows."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def audio_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata without retaining decoded audio arrays or bytes."""
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return {"present": audio is not None}

    metadata: dict[str, Any] = {
        "present": bool(audio),
        "path": audio.get("path"),
        "has_bytes": bool(audio.get("bytes")),
        "sampling_rate": audio.get("sampling_rate"),
    }

    duration = as_number(audio.get("duration"))
    if duration is not None:
        metadata["duration"] = duration

    # This is only used if the caller explicitly enables audio decoding.
    array = audio.get("array")
    sampling_rate = as_number(audio.get("sampling_rate"))
    if "duration" not in metadata and array is not None and sampling_rate:
        try:
            metadata["duration"] = len(array) / sampling_rate
        except TypeError:
            pass

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a limited number of rows from a streamed ASR dataset."
    )
    parser.add_argument(
        "--dataset",
        default="Digital-Divide-Data/khmer-speech-dataset",
        help="Hugging Face dataset ID.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to inspect.")
    parser.add_argument(
        "--text-column",
        default="transcript",
        help="Column containing the ground-truth transcript.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300,
        help="Maximum number of streamed rows to inspect (default: 300).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of example rows to include in the report.",
    )
    parser.add_argument(
        "--decode-audio",
        action="store_true",
        help="Allow audio decoding when duration metadata is unavailable.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face dataset revision for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: reports/asr/dataset_inspection.json).",
    )
    args = parser.parse_args()

    if args.max_rows < 1:
        parser.error("--max-rows must be at least 1")
    if args.preview < 0:
        parser.error("--preview cannot be negative")
    return args


def inspect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements-asr.txt'."
        ) from exc

    load_kwargs: dict[str, Any] = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    dataset = load_dataset(**load_kwargs)
    columns = list(getattr(dataset, "column_names", []) or [])
    if "audio" not in columns:
        raise RuntimeError(f"Expected an 'audio' column; available columns: {columns}")
    if args.text_column not in columns:
        raise RuntimeError(
            f"Expected a '{args.text_column}' column; available columns: {columns}"
        )

    decode_mode = "enabled" if args.decode_audio else "disabled"
    if not args.decode_audio:
        # Metadata inspection must not unexpectedly download and decode every clip.
        dataset = dataset.cast_column("audio", Audio(decode=False))

    durations: list[float] = []
    sampling_rates: Counter[str] = Counter()
    speakers: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    transcript_counts: Counter[str] = Counter()
    previews: list[dict[str, Any]] = []
    missing_audio = 0
    empty_transcripts = 0
    inspected_rows = 0

    for row_index, row in enumerate(dataset.take(args.max_rows)):
        inspected_rows += 1
        transcript = str(row.get(args.text_column) or "")
        normalized = normalize_asr_text(transcript)
        transcript_counts[normalized] += 1
        if not normalized:
            empty_transcripts += 1

        audio = audio_metadata(row)
        if not audio.get("present"):
            missing_audio += 1
        if audio.get("sampling_rate") is not None:
            sampling_rates[str(audio["sampling_rate"])] += 1

        duration = as_number(row.get("duration"))
        if duration is None:
            duration = as_number(audio.get("duration"))
        if duration is not None:
            durations.append(duration)

        speaker = str(row.get("speaker_id") or "").strip()
        topic = str(row.get("topic") or "").strip()
        if speaker:
            speakers[speaker] += 1
        if topic:
            topics[topic] += 1

        if len(previews) < args.preview:
            previews.append(
                {
                    "row_index": row_index,
                    "transcript": transcript,
                    "normalized_transcript": normalized,
                    "speaker_id": row.get("speaker_id"),
                    "topic": row.get("topic"),
                    "subtopic": row.get("subtopic"),
                    "paragraph_id": row.get("paragraph_id"),
                    "sentence_id": row.get("sentence_id"),
                    "duration": duration,
                    "audio": audio,
                }
            )

    duplicate_rows = sum(count - 1 for count in transcript_counts.values() if count > 1)
    report = {
        "dataset": args.dataset,
        "revision": args.revision or "default",
        "split": args.split,
        "streaming": True,
        "audio_decode": decode_mode,
        "normalization_version": ASR_NORMALIZATION_VERSION,
        "requested_rows": args.max_rows,
        "inspected_rows": inspected_rows,
        "columns": columns,
        "features": feature_description(getattr(dataset, "features", None)),
        "quality": {
            "missing_audio": missing_audio,
            "empty_transcripts": empty_transcripts,
            "unique_normalized_transcripts": len(transcript_counts),
            "duplicate_transcript_rows": duplicate_rows,
        },
        "duration_seconds": numeric_summary(durations),
        "sampling_rates": dict(sampling_rates),
        "speaker_counts": dict(speakers),
        "topic_counts": dict(topics),
        "preview": previews,
        "note": "This report summarizes only the requested streamed rows; it is not a full-dataset count.",
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = inspect_dataset(args)
    except Exception as exc:
        print(f"Dataset inspection failed: {exc}")
        return 1

    output_path = args.output if args.output.is_absolute() else ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    print(f"Inspected rows: {report['inspected_rows']}")
    print(f"Unique normalized transcripts: {report['quality']['unique_normalized_transcripts']}")
    print(f"Saved report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
