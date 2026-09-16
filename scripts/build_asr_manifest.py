"""Build a small, balanced, speaker-disjoint ASR pilot manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "voice"


def as_number(value: Any) -> float | None:
    """Convert a duration-like value to a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def duration_bucket(duration: float | None) -> str:
    """Classify clips so selection can preserve short, medium, and long audio."""
    if duration is None:
        return "unknown"
    if duration < 5:
        return "short_lt_5s"
    if duration <= 10:
        return "medium_5_to_10s"
    return "long_gt_10s"


def audio_path(row: dict[str, Any]) -> str:
    """Return the source audio identifier without retaining audio bytes."""
    audio = row.get("audio")
    if isinstance(audio, dict):
        return str(audio.get("path") or "")
    return ""


def make_candidate(row_index: int, row: dict[str, Any], dataset_id: str, split: str) -> dict[str, Any] | None:
    """Convert a streamed row into a compact manifest record."""
    transcript = str(row.get("transcript") or "")
    normalized = normalize_asr_text(transcript)
    source_audio = audio_path(row)
    if not normalized or not source_audio:
        return None

    duration = as_number(row.get("duration"))
    speaker = str(row.get("speaker_id") or "").strip() or "unknown"
    topic = str(row.get("topic") or "").strip() or "unknown"
    subtopic = str(row.get("subtopic") or "").strip() or "unknown"
    sentence_id = str(row.get("sentence_id") or "").strip()
    paragraph_id = row.get("paragraph_id")

    # Prefer the dataset's sentence identity. The normalized transcript remains
    # part of the key to guard against malformed or reused sentence IDs.
    group_key = sentence_id or normalized
    if paragraph_id is not None:
        group_key = f"{paragraph_id}:{group_key}"

    return {
        "dataset_id": dataset_id,
        "dataset_split": split,
        "source_row_index": row_index,
        "audio_path": source_audio,
        "transcript": transcript,
        "normalized_transcript": normalized,
        "speaker_id": speaker,
        "topic": topic,
        "subtopic": subtopic,
        "paragraph_id": paragraph_id,
        "sentence_id": sentence_id,
        "duration": duration,
        "duration_bucket": duration_bucket(duration),
        "group_key": group_key,
    }


def collect_candidates(dataset: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    """Collect unique transcript groups from a deterministic shuffled stream."""
    candidates_by_group: dict[str, dict[str, Any]] = {}
    scanned = 0

    for row_index, row in enumerate(dataset.take(args.scan_rows)):
        scanned += 1
        candidate = make_candidate(row_index, row, args.dataset, args.split)
        if candidate is None:
            continue
        candidates_by_group.setdefault(candidate["group_key"], candidate)

    return list(candidates_by_group.values()), scanned


def choose_balanced_candidates(candidates: list[dict[str, Any]], target_rows: int, seed: int) -> list[dict[str, Any]]:
    """Greedily select rows while balancing speakers, topics, and durations."""
    rng = random.Random(seed)
    remaining = list(candidates)
    tie_breakers = {id(candidate): rng.random() for candidate in remaining}
    selected: list[dict[str, Any]] = []
    speaker_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    duration_counts: Counter[str] = Counter()

    while remaining and len(selected) < target_rows:
        remaining.sort(
            key=lambda candidate: (
                speaker_counts[candidate["speaker_id"]],
                topic_counts[candidate["topic"]],
                duration_counts[candidate["duration_bucket"]],
                tie_breakers[id(candidate)],
            )
        )
        candidate = remaining.pop(0)
        selected.append(candidate)
        speaker_counts[candidate["speaker_id"]] += 1
        topic_counts[candidate["topic"]] += 1
        duration_counts[candidate["duration_bucket"]] += 1

    return selected


def speaker_partition(
    rows: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Assign complete speakers to splits with approximately requested sizes."""
    rows_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[row["speaker_id"]].append(row)

    speakers = list(rows_by_speaker)
    if len(speakers) < 3:
        raise RuntimeError(
            "At least three speakers are required for speaker-disjoint train, "
            "validation, and test splits."
        )

    rng = random.Random(seed)
    rng.shuffle(speakers)
    total = len(rows)
    targets = {
        "train": total * train_ratio,
        "validation": total * validation_ratio,
        "test": total * test_ratio,
    }
    assignments: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    counts = {name: 0 for name in assignments}

    # Put each speaker into the currently most-underfilled split, while leaving
    # at least one speaker available for each later split.
    for speaker_index, speaker in enumerate(speakers):
        remaining_speakers = len(speakers) - speaker_index - 1
        empty_split_names = [name for name in assignments if not assignments[name]]
        if empty_split_names and remaining_speakers == len(empty_split_names):
            available = empty_split_names
        else:
            available = list(assignments)

        split_name = min(
            available,
            key=lambda name: (counts[name] - targets[name], rng.random()),
        )
        assignments[split_name].append(speaker)
        counts[split_name] += len(rows_by_speaker[speaker])

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split_name, split_speakers in assignments.items():
        for speaker in split_speakers:
            result[split_name].extend(rows_by_speaker[speaker])
        result[split_name].sort(key=lambda row: row["audio_path"])
    return result


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one manifest CSV without audio bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "dataset_split",
        "source_row_index",
        "audio_path",
        "transcript",
        "normalized_transcript",
        "speaker_id",
        "topic",
        "subtopic",
        "paragraph_id",
        "sentence_id",
        "duration",
        "duration_bucket",
        "group_key",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a balanced pilot subset from a streamed ASR dataset."
    )
    parser.add_argument(
        "--dataset",
        default="Digital-Divide-Data/khmer-speech-dataset",
        help="Hugging Face dataset ID.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to stream.")
    parser.add_argument("--target-rows", type=int, default=300)
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=5000,
        help="Maximum shuffled rows to inspect while collecting candidates.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.target_rows < 3:
        parser.error("--target-rows must be at least 3")
    if args.scan_rows < args.target_rows:
        parser.error("--scan-rows must be at least --target-rows")
    if args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be at least 1")
    ratios = [args.train_ratio, args.validation_ratio, args.test_ratio]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("train, validation, and test ratios must be positive and sum to 1")
    return args


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
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
    required_columns = {"audio", "transcript", "speaker_id"}
    missing_columns = sorted(required_columns - set(columns))
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {missing_columns}; available: {columns}")

    dataset = dataset.cast_column("audio", Audio(decode=False))
    shuffled = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    candidates, scanned_rows = collect_candidates(shuffled, args)
    if len(candidates) < args.target_rows:
        raise RuntimeError(
            f"Only {len(candidates)} unique candidate groups were found after scanning "
            f"{scanned_rows} rows; increase --scan-rows."
        )

    selected = choose_balanced_candidates(candidates, args.target_rows, args.seed)
    split_rows = speaker_partition(
        selected,
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
        args.seed,
    )

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir
    for split_name, rows in split_rows.items():
        write_manifest(output_dir / f"{split_name}.csv", rows)

    summary = {
        "dataset": args.dataset,
        "revision": args.revision or "default",
        "normalization_version": ASR_NORMALIZATION_VERSION,
        "source_split": args.split,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "scanned_rows": scanned_rows,
        "candidate_groups": len(candidates),
        "selected_rows": len(selected),
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "speaker_counts": {
            name: dict(Counter(row["speaker_id"] for row in rows))
            for name, rows in split_rows.items()
        },
        "topic_counts": {
            name: dict(Counter(row["topic"] for row in rows))
            for name, rows in split_rows.items()
        },
        "duration_bucket_counts": {
            name: dict(Counter(row["duration_bucket"] for row in rows))
            for name, rows in split_rows.items()
        },
        "group_overlap": {
            "train_validation": len(
                {row["group_key"] for row in split_rows["train"]}
                & {row["group_key"] for row in split_rows["validation"]}
            ),
            "train_test": len(
                {row["group_key"] for row in split_rows["train"]}
                & {row["group_key"] for row in split_rows["test"]}
            ),
            "validation_test": len(
                {row["group_key"] for row in split_rows["validation"]}
                & {row["group_key"] for row in split_rows["test"]}
            ),
        },
        "note": "Rows are selected with one record per group and assigned by complete speaker groups.",
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = build_manifests(args)
    except Exception as exc:
        print(f"Manifest creation failed: {exc}")
        return 1

    print(f"Scanned rows: {summary['scanned_rows']}")
    print(f"Selected rows: {summary['selected_rows']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Saved manifests to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
