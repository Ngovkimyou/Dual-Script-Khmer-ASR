"""Build balanced speaker-disjoint ASR manifests, optionally with a fixed test split."""

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
from khmer_transliteration.asr_data import (
    as_number,
    load_manifest_configuration,
    read_manifest_rows,
    resolve_path,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "voice"


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
        if empty_split_names and remaining_speakers + 1 == len(empty_split_names):
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


def speaker_partition_train_validation(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows by speaker when a fixed test set is supplied separately."""
    rows_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[row["speaker_id"]].append(row)

    speakers = list(rows_by_speaker)
    if len(speakers) < 2:
        raise RuntimeError(
            "At least two non-test speakers are required for train and validation."
        )

    rng = random.Random(seed)
    rng.shuffle(speakers)
    total = len(rows)
    targets = {
        "train": total * (1.0 - validation_ratio),
        "validation": total * validation_ratio,
    }
    assignments: dict[str, list[str]] = {"train": [], "validation": []}
    counts = {name: 0 for name in assignments}

    for speaker_index, speaker in enumerate(speakers):
        remaining_speakers = len(speakers) - speaker_index - 1
        empty_splits = [name for name, values in assignments.items() if not values]
        if empty_splits and remaining_speakers + 1 == len(empty_splits):
            available = empty_splits
        else:
            available = list(assignments)
        split_name = min(
            available,
            key=lambda name: (counts[name] - targets[name], rng.random()),
        )
        assignments[split_name].append(speaker)
        counts[split_name] += len(rows_by_speaker[speaker])

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
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
        description="Build balanced speaker-disjoint ASR manifests from a streamed dataset."
    )
    parser.add_argument(
        "--dataset",
        help="Hugging Face dataset ID. In fixed-test mode, defaults to the test manifest's dataset.",
    )
    parser.add_argument("--split", help="Dataset split to stream.")
    parser.add_argument("--target-rows", type=int, default=300)
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=5000,
        help="Maximum shuffled rows to inspect while collecting candidates.",
    )
    parser.add_argument("--shuffle-buffer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision.")
    parser.add_argument(
        "--fixed-test-manifest-dir",
        type=Path,
        help=(
            "Keep the test.csv from this existing manifest directory unchanged, "
            "exclude its speakers and transcript groups from new train/validation rows, "
            "and build a larger total dataset around that fixed test set."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.fixed_test_manifest_dir is None:
        args.dataset = args.dataset or "Digital-Divide-Data/khmer-speech-dataset"
        args.split = args.split or "train"
        args.shuffle_buffer = (
            args.shuffle_buffer if args.shuffle_buffer is not None else 10000
        )
        args.seed = args.seed if args.seed is not None else 42

    if args.target_rows < 3:
        parser.error("--target-rows must be at least 3")
    if args.scan_rows < args.target_rows:
        parser.error("--scan-rows must be at least --target-rows")
    if args.shuffle_buffer is not None and args.shuffle_buffer < 1:
        parser.error("--shuffle-buffer must be at least 1")
    ratios = [args.train_ratio, args.validation_ratio, args.test_ratio]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("train, validation, and test ratios must be positive and sum to 1")
    return args


def build_manifests(args: argparse.Namespace) -> dict[str, Any]:
    fixed_test_rows: list[dict[str, Any]] | None = None
    fixed_test_dir: Path | None = None
    if args.fixed_test_manifest_dir is not None:
        fixed_test_dir = resolve_path(args.fixed_test_manifest_dir)
        fixed_configuration = load_manifest_configuration(fixed_test_dir)
        expected = {
            "dataset": fixed_configuration["dataset"],
            "split": fixed_configuration["source_split"],
            "seed": fixed_configuration["seed"],
            "shuffle_buffer": fixed_configuration["shuffle_buffer"],
        }
        for argument_name, expected_value in expected.items():
            argument_value = getattr(args, argument_name)
            if argument_value is not None and argument_value != expected_value:
                raise ValueError(
                    f"--{argument_name.replace('_', '-')}={argument_value!r} does not "
                    f"match the fixed test manifest value {expected_value!r}."
                )
            setattr(args, argument_name, expected_value)
        manifest_revision = fixed_configuration.get("revision")
        if args.revision is not None and args.revision != manifest_revision:
            raise ValueError(
                f"--revision={args.revision!r} does not match the fixed test manifest "
                f"revision {manifest_revision!r}."
            )
        if args.revision is None and manifest_revision not in (None, "default"):
            args.revision = manifest_revision
        elif args.revision == "default":
            args.revision = None
        fixed_test_rows = read_manifest_rows(fixed_test_dir, "test")
        if len(fixed_test_rows) >= args.target_rows:
            raise ValueError(
                "--target-rows must be larger than the fixed test split "
                f"({len(fixed_test_rows)} rows)."
            )
        if any(not str(row.get("speaker_id") or "").strip() for row in fixed_test_rows):
            raise ValueError("The fixed test split contains a row without a speaker ID.")
    else:
        args.dataset = args.dataset or "Digital-Divide-Data/khmer-speech-dataset"
        args.split = args.split or "train"
        args.shuffle_buffer = (
            args.shuffle_buffer if args.shuffle_buffer is not None else 10000
        )
        args.seed = args.seed if args.seed is not None else 42

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
    required_candidates = args.target_rows - (
        len(fixed_test_rows) if fixed_test_rows is not None else 0
    )
    if len(candidates) < required_candidates:
        raise RuntimeError(
            f"Only {len(candidates)} unique candidate groups were found after scanning "
            f"{scanned_rows} rows; {required_candidates} are needed before fixed-test "
            "exclusions, so increase --scan-rows."
        )

    if fixed_test_rows is None:
        selected = choose_balanced_candidates(candidates, args.target_rows, args.seed)
        split_rows = speaker_partition(
            selected,
            args.train_ratio,
            args.validation_ratio,
            args.test_ratio,
            args.seed,
        )
        fixed_test_source = None
    else:
        fixed_test_speakers = {str(row["speaker_id"]).strip() for row in fixed_test_rows}
        fixed_test_groups = {
            str(row.get("group_key") or normalize_asr_text(row.get("transcript") or ""))
            for row in fixed_test_rows
        }
        fixed_test_indices = {int(row["source_row_index"]) for row in fixed_test_rows}
        eligible_candidates = [
            row
            for row in candidates
            if row["speaker_id"] not in fixed_test_speakers
            and row["group_key"] not in fixed_test_groups
            and int(row["source_row_index"]) not in fixed_test_indices
        ]
        non_test_target = args.target_rows - len(fixed_test_rows)
        if len(eligible_candidates) < non_test_target:
            raise RuntimeError(
                f"Only {len(eligible_candidates)} candidates remain after excluding the "
                f"fixed test speakers/groups; {non_test_target} are required. Increase "
                "--scan-rows."
            )
        selected = choose_balanced_candidates(
            eligible_candidates,
            non_test_target,
            args.seed,
        )
        train_validation = speaker_partition_train_validation(
            selected,
            validation_ratio=args.validation_ratio
            / (args.train_ratio + args.validation_ratio),
            seed=args.seed,
        )
        split_rows = {
            **train_validation,
            "test": fixed_test_rows,
        }
        fixed_test_source = str(fixed_test_dir)

    all_source_indices = [
        int(row["source_row_index"])
        for split_rows_for_name in split_rows.values()
        for row in split_rows_for_name
    ]
    if len(all_source_indices) != len(set(all_source_indices)):
        raise RuntimeError("The resulting manifests contain duplicate source row indexes.")
    if any(
        {row["speaker_id"] for row in split_rows[left]}
        & {row["speaker_id"] for row in split_rows[right]}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("The resulting manifests contain speaker overlap between splits.")

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
        "selected_rows": sum(len(rows) for rows in split_rows.values()),
        "fixed_test_manifest_dir": fixed_test_source,
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
        "note": (
            "The test split is copied unchanged from the fixed test manifest; its speakers "
            "and transcript groups are excluded from train/validation. Other rows are "
            "selected once per group and assigned by complete speaker groups."
            if fixed_test_rows is not None
            else "Rows are selected with one record per group and assigned by complete speaker groups."
        ),
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
