"""Shared manifest and streamed-audio helpers for the Khmer ASR workflow."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    """Resolve a project-relative path while preserving absolute paths."""
    return path if path.is_absolute() else ROOT_DIR / path


def as_number(value: Any) -> float | None:
    """Convert a value to a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def read_manifest_rows(
    manifest_dir: Path,
    split: str,
    *,
    include_split: bool = False,
) -> list[dict[str, Any]]:
    """Read and validate one CSV manifest split."""
    path = resolve_path(manifest_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            try:
                row["source_row_index"] = int(row["source_row_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source_row_index in {path}: {row}") from exc
            if include_split:
                row["manifest_split"] = split
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def load_manifest_configuration(
    manifest_dir: Path,
    *,
    dataset: str | None = None,
    revision: str | None = None,
    source_split: str | None = None,
    seed: int | None = None,
    shuffle_buffer: int | None = None,
) -> dict[str, Any]:
    """Load the dataset replay settings saved beside the manifests."""
    manifest_dir = resolve_path(manifest_dir)
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configuration = {
        "manifest_dir": str(manifest_dir),
        "dataset": dataset or summary["dataset"],
        "source_split": source_split or summary.get("source_split", "train"),
        "revision": revision or summary.get("revision"),
        "seed": seed if seed is not None else int(summary.get("seed", 42)),
        "shuffle_buffer": (
            shuffle_buffer
            if shuffle_buffer is not None
            else int(summary.get("shuffle_buffer", 1))
        ),
    }
    if configuration["shuffle_buffer"] < 1:
        raise ValueError("shuffle_buffer must be at least 1")
    return configuration


def stream_selected_audio(
    configuration: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Replay a manifest stream and return selected audio bytes.

    When ``cache_dir`` is provided, selected audio is saved by source index and
    reused on later runs. A complete cache avoids contacting Hugging Face at
    all, which makes repeated baseline evaluations much faster.
    """
    if not manifest_rows:
        raise ValueError("At least one manifest row is required")

    cache_path = resolve_path(cache_dir) if cache_dir is not None else None
    by_index = {row["source_row_index"]: row for row in manifest_rows}

    def cached_audio_path(source_index: int) -> Path | None:
        if cache_path is None:
            return None
        return cache_path / f"source_{source_index:06d}.wav"

    if cache_path is not None:
        cached_rows: list[dict[str, Any]] = []
        for row in manifest_rows:
            path = cached_audio_path(row["source_row_index"])
            if path is None or not path.is_file() or path.stat().st_size == 0:
                break
            selected = dict(row)
            selected["audio_bytes"] = path.read_bytes()
            selected["source_audio_path"] = str(row.get("audio_path") or "")
            cached_rows.append(selected)

        if len(cached_rows) == len(manifest_rows):
            if progress_callback:
                progress_callback(
                    f"Loaded {len(cached_rows)}/{len(manifest_rows)} audio files from cache: {cache_path}"
                )
            return cached_rows

        cache_path.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback(
                f"Audio cache has {len(cached_rows)}/{len(manifest_rows)} files; replaying Hugging Face stream..."
            )
    elif progress_callback:
        progress_callback(
            f"Replaying Hugging Face stream for {len(manifest_rows)} selected audio files..."
        )

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements-asr.txt first."
        ) from exc

    load_kwargs: dict[str, Any] = {
        "path": configuration["dataset"],
        "split": configuration["source_split"],
        "streaming": True,
    }
    if configuration.get("revision") and configuration["revision"] != "default":
        load_kwargs["revision"] = configuration["revision"]

    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    shuffled = dataset.shuffle(
        seed=configuration["seed"],
        buffer_size=configuration["shuffle_buffer"],
    )

    max_index = max(by_index)
    found: list[dict[str, Any]] = []
    found_indices: set[int] = set()

    for stream_index, row in enumerate(shuffled.take(max_index + 1)):
        if progress_callback and stream_index and stream_index % 500 == 0:
            progress_callback(
                f"Scanned {stream_index}/{max_index + 1} streamed rows; "
                f"found {len(found)}/{len(by_index)} selected files"
            )
        if stream_index not in by_index:
            continue
        audio = row.get("audio") if isinstance(row.get("audio"), dict) else {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            raise RuntimeError(
                f"Audio bytes were unavailable for source row {stream_index} "
                f"({audio.get('path', 'unknown path')})."
            )
        selected = dict(by_index[stream_index])
        selected["audio_bytes"] = bytes(audio_bytes)
        selected["source_audio_path"] = str(audio.get("path") or "")
        found.append(selected)
        found_indices.add(stream_index)
        if cache_path is not None:
            cache_file = cached_audio_path(stream_index)
            if cache_file is not None:
                cache_file.write_bytes(selected["audio_bytes"])
        if progress_callback:
            progress_callback(
                f"Fetched {len(found)}/{len(by_index)} selected audio files "
                f"(stream row {stream_index})"
            )

    missing = sorted(set(by_index) - found_indices)
    if missing:
        raise RuntimeError(
            "Could not replay all selected manifest rows. "
            f"Missing source indexes: {missing[:10]}"
        )
    if progress_callback:
        progress_callback(f"Audio preparation complete: {len(found)} files ready")
    return found
