"""Evaluate an existing Khmer ASR checkpoint on manifest rows.

The manifest stores metadata and the source row index, not local audio files.
This script replays the deterministic streamed dataset order, retrieves the
selected audio bytes from Hugging Face, resamples audio to 16 kHz, and runs
inference with a Wav2Vec2/XLS-R CTC checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_text import (
    ASR_NORMALIZATION_VERSION,
    normalize_asr_text,
)


DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-khmer"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MANIFEST_DIR = ROOT_DIR / "data" / "voice_1000"
DEFAULT_OUTPUT = ROOT_DIR / "reports" / "asr" / "baseline_smoke_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline inference on selected Khmer ASR manifest rows."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Number of manifest rows to evaluate; use 0 for the entire split.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", help="Override the dataset ID from manifest_summary.json.")
    parser.add_argument("--revision", help="Override the dataset revision from the summary.")
    parser.add_argument("--seed", type=int, help="Override the stream shuffle seed.")
    parser.add_argument("--shuffle-buffer", type=int, help="Override the stream shuffle buffer.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda (default: auto).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be zero or greater")
    if args.device not in {"auto", "cpu", "cuda"}:
        parser.error("--device must be auto, cpu, or cuda")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def read_manifest_rows(manifest_dir: Path, split: str) -> list[dict[str, Any]]:
    path = manifest_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            try:
                row["source_row_index"] = int(row["source_row_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source_row_index in {path}: {row}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_dir = resolve_path(args.manifest_dir)
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_manifest_rows(manifest_dir, args.split)
    if args.max_samples:
        rows = rows[: args.max_samples]

    configuration = {
        "manifest_dir": str(manifest_dir),
        "dataset": args.dataset or summary["dataset"],
        "source_split": summary.get("source_split", "train"),
        "revision": args.revision or summary.get("revision"),
        "seed": args.seed if args.seed is not None else int(summary["seed"]),
        "shuffle_buffer": (
            args.shuffle_buffer
            if args.shuffle_buffer is not None
            else int(summary["shuffle_buffer"])
        ),
    }
    if configuration["shuffle_buffer"] < 1:
        raise ValueError("shuffle_buffer must be at least 1")
    return configuration, rows


def stream_selected_audio(
    configuration: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replay the manifest stream and return bytes for selected rows."""
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
    if configuration["revision"] and configuration["revision"] != "default":
        load_kwargs["revision"] = configuration["revision"]

    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    shuffled = dataset.shuffle(
        seed=configuration["seed"],
        buffer_size=configuration["shuffle_buffer"],
    )

    by_index = {row["source_row_index"]: row for row in manifest_rows}
    max_index = max(by_index)
    found: list[dict[str, Any]] = []
    found_indices: set[int] = set()

    for stream_index, row in enumerate(shuffled.take(max_index + 1)):
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

    missing = sorted(set(by_index) - found_indices)
    if missing:
        raise RuntimeError(
            "Could not replay all selected manifest rows. "
            f"Missing source indexes: {missing[:10]}"
        )
    return found


def decode_and_resample(audio_bytes: bytes) -> tuple[Any, int, int, bool]:
    """Decode WAV bytes and resample to the model's required rate."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Install soundfile and scipy before running ASR inference."
        ) from exc

    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Decoded audio must contain a non-empty mono waveform.")

    sample_rate = int(sample_rate)
    if sample_rate == TARGET_SAMPLE_RATE:
        return samples, sample_rate, sample_rate, False
    if sample_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {sample_rate}")

    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
    samples = resample_poly(
        samples,
        TARGET_SAMPLE_RATE // divisor,
        sample_rate // divisor,
    ).astype("float32")
    return samples, TARGET_SAMPLE_RATE, sample_rate, True


def resolve_device(requested: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; use the Colab runtime's PyTorch.") from exc

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is available in this runtime.")
    return torch.device(requested)


def load_asr_model(model_id: str, device: Any) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForCTC,
            AutoTokenizer,
            Wav2Vec2Processor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers and use a runtime with PyTorch before loading the model."
        ) from exc

    # The checkpoint also publishes an optional KenLM/pyctcdecode processor.
    # Use ordinary greedy CTC decoding for the baseline so the experiment does
    # not depend on compiling KenLM in the Colab runtime.
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    model = AutoModelForCTC.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def transcribe(
    samples: Any,
    processor: Any,
    model: Any,
    device: Any,
) -> tuple[str, float]:
    import torch

    inputs = processor(
        samples,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    prediction = processor.batch_decode(predicted_ids, group_tokens=True)[0]
    return prediction, latency


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from jiwer import cer, wer
    except ImportError as exc:
        raise RuntimeError("Install jiwer before calculating CER and WER.") from exc

    references = [row["reference"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    if not predictions:
        return {
            "cer": None,
            "wer": None,
            "sentence_accuracy": None,
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "mean_real_time_factor": None,
        }

    latencies = [row["latency_seconds"] for row in predictions]
    real_time_factors = [
        row["real_time_factor"]
        for row in predictions
        if row["real_time_factor"] is not None
    ]
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    sorted_latencies = sorted(latencies)
    return {
        "cer": float(cer(references, hypotheses)),
        "wer": float(wer(references, hypotheses)),
        "sentence_accuracy": sum(
            row["reference"] == row["prediction"] for row in predictions
        )
        / len(predictions),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": sorted_latencies[p95_index],
        "mean_real_time_factor": (
            statistics.mean(real_time_factors) if real_time_factors else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    configuration, manifest_rows = load_configuration(args)
    selected_audio = stream_selected_audio(configuration, manifest_rows)
    device = resolve_device(args.device)
    processor, model = load_asr_model(args.model, device)

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    resampled_count = 0

    for row in selected_audio:
        try:
            samples, sample_rate, original_sample_rate, was_resampled = decode_and_resample(
                row["audio_bytes"]
            )
            resampled_count += int(was_resampled)
            prediction, latency = transcribe(samples, processor, model, device)
            reference = normalize_asr_text(row.get("normalized_transcript") or row.get("transcript"))
            normalized_prediction = normalize_asr_text(prediction)
            duration = float(row["duration"]) if row.get("duration") else None
            predictions.append(
                {
                    "source_row_index": row["source_row_index"],
                    "speaker_id": row.get("speaker_id"),
                    "audio_path": row.get("audio_path"),
                    "sample_rate_before_resampling": original_sample_rate,
                    "resampled_to_target_rate": was_resampled,
                    "reference": reference,
                    "prediction": normalized_prediction,
                    "latency_seconds": latency,
                    "audio_duration_seconds": duration,
                    "real_time_factor": latency / duration if duration and duration > 0 else None,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "source_row_index": row.get("source_row_index"),
                    "error": str(exc),
                }
            )

    report = {
        "model": args.model,
        "decoder": "greedy_ctc",
        "device": str(device),
        "split": args.split,
        "dataset": configuration["dataset"],
        "source_split": configuration["source_split"],
        "manifest_dir": configuration["manifest_dir"],
        "normalization_version": ASR_NORMALIZATION_VERSION,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "requested_samples": len(manifest_rows),
        "successful_samples": len(predictions),
        "error_count": len(errors),
        "resampled_count": resampled_count,
        "metrics": calculate_metrics(predictions),
        "predictions": predictions,
        "errors": errors,
        "note": (
            "This report evaluates the existing checkpoint before project-specific "
            "fine-tuning with greedy CTC decoding. The first inference may include "
            "GPU warm-up overhead."
        ),
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"ASR baseline failed: {exc}")
        return 1

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Model: {report['model']}")
    print(f"Device: {report['device']}")
    print(f"Successful samples: {report['successful_samples']}")
    print(f"Errors: {report['error_count']}")
    print(f"Resampled to 16 kHz: {report['resampled_count']}")
    print(f"Metrics: {report['metrics']}")
    print(f"Saved report to: {output_path}")
    return 0 if report["successful_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
