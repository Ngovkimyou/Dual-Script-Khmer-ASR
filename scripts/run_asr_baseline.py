"""Evaluate an existing Khmer ASR checkpoint on manifest rows.

The manifest stores metadata and the source row index, not local audio files.
This script replays the deterministic streamed dataset order, retrieves the
selected audio bytes from Hugging Face, resamples audio to 16 kHz, and runs
inference with a Wav2Vec2/XLS-R CTC checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_text import (
    ASR_NORMALIZATION_VERSION,
    normalize_asr_text,
)
from khmer_transliteration.asr_audio import TARGET_SAMPLE_RATE, decode_and_resample
from khmer_transliteration.asr_data import (
    load_manifest_configuration,
    read_manifest_rows,
    resolve_path,
    stream_selected_audio,
)
from khmer_transliteration.asr_inference import (
    DEFAULT_MODEL_ID,
    load_asr_model,
    resolve_device,
    transcribe,
)
from khmer_transliteration.asr_metrics import (
    METRIC_SCHEMA_VERSION,
    calculate_metrics,
    calculate_prediction_diagnostics,
    metric_definitions,
    word_segmentation_examples,
)


DEFAULT_MODEL = DEFAULT_MODEL_ID
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
    parser.add_argument(
        "--decoder",
        choices=("greedy_ctc", "beam_ctc"),
        default="greedy_ctc",
        help="CTC decoder; beam_ctc is a no-language-model experiment.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Number of prefixes retained by beam_ctc (default: 10).",
    )
    parser.add_argument("--dataset", help="Override the dataset ID from manifest_summary.json.")
    parser.add_argument("--revision", help="Override the dataset revision from the summary.")
    parser.add_argument("--seed", type=int, help="Override the stream shuffle seed.")
    parser.add_argument("--shuffle-buffer", type=int, help="Override the stream shuffle buffer.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda (default: auto).",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=Path,
        help="Local cache for selected WAV files (default: <manifest-dir>/audio_cache).",
    )
    parser.add_argument(
        "--no-audio-cache",
        action="store_true",
        help="Disable local audio caching for this run.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be zero or greater")
    if args.device not in {"auto", "cpu", "cuda"}:
        parser.error("--device must be auto, cpu, or cuda")
    if args.beam_width < 1:
        parser.error("--beam-width must be at least 1")
    return args


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_dir = resolve_path(args.manifest_dir)
    rows = read_manifest_rows(manifest_dir, args.split)
    if args.max_samples:
        rows = rows[: args.max_samples]

    configuration = load_manifest_configuration(
        manifest_dir,
        dataset=args.dataset,
        revision=args.revision,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    return configuration, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    configuration, manifest_rows = load_configuration(args)
    cache_dir = None
    if not args.no_audio_cache:
        cache_dir = (
            resolve_path(args.audio_cache_dir)
            if args.audio_cache_dir
            else resolve_path(args.manifest_dir) / "audio_cache"
        )

    def report_progress(message: str) -> None:
        print(message, flush=True)

    selected_audio = stream_selected_audio(
        configuration,
        manifest_rows,
        cache_dir=cache_dir,
        progress_callback=report_progress,
    )
    device = resolve_device(args.device)
    print(f"Loading ASR model on {device}: {args.model}", flush=True)
    processor, model = load_asr_model(args.model, device)
    print(f"Running inference on {len(selected_audio)} samples...", flush=True)

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    resampled_count = 0

    for sample_number, row in enumerate(selected_audio, start=1):
        try:
            samples, _, original_sample_rate, was_resampled = decode_and_resample(
                row["audio_bytes"]
            )
            resampled_count += int(was_resampled)
            (
                prediction,
                latency,
                model_latency,
                decoder_latency,
                filtered_unk_count,
            ) = transcribe(
                samples,
                processor,
                model,
                device,
                decoder=args.decoder,
                beam_width=args.beam_width,
            )
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
                    "model_latency_seconds": model_latency,
                    "decoder_latency_seconds": decoder_latency,
                    "filtered_unk_count": filtered_unk_count,
                    "audio_duration_seconds": duration,
                    "real_time_factor": latency / duration if duration and duration > 0 else None,
                }
            )
            print(f"Inference progress: {sample_number}/{len(selected_audio)}", flush=True)
        except Exception as exc:
            errors.append(
                {
                    "source_row_index": row.get("source_row_index"),
                    "error": str(exc),
                }
            )
            print(
                f"Inference progress: {sample_number}/{len(selected_audio)} "
                f"(error: {exc})",
                flush=True,
            )

    report = {
        "model": args.model,
        "decoder": args.decoder,
        "beam_width": args.beam_width if args.decoder == "beam_ctc" else None,
        "device": str(device),
        "split": args.split,
        "dataset": configuration["dataset"],
        "source_split": configuration["source_split"],
        "manifest_dir": configuration["manifest_dir"],
        "audio_cache_dir": str(cache_dir) if cache_dir is not None else None,
        "normalization_version": ASR_NORMALIZATION_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "metric_definitions": metric_definitions(),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "requested_samples": len(manifest_rows),
        "successful_samples": len(predictions),
        "error_count": len(errors),
        "resampled_count": resampled_count,
        "metrics": calculate_metrics(predictions),
        "diagnostics": calculate_prediction_diagnostics(predictions),
        "word_segmentation_examples": word_segmentation_examples(predictions),
        "predictions": predictions,
        "errors": errors,
        "note": (
            f"Evaluation uses {args.decoder} decoding. The first inference may include "
            "GPU warm-up overhead. beam_ctc is a no-language-model experiment. "
            "Visible special tokens are removed before scoring; suppressed [UNK] "
            "emissions are counted in diagnostics."
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
    metrics = report["metrics"]
    print("Accuracy rates (decimal; percentage in parentheses):")
    for label, key in (
        ("Code-point CER", "cer"),
        ("Grapheme-cluster CER", "grapheme_cluster_cer"),
        ("Khmer-segmented WER", "wer"),
        ("Legacy whitespace WER", "whitespace_wer"),
    ):
        value = metrics.get(key)
        formatted = (
            "n/a"
            if value is None
            else f"{value:.4f} ({value * 100:.2f}%)"
        )
        print(f"  {label}: {formatted}")
    latency_values = (
        metrics.get("mean_latency_seconds"),
        metrics.get("median_latency_seconds"),
        metrics.get("p95_latency_seconds"),
    )
    if all(value is not None for value in latency_values):
        print(
            "Latency: "
            f"mean {latency_values[0]:.3f}s, "
            f"median {latency_values[1]:.3f}s, "
            f"p95 {latency_values[2]:.3f}s"
        )
    else:
        print("Latency: unavailable")
    print(f"Diagnostics: {report['diagnostics']}")
    print(f"Saved report to: {output_path}")
    return 0 if report["successful_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
