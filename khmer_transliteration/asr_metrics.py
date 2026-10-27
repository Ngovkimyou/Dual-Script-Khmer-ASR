"""Shared ASR evaluation metrics."""

from __future__ import annotations

import math
import statistics
from typing import Any


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate normalized CER, WER, exact sentence accuracy, and latency."""
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

    try:
        from jiwer import cer, wer
    except ImportError as exc:
        raise RuntimeError("Install jiwer before calculating CER and WER.") from exc

    references = [row["reference"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
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


def calculate_prediction_diagnostics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Report failure patterns that CER/WER alone do not explain."""
    unknown_rows = [
        row for row in predictions if "[UNK]" in str(row.get("prediction", ""))
    ]
    trailing_unknown_rows = [
        row
        for row in unknown_rows
        if str(row.get("prediction", "")).rstrip().endswith("[UNK]")
    ]
    missing_space_rows = [
        row
        for row in predictions
        if " " in str(row.get("reference", ""))
        and " " not in str(row.get("prediction", ""))
    ]
    subscript_rows = [
        row
        for row in predictions
        if "\u17d2" in str(row.get("reference", ""))
    ]

    subscript_metrics: dict[str, Any] = {
        "count": len(subscript_rows),
        "cer": None,
        "examples": [
            {
                "source_row_index": row.get("source_row_index"),
                "reference": row.get("reference"),
                "prediction": row.get("prediction"),
            }
            for row in subscript_rows[:5]
        ],
    }
    if subscript_rows:
        try:
            from jiwer import cer

            subscript_metrics["cer"] = float(
                cer(
                    [row["reference"] for row in subscript_rows],
                    [row["prediction"] for row in subscript_rows],
                )
            )
        except ImportError as exc:
            raise RuntimeError("Install jiwer before calculating ASR diagnostics.") from exc

    return {
        "prediction_contains_unk_count": len(unknown_rows),
        "prediction_trailing_unk_count": len(trailing_unknown_rows),
        "prediction_without_spaces_count": len(missing_space_rows),
        "subscript_reference": subscript_metrics,
    }
