"""Recalculate Khmer-aware metrics from a saved ASR JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_metrics import (
    METRIC_SCHEMA_VERSION,
    calculate_prediction_diagnostics,
    calculate_text_metrics,
    metric_definitions,
    word_segmentation_examples,
)
from khmer_transliteration.asr_text import normalize_asr_text


VISIBLE_SPECIAL_TOKENS = ("[UNK]", "[PAD]", "<s>", "</s>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score saved ASR predictions with Khmer-aware WER without "
            "downloading audio or loading an ASR model."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <report>_rescored.json).",
    )
    return parser.parse_args()


def rescore_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with current metrics and explicit metric definitions."""
    predictions = report.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("The report must contain a predictions list.")

    previous_metrics = report.get("metrics", {})
    if not isinstance(previous_metrics, dict):
        previous_metrics = {}
    scoring_predictions: list[dict[str, Any]] = []
    for row in predictions:
        scored_row = dict(row)
        raw_prediction = str(row.get("prediction") or "")
        cleaned_prediction = raw_prediction
        filtered_unknowns = 0
        for token in VISIBLE_SPECIAL_TOKENS:
            if token == "[UNK]":
                filtered_unknowns += cleaned_prediction.count(token)
            cleaned_prediction = cleaned_prediction.replace(token, " ")
        cleaned_prediction = normalize_asr_text(cleaned_prediction)
        if cleaned_prediction != raw_prediction:
            scored_row.setdefault("raw_prediction", raw_prediction)
            scored_row["prediction"] = cleaned_prediction
            scored_row["filtered_unk_count"] = (
                int(row.get("filtered_unk_count") or 0) + filtered_unknowns
            )
        scoring_predictions.append(scored_row)

    references = [str(row.get("reference") or "") for row in scoring_predictions]
    hypotheses = [str(row.get("prediction") or "") for row in scoring_predictions]
    updated = dict(report)
    updated["predictions"] = scoring_predictions
    updated["previous_metrics"] = previous_metrics
    if report.get("metric_schema_version") is None:
        updated["previous_metric_definitions"] = {
            "wer": (
                "Legacy jiwer WER with whitespace-delimited tokens; not Khmer-aware."
            )
        }
    updated["metrics"] = {
        **{
            key: value
            for key, value in previous_metrics.items()
            if key
            in {
                "mean_latency_seconds",
                "median_latency_seconds",
                "p95_latency_seconds",
                "mean_real_time_factor",
            }
        },
        **calculate_text_metrics(references, hypotheses),
    }
    updated["diagnostics"] = calculate_prediction_diagnostics(scoring_predictions)
    updated["word_segmentation_examples"] = word_segmentation_examples(
        scoring_predictions
    )
    updated["metric_schema_version"] = METRIC_SCHEMA_VERSION
    updated["metric_definitions"] = metric_definitions()
    updated["rescore_note"] = (
        "Text metrics were recalculated from the saved predictions using Khmer-aware "
        "word segmentation. Visible ASR special-token strings were removed from the "
        "scoring hypotheses; raw_prediction preserves any changed original text. "
        "previous_metrics preserves the values in the source report. "
        "diagnostics are computed from the cleaned hypotheses and retain the count "
        "of removed [UNK] tokens."
    )
    return updated


def main() -> int:
    args = parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT_DIR / args.report
    output_path = args.output
    if output_path is None:
        output_path = report_path.with_name(f"{report_path.stem}_rescored.json")
    elif not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    report = json.loads(report_path.read_text(encoding="utf-8"))
    updated = rescore_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Recalculated metrics:", updated["metrics"])
    print("Saved rescored report to:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
