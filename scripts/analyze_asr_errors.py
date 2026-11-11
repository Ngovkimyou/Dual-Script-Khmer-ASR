"""Analyze Khmer Unicode-cluster errors in an ASR evaluation report."""

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
    grapheme_clusters,
    sequence_edit_distance,
)


COENG = "\u17d2"
DEFAULT_REPORT = ROOT_DIR / "reports" / "asr" / "finetuned_test_clean.json"


def analyze_report(
    report: dict[str, Any],
    *,
    minimum_coeng_count: int = 8,
    maximum_examples: int = 10,
) -> dict[str, Any]:
    """Summarize Unicode-cluster and coeng error patterns."""
    predictions = report.get("predictions", [])
    analyzed_rows: list[dict[str, Any]] = []

    for row in predictions:
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        reference_clusters = grapheme_clusters(reference)
        prediction_clusters = grapheme_clusters(prediction)
        reference_coeng = reference.count(COENG)
        prediction_coeng = prediction.count(COENG)
        distance = sequence_edit_distance(reference_clusters, prediction_clusters)
        analyzed_rows.append(
            {
                "source_row_index": row.get("source_row_index"),
                "reference_grapheme_clusters": len(reference_clusters),
                "prediction_grapheme_clusters": len(prediction_clusters),
                "grapheme_cluster_edit_distance": distance,
                "grapheme_cluster_error_rate": (
                    distance / len(reference_clusters) if reference_clusters else 0.0
                ),
                "reference_coeng_count": reference_coeng,
                "prediction_coeng_count": prediction_coeng,
                "coeng_count_difference": prediction_coeng - reference_coeng,
                "prediction_has_double_coeng": f"{COENG}{COENG}" in prediction,
                "prediction_has_trailing_unk": prediction.rstrip().endswith("[UNK]"),
                "subscript_heavy": reference_coeng >= minimum_coeng_count,
                "reference": reference,
                "prediction": prediction,
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        reference_cluster_count = sum(
            row["reference_grapheme_clusters"] for row in rows
        )
        total_distance = sum(row["grapheme_cluster_edit_distance"] for row in rows)
        return {
            "sample_count": len(rows),
            "grapheme_cluster_error_rate": (
                total_distance / reference_cluster_count
                if reference_cluster_count
                else 0.0
            ),
            "reference_coeng_total": sum(
                row["reference_coeng_count"] for row in rows
            ),
            "prediction_coeng_total": sum(
                row["prediction_coeng_count"] for row in rows
            ),
            "rows_with_coeng_count_mismatch": sum(
                row["coeng_count_difference"] != 0 for row in rows
            ),
            "rows_with_double_coeng_prediction": sum(
                row["prediction_has_double_coeng"] for row in rows
            ),
            "rows_with_trailing_unk": sum(
                row["prediction_has_trailing_unk"] for row in rows
            ),
        }

    subscript_rows = [row for row in analyzed_rows if row["subscript_heavy"]]
    worst_examples = sorted(
        analyzed_rows,
        key=lambda row: (
            row["prediction_has_double_coeng"],
            abs(row["coeng_count_difference"]),
            row["grapheme_cluster_edit_distance"],
        ),
        reverse=True,
    )[:maximum_examples]

    return {
        "model": report.get("model"),
        "split": report.get("split"),
        "normalization_version": report.get("normalization_version"),
        "reported_metrics": report.get("metrics", {}),
        "minimum_coeng_count_for_subscript_heavy": minimum_coeng_count,
        "all_rows": aggregate(analyzed_rows),
        "subscript_heavy_rows": aggregate(subscript_rows),
        "worst_examples": worst_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Khmer grapheme-cluster and coeng errors in an ASR report."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--minimum-coeng-count",
        type=int,
        default=8,
        help="Minimum reference coeng count for the subscript-heavy subset.",
    )
    parser.add_argument(
        "--maximum-examples",
        type=int,
        default=10,
        help="Number of worst examples to include in the output.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT_DIR / args.report
    if args.minimum_coeng_count < 1:
        raise SystemExit("--minimum-coeng-count must be at least 1")
    if args.maximum_examples < 0:
        raise SystemExit("--maximum-examples cannot be negative")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = analyze_report(
        report,
        minimum_coeng_count=args.minimum_coeng_count,
        maximum_examples=args.maximum_examples,
    )
    output_path = args.output
    if output_path is None:
        output_path = report_path.with_name(f"{report_path.stem}_error_analysis.json")
    elif not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("All rows:", analysis["all_rows"])
    print("Subscript-heavy rows:", analysis["subscript_heavy_rows"])
    print("Saved analysis to:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
