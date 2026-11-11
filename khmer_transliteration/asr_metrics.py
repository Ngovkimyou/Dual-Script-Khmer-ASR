"""Shared Khmer-aware ASR evaluation metrics."""

from __future__ import annotations

import importlib.metadata
import math
import statistics
from functools import lru_cache
from typing import Any, Callable, Sequence

from khmer_transliteration.asr_text import normalize_asr_text


METRIC_SCHEMA_VERSION = 2
WORD_TOKENIZER_NAME = "khmercut.nn"


def grapheme_clusters(text: str) -> list[str]:
    """Split text into Unicode grapheme clusters (not Khmer lexical words)."""
    try:
        import regex
    except ImportError as exc:
        raise RuntimeError(
            "Install the ASR dependencies to calculate grapheme-cluster CER."
        ) from exc
    return regex.findall(r"\X", text)


def sequence_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return Levenshtein distance using O(min(n, m)) extra memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


@lru_cache(maxsize=1)
def _load_word_segmenter() -> tuple[Callable[[str], list[str]], str]:
    """Load and cache the neural Khmer word segmenter used for WER."""
    try:
        from khmercut.nn import tokenize
    except ImportError as exc:
        raise RuntimeError(
            "Khmer-aware WER requires khmercut's neural tokenizer. "
            "Install it with: python -m pip install -r requirements-asr.txt"
        ) from exc

    try:
        version = importlib.metadata.version("khmercut")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return tokenize, version


def tokenize_khmer_words(text: str) -> list[str]:
    """Segment a transcript and omit separator whitespace tokens."""
    segment, _ = _load_word_segmenter()
    try:
        tokens = segment(normalize_asr_text(text))
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            "khmercut's neural model data must be opened as UTF-8. On Windows, "
            "run metric commands with 'python -X utf8'."
        ) from exc
    return [
        token
        for token in tokens
        if isinstance(token, str) and token and not token.isspace()
    ]


def word_segmentation_examples(
    predictions: Sequence[dict[str, Any]],
    *,
    maximum_examples: int = 5,
) -> list[dict[str, Any]]:
    """Expose a few word tokenizations so the WER boundaries can be reviewed."""
    if maximum_examples < 0:
        raise ValueError("maximum_examples cannot be negative")
    examples: list[dict[str, Any]] = []
    for row in predictions[:maximum_examples]:
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        examples.append(
            {
                "source_row_index": row.get("source_row_index"),
                "reference_tokens": tokenize_khmer_words(reference),
                "prediction_tokens": tokenize_khmer_words(prediction),
            }
        )
    return examples


def metric_definitions() -> dict[str, Any]:
    """Describe metric units and implementation for evaluation reports."""
    _, tokenizer_version = _load_word_segmenter()
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "cer": "Unicode code-point CER after NFC and whitespace normalization.",
        "grapheme_cluster_cer": (
            "Edit rate over Unicode grapheme clusters matched by regex \\X. "
            "This is an additional script-aware view, not the published KHCWER metric."
        ),
        "wer": (
            "Micro-averaged token edit rate after independently segmenting each "
            "reference and hypothesis with khmercut.nn; whitespace separator tokens "
            "are discarded."
        ),
        "wer_tokenizer": WORD_TOKENIZER_NAME,
        "wer_tokenizer_version": tokenizer_version,
        "whitespace_wer": (
            "Legacy jiwer WER using whitespace-delimited tokens; included only "
            "for comparison with earlier project reports."
        ),
    }


def calculate_character_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float | None]:
    """Calculate code-point and grapheme-cluster CER without word segmentation."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same number of rows")
    if not references:
        return {
            "cer": None,
            "grapheme_cluster_cer": None,
            "sentence_accuracy": None,
        }

    try:
        from jiwer import cer
    except ImportError as exc:
        raise RuntimeError(
            "Install the ASR dependencies before calculating CER."
        ) from exc

    normalized_references = [normalize_asr_text(text) for text in references]
    normalized_hypotheses = [normalize_asr_text(text) for text in hypotheses]
    codepoint_cer = float(cer(normalized_references, normalized_hypotheses))

    reference_clusters = [grapheme_clusters(text) for text in normalized_references]
    hypothesis_clusters = [grapheme_clusters(text) for text in normalized_hypotheses]
    cluster_denominator = sum(map(len, reference_clusters))
    cluster_errors = sum(
        sequence_edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(reference_clusters, hypothesis_clusters)
    )

    return {
        "cer": codepoint_cer,
        "grapheme_cluster_cer": (
            cluster_errors / cluster_denominator if cluster_denominator else None
        ),
        "sentence_accuracy": sum(
            reference == hypothesis
            for reference, hypothesis in zip(
                normalized_references, normalized_hypotheses
            )
        )
        / len(normalized_references),
    }


def calculate_text_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> dict[str, float | None]:
    """Calculate CER, Khmer-segmented WER, and legacy whitespace WER."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same number of rows")
    if not references:
        return {
            **calculate_character_metrics([], []),
            "wer": None,
            "whitespace_wer": None,
        }

    try:
        from jiwer import wer
    except ImportError as exc:
        raise RuntimeError(
            "Install the ASR dependencies before calculating WER."
        ) from exc

    character_metrics = calculate_character_metrics(references, hypotheses)
    normalized_references = [normalize_asr_text(text) for text in references]
    normalized_hypotheses = [normalize_asr_text(text) for text in hypotheses]
    reference_words = [tokenize_khmer_words(text) for text in normalized_references]
    hypothesis_words = [tokenize_khmer_words(text) for text in normalized_hypotheses]
    word_denominator = sum(map(len, reference_words))
    word_errors = sum(
        sequence_edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(reference_words, hypothesis_words)
    )
    if word_denominator:
        khmer_wer: float | None = word_errors / word_denominator
    else:
        khmer_wer = 0.0 if not any(hypothesis_words) else None

    return {
        **character_metrics,
        "wer": khmer_wer,
        "whitespace_wer": float(wer(normalized_references, normalized_hypotheses)),
    }


def calculate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate text accuracy, sentence accuracy, and inference latency."""
    if not predictions:
        return {
            **calculate_text_metrics([], []),
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "mean_real_time_factor": None,
        }

    text_metrics = calculate_text_metrics(
        [str(row.get("reference") or "") for row in predictions],
        [str(row.get("prediction") or "") for row in predictions],
    )
    latencies = [float(row["latency_seconds"]) for row in predictions]
    real_time_factors = [
        float(row["real_time_factor"])
        for row in predictions
        if row.get("real_time_factor") is not None
    ]
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    sorted_latencies = sorted(latencies)
    return {
        **text_metrics,
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": sorted_latencies[p95_index],
        "mean_real_time_factor": (
            statistics.mean(real_time_factors) if real_time_factors else None
        ),
    }


def calculate_prediction_diagnostics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report failure patterns that CER and WER alone do not explain."""
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
        row for row in predictions if "\u17d2" in str(row.get("reference", ""))
    ]
    subscript_metrics: dict[str, Any] = {
        "count": len(subscript_rows),
        "cer": None,
        "grapheme_cluster_cer": None,
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
        subset_metrics = calculate_character_metrics(
            [str(row.get("reference") or "") for row in subscript_rows],
            [str(row.get("prediction") or "") for row in subscript_rows],
        )
        subscript_metrics.update(
            {
                key: subset_metrics[key]
                for key in ("cer", "grapheme_cluster_cer")
            }
        )

    return {
        "prediction_contains_unk_count": len(unknown_rows),
        "prediction_trailing_unk_count": len(trailing_unknown_rows),
        "unknown_token_emissions_suppressed": sum(
            int(row.get("filtered_unk_count") or 0) for row in predictions
        ),
        "prediction_without_spaces_count": len(missing_space_rows),
        "subscript_reference": subscript_metrics,
    }
