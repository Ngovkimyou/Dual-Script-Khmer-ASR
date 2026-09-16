"""Shared text normalization for Khmer ASR labels and evaluation."""

from __future__ import annotations

import unicodedata
from typing import Any


ASR_NORMALIZATION_VERSION = "nfc_trim_collapse_whitespace_v1"


def normalize_asr_text(value: Any) -> str:
    """Normalize an ASR transcript without changing its linguistic content."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().split())
