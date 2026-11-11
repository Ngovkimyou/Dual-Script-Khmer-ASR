"""Dictionary-backed canonical Khmer-to-Romanized output.

The existing keyboard is primarily Romanized Khmer -> Khmer. A reverse lookup
is ambiguous, so this module chooses the highest-frequency romanization for a
known Khmer token and preserves unknown text. It is a transparent baseline,
not a claim that every Khmer spelling has one unique romanization.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from khmer_transliteration.asr_text import normalize_asr_text
from khmer_transliteration.paths import ALL_WORDS_FILE


_WHITESPACE_OR_TOKEN = re.compile(r"(\s+)")
_PUNCTUATION = ",.;:!?()[]{}\"'-"


@lru_cache(maxsize=1)
def _load_reverse_index() -> dict[str, str]:
    """Build Khmer -> canonical romanization from the existing word dataset."""
    candidates: dict[str, list[tuple[int, str]]] = {}
    data_path = Path(ALL_WORDS_FILE)
    if not data_path.exists():
        return {}

    with data_path.open("r", encoding="utf-8-sig", newline="") as data_file:
        for row in csv.DictReader(data_file):
            khmer = str(row.get("khmer") or "").strip()
            romanized = normalize_asr_text(row.get("romanized") or "")
            if not khmer or not romanized:
                continue
            try:
                frequency = int(row.get("frequency") or 0)
            except (TypeError, ValueError):
                frequency = 0
            candidates.setdefault(khmer, []).append((frequency, romanized))

    return {
        khmer: sorted(options, key=lambda item: (-item[0], len(item[1]), item[1]))[0][1]
        for khmer, options in candidates.items()
    }


def _romanize_token(token: str, reverse_index: dict[str, str]) -> str:
    if token in reverse_index:
        return reverse_index[token]

    leading = token[:1] if token[:1] in _PUNCTUATION else ""
    trailing = token[-1:] if token[-1:] in _PUNCTUATION else ""
    core_start = 1 if leading else 0
    core_end = -1 if trailing else None
    core = token[core_start:core_end]
    if core and core in reverse_index:
        return f"{leading}{reverse_index[core]}{trailing}"
    return token


def romanize_khmer_text(text: str) -> str:
    """Return a canonical dictionary-backed Romanized Khmer approximation."""
    normalized = normalize_asr_text(text)
    if not normalized:
        return ""

    reverse_index = _load_reverse_index()
    return "".join(
        part if part.isspace() else _romanize_token(part, reverse_index)
        for part in _WHITESPACE_OR_TOKEN.split(normalized)
        if part
    )
