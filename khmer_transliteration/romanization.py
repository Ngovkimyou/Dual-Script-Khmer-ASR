"""Best-effort dictionary-backed Khmer-to-Romanized output.

The existing keyboard is primarily Romanized Khmer -> Khmer. A reverse lookup
is ambiguous, so this module segments Khmer text, chooses the highest-frequency
romanization for known words, and preserves words it cannot map. It is a
transparent baseline, not a claim that every Khmer spelling has one unique
romanization.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from khmer_transliteration.asr_text import normalize_asr_text
from khmer_transliteration.paths import ALL_WORDS_FILE


_WHITESPACE_OR_TOKEN = re.compile(r"(\s+)")
_PUNCTUATION = ",.;:!?()[]{}\"'-\u17d4\u17d5\u17d6\u17da\u00ab\u00bb\u201c\u201d\u2018\u2019"
_CLOSING_PUNCTUATION = ",.;:!?)]}\u17d4\u17d5\u17d6\u17da"


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

    parts = _split_known_compound(core, reverse_index) if core else None
    if parts:
        romanized = " ".join(reverse_index[part] for part in parts)
        return f"{leading}{romanized}{trailing}"
    return token


def _split_known_compound(
    token: str,
    reverse_index: dict[str, str],
    *,
    max_parts: int = 4,
) -> tuple[str, ...] | None:
    """Split an unknown compound only when known dictionary words cover it."""
    if not token:
        return None
    try:
        import regex

        clusters = regex.findall(r"\X", token)
    except ImportError:
        clusters = list(token)

    best: list[tuple[str, ...] | None] = [None] * (len(clusters) + 1)
    best[0] = ()
    for start in range(len(clusters)):
        previous = best[start]
        if previous is None or len(previous) >= max_parts:
            continue
        for end in range(start + 1, len(clusters) + 1):
            word = "".join(clusters[start:end])
            if word not in reverse_index:
                continue
            candidate = (*previous, word)
            current = best[end]
            if current is None or (
                len(candidate), tuple(-len(part) for part in candidate)
            ) < (len(current), tuple(-len(part) for part in current)):
                best[end] = candidate

    result = best[-1]
    return result if result and len(result) >= 2 else None


@lru_cache(maxsize=1)
def _load_word_segmenter():
    """Load the same Khmer-aware segmenter used by the ASR WER metric."""
    try:
        from khmercut.nn import tokenize
    except ImportError:
        return None
    return tokenize


def romanize_khmer_text(text: str) -> str:
    """Return a segmented, dictionary-backed Romanized Khmer approximation.

    Words absent from the dictionary remain in Khmer. Unknown compounds are
    romanized only when they can be split completely into known dictionary
    words.
    """
    normalized = normalize_asr_text(text)
    if not normalized:
        return ""

    reverse_index = _load_reverse_index()
    segment = _load_word_segmenter()
    if segment is None:
        parts = _WHITESPACE_OR_TOKEN.split(normalized)
    else:
        try:
            parts = segment(normalized)
        except (OSError, RuntimeError, UnicodeDecodeError):
            # Keep transcription available if optional word segmentation fails.
            parts = _WHITESPACE_OR_TOKEN.split(normalized)

    output = ""
    for part in parts:
        if not part:
            continue
        if part.isspace():
            if output and not output.endswith(" "):
                output += " "
            continue
        if (
            output
            and not output.endswith(" ")
            and not all(character in _CLOSING_PUNCTUATION for character in part)
        ):
            output += " "
        output += _romanize_token(part, reverse_index)

    return normalize_asr_text(output)
    