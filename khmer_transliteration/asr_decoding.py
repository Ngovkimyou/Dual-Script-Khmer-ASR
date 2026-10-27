"""Small, dependency-free CTC decoding helpers.

The production path currently uses greedy CTC decoding. This module also
contains an opt-in prefix beam search so decoding can be evaluated without
requiring KenLM or another external language-model package.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


NEGATIVE_INFINITY = float("-inf")


def _logaddexp(left: float, right: float) -> float:
    """Add two log probabilities without underflowing."""
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    if left < right:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))


def ctc_prefix_beam_search(
    logits: Any,
    *,
    blank_id: int,
    beam_width: int = 10,
    blocked_ids: Iterable[int] = (),
) -> list[int]:
    """Decode one CTC logit sequence with prefix beam search.

    ``blank_id`` must be the model's CTC blank class. For the clean Khmer
    XLS-R checkpoint this is the ``[PAD]`` token ID, not the ``|`` delimiter
    ID. ``blocked_ids`` is useful for token classes that were not included in
    training labels, such as BOS/EOS tokens.

    This is intentionally a no-language-model decoder. It is an experiment
    to determine whether a better CTC path alone reduces errors before adding
    a Khmer language model or changing the training data.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the ASR stack already needs torch
        raise RuntimeError("PyTorch is required for CTC beam decoding.") from exc

    if not hasattr(logits, "ndim") or logits.ndim != 2:
        raise ValueError("logits must have shape [time, vocabulary]")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        return []
    vocabulary_size = int(logits.shape[1])
    if not 0 <= blank_id < vocabulary_size:
        raise ValueError("blank_id must be a valid vocabulary index")

    blocked = {int(token_id) for token_id in blocked_ids}
    blocked.add(int(blank_id))
    log_probs = torch.log_softmax(logits.detach(), dim=-1).to("cpu").tolist()

    # Each prefix stores (probability ending in blank, probability ending in
    # a non-blank token), both in log space.
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, NEGATIVE_INFINITY)
    }

    for timestep in log_probs:
        next_beams: dict[tuple[int, ...], list[float]] = {}

        def scores_for(prefix: tuple[int, ...]) -> list[float]:
            return next_beams.setdefault(
                prefix,
                [NEGATIVE_INFINITY, NEGATIVE_INFINITY],
            )

        for prefix, (prefix_blank, prefix_nonblank) in beams.items():
            prefix_total = _logaddexp(prefix_blank, prefix_nonblank)

            # Staying on the same prefix after emitting the CTC blank.
            blank_score = timestep[blank_id]
            current = scores_for(prefix)
            current[0] = _logaddexp(current[0], prefix_total + blank_score)

            for token_id, token_score in enumerate(timestep):
                if token_id in blocked:
                    continue

                if prefix and token_id == prefix[-1]:
                    # A repeated token without a blank is still one CTC
                    # token. A repeated token after a blank starts a new
                    # prefix, so both paths must be retained.
                    current[1] = _logaddexp(
                        current[1], prefix_nonblank + token_score
                    )
                    extended = prefix + (token_id,)
                    extended_scores = scores_for(extended)
                    extended_scores[1] = _logaddexp(
                        extended_scores[1], prefix_blank + token_score
                    )
                else:
                    extended = prefix + (token_id,)
                    extended_scores = scores_for(extended)
                    extended_scores[1] = _logaddexp(
                        extended_scores[1], prefix_total + token_score
                    )

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _logaddexp(item[1][0], item[1][1]),
            reverse=True,
        )
        beams = {
            prefix: (scores[0], scores[1])
            for prefix, scores in ranked[:beam_width]
        }

    if not beams:
        return []
    best_prefix = max(
        beams.items(),
        key=lambda item: _logaddexp(item[1][0], item[1][1]),
    )[0]
    return list(best_prefix)
