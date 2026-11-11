import torch

from khmer_transliteration.asr_decoding import ctc_prefix_beam_search
from khmer_transliteration.asr_inference import extend_ctc_vocabulary


def test_ctc_beam_search_collapses_repeated_tokens_without_blank():
    logits = torch.tensor(
        [
            [0.0, 8.0, 0.0],
            [0.0, 8.0, 0.0],
        ]
    )

    assert ctc_prefix_beam_search(logits, blank_id=0, beam_width=3) == [1]


def test_ctc_beam_search_keeps_repeated_tokens_after_blank():
    logits = torch.tensor(
        [
            [0.0, 8.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
        ]
    )

    assert ctc_prefix_beam_search(logits, blank_id=0, beam_width=3) == [1, 1]


def test_ctc_beam_search_can_block_special_tokens():
    logits = torch.tensor(
        [
            [0.0, 8.0, 7.0],
        ]
    )

    assert ctc_prefix_beam_search(
        logits,
        blank_id=0,
        beam_width=3,
        blocked_ids={1},
    ) == [2]


def test_vocabulary_expansion_preserves_existing_ids():
    base = {"|": 0, "ក": 1, "[UNK]": 2, "[PAD]": 3}

    expanded = extend_ctc_vocabulary(base, ["។", "ក", " ", "|"])

    assert expanded["|"] == 0
    assert expanded["ក"] == 1
    assert expanded["[UNK]"] == 2
    assert expanded["[PAD]"] == 3
    assert expanded["។"] == 4
