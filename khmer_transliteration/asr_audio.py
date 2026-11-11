"""Audio decoding and sample-rate conversion shared by ASR components."""

from __future__ import annotations

import io
import math
from typing import Any


TARGET_SAMPLE_RATE = 16_000


def decode_and_resample(audio_bytes: bytes) -> tuple[Any, int, int, bool]:
    """Decode WAV bytes, convert stereo to mono, and resample to 16 kHz."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Install soundfile and scipy before processing ASR audio."
        ) from exc

    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Decoded audio must contain a non-empty mono waveform.")

    original_sample_rate = int(sample_rate)
    if original_sample_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {original_sample_rate}")
    if original_sample_rate == TARGET_SAMPLE_RATE:
        return samples, TARGET_SAMPLE_RATE, original_sample_rate, False

    divisor = math.gcd(original_sample_rate, TARGET_SAMPLE_RATE)
    samples = resample_poly(
        samples,
        TARGET_SAMPLE_RATE // divisor,
        original_sample_rate // divisor,
    ).astype("float32")
    return samples, TARGET_SAMPLE_RATE, original_sample_rate, True
