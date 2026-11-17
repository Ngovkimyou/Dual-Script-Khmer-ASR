"""Audio decoding and sample-rate conversion shared by ASR components."""

from __future__ import annotations

import io
import math
from typing import Any


TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION_SECONDS = 60.0


def decode_and_resample(
    audio_bytes: bytes,
    *,
    require_wav_container: bool = False,
    max_duration_seconds: float | None = None,
) -> tuple[Any, int, int, bool]:
    """Decode audio, optionally validate a WAV upload, and resample to 16 kHz."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Install soundfile and scipy before processing ASR audio."
        ) from exc

    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")

    info = None
    if require_wav_container:
        is_wave_container = (
            len(audio_bytes) >= 12
            and audio_bytes[:4] in {b"RIFF", b"RIFX", b"RF64"}
            and audio_bytes[8:12] == b"WAVE"
        )
        if not is_wave_container:
            raise ValueError("The uploaded file is not a WAV audio file.")

    if require_wav_container or max_duration_seconds is not None:
        try:
            info = sf.info(io.BytesIO(audio_bytes))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("The uploaded WAV file is invalid or unsupported.") from exc
        if not str(info.format).startswith("WAV") and info.format != "RF64":
            raise ValueError("The uploaded file is not a supported WAV format.")
        if info.samplerate <= 0 or info.frames <= 0:
            raise ValueError("The WAV file must contain audio frames at a valid sample rate.")
        if max_duration_seconds is not None:
            duration_seconds = info.frames / info.samplerate
            if duration_seconds > max_duration_seconds:
                raise ValueError(
                    "Audio is too long. The maximum supported duration is "
                    f"{max_duration_seconds:g} seconds."
                )

    try:
        samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Could not decode the uploaded audio file.") from exc
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        if samples.shape[1] > 2:
            raise ValueError("Only mono or stereo audio is supported.")
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Decoded audio must contain a non-empty mono waveform.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains invalid numeric samples.")

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
