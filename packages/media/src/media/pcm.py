from __future__ import annotations

import numpy as np


def validate_pcm_unit(samples: bytes, *, sample_rate_hz: int = 24_000, channels: int = 1) -> None:
    if sample_rate_hz not in {16000, 24000} or channels != 1:
        raise ValueError("realtime PCM must be 16 kHz microphone or 24 kHz speech mono")
    if len(samples) != sample_rate_hz // 5 * 4:
        raise ValueError("one PCM unit must contain exactly 200 ms of float32 samples")
    values = np.frombuffer(samples, dtype=np.float32)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
        raise ValueError("PCM must contain finite samples in [-1, 1]")


def silence_pcm(samples: int = 4800) -> bytes:
    """Physical test/capture utility; model output never uses this as a gate."""
    if samples != 4800:
        raise ValueError("realtime speech output is exactly one 200 ms unit")
    return np.zeros(samples, dtype=np.float32).tobytes()


def resample_pcm(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited conversion of a complete capture/target timeline."""
    import math

    from scipy.signal import resample_poly

    if values.ndim != 1 or source_rate <= 0 or target_rate <= 0:
        raise ValueError("resampling requires mono PCM and positive sample rates")
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(values, target_rate // divisor, source_rate // divisor).astype(np.float32)
