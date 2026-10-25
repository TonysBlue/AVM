from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 3_200


def write_flac(path: Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_16")


def read_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    values, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    if values.shape[1] != 1:
        raise ValueError(f"{path} must be mono")
    waveform = values[:, 0]
    if actual_rate != sample_rate:
        raise ValueError(f"{path} must have sample rate {sample_rate}, got {actual_rate}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"{path} contains non-finite audio")
    return waveform


def fixture_voice(text: str, voice: int, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    duration = min(1.2, max(0.32, len(text) * 0.035))
    samples = max(int(duration * sample_rate), int(0.3 * sample_rate))
    phase = np.arange(samples, dtype=np.float32) / sample_rate
    frequency = 155 + 23 * (voice % 7)
    envelope = np.minimum(phase / 0.02, 1.0) * np.minimum((duration - phase) / 0.03, 1.0)
    envelope = np.clip(envelope, 0, 1)
    waveform = 0.08 * np.sin(2 * math.pi * frequency * phase) * envelope
    return waveform.astype(np.float32)


def quality_metrics(path: Path) -> dict[str, float | int | bool]:
    waveform = read_mono(path)
    peak = float(np.abs(waveform).max(initial=0.0))
    clipping = float(np.mean(np.abs(waveform) >= 0.999))
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    return {
        "samples": int(waveform.size),
        "duration_seconds": waveform.size / SAMPLE_RATE,
        "peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "clipping_fraction": clipping,
        "finite": bool(np.isfinite(waveform).all()),
    }


def align_up(sample: int, frame_samples: int = FRAME_SAMPLES) -> int:
    return -(-sample // frame_samples) * frame_samples
