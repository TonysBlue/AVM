"""Shared physical video normalization and per-session temporal validation."""

from __future__ import annotations

import math

import numpy as np
import torch
from contracts import ObservationSignal, VideoSignal
from PIL import Image


def video_tensor(
    video: VideoSignal, *, height: int = 448, width: int = 448, patch_size: int = 14
) -> torch.Tensor:
    channels = 4 if video.pixel_format == "rgba32" else 3
    pixels = np.frombuffer(video.image, dtype=np.uint8).reshape(video.height, video.width, channels)
    ratio = video.width / video.height
    target_height = int(math.sqrt(height * width) / math.sqrt(ratio))
    target_width = int(target_height * ratio)
    size = (
        max(round(target_width / patch_size) * patch_size, patch_size),
        max(round(target_height / patch_size) * patch_size, patch_size),
    )
    image = Image.fromarray(pixels[..., :3]).resize(size, Image.Resampling.BICUBIC)
    values = np.asarray(image).copy()
    return torch.from_numpy(values).permute(2, 0, 1).float()[None] / 255


def validate_video_time(
    observation: ObservationSignal, previous: ObservationSignal | None, max_age_ms: int
) -> None:
    current = observation.video
    if observation.timestamp_ms - current.captured_at_ms > max_age_ms:
        raise ValueError("video source is stale")
    if previous is None:
        return
    if observation.timestamp_ms <= previous.timestamp_ms:
        raise ValueError("observation time moved backwards")
    old = previous.video
    if current.revision < old.revision or current.captured_at_ms < old.captured_at_ms:
        raise ValueError("video time or revision moved backwards")
    if current.revision == old.revision and current != old:
        raise ValueError("reused video revision changed its frame")
