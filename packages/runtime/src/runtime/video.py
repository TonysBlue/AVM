"""Shared physical video normalization and per-session temporal validation."""

from __future__ import annotations

import numpy as np
import torch
from contracts import ObservationSignal, VideoSignal
from torch.nn import functional as F


def video_tensor(video: VideoSignal) -> torch.Tensor:
    channels = 4 if video.pixel_format == "rgba32" else 3
    pixels = np.frombuffer(video.image, dtype=np.uint8).reshape(video.height, video.width, channels)
    tensor = torch.from_numpy(pixels[..., :3].copy()).permute(2, 0, 1).float()[None] / 255
    if tensor.shape[-2:] != (224, 224):
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    return tensor


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
