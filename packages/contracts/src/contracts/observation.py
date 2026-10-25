from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MicSignal:
    samples: bytes
    sample_rate_hz: int = 16_000
    channels: int = 1
    encoding: str = "pcm_f32le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000 or self.channels != 1:
            raise ValueError("the realtime microphone clock requires 16 kHz mono")
        if self.encoding not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("unsupported microphone encoding")
        if not self.samples:
            raise ValueError("microphone samples are required")
        bytes_per_sample = 4 if self.encoding == "pcm_f32le" else 2
        expected = 3200 * bytes_per_sample
        if len(self.samples) != expected:
            raise ValueError("microphone PCM must contain exactly one 200 ms unit")


@dataclass(frozen=True, slots=True)
class VideoSignal:
    image: bytes
    width: int
    height: int
    captured_at_ms: int
    revision: int
    pixel_format: str = "rgb24"
    encoding: str = "raw"

    def __post_init__(self) -> None:
        if self.captured_at_ms < 0 or self.revision < 0:
            raise ValueError("video timestamp and revision must be non-negative")
        if self.width < 1 or self.height < 1:
            raise ValueError("video dimensions must be positive")
        if self.pixel_format not in {"rgb24", "rgba32"}:
            raise ValueError("unsupported video pixel format")
        if self.encoding != "raw":
            raise ValueError("physical video requires raw RGB or RGBA pixels")
        if not self.image:
            raise ValueError("video image is required")
        if self.encoding == "raw":
            channels = 3 if self.pixel_format == "rgb24" else 4
            if len(self.image) != self.width * self.height * channels:
                raise ValueError("raw video image size does not match its dimensions")


@dataclass(frozen=True, slots=True)
class ObservationSignal:
    session_id: str
    unit_index: int
    timestamp_ms: int
    delta_ms: int
    mic: MicSignal
    video: VideoSignal

    def __post_init__(self) -> None:
        if self.video.captured_at_ms > self.timestamp_ms:
            raise ValueError("video frame cannot be from the future")
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.unit_index < 0 or self.timestamp_ms < 0 or self.delta_ms != 200:
            raise ValueError("invalid observation clock")
