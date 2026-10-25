from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeechSignal:
    pcm: bytes
    sample_rate_hz: int = 24_000
    channels: int = 1
    encoding: str = "pcm_f32le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 24_000 or self.channels != 1:
            raise ValueError("speech output must be 24 kHz mono")
        if not self.pcm:
            raise ValueError("speech PCM is required")
        if self.encoding not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("unsupported speech encoding")
        bytes_per_sample = 4 if self.encoding == "pcm_f32le" else 2
        expected = 4800 * bytes_per_sample
        if len(self.pcm) != expected:
            raise ValueError("speech PCM must contain exactly one 200 ms unit")


@dataclass(frozen=True, slots=True)
class ActuationSignal:
    session_id: str
    unit_index: int
    speech: SpeechSignal | None
    speech_unit_index: int | None = None
    play_at_ms: int | None = None
    generation_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.session_id or self.unit_index < 0:
            raise ValueError("invalid actuation identity")
        if self.generation_epoch < 0 or (self.play_at_ms is not None and self.play_at_ms < 0):
            raise ValueError("invalid playback clock or generation epoch")
        if self.speech is None:
            if self.speech_unit_index is not None or self.play_at_ms is not None:
                raise ValueError("an actuation without PCM cannot claim a playback unit")
        elif self.speech_unit_index is None:
            object.__setattr__(self, "speech_unit_index", self.unit_index)
