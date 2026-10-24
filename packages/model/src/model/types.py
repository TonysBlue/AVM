from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from torch import Tensor
from torch.utils import _pytree


class SpeechMode(IntEnum):
    SILENCE = 0
    SPEECH = 1


@dataclass(slots=True)
class StreamUnit:
    """One batched, time-aligned multimodal training unit."""

    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    video: Tensor
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor

    @property
    def batch_size(self) -> int:
        return self.mic_audio.shape[0]

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> StreamUnit:
        return StreamUnit(
            timestamp_ms=self.timestamp_ms.to(device),
            delta_ms=self.delta_ms.to(device),
            mic_audio=self.mic_audio.to(device=device, dtype=dtype),
            video=self.video.to(device=device, dtype=dtype),
            speech_mode=self.speech_mode.to(device),
            speech_mode_mask=self.speech_mode_mask.to(device),
            speech_codes=self.speech_codes.to(device),
            speech_codec_mask=self.speech_codec_mask.to(device),
        )

    def validate(
        self,
        *,
        audio_samples: int,
        speech_frames: int,
        speech_codebooks: int,
        speech_codebook_size: int = 2**16,
        **_: int,
    ) -> None:
        batch = self.batch_size
        if self.mic_audio.shape != (batch, audio_samples):
            raise ValueError(f"mic_audio must have shape [B, {audio_samples}]")
        if self.video.shape != (batch, 3, 224, 224):
            raise ValueError("video must have shape [B, 3, 224, 224]")
        if self.speech_mode.shape != (batch,) or self.speech_mode_mask.shape != (batch,):
            raise ValueError("speech mode tensors must have shape [B]")
        if torch.any((self.speech_mode < 0) | (self.speech_mode > 1)):
            raise ValueError("speech mode must be SILENCE or SPEECH")
        if self.speech_codes.shape != (batch, speech_frames, speech_codebooks):
            raise ValueError("speech_codes has an incompatible shape")
        if self.speech_codec_mask.shape != (batch, speech_frames):
            raise ValueError("speech_codec_mask has an incompatible shape")
        valid_codes = self.speech_codes[self.speech_codec_mask]
        if valid_codes.numel() and (
            valid_codes.min() < 0 or valid_codes.max() >= speech_codebook_size
        ):
            raise ValueError("speech code is outside the configured codebook")
        if torch.any(self.delta_ms <= 0):
            raise ValueError("delta_ms must be positive")


@dataclass(slots=True)
class Episode:
    episode_id: str
    units: list[StreamUnit]
    metadata: dict[str, Any]
    target_speech: Tensor | None = None
    ordered_shard_index: int = 0
    sample_index_in_shard: int = 0

    def validate(self, **unit_shape: int) -> None:
        if not self.units:
            raise ValueError("episode must contain at least one stream unit")
        previous = -1
        for unit in self.units:
            unit.validate(**unit_shape)
            timestamp = int(unit.timestamp_ms[0].item())
            if timestamp <= previous:
                raise ValueError("episode timestamps must be strictly increasing")
            previous = timestamp
        if self.target_speech is not None:
            expected = len(self.units) * unit_shape["audio_samples"]
            if self.target_speech.numel() != expected:
                raise ValueError("target_speech must exactly cover the episode timeline")


@dataclass(slots=True)
class LayerKV:
    key: Tensor
    value: Tensor

    def detach(self) -> LayerKV:
        return LayerKV(self.key.detach(), self.value.detach())


@dataclass(slots=True)
class SlowMemoryState:
    """Fixed-capacity per-layer associative memory."""

    matrix: Tensor
    normalizer: Tensor

    def detach(self) -> SlowMemoryState:
        return SlowMemoryState(self.matrix.detach(), self.normalizer.detach())


@dataclass(slots=True)
class SpeechLocalState:
    temporal: Tensor
    previous_codes: Tensor

    def detach(self) -> SpeechLocalState:
        return SpeechLocalState(self.temporal.detach(), self.previous_codes.detach())


@dataclass(slots=True)
class RecurrentState:
    layer_kv: tuple[LayerKV, ...]
    semantic_memory: Tensor
    slow_memory: tuple[SlowMemoryState, ...]
    audio_cache: Tensor
    hidden: Tensor
    speech_local: SpeechLocalState
    unit_index: Tensor

    def detach(self) -> RecurrentState:
        return RecurrentState(
            layer_kv=tuple(cache.detach() for cache in self.layer_kv),
            semantic_memory=self.semantic_memory.detach(),
            slow_memory=tuple(memory.detach() for memory in self.slow_memory),
            audio_cache=self.audio_cache.detach(),
            hidden=self.hidden.detach(),
            speech_local=self.speech_local.detach(),
            unit_index=self.unit_index.detach(),
        )


@dataclass(slots=True)
class StepOutput:
    state: RecurrentState
    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    hidden: Tensor
    perceiver_slots: Tensor
    jepa_prediction: Tensor
    value: Tensor
    selected_speech_mode: Tensor

    def sampled_logprob(self, speech_mode: Tensor, speech_codes: Tensor) -> Tensor:
        mode = torch.log_softmax(self.speech_mode_logits, dim=-1).gather(
            -1, speech_mode.unsqueeze(-1)
        ).squeeze(-1)
        codec = torch.log_softmax(self.speech_codec_logits, dim=-1).gather(
            -1, speech_codes.unsqueeze(-1)
        ).squeeze(-1).sum(dim=(-1, -2))
        return mode + torch.where(speech_mode == int(SpeechMode.SPEECH), codec, 0.0)


@dataclass(slots=True)
class TrainingStepOutput:
    """Per-unit differentiable outputs without a retained recurrent state."""

    speech_mode_logits: Tensor
    speech_codec_logits: Tensor
    perceiver_slots: Tensor
    jepa_prediction: Tensor
    value: Tensor


def training_output(output: StepOutput) -> TrainingStepOutput:
    return TrainingStepOutput(
        speech_mode_logits=output.speech_mode_logits,
        speech_codec_logits=output.speech_codec_logits,
        perceiver_slots=output.perceiver_slots,
        jepa_prediction=output.jepa_prediction,
        value=output.value,
    )


@dataclass(frozen=True, slots=True)
class SpeechSamplingConfig:
    temperature: float = 0.8
    top_k: int = 250
    greedy: bool = False


@dataclass(slots=True)
class GenerationOutput:
    output: StepOutput
    speech_mode: Tensor
    speech_codes: Tensor


# Activation checkpointing must see every tensor nested in recurrent and step
# dataclasses; otherwise recomputation silently drops parts of the autograd graph.
for _checkpoint_dataclass in (
    LayerKV,
    SlowMemoryState,
    SpeechLocalState,
    RecurrentState,
    StepOutput,
    TrainingStepOutput,
):
    _pytree.register_dataclass(_checkpoint_dataclass)
