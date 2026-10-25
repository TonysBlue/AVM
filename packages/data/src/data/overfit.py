from __future__ import annotations

from collections.abc import Iterator

import torch
from model.types import Episode, StreamUnit
from runtime.config import DataConfig, ModelConfig


class SpeechOverfitDataset:
    """Small deterministic direct-speech trajectories for the overfit gate."""

    def __init__(self, data: DataConfig, model: ModelConfig) -> None:
        if data.episode_units < 12:
            raise ValueError("speech overfit trajectories require at least 12 units")
        self.data = data
        self.model = model

    def __len__(self) -> int:
        return self.data.train_episodes

    def __iter__(self) -> Iterator[Episode]:
        for episode_index in range(self.data.train_episodes):
            yield self.make_episode(episode_index)

    def make_episode(self, episode_index: int) -> Episode:
        response_class = episode_index % 8
        variant = episode_index // 8
        mic = self._microphone_prompt(response_class, variant)
        target = self._target_response(response_class)
        start_tick, active_ticks = 5, 8
        units = [
            self._unit(index, mic, response_class, start_tick, active_ticks)
            for index in range(self.data.episode_units)
        ]
        episode = Episode(
            episode_id=f"direct-speech-overfit-{episode_index:04d}",
            units=units,
            metadata={
                "source": "synthetic",
                "source_license": "project-generated",
                "redistribution_allowed": True,
                "language": "zxx-procedural",
                "split": "train",
                "scenario": "direct-speech-codec-overfit-gate",
                "device_id_hash": "direct-speech-overfit-device",
                "session_id_hash": f"direct-speech-overfit-session-{episode_index:04d}",
                "sample_rate": self.data.audio_sample_rate,
                "unit_ms": self.data.unit_ms,
                "codec_frame_rate": self.data.codec_frame_rate,
                "codec_id": self.data.codec_id,
                "codec_weight_hash": self.data.codec_weight_hash,
                "codec_revision": self.data.codec_revision,
                "speech_codes_encoded": False,
                "response_class": response_class,
                "variant": variant,
                "stage": "sft",
                "schema_id": "avm-episode-v2",
                "dataset_scale": self.data.dataset,
                "sample_kind": "supervised_episode",
                "supervision_kind": "speech",
                "task_id": f"overfit-task-{episode_index}",
                "environment_id": "synthetic-test-only",
                "environment_version": "1",
                "protocol_version": "avm-realtime-v2",
            },
            target_speech=target,
            sample_index_in_shard=episode_index,
        )
        episode.validate(
            audio_samples=self.data.unit_audio_samples,
            speech_frames=self.model.speech_frames_per_unit,
            speech_codebooks=self.model.speech_codebooks,
            speech_codebook_size=self.model.speech_codebook_size,
        )
        return episode

    def _microphone_prompt(self, response_class: int, variant: int) -> torch.Tensor:
        total = self.data.episode_units * self.data.unit_audio_samples
        waveform = torch.zeros(total)
        prompt_samples = 3 * self.data.unit_audio_samples
        time = torch.arange(prompt_samples) / self.data.audio_sample_rate
        fundamental = 150.0 + 25.0 * response_class
        secondary = 410.0 + 15.0 * variant
        prompt = 0.12 * torch.sin(2 * torch.pi * fundamental * time)
        prompt += 0.04 * torch.sin(2 * torch.pi * secondary * time)
        prompt *= self._edge_envelope(prompt_samples)
        waveform[:prompt_samples] = prompt
        return waveform

    def _target_response(self, response_class: int) -> torch.Tensor:
        total = self.data.episode_units * self.data.unit_audio_samples
        waveform = torch.zeros(total)
        start_tick, active_ticks = 5, 8
        active_samples = active_ticks * self.data.unit_audio_samples
        time = torch.arange(active_samples) / self.data.audio_sample_rate
        fundamental = 180.0 + 30.0 * response_class
        response = 0.16 * torch.sin(2 * torch.pi * fundamental * time)
        response += 0.05 * torch.sin(2 * torch.pi * fundamental * 2.0 * time + 0.2)
        response += 0.02 * torch.sin(2 * torch.pi * fundamental * 3.0 * time + 0.4)
        response *= self._edge_envelope(active_samples)
        offset = start_tick * self.data.unit_audio_samples
        waveform[offset : offset + active_samples] = response
        return waveform

    def _edge_envelope(self, samples: int) -> torch.Tensor:
        ramp_samples = min(self.data.audio_sample_rate // 100, samples // 2)
        envelope = torch.ones(samples)
        ramp = torch.linspace(0.0, 1.0, ramp_samples)
        envelope[:ramp_samples] = ramp
        envelope[-ramp_samples:] = ramp.flip(0)
        return envelope

    def _unit(
        self,
        index: int,
        microphone: torch.Tensor,
        response_class: int,
        start_tick: int,
        active_ticks: int,
    ) -> StreamUnit:
        frame = self.data.unit_audio_samples
        # The direct codec target is filled by the codec preparation stage.
        return StreamUnit(
            timestamp_ms=torch.tensor([index * self.data.unit_ms]),
            delta_ms=torch.tensor([self.data.unit_ms]),
            mic_audio=microphone[None, index * frame : (index + 1) * frame],
            video=torch.zeros(1, 3, self.data.video_height, self.data.video_width),
            speech_codes=torch.zeros(
                1, self.model.speech_frames_per_unit, self.model.speech_codebooks, dtype=torch.long
            ),
            speech_codec_mask=torch.ones((1, 5), dtype=torch.bool),
        )
