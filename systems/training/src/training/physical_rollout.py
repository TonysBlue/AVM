"""Physical lifetime rollout through the same continuous speech executor as serving."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import torch
from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal
from harness.transport.control import HarnessControlClient
from model.types import StreamUnit
from runtime.config import ProjectConfig
from runtime.continuous_speech import ContinuousSpeechClock, ContinuousSpeechExecutor
from runtime.native_acoustics import build_native_decoder
from runtime.video import validate_video_time, video_tensor


class PhysicalRolloutClient:
    def __init__(self, config: ProjectConfig) -> None:
        if not config.training.rl.environment_socket:
            raise ValueError("physical PPO requires a dialogue Harness socket")
        self.config = config
        self.harness = HarnessControlClient(config.training.rl.environment_socket)
        decoder = build_native_decoder(config)
        clock = ContinuousSpeechClock(
            lookahead_tokens=decoder.lookahead_tokens,
            vocoder_tail_samples=decoder.tail_samples,
            playback_delay_ms=config.runtime.playback_delay_ms,
        )
        self.executor = ContinuousSpeechExecutor(
            decoder, clock, vocab_size=config.model.speech_codebook_size
        )
        self.session_id: str | None = None

    def identity(self) -> dict[str, str]:
        return self.harness.identity()

    def start_lifetime_session(self, scenario_id: str, seed: int, session_id: str):
        self.session_id = session_id
        self.executor.reset(session_id)
        observation = self.harness.start_lifetime_session(scenario_id, seed, session_id)
        validate_video_time(observation, None, self.config.data.video_max_age_ms)
        return observation

    def resume_lifetime_session(
        self,
        session_id: str,
        expected_next_unit: int,
        *,
        speech_history: Iterable[torch.Tensor] = (),
    ):
        observation = self.harness.resume_lifetime_session(session_id, expected_next_unit)
        self.session_id = session_id
        self.executor.reset(session_id)
        stream = self.executor.stream
        origin = observation.timestamp_ms - expected_next_unit * 200
        count = 0
        for index, codes in enumerate(speech_history):
            timestamp = origin + index * 200
            stream.push(index, timestamp, codes.reshape(-1).long(), epoch=self.executor.epoch)
            if stream.next_output_unit < stream.next_input_unit:
                stream.take(stream.next_output_unit, now_ms=timestamp, epoch=self.executor.epoch)
            count += 1
        if count != expected_next_unit:
            raise ValueError("native speech recovery requires the complete sampled token timeline")
        return observation

    def actuation(self, observation: ObservationSignal, codes: torch.Tensor) -> ActuationSignal:
        elapsed = (time.monotonic() * 1000 - observation.timestamp_ms
                   if self.config.runtime.clock_kind == "monotonic" else 0.0)
        return self.executor.step(observation, codes, elapsed_ms=elapsed)

    def step(
        self, observation: ObservationSignal, codes: torch.Tensor
    ) -> tuple[ObservationSignal, EnvironmentReceipt, ActuationSignal]:
        output = self.actuation(observation, codes)
        next_observation, receipt = self.harness.apply(output)
        validate_video_time(next_observation, observation, self.config.data.video_max_age_ms)
        return next_observation, receipt, output

    def close(self) -> None:
        if self.session_id is not None:
            outputs = self.executor.finish()
            self.harness.flush_speech(self.session_id, outputs)
            self.harness.close(self.session_id)
            self.session_id = None
        self.executor.close()

    def detach(self) -> None:
        self.session_id = None
        self.executor.close()


def observation_to_stream_unit(observation: ObservationSignal, config: ProjectConfig) -> StreamUnit:
    dtype = np.float32 if observation.mic.encoding == "pcm_f32le" else np.int16
    mic = np.frombuffer(observation.mic.samples, dtype=dtype)
    if dtype == np.int16:
        mic = mic.astype(np.float32) / 32768.0
    if mic.size != 3200 or not np.isfinite(mic).all() or np.max(np.abs(mic)) > 1:
        raise ValueError("physical observation must contain normalized 16 kHz / 200 ms PCM")
    video = video_tensor(
        observation.video,
        height=config.data.video_height,
        width=config.data.video_width,
        patch_size=config.model.vision_patch_size,
    )
    return StreamUnit(
        timestamp_ms=torch.tensor([observation.timestamp_ms], dtype=torch.long),
        delta_ms=torch.tensor([200], dtype=torch.long),
        mic_audio=torch.from_numpy(mic.copy()).reshape(1, -1),
        video=video,
        speech_codes=torch.zeros(1, 5, 1, dtype=torch.long),
        speech_codec_mask=torch.zeros(1, 5, dtype=torch.bool),
    )
