"""Physical lifetime rollout client used by Online Recurrent PPO."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    ObservationSignal,
    SpeechSignal,
)
from harness.transport.control import HarnessControlClient
from model.types import SpeechMode, StreamUnit
from runtime.codec_worker import CodecWorkerClient
from runtime.config import ProjectConfig
from runtime.video import video_tensor


class PhysicalRolloutClient:
    """Bridges Model Core token samples to Harness physical signals.

    The Harness never receives token arrays.  Token log-probabilities remain in
    the training trace while this client decodes speech output at the
    training-to-environment boundary.
    """

    def __init__(self, config: ProjectConfig) -> None:
        socket_path = config.training.rl.environment_socket
        codec_socket = config.training.rl.codec_socket
        if not socket_path:
            raise ValueError("Online Recurrent PPO requires training.rl.environment_socket")
        if not codec_socket:
            raise ValueError("Online Recurrent PPO requires training.rl.codec_socket")
        self.config = config
        self.harness = HarnessControlClient(socket_path)
        from runtime.codec import CodecIdentity

        self.codec = CodecWorkerClient(
            Path(codec_socket).expanduser(),
            CodecIdentity(
                config.data.codec_id,
                config.data.codec_weight_hash,
                config.data.codec_revision,
                sample_rate=config.data.audio_sample_rate,
                frame_rate=config.data.codec_frame_rate,
                frame_samples=config.data.unit_audio_samples,
                codebooks=config.data.codec_codebooks,
                codebook_size=config.data.codec_codebook_size,
            ),
        )
        self.codec.health()
        self.session_id: str | None = None
        self.speech_active = False

    def identity(self) -> dict[str, str]:
        return self.harness.identity()

    def start_lifetime_session(
        self, scenario_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        self.session_id = session_id
        self.codec.reset(session_id, replay=False)
        self.speech_active = False
        return self.harness.start_lifetime_session(scenario_id, seed, session_id)

    def resume_lifetime_session(
        self,
        session_id: str,
        expected_next_unit: int,
        *,
        speech_history: Iterable[torch.Tensor] = (),
    ) -> ObservationSignal:
        self.session_id = session_id
        history = tuple(speech_history)
        self.codec.restore_decode_history(session_id, history)
        self.speech_active = bool(history)
        return self.harness.resume_lifetime_session(session_id, expected_next_unit)

    def actuation(
        self,
        observation: ObservationSignal,
        mode: torch.Tensor,
        codes: torch.Tensor,
    ) -> ActuationSignal:
        selected_mode = int(mode.reshape(-1)[0].item())
        if selected_mode == int(SpeechMode.SILENCE):
            self.speech_active = False
            pcm = np.zeros(self.config.data.unit_audio_samples, dtype=np.float32).tobytes()
            speech = SpeechSignal(pcm, silent=True)
        else:
            if not self.speech_active:
                self.codec.reset(observation.session_id, replay=False)
            code_tensor = codes.detach().to("cpu").long().reshape(
                self.config.model.speech_frames_per_unit, self.config.model.speech_codebooks
            )
            decoded = self.codec.decode_step(
                code_tensor.transpose(0, 1).unsqueeze(0), observation.session_id
            )
            values = decoded.detach().cpu().float().reshape(-1).numpy()
            if values.size != self.config.data.unit_audio_samples:
                raise RuntimeError("codec decoder returned an invalid speech unit")
            if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
                raise RuntimeError("codec decoder returned invalid PCM values")
            speech = SpeechSignal(values.tobytes())
            self.speech_active = True
        return ActuationSignal(observation.session_id, observation.unit_index, speech)

    def step(
        self,
        observation: ObservationSignal,
        mode: torch.Tensor,
        codes: torch.Tensor,
    ) -> tuple[ObservationSignal, EnvironmentReceipt, ActuationSignal]:
        output = self.actuation(observation, mode, codes)
        next_observation, receipt = self.harness.apply(output)
        return next_observation, receipt, output

    def close(self) -> None:
        if self.session_id is not None:
            self.harness.close(self.session_id)
            self.session_id = None

    def detach(self) -> None:
        """Disconnect this client without closing the persistent lifetime session."""
        self.session_id = None


def observation_to_stream_unit(observation: ObservationSignal, config: ProjectConfig) -> StreamUnit:
    dtype = np.float32 if observation.mic.encoding == "pcm_f32le" else np.int16
    mic = np.frombuffer(observation.mic.samples, dtype=dtype)
    if dtype == np.int16:
        mic = mic.astype(np.float32) / 32768.0
    if mic.size != config.data.unit_audio_samples:
        raise ValueError("physical observation microphone has wrong unit size")
    if not np.isfinite(mic).all() or np.max(np.abs(mic)) > 1.0:
        raise ValueError("physical observation microphone PCM is not normalized and finite")
    video = video_tensor(observation.video)
    return StreamUnit(
        timestamp_ms=torch.tensor([observation.timestamp_ms], dtype=torch.long),
        delta_ms=torch.tensor([observation.delta_ms], dtype=torch.long),
        mic_audio=torch.from_numpy(mic.copy()).reshape(1, -1),
        video=video,
        speech_mode=torch.zeros(1, dtype=torch.long),
        speech_mode_mask=torch.zeros(1, dtype=torch.bool),
        speech_codes=torch.zeros(
            1, config.model.speech_frames_per_unit, config.model.speech_codebooks, dtype=torch.long
        ),
        speech_codec_mask=torch.zeros(1, config.model.speech_frames_per_unit, dtype=torch.bool),
    )
