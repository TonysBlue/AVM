from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from contracts import ActuationSignal, ObservationSignal
from model import StreamingLatentLoop
from model.types import AudioVideoInput, SpeechSamplingConfig
from runtime.config import ProjectConfig
from runtime.continuous_speech import ContinuousSpeechClock, ContinuousSpeechExecutor
from runtime.native_acoustics import build_native_decoder
from runtime.video import validate_video_time, video_tensor


@dataclass(slots=True)
class ModelSession:
    session_id: str
    state: object
    executor: ContinuousSpeechExecutor
    next_unit: int = 0
    last_observation: ObservationSignal | None = None


class ModelService:
    """Physical signals only; capture belongs to Harness, value stays private."""

    def __init__(
        self,
        config: ProjectConfig,
        checkpoint: str | None = None,
        device: str = "cpu",
        speech_decoder: object | None = None,
    ) -> None:
        self.config, self.device = config, torch.device(device)
        if config.data.dataset != "synthetic":
            from runtime.native_assets import audit_native_assets, require_native_qualification

            audit = audit_native_assets(config.model.native_source_lock, model_weights=False)
            require_native_qualification(audit, config.model.native_qualification)
        self.model = StreamingLatentLoop(config.model, load_native=checkpoint is None)
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if (
                not isinstance(payload.get("model"), dict)
                or payload.get("metadata", {}).get("model_schema_id") != "avm-model-v3"
            ):
                raise ValueError("model service requires a complete native AVM checkpoint")
            from runtime.native_assets import validate_native_checkpoint_identity

            validate_native_checkpoint_identity(
                payload["metadata"], config.model.native_source_lock,
                config.model.native_qualification)
            for key in ("codec_id", "codec_revision", "codec_weight_hash"):
                if payload["metadata"].get(key) != getattr(config.data, key):
                    raise ValueError(f"checkpoint {key} differs from service identity")
            self.model.load_state_dict(payload["model"], strict=True, assign=True)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self._sessions: dict[str, ModelSession] = {}
        self._lock = threading.RLock()
        if speech_decoder is not None and config.data.dataset != "synthetic":
            raise ValueError("injected speech decoders are restricted to explicit synthetic tests")
        self._speech_decoder = speech_decoder

    def identity(self) -> dict[str, str]:
        return {
            "service": "model-service",
            "protocol_version": "avm-realtime-v2",
            "model_schema_id": "avm-model-v3",
            "codec_id": self.config.data.codec_id,
        }

    def open_session(self, session_id: str | None = None) -> str:
        selected = session_id or uuid.uuid4().hex
        with self._lock:
            if self._sessions:
                raise ValueError("model service already has an active lifetime session")
            decoder = self._speech_decoder
            if decoder is None:
                decoder = build_native_decoder(self.config)
            clock = ContinuousSpeechClock(
                lookahead_tokens=decoder.lookahead_tokens,
                vocoder_tail_samples=decoder.tail_samples,
                playback_delay_ms=self.config.runtime.playback_delay_ms,
            )
            executor = ContinuousSpeechExecutor(
                decoder, clock, vocab_size=self.config.model.speech_codebook_size
            )
            executor.reset(selected)
            self._sessions[selected] = ModelSession(
                selected, self.model.initial_state(1, self.device), executor
            )
        return selected

    def finish_session(self, session_id: str) -> list[ActuationSignal]:
        with self._lock:
            return self._sessions[session_id].executor.finish()

    def cancel_speech(self, session_id: str) -> int:
        with self._lock:
            session = self._sessions[session_id]
            session.executor.reset(session_id, unit_offset=session.next_unit)
            session.state.speech_local = self.model.speech_head.initial_state(
                1,
                self.device,
                next(self.model.parameters()).dtype,
            )
            return session.executor.epoch

    def close_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id)
            session.executor.close()

    @staticmethod
    def _video_tensor(observation: ObservationSignal) -> torch.Tensor:
        return video_tensor(observation.video)

    def _unit(self, observation: ObservationSignal) -> AudioVideoInput:
        dtype = np.float32 if observation.mic.encoding == "pcm_f32le" else np.int16
        mic = np.frombuffer(observation.mic.samples, dtype=dtype)
        if dtype == np.int16:
            mic = mic.astype(np.float32) / 32768.0
        if mic.size != 3200 or not np.isfinite(mic).all() or np.max(np.abs(mic)) > 1:
            raise ValueError("mic signal must contain finite normalized 16 kHz / 200 ms PCM")
        video = video_tensor(
            observation.video,
            height=self.config.data.video_height,
            width=self.config.data.video_width,
            patch_size=self.config.model.vision_patch_size,
        )
        return AudioVideoInput(torch.from_numpy(mic.copy()).reshape(1, -1), video)

    @torch.inference_mode()
    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        with self._lock:
            session = self._sessions.get(observation.session_id)
            if session is None:
                raise KeyError("unknown model session")
            if observation.unit_index != session.next_unit:
                raise ValueError("model unit index is out of order")
            validate_video_time(
                observation, session.last_observation, self.config.data.video_max_age_ms
            )
            raw = self._unit(observation)
            dtype = next(self.model.parameters()).dtype
            unit = AudioVideoInput(
                raw.mic_audio.to(self.device, dtype=dtype), raw.video.to(self.device, dtype=dtype)
            )
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                generated = self.model.generate_step(
                    unit, session.state, SpeechSamplingConfig(top_k=250)
                )
            output = session.executor.step(
                observation, generated.speech_codes, elapsed_ms=(
                    time.monotonic() * 1000 - observation.timestamp_ms
                    if self.config.runtime.clock_kind == "monotonic"
                    else 0.0),
            )
            session.state = generated.output.state
            session.next_unit += 1
            session.last_observation = observation
            return output
