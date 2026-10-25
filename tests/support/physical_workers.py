from __future__ import annotations

import base64
import json
import socket
import threading
from pathlib import Path

import numpy as np
import torch
from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    VideoSignal,
)
from contracts.framing import frame, read_frame
from runtime.codec import CodecIdentity
from runtime.config import ProjectConfig
from runtime.native_acoustics import NativeAcousticServer


class CodecWorker:
    def __init__(self, path: Path, identity: CodecIdentity) -> None:
        self.path = path
        self.server = NativeAcousticServer(ProtocolAcousticDecoder(identity), path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        import time
        for _ in range(100):
            if path.exists():
                break
            time.sleep(0.01)

    def close(self) -> None:
        self.server.close()
        self.thread.join(timeout=1)


class ProtocolAcousticDecoder:
    """Explicit test-only timing protocol; provides no acoustic quality evidence."""

    lookahead_tokens = 3
    tail_samples = 3840

    def __init__(self, identity: CodecIdentity) -> None:
        self.vocab_size = identity.codebook_size
        self.identity = {
            "acoustic_protocol_id": "avm-native-acoustics-v1",
            "codec_id": identity.codec_id,
            "codec_revision": identity.revision,
            "codec_weight_sha256": identity.weight_sha256,
            "source_lock_sha256": "synthetic",
            "vocab_size": self.vocab_size,
            "lookahead_tokens": self.lookahead_tokens,
            "tail_samples": self.tail_samples,
            "target_encoding_context": {"kind": "explicit-test-fixture"},
        }
        self.reset()

    def reset(self) -> None:
        self.started = False

    def synchronize(self) -> None:
        pass

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.arange(waveform.numel() // 640) % (self.vocab_size - 1)

    def decode(self, tokens: list[int], *, final: bool) -> torch.Tensor:
        count = len(tokens) if final else len(tokens) - self.lookahead_tokens
        samples = count * 960
        if final and self.started:
            samples += self.tail_samples
        elif not final and not self.started:
            samples -= self.tail_samples
        self.started = True
        return torch.full((samples,), 0.01)

    def close(self) -> None:
        pass


class PhysicalBackend:
    environment_id = "test-physical"
    environment_version = "1"

    def __init__(self) -> None:
        self.start_count = 0
        self.seen: list[ActuationSignal] = []

    def start_lifetime_session(self, scenario_id: str, seed: int, session_id: str):
        self.start_count += 1
        self.session_id = session_id
        return ObservationSignal(
            session_id,
            0,
            0,
            200,
            MicSignal(np.zeros(3200, dtype=np.float32).tobytes()),
            VideoSignal(b"x" * (224 * 224 * 3), 224, 224, captured_at_ms=0, revision=0),
        )

    def apply(self, output: ActuationSignal):
        if not isinstance(output, ActuationSignal):
            raise TypeError("test physical backend requires ActuationSignal")
        self.seen.append(output)
        receipt = EnvironmentReceipt(output.session_id, output.unit_index, True)
        return (
            ObservationSignal(
                output.session_id,
                output.unit_index + 1,
                (output.unit_index + 1) * 200,
                200,
                MicSignal(np.zeros(3200, dtype=np.float32).tobytes()),
                VideoSignal(b"x" * (224 * 224 * 3), 224, 224,
                            captured_at_ms=(output.unit_index + 1) * 200,
                            revision=output.unit_index + 1),
            ),
            receipt,
        )

    def close(self) -> None:
        return None

    def flush_speech(self, outputs: list[ActuationSignal]) -> None:
        self.seen.extend(outputs)

    def cancel_speech(self, generation_epoch: int) -> None:
        pass


class RewardWorker:
    def __init__(self, path: Path, config: ProjectConfig, *, delayed: bool = False) -> None:
        self.path = path
        self.config = config
        self.delayed = delayed
        self.chain_by_unit: dict[int, str] = {}
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(16)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            block = connection.recv(size - len(output))
            if not block:
                raise ConnectionError
            output.extend(block)
        return bytes(output)

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                try:
                    payload = read_frame(
                        lambda size, active=connection: self._read_exact(active, size)
                    )
                    request = json.loads(payload)
                    if request["operation"] == "identity":
                        response = {
                            "ok": True,
                            "identity": {
                                "spec_id": self.config.training.rl.reward_spec_id,
                                "judge_model_id": self.config.training.rl.judge_model_id,
                                "judge_revision": self.config.training.rl.judge_revision,
                                "rubric_sha256": self.config.training.rl.rubric_sha256,
                            },
                        }
                    else:
                        if not base64.b64decode(request["observation"], validate=True):
                            raise ValueError("empty canonical observation")
                        unit_index = int(request["unit_index"])
                        self.chain_by_unit[unit_index] = request["observation_chain_sha256"]
                        events = []
                        event_unit = 9 if self.delayed else 7
                        if unit_index == event_unit:
                            events.append(
                                {
                                    "event_id": "goal-success",
                                    "lineage_id": request["lineage_id"],
                                    "session_id": request["session_id"],
                                    "goal_id": "goal",
                                    "goal_start_unit": 0,
                                    "outcome_unit": 7,
                                    "evidence_start_unit": 6,
                                    "evidence_end_unit": 7,
                                    "status": "finalized",
                                    "outcome": "success",
                                    "reward": {
                                        "task": 1.0,
                                        "speech_quality": 0.5,
                                        "latency_quality": 0.5,
                                        "safety_quality": 0.0,
                                    },
                                    "spec_id": self.config.training.rl.reward_spec_id,
                                    "judge_model_id": self.config.training.rl.judge_model_id,
                                    "judge_revision": self.config.training.rl.judge_revision,
                                    "rubric_sha256": self.config.training.rl.rubric_sha256,
                                    "observation_chain_end_sha256": self.chain_by_unit[7],
                                }
                            )
                        response = {
                            "ok": True,
                            "finalized_through_unit": (
                                min(unit_index, 6)
                                if self.delayed and unit_index < event_unit
                                else unit_index
                            ),
                            "events": events,
                        }
                    connection.sendall(frame(json.dumps(response).encode()))
                except (ConnectionError, OSError):
                    return

    def close(self) -> None:
        self.server.close()
        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(str(self.path))
        except OSError:
            pass
        self.thread.join(timeout=1)
