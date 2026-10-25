from __future__ import annotations

import socket
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from data import SyntheticEpisodeDataset
from media.metrics import boundary_discontinuity_db, codec_accuracy
from model import StreamingLatentLoop
from model.types import SpeechSamplingConfig
from runtime.codec import CodecIdentity
from runtime.codec_worker import (
    CodecWorkerClient,
    receive_message,
    send_message,
)
from runtime.config import ProjectConfig


def test_later_teacher_codebooks_do_not_change_earlier_logits(
    smoke_config: ProjectConfig,
) -> None:
    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model.initial_state(1, "cpu")
    first_codes = unit.speech_codes.clone()
    changed_codes = first_codes.clone()
    changed_codes[:, :, 0] = (changed_codes[:, :, 0] + 1) % 32

    first = model(unit.observation, state, first_codes).speech_codec_logits
    changed = model(unit.observation, state, changed_codes).speech_codec_logits

    assert torch.equal(first[:, :, 0], changed[:, :, 0])
    assert not torch.equal(first[:, :, 1], changed[:, :, 1])


def test_generate_step_returns_one_complete_codec_frame(
    smoke_config: ProjectConfig,
) -> None:
    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    generated = model.generate_step(
        unit.observation,
        model.initial_state(1, "cpu"),
        SpeechSamplingConfig(greedy=True),
    )

    assert generated.speech_codes.shape == (1, 1, smoke_config.model.speech_codebooks)
    assert generated.output.state.speech_local.previous_codes.shape == (
        1,
        smoke_config.model.speech_codebooks,
    )


def test_speech_mode_is_binary() -> None:
    logits = torch.tensor([[0.0, 1.0], [9.0, 8.0]])
    assert logits.argmax(dim=-1).tolist() == [1, 0]


def test_codec_accuracy_reports_each_codebook() -> None:
    logits = torch.zeros(1, 1, 2, 3)
    logits[0, 0, 0, 1] = 1
    logits[0, 0, 1, 2] = 1
    accuracy = codec_accuracy(logits, torch.tensor([[[1, 0]]]), torch.tensor([[True]]))
    assert accuracy.tolist() == [1.0, 0.0]


def test_boundary_metric_detects_large_jump() -> None:
    previous = torch.linspace(0, 0.1, 1_920)
    current = torch.linspace(1.0, 1.1, 1_920)
    assert boundary_discontinuity_db(previous, current) > 20


class FakeCodecServer:
    def __init__(self, path: Path, identity: CodecIdentity) -> None:
        self.path = path
        self.identity = identity
        self.operations: list[str] = []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen(8)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                try:
                    header, _ = receive_message(connection)
                except ConnectionError:
                    return
                operation = header["operation"]
                self.operations.append(operation)
                if operation == "health":
                    send_message(
                        connection,
                        {"ok": True, "identity": asdict(self.identity)},
                    )
                elif operation == "reset":
                    send_message(connection, {"ok": True})
                elif operation == "decode_step":
                    values = np.zeros((1, 1, 1_920), dtype=np.float32)
                    send_message(
                        connection,
                        {"ok": True, "dtype": "float32", "shape": values.shape},
                        values.tobytes(),
                    )
                elif operation == "encode_step":
                    values = np.zeros((1, self.identity.codebooks, 1), dtype=np.uint16)
                    send_message(
                        connection,
                        {"ok": True, "dtype": "uint16", "shape": values.shape},
                        values.tobytes(),
                    )
                else:
                    send_message(connection, {"ok": False, "error": "unsupported"})

    def close(self) -> None:
        self.server.close()
        try:
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(str(self.path))
        except OSError:
            pass
        self.thread.join(timeout=1)


def test_codec_client_validates_identity_and_replays_history(tmp_path: Path) -> None:
    identity = CodecIdentity("test", "hash", "revision", codebooks=2, codebook_size=32)
    server = FakeCodecServer(tmp_path / "codec.sock", identity)
    try:
        client = CodecWorkerClient(server.path, identity)
        assert client.health()["ok"]
        client.reset("one", replay=False)
        client.decode_step(torch.zeros(1, 2, 1, dtype=torch.long), "one")
        client.reset("one", replay=True)
        assert server.operations == ["health", "reset", "decode_step", "reset", "decode_step"]
    finally:
        server.close()


def test_codec_client_does_not_replay_history_into_new_session(tmp_path: Path) -> None:
    identity = CodecIdentity("test", "hash", "revision", codebooks=2, codebook_size=32)
    server = FakeCodecServer(tmp_path / "codec.sock", identity)
    try:
        client = CodecWorkerClient(server.path, identity)
        client.reset("one", replay=False)
        client.decode_step(torch.zeros(1, 2, 1, dtype=torch.long), "one")
        client.reset("two", replay=True)
        assert server.operations == ["reset", "decode_step", "reset"]
    finally:
        server.close()


def test_codec_client_rejects_wrong_shape(tmp_path: Path) -> None:
    identity = CodecIdentity("test", "hash", "revision", codebooks=2, codebook_size=32)
    client = CodecWorkerClient(tmp_path / "absent.sock", identity)
    with pytest.raises(ValueError, match="shape"):
        client.decode_step(torch.zeros(1, 2, dtype=torch.long), "one")


def test_codec_client_rejects_out_of_range_codes_before_uint16_conversion(
    tmp_path: Path,
) -> None:
    identity = CodecIdentity("test", "hash", "revision", codebooks=2, codebook_size=32)
    client = CodecWorkerClient(tmp_path / "absent.sock", identity)

    with pytest.raises(ValueError, match="outside the configured codebook"):
        client.decode_step(torch.tensor([[[-1], [32]]]), "one")


def test_codec_client_rejects_invalid_waveform_before_request(tmp_path: Path) -> None:
    identity = CodecIdentity(
        "test", "hash", "revision", frame_samples=4, codebooks=2, codebook_size=32
    )
    client = CodecWorkerClient(tmp_path / "absent.sock", identity)

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        client.encode_step(torch.full((1, 1, 4), 1.1), "one")


def test_codec_client_encodes_one_stream_frame(tmp_path: Path) -> None:
    identity = CodecIdentity("test", "hash", "revision", codebooks=2, codebook_size=32)
    server = FakeCodecServer(tmp_path / "codec.sock", identity)
    try:
        client = CodecWorkerClient(server.path, identity)
        client.reset("one", replay=False)
        codes = client.encode_step(torch.zeros(1, 1, 1_920), "one")
        assert codes.shape == (1, 2, 1)
        assert codes.dtype == torch.long
    finally:
        server.close()
