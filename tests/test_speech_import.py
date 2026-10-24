from __future__ import annotations

import json
import socket
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from data import EpisodeShardReader, import_speech_manifest, write_episode_shards
from data.cli import main
from model.types import SpeechMode
from runtime.codec import CodecIdentity
from runtime.codec_worker import receive_message, send_message
from runtime.config import ProjectConfig


def _write_audio(path: Path, values: np.ndarray, sample_rate: int) -> None:
    sf.write(path, values, sample_rate, subtype="FLOAT")


def _record(**overrides: object) -> dict[str, object]:
    return {
        "episode_id": "speech-0001",
        "mic_audio": "mic.wav",
        "target_speech": "target.wav",
        "source": "licensed-test-corpus",
        "source_license": "CC-BY-4.0",
        "redistribution_allowed": True,
        "language": "zh-CN",
        "split": "train",
        "session_id_hash": "session-a",
        **overrides,
    }


def _write_manifest(path: Path, record: dict[str, object]) -> None:
    if "videos" not in record:
        video_path = path.parent / "test-video.npz"
        np.savez_compressed(
            video_path, ticks=np.array([0]), frames=np.ones((1, 3, 224, 224), dtype=np.float32)
        )
        record = {**record, "videos": str(video_path)}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_import_requires_configured_sample_rate_and_mono(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    samples = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "target.wav", np.zeros(samples), 4_000)
    manifest = tmp_path / "source.jsonl"

    _write_audio(tmp_path / "mic.wav", np.zeros(samples), 8_000)
    _write_manifest(manifest, _record())
    with pytest.raises(ValueError, match="pre-resampled to 4000 Hz"):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))

    _write_audio(tmp_path / "mic.wav", np.zeros((samples, 2)), 4_000)
    with pytest.raises(ValueError, match="must be mono"):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))


def test_import_pads_to_80_ms_and_sets_speech_only_masks(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.full(frame + frame // 2, 0.1), 4_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 4_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, _record())

    episode = next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))

    assert len(episode.units) == 2
    assert episode.target_speech is not None
    assert episode.target_speech.numel() == 2 * frame
    assert torch.allclose(
        episode.units[1].mic_audio[:, : frame // 2],
        torch.full((1, frame // 2), 0.1),
    )
    assert torch.equal(episode.units[1].mic_audio[:, frame // 2 :], torch.zeros(1, frame // 2))
    for unit in episode.units:
        assert unit.speech_mode_mask.all()
        assert unit.speech_mode.item() == SpeechMode.SILENCE


def test_import_generates_start_continue_and_stop(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    target = np.concatenate([np.full(frame, 0.25), np.full(frame, 0.25), np.zeros(frame)])
    _write_audio(tmp_path / "mic.wav", np.zeros(target.size), 4_000)
    _write_audio(tmp_path / "target.wav", target, 4_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, _record())

    episode = next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))
    assert [unit.speech_mode.item() for unit in episode.units] == [
        SpeechMode.SPEECH,
        SpeechMode.SPEECH,
        SpeechMode.SILENCE,
    ]


def test_import_uses_explicit_segments_for_controls_and_masks(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    target = np.concatenate(
        [np.zeros(frame), np.full(frame + frame // 2, 0.25), np.zeros(2 * frame)]
    )
    _write_audio(tmp_path / "mic.wav", np.zeros(target.size), 4_000)
    _write_audio(tmp_path / "target.wav", target, 4_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(
        manifest,
        _record(
            target_segments=[
                {
                    "turn_id": "assistant-1",
                    "start_sample": frame,
                    "end_sample": frame + frame + frame // 2,
                }
            ]
        ),
    )

    episode = next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))

    assert [unit.speech_mode.item() for unit in episode.units] == [
        SpeechMode.SILENCE,
        SpeechMode.SPEECH,
        SpeechMode.SPEECH,
        SpeechMode.SILENCE,
        SpeechMode.SILENCE,
    ]
    assert [bool(unit.speech_codec_mask.item()) for unit in episode.units] == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_import_rejects_unaligned_explicit_segment(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(3 * frame), 4_000)
    _write_audio(tmp_path / "target.wav", np.zeros(3 * frame), 4_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(
        manifest,
        _record(
            target_segments=[{"turn_id": "assistant-1", "start_sample": 1, "end_sample": frame}]
        ),
    )

    with pytest.raises(ValueError, match="80 ms boundary"):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_license": ""}, "source_license must be a non-empty string"),
        ({"redistribution_allowed": 1}, "redistribution_allowed must be a boolean"),
    ],
)
def test_import_rejects_invalid_license_metadata(
    tmp_path: Path,
    smoke_config: ProjectConfig,
    overrides: dict[str, object],
    message: str,
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 4_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 4_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, _record(**overrides))

    with pytest.raises(ValueError, match=message):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))


def test_staging_shard_is_rejected_until_speech_is_encoded(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 4_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 4_000)
    source_manifest = tmp_path / "source.jsonl"
    _write_manifest(source_manifest, _record(content_sha256="untrusted"))
    episode = next(import_speech_manifest(source_manifest, smoke_config.data, smoke_config.model))
    written = write_episode_shards([episode], tmp_path / "staging-%06d.tar")
    reader = EpisodeShardReader(
        str(tmp_path / "staging-*.tar"), smoke_config.data, smoke_config.model
    )

    assert written[0]["content_sha256"] != "untrusted"
    with pytest.raises(ValueError, match="speech codes are not encoded"):
        next(iter(reader))


class _EncodingCodecServer:
    def __init__(self, path: Path, identity: CodecIdentity) -> None:
        self.path = path
        self.identity = identity
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
                if operation == "health":
                    send_message(connection, {"ok": True, "identity": asdict(self.identity)})
                elif operation == "reset":
                    send_message(connection, {"ok": True})
                elif operation == "encode_step":
                    codes = np.zeros((1, self.identity.codebooks, 1), dtype=np.uint16)
                    send_message(
                        connection,
                        {"ok": True, "dtype": "uint16", "shape": codes.shape},
                        codes.tobytes(),
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


def test_cli_import_encode_validate_workflow(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 4_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 4_000)
    source_manifest = tmp_path / "source.jsonl"
    _write_manifest(source_manifest, _record())
    staging = tmp_path / "staging-%06d.tar"
    processed = tmp_path / "processed-%06d.tar"
    identity = CodecIdentity(
        codec_id=smoke_config.data.codec_id,
        weight_sha256=smoke_config.data.codec_weight_hash,
        revision=smoke_config.data.codec_revision,
        sample_rate=smoke_config.data.audio_sample_rate,
        frame_rate=smoke_config.data.codec_frame_rate,
        frame_samples=smoke_config.data.unit_audio_samples,
        codebooks=smoke_config.data.codec_codebooks,
        codebook_size=smoke_config.data.codec_codebook_size,
    )
    server = _EncodingCodecServer(tmp_path / "codec.sock", identity)
    smoke_config.data.manifest = str(tmp_path / "unrelated-processed-manifest.jsonl")
    try:
        assert (
            main(
                [
                    "import-speech",
                    "--config",
                    "configs/smoke.yaml",
                    "--manifest",
                    str(source_manifest),
                    "--output",
                    str(staging),
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "encode-speech",
                    "--config",
                    "configs/smoke.yaml",
                    "--set",
                    f"data.manifest={smoke_config.data.manifest}",
                    "--shards",
                    str(tmp_path / "staging-*.tar"),
                    "--output",
                    str(processed),
                    "--socket",
                    str(server.path),
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "validate-data",
                    "--config",
                    "configs/smoke.yaml",
                    "--shards",
                    str(tmp_path / "processed-*.tar"),
                ]
            )
            == 0
        )
    finally:
        server.close()
