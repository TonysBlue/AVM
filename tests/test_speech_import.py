from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from data import EpisodeShardReader, import_speech_manifest, write_episode_shards
from data.cli import main
from runtime.codec import CodecIdentity
from runtime.config import ProjectConfig

from tests.support.physical_workers import CodecWorker


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
    _write_audio(tmp_path / "target.wav", np.zeros(samples), 16_000)
    manifest = tmp_path / "source.jsonl"

    _write_audio(tmp_path / "mic.wav", np.zeros(samples), 8_000)
    _write_manifest(manifest, _record())
    with pytest.raises(ValueError, match="pre-resampled to 16000 Hz"):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))

    _write_audio(tmp_path / "mic.wav", np.zeros((samples, 2)), 16_000)
    with pytest.raises(ValueError, match="must be mono"):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))


def test_import_pads_to_200_ms_and_supervises_silence(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.full(frame + frame // 2, 0.1), 16_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 16_000)
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
        assert unit.speech_codec_mask.all()


def test_import_retains_speech_and_silence_targets(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    target = np.concatenate([np.full(frame, 0.25), np.full(frame, 0.25), np.zeros(frame)])
    _write_audio(tmp_path / "mic.wav", np.zeros(target.size), 16_000)
    _write_audio(tmp_path / "target.wav", target, 16_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, _record())

    episode = next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))
    assert all(unit.speech_codec_mask.all() for unit in episode.units)


def test_import_retains_silence_supervision_outside_explicit_segments(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    target = np.concatenate(
        [np.zeros(frame), np.full(frame + frame // 2, 0.25), np.zeros(2 * frame)]
    )
    _write_audio(tmp_path / "mic.wav", np.zeros(target.size), 16_000)
    _write_audio(tmp_path / "target.wav", target, 16_000)
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

    assert all(unit.speech_codec_mask.all() for unit in episode.units)
    assert len(episode.units) == 5


def test_import_rejects_unaligned_explicit_segment(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(3 * frame), 16_000)
    _write_audio(tmp_path / "target.wav", np.zeros(3 * frame), 16_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(
        manifest,
        _record(
            target_segments=[{"turn_id": "assistant-1", "start_sample": 1, "end_sample": frame}]
        ),
    )

    with pytest.raises(ValueError, match="200 ms boundary"):
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
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 16_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 16_000)
    manifest = tmp_path / "source.jsonl"
    _write_manifest(manifest, _record(**overrides))

    with pytest.raises(ValueError, match=message):
        next(import_speech_manifest(manifest, smoke_config.data, smoke_config.model))


def test_staging_shard_is_rejected_until_speech_is_encoded(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 16_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 16_000)
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


def test_cli_import_encode_validate_workflow(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    frame = smoke_config.data.unit_audio_samples
    _write_audio(tmp_path / "mic.wav", np.zeros(frame), 16_000)
    _write_audio(tmp_path / "target.wav", np.zeros(frame), 16_000)
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
    server = CodecWorker(tmp_path / "codec.sock", identity)
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
                    "--set",
                    f"runtime.native_acoustic_socket={server.path}",
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
