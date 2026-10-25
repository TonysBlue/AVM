from __future__ import annotations

from pathlib import Path

import pytest
import torch
from data import (
    EpisodeShardReader,
    SyntheticEpisodeDataset,
    load_manifest,
    write_episode_shards,
)
from data.cli import main
from runtime.config import ProjectConfig


def test_webdataset_episode_round_trip(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    original = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(2)
    manifest = write_episode_shards([original], tmp_path / "train-%06d.tar", max_size=10_000_000)
    reader = EpisodeShardReader(
        str(tmp_path / "train-*.tar"), smoke_config.data, smoke_config.model
    )
    decoded = next(iter(reader))

    assert manifest[0]["episode_id"] == original.episode_id
    assert len(manifest[0]["content_sha256"]) == 64
    assert "schema_version" not in manifest[0]
    assert manifest[0]["codec_id"] == smoke_config.data.codec_id
    assert decoded.episode_id == original.episode_id
    assert decoded.ordered_shard_index == 0
    assert decoded.sample_index_in_shard == 0
    assert len(decoded.units) == len(original.units)
    assert torch.equal(decoded.units[0].speech_codes, original.units[0].speech_codes)
    assert torch.equal(decoded.target_speech, original.target_speech)
    assert torch.equal(decoded.units[0].speech_codec_mask, original.units[0].speech_codec_mask)


def test_webdataset_preserves_variable_video_geometry(tmp_path, smoke_config) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    episode.units[0].video = torch.rand(1, 3, 224, 448)
    episode.units[1].video = torch.rand(1, 2, 3, 224, 224)
    write_episode_shards([episode], tmp_path / "train-%06d.tar")
    decoded = next(iter(EpisodeShardReader(str(tmp_path / "train-*.tar"),
                                          smoke_config.data, smoke_config.model)))
    for before, after in zip(episode.units, decoded.units, strict=True):
        torch.testing.assert_close(before.video, after.video)


def test_validation_rejects_non_monotonic_timestamps(smoke_config: ProjectConfig) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    episode.units[1].timestamp_ms.copy_(episode.units[0].timestamp_ms)
    try:
        episode.validate(
            audio_samples=smoke_config.data.unit_audio_samples,
            speech_frames=smoke_config.model.speech_frames_per_unit,
            speech_codebooks=smoke_config.model.speech_codebooks,
        )
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-monotonic timestamps must be rejected")


def test_validation_rejects_code_outside_codec_vocabulary(
    smoke_config: ProjectConfig,
) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    episode.units[0].speech_codes[0, 0, 0] = smoke_config.model.speech_codebook_size

    with pytest.raises(ValueError, match="outside the configured codebook"):
        episode.validate(
            audio_samples=smoke_config.data.unit_audio_samples,
            speech_frames=smoke_config.model.speech_frames_per_unit,
            speech_codebooks=smoke_config.model.speech_codebooks,
            speech_codebook_size=smoke_config.model.speech_codebook_size,
        )


def test_reader_rejects_manifest_content_mismatch(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    write_episode_shards([episode], tmp_path / "train-%06d.tar", max_size=10_000_000)
    manifest_path = tmp_path / "train-manifest.jsonl"
    manifest_path.write_text(
        manifest_path.read_text().replace('"content_sha256": "', '"content_sha256": "bad'),
        encoding="utf-8",
    )
    smoke_config.data.manifest = str(manifest_path)
    reader = EpisodeShardReader(
        str(tmp_path / "train-*.tar"), smoke_config.data, smoke_config.model
    )

    with pytest.raises(ValueError, match="content hash mismatch"):
        next(iter(reader))


def test_reader_rejects_incomplete_metadata(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    episode.metadata.pop("task_id")
    write_episode_shards([episode], tmp_path / "train-%06d.tar", max_size=10_000_000)
    reader = EpisodeShardReader(
        str(tmp_path / "train-*.tar"), smoke_config.data, smoke_config.model
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        next(iter(reader))


def test_manifest_rejects_session_split_leak(tmp_path: Path, smoke_config: ProjectConfig) -> None:
    episodes = [
        SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(index)
        for index in range(2)
    ]
    episodes[1].metadata["device_id_hash"] = episodes[0].metadata["device_id_hash"]
    episodes[1].metadata["session_id_hash"] = episodes[0].metadata["session_id_hash"]
    episodes[1].metadata["split"] = "validation"
    write_episode_shards(episodes, tmp_path / "train-%06d.tar", max_size=10_000_000)

    with pytest.raises(ValueError, match="sessions cross dataset splits"):
        load_manifest(tmp_path / "train-manifest.jsonl")


def test_validate_data_explicit_shards_override_configured_source(
    tmp_path: Path, smoke_config: ProjectConfig
) -> None:
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    write_episode_shards([episode], tmp_path / "actual-%06d.tar")

    assert (
        main(
            [
                "validate-data",
                "--config",
                "configs/smoke.yaml",
                "--set",
                "data.source=webdataset",
                "--set",
                f"data.shards={tmp_path / 'missing-*.tar'}",
                "--shards",
                str(tmp_path / "actual-*.tar"),
            ]
        )
        == 0
    )
