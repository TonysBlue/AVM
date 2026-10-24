from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from data import SpeechOverfitDataset, write_episode_shards
from model import StreamingLatentLoop
from model.types import SpeechMode
from runtime.config import ProjectConfig
from training.checkpoint import CheckpointMetadata, file_sha256
from training.evaluation import evaluate_overfit_checkpoint


def test_speech_overfit_fixture_has_expected_timeline(
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.data.episode_units = 16
    smoke_config.data.train_episodes = 32
    dataset = SpeechOverfitDataset(smoke_config.data, smoke_config.model)
    episode = dataset.make_episode(9)
    modes = [int(unit.speech_mode.item()) for unit in episode.units]
    assert len(dataset) == 32
    assert (
        modes
        == [int(SpeechMode.SILENCE)] * 5
        + [int(SpeechMode.SPEECH)] * 8
        + [int(SpeechMode.SILENCE)] * 3
    )
    assert episode.target_speech is not None
    assert episode.target_speech[: 5 * smoke_config.data.unit_audio_samples].count_nonzero() == 0


def test_overfit_evaluation_covers_complete_encoded_dataset(
    tmp_path: Path,
    smoke_config: ProjectConfig,
) -> None:
    smoke_config.data.episode_units = 16
    smoke_config.data.train_episodes = 2
    episodes = list(SpeechOverfitDataset(smoke_config.data, smoke_config.model))
    for episode in episodes:
        episode.metadata["speech_codes_encoded"] = True
    processed = tmp_path / "processed"
    write_episode_shards(episodes, processed / "train-%06d.tar")
    smoke_config.data.source = "webdataset"
    smoke_config.data.shards = str(processed / "train-*.tar")
    smoke_config.data.manifest = str(processed / "train-manifest.jsonl")
    model = StreamingLatentLoop(smoke_config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = CheckpointMetadata(
        data_identity=file_sha256(processed / "train-manifest.jsonl"),
        codec_id=smoke_config.data.codec_id,
        codec_weight_hash=smoke_config.data.codec_weight_hash,
        git_commit="test",
        codec_revision=smoke_config.data.codec_revision,
    )
    torch.save(
        {
            "metadata": asdict(metadata),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    result = evaluate_overfit_checkpoint(
        smoke_config,
        checkpoint,
        device="cpu",
        codec_threshold=0.0,
    )
    assert result.episodes == 2
    assert len(result.speech_codec_accuracy) == smoke_config.model.speech_codebooks
    assert result.passed
