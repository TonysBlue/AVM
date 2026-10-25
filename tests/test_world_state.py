from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop


def test_semantic_memory_is_updated_inside_backbone(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model.initial_state(1, "cpu")
    output = model(unit.observation, state)
    assert output.state.semantic_memory.shape == (
        1, smoke_config.model.semantic_memory_slots, smoke_config.model.model_dim
    )
    assert not torch.equal(output.state.semantic_memory, state.semantic_memory)


def test_clock_metadata_does_not_enter_model(smoke_config) -> None:
    from dataclasses import fields

    from model import AudioVideoInput

    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    state = model.initial_state(1, "cpu")
    first = model(unit.observation, state)
    unit.timestamp_ms = unit.timestamp_ms + 10_000
    unit.delta_ms = torch.tensor([1_000])
    second = model(unit.observation, state)
    assert {field.name for field in fields(AudioVideoInput)} == {"mic_audio", "video"}
    assert model.type_embedding.num_embeddings == 2
    assert not any("delta_time" in name for name, _ in model.named_parameters())
    assert torch.equal(first.perceiver_slots, second.perceiver_slots)
    assert torch.equal(first.hidden, second.hidden)
    assert torch.equal(first.speech_codec_logits, second.speech_codec_logits)
    assert torch.equal(first.state.semantic_memory, second.state.semantic_memory)
