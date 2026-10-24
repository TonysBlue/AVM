from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop


def test_vision_encoder_emits_sixteen_spatial_tokens(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    tokens = model.vision_encoder(episode.units[0].video)

    assert tokens.shape == (1, smoke_config.model.vision_tokens, smoke_config.model.model_dim)
    assert smoke_config.model.vision_tokens == 16


def test_vision_encoder_has_no_nondeterministic_adaptive_pool(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    assert not any(
        isinstance(layer, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d))
        for layer in model.vision_encoder.modules()
    )


def test_static_and_dynamic_frames_use_the_same_visual_path(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    static = torch.zeros(1, 3, 224, 224)
    dynamic = static.clone()
    dynamic[:, :, 40:80, 80:120] = 1

    static_tokens = model.vision_encoder(static)
    dynamic_tokens = model.vision_encoder(dynamic)

    assert static_tokens.shape == dynamic_tokens.shape == (1, 16, 64)
    assert not torch.equal(static_tokens, dynamic_tokens)


def test_perceiver_is_the_only_backbone_token_layout(smoke_config) -> None:
    assert smoke_config.model.tokens_per_unit == smoke_config.model.perceiver_slots == 16




def test_kv_horizon_is_unified_over_perceiver_slots(smoke_config) -> None:
    assert smoke_config.model.kv_units == 4
    assert smoke_config.model.kv_units * smoke_config.model.perceiver_slots == 64
