from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from model.native_weights import NATIVE_PREFIXES, initialize_native_model
from safetensors.torch import save_file


@pytest.mark.parametrize("damage", [None, "missing", "unknown"])
def test_strict_native_import_materializes_model_and_preserves_weights(
    tmp_path, smoke_config, monkeypatch, damage
) -> None:
    source = StreamingLatentLoop(smoke_config.model)
    tensors = {}
    imported = {}
    for name, parameter in source.named_parameters():
        for native, destination in NATIVE_PREFIXES:
            if name.startswith(destination):
                tensors[native + name[len(destination):]] = parameter.detach().contiguous()
                imported[name] = parameter.detach()
                break
    tensors["llm.lm_head.weight"] = torch.ones(3, 4)
    if damage == "missing":
        tensors.pop("llm.model.norm.weight")
    elif damage == "unknown":
        tensors["unclassified.weight"] = torch.ones(1)
    shard = "fixture.safetensors"
    save_file(tensors, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": dict.fromkeys(tensors, shard),
    }))
    monkeypatch.setattr(
        "model.native_weights.audit_native_assets",
        lambda *args, **kwargs: {"root": str(tmp_path), "explicit_test_fixture": True},
    )
    candidate = StreamingLatentLoop(
        replace(smoke_config.model, native_source_lock="explicit-test-fixture"),
        load_native=False,
    )
    if damage:
        with pytest.raises(ValueError, match="initialize parameters|unclassified"):
            initialize_native_model(candidate, "explicit-test-fixture")
        return
    report = initialize_native_model(candidate, "explicit-test-fixture")
    assert report["imported_tensor_count"] == len(imported)
    assert report["unused_native_parameters"] == {"llm.lm_head.weight": [3, 4]}
    assert all(not value.is_meta and torch.isfinite(value).all()
               for value in candidate.parameters())
    for name, value in candidate.named_parameters():
        if name in imported:
            assert torch.equal(value, imported[name])
    episode = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0)
    output = candidate(episode.units[0].observation, candidate.initial_state(1, "cpu"))
    output.speech_codec_logits.square().mean().backward()
    assert candidate.perceiver.queries.grad is not None
    assert candidate.audio_encoder.encoder.conv1.weight.grad is not None
