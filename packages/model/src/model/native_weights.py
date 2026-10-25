"""Strict, memory-mapped import of the native non-text model parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from runtime.native_assets import audit_native_assets
from safetensors import safe_open
from torch import nn

NATIVE_PREFIXES = (
    ("llm.model.layers.", "layers."),
    ("llm.model.norm.", "final_norm."),
    ("apm.", "audio_encoder.encoder."),
    ("audio_projection_layer.", "audio_encoder.projection."),
    ("vpm.", "vision_encoder.encoder."),
    ("resampler.", "vision_encoder.resampler."),
    ("tts.projector_semantic.", "speech_head.projector_semantic."),
    ("tts.model.layers.", "speech_head.layers."),
    ("tts.model.norm.", "speech_head.norm."),
    ("tts.emb_code.", "speech_head.emb_code."),
    ("tts.head_code.", "speech_head.head_code."),
)
UNUSED_NATIVE_PREFIXES = (
    "llm.model.embed_tokens.",
    "llm.lm_head.",
    "tts.emb_text.",
    "tts.model.embed_tokens.",
    "tts.projector_spk.",
)
NEW_PREFIXES = (
    "type_embedding.",
    "perceiver.",
    "jepa_head.",
    "semantic_slot_identity",
    "slow_memories.",
    "value_head.",
)


def native_destination(name: str) -> str | None:
    for source, target in NATIVE_PREFIXES:
        if name.startswith(source):
            return target + name[len(source) :]
    if name.startswith(UNUSED_NATIVE_PREFIXES):
        return None
    raise ValueError(f"unclassified native parameter: {name}")


def _install_parameter(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent, _, leaf = name.rpartition(".")
    module = model.get_submodule(parent) if parent else model
    previous = module.get_parameter(leaf)
    if previous.shape != value.shape:
        raise ValueError(f"native parameter shape mismatch: {name}")
    module._parameters[leaf] = nn.Parameter(value, requires_grad=previous.requires_grad)


def initialize_native_model(model: nn.Module, source_lock: str | Path) -> dict[str, Any]:
    audit = audit_native_assets(source_lock, model_weights=True)
    root = Path(audit["root"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    current = dict(model.named_parameters())
    imported, unused = {}, {}
    for shard in sorted(set(index.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as source:
            for native_name in source.keys():
                destination = native_destination(native_name)
                if destination is None:
                    unused[native_name] = source.get_slice(native_name).get_shape()
                    continue
                if destination not in current or destination in imported:
                    raise ValueError(f"unexpected or duplicate native parameter: {native_name}")
                tensor = source.get_tensor(native_name)
                _install_parameter(model, destination, tensor)
                imported[destination] = tensor.numel()
    missing = {
        name for name in current if name not in imported and not name.startswith(NEW_PREFIXES)
    }
    if missing:
        raise ValueError(f"native import did not initialize parameters: {sorted(missing)}")
    dtype = model.final_norm.weight.dtype
    for name in ("type_embedding", "perceiver", "jepa_head", "slow_memories", "value_head"):
        module = model.get_submodule(name)
        module.to_empty(device="cpu")
        module.to(dtype=dtype)
        for child in module.modules():
            if isinstance(child, nn.MultiheadAttention):
                child._reset_parameters()
            elif hasattr(child, "reset_parameters"):
                child.reset_parameters()
    model.semantic_slot_identity = nn.Parameter(
        torch.randn(
            model.config.semantic_memory_slots,
            model.config.model_dim,
            dtype=dtype,
        )
        * 0.02
    )
    nn.init.normal_(model.perceiver.queries, std=0.02)
    nn.init.normal_(model.jepa_head.slot_identity, std=0.02)
    for memory in model.slow_memories:
        nn.init.constant_(memory.decay_logit, 4.0)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("native initialization left meta parameters")
    return {
        **audit,
        "imported_tensor_count": len(imported),
        "imported_parameters": sum(imported.values()),
        "unused_native_parameters": unused,
        "avm_parameters": sum(value.numel() for value in model.parameters()),
    }
