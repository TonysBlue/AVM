"""Explicit one-way LatentLoop weight import into a new AVM lineage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from model import StreamingLatentLoop
from runtime.config import ProjectConfig

from training.checkpoint import CheckpointManager, DataCursor, file_sha256
from training.training import _checkpoint_metadata


def migrate_latentloop_weights(
    config: ProjectConfig, source: str | Path, output: str | Path
) -> Path:
    if config.training.stage != "pretrain":
        raise ValueError("migrated weights must start a new AVM Pretrain lineage")
    destination = Path(output).expanduser().resolve()
    if destination.suffix != ".pt" or destination.exists():
        raise ValueError("migration requires a new .pt output path")
    payload = torch.load(Path(source).expanduser(), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("action_schema_id") != "structured-action-v1":
        raise ValueError("migration requires an identified LatentLoop checkpoint")
    for name in ("codec_id", "codec_revision", "codec_weight_hash"):
        if metadata.get(name) != getattr(config.data, name):
            raise ValueError(f"migration {name} mismatch")
    weights = payload.get("model")
    if not isinstance(weights, dict):
        raise ValueError("source checkpoint has no model weights")
    model = StreamingLatentLoop(config.model)
    current = model.state_dict()
    shared = {key: value for key, value in weights.items()
              if not key.startswith(("action_head.", "value_head."))}
    expected = {key for key in current if not key.startswith("value_head.")}
    if shared.keys() != expected or any(
        shared[key].shape != current[key].shape for key in expected
    ):
        raise ValueError("LatentLoop shared model keys or shapes do not match AVM")
    model.load_state_dict({**current, **shared}, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.backbone_learning_rate)
    identity = replace(_checkpoint_metadata(config), parent_sha256=file_sha256(source))
    path, _ = CheckpointManager(destination.parent).save(
        destination.stem, model=model, optimizer=optimizer, scheduler=None, scaler=None,
        recurrent_state=None, train_state={"update": 0, "weight_import_only": True},
        data_cursor=DataCursor(), metadata=identity, config=config.as_dict(),
    )
    return path
