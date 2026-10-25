from __future__ import annotations

import copy

import pytest
import torch
from training.optimizer_state import OptimizerStateSnapshot


def test_cpu_snapshot_preserves_adamw_moments_and_next_update() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 4)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    snapshot = OptimizerStateSnapshot(optimizer)
    restored_model = copy.deepcopy(model)
    restored = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored.load_state_dict(copy.deepcopy(snapshot.state_dict()))
    for current, active in ((model, optimizer), (restored_model, restored)):
        current(torch.ones(1, 4)).square().sum().backward()
        active.step()
    for actual, expected in zip(restored_model.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(values["exp_avg"].device.type == "cpu"
               for values in snapshot.state_dict()["state"].values())


def test_snapshot_rejects_incompatible_moments() -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    snapshot = OptimizerStateSnapshot(optimizer)
    invalid = copy.deepcopy(snapshot.state_dict())
    invalid["state"][0] = {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}
    with pytest.raises(ValueError, match="moment shape"):
        snapshot.load_state_dict(invalid)
