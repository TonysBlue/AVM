"""CPU AdamW state ownership for an actor that never performs optimizer steps."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils import _pytree


class OptimizerStateSnapshot:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._shapes = [tuple(parameter.shape) for group in optimizer.param_groups
                        for parameter in group["params"]]
        self._group_sizes = [len(group["params"]) for group in optimizer.param_groups]
        self.load_state_dict(optimizer.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return self._state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        groups = state.get("param_groups", [])
        if [len(group["params"]) for group in groups] != self._group_sizes:
            raise ValueError("optimizer snapshot parameter groups differ from policy")
        indices = [index for group in groups for index in group["params"]]
        if len(set(indices)) != len(indices):
            raise ValueError("optimizer snapshot has duplicate parameter identities")
        shapes = dict(zip(indices, self._shapes, strict=True))
        for index, values in state.get("state", {}).items():
            if index not in shapes:
                raise ValueError("optimizer snapshot contains an unknown parameter")
            for name in ("exp_avg", "exp_avg_sq"):
                value = values.get(name)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[index]:
                    raise ValueError("optimizer snapshot moment shape differs from policy")
        self._state = _pytree.tree_map(
            lambda value: value.detach().cpu() if isinstance(value, torch.Tensor) else value,
            state,
        )
