from __future__ import annotations

import torch
from torch import Tensor, nn

from model.types import LayerKV, SlowMemoryState


class SlowMemory(nn.Module):
    """Fixed-size linear associative memory with gated delta writes."""

    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.write_gate = nn.Linear(1, 1)
        self.decay_logit = nn.Parameter(torch.tensor(4.0))

    def _phi(self, value: Tensor) -> Tensor:
        return torch.nn.functional.elu(value.float()) + 1.0

    def read(self, query: Tensor, state: SlowMemoryState) -> Tensor:
        original_shape = query.shape
        if query.ndim == 3:
            batch, tokens, dim = query.shape
            query = query.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
            flatten = True
        else:
            flatten = False
        features = self._phi(query)
        numerator = torch.einsum("bhqd,bhde->bhqe", features, state.matrix.float())
        denominator = torch.einsum("bhqd,bhd->bhq", features, state.normalizer.float())
        output = (numerator / denominator.clamp_min(1e-4).unsqueeze(-1)).to(query.dtype)
        if flatten:
            output = output.transpose(1, 2).reshape(original_shape)
        return output

    def write(self, state: SlowMemoryState, expired: LayerKV) -> SlowMemoryState:
        if expired.key.shape[2] == 0:
            return state
        keys = self._phi(expired.key)
        values = expired.value.float()
        features = keys
        numerator = torch.einsum("bhqd,bhde->bhqe", features, state.matrix.float())
        denominator = torch.einsum("bhqd,bhd->bhq", features, state.normalizer.float())
        predicted = (numerator / denominator.clamp_min(1e-4).unsqueeze(-1)).float()
        error = values - predicted
        surprise = error.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        gate_input = surprise.reshape(-1, 1).to(self.write_gate.weight.dtype)
        gate = torch.sigmoid(self.write_gate(gate_input)).float().reshape(-1, 1, 1, 1)
        decay = torch.sigmoid(self.decay_logit)
        matrix = state.matrix.float() * decay
        normalizer = state.normalizer.float() * decay
        matrix = matrix + gate * torch.einsum("bhqd,bhqe->bhde", keys, error)
        normalizer = normalizer + gate.squeeze(-1) * keys.sum(dim=2)
        return SlowMemoryState(matrix.to(state.matrix.dtype), normalizer.to(state.normalizer.dtype))
