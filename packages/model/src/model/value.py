from __future__ import annotations

from torch import Tensor, nn


class ValueHead(nn.Module):
    """Training-only value estimate from the causal recurrent representation."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, min(model_dim, 256)),
            nn.GELU(),
            nn.Linear(min(model_dim, 256), 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden.mean(dim=1)).squeeze(-1)


__all__ = ["ValueHead"]
