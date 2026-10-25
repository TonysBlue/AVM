"""Shared distribution for five continuous acoustic actions, including silence."""

from __future__ import annotations

import torch
from torch import Tensor


def acoustic_distribution_logits(
    logits: Tensor,
    *,
    forbidden_tokens: tuple[int, ...],
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("acoustic logits must have batch and vocabulary dimensions")
    if not forbidden_tokens or len(set(forbidden_tokens)) != len(forbidden_tokens):
        raise ValueError("unique locked acoustic EOS/control token exclusions are required")
    vocab = logits.shape[-1]
    if any(type(token) is not int or not 0 <= token < vocab for token in forbidden_tokens):
        raise ValueError("acoustic exclusions must be valid vocabulary indices")
    if len(forbidden_tokens) == vocab:
        raise ValueError("acoustic vocabulary contains no valid actions")
    if not temperature > 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("acoustic sampling temperature must be finite and positive")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("acoustic top_k must be a non-negative integer")
    if not torch.isfinite(logits).all():
        raise ValueError("raw acoustic logits must be finite")
    result = logits.float() / temperature
    excluded = torch.tensor(forbidden_tokens, device=logits.device)
    result = result.index_fill(-1, excluded, -torch.inf)
    if top_k:
        count = min(top_k, vocab - len(forbidden_tokens))
        indices = result.topk(count, dim=-1).indices
        legal = torch.zeros_like(result, dtype=torch.bool).scatter_(-1, indices, True)
        result = result.masked_fill(~legal, -torch.inf)
    return result


def acoustic_action_logprob(
    logits: Tensor,
    codes: Tensor,
    *,
    forbidden_tokens: tuple[int, ...],
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tensor:
    """Evaluate the actual constrained sampling law, not an unmasked softmax."""
    if logits.ndim != 3 or logits.shape[1] != 5:
        raise ValueError("each acoustic action must contain exactly five token logits")
    if codes.shape != logits.shape[:-1] or codes.dtype != torch.long:
        raise ValueError("acoustic codes must have shape [B,5] and dtype int64")
    if torch.any((codes < 0) | (codes >= logits.shape[-1])):
        raise ValueError("acoustic code is outside the vocabulary")
    if any(torch.any(codes == token) for token in forbidden_tokens):
        raise ValueError("acoustic action contains an EOS/control token")
    scores = acoustic_distribution_logits(
        logits, forbidden_tokens=forbidden_tokens, temperature=temperature, top_k=top_k
    )
    return scores.log_softmax(-1).gather(-1, codes[..., None]).squeeze(-1).sum(-1)


def acoustic_token_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    forbidden_tokens: tuple[int, ...],
) -> Tensor:
    """Mask missing labels only; silence uses the same five-token supervision."""
    if logits.ndim != 3 or logits.shape[1] != 5:
        raise ValueError("acoustic supervision requires [B,5,V] logits")
    if targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("acoustic targets must have shape [B,5] and dtype int64")
    if valid_mask.shape != targets.shape or valid_mask.dtype != torch.bool:
        raise ValueError("acoustic valid_mask must have shape [B,5] and dtype bool")
    valid_targets = targets[valid_mask]
    if torch.any((valid_targets < 0) | (valid_targets >= logits.shape[-1])):
        raise ValueError("supervised acoustic target is outside the vocabulary")
    if any(torch.any(valid_targets == token) for token in forbidden_tokens):
        raise ValueError("supervised acoustic target contains an EOS/control token")
    scores = acoustic_distribution_logits(logits, forbidden_tokens=forbidden_tokens)
    selected = scores[valid_mask]
    if not selected.numel():
        return logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(selected, valid_targets)
