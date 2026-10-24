from __future__ import annotations

import torch
from torch import Tensor


def time_discounts(delta_ms: Tensor, time_constant_ms: float) -> Tensor:
    if time_constant_ms <= 0:
        raise ValueError("discount time constant must be positive")
    if torch.any(delta_ms <= 0):
        raise ValueError("PPO delta_ms must be positive")
    return torch.exp(-delta_ms.to(torch.float32) / time_constant_ms)


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_value: Tensor,
    discounts: Tensor,
    bootstrap_mask: Tensor,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    if rewards.ndim != 1 or values.shape != rewards.shape:
        raise ValueError("PPO rewards and values must be matching one-dimensional tensors")
    if discounts.shape != rewards.shape or bootstrap_mask.shape != rewards.shape:
        raise ValueError("PPO discount and bootstrap mask shapes must match rewards")
    if not 0 < gae_lambda <= 1:
        raise ValueError("GAE lambda must be in (0, 1]")
    advantage = torch.zeros_like(values)
    carry = torch.zeros((), device=values.device, dtype=values.dtype)
    following = next_value.to(values).reshape(())
    for index in range(rewards.numel() - 1, -1, -1):
        mask = bootstrap_mask[index].to(values)
        delta = rewards[index] + discounts[index].to(values) * mask * following - values[index]
        carry = delta + discounts[index].to(values) * gae_lambda * mask * carry
        advantage[index] = carry
        following = values[index]
    return advantage, advantage + values


def clipped_policy_loss(
    current_logprob: Tensor,
    old_logprob: Tensor,
    advantage: Tensor,
    clip_epsilon: float,
    mask: Tensor | None = None,
) -> Tensor:
    if current_logprob.shape != old_logprob.shape or advantage.shape != current_logprob.shape:
        raise ValueError("PPO policy tensors must have matching shapes")
    ratio = torch.exp((current_logprob - old_logprob).clamp(-30, 30))
    unclipped = ratio * advantage.detach()
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage.detach()
    values = torch.minimum(unclipped, clipped)
    if mask is None:
        return -values.mean()
    weights = mask.to(values)
    return -(values * weights).sum() / weights.sum().clamp_min(1)


def clipped_value_loss(
    current: Tensor,
    old: Tensor,
    target: Tensor,
    clip_epsilon: float,
) -> Tensor:
    if current.shape != old.shape or target.shape != current.shape:
        raise ValueError("PPO value tensors must have matching shapes")
    clipped = old + (current - old).clamp(-clip_epsilon, clip_epsilon)
    raw_error = (current - target.detach()).square()
    clipped_error = (clipped - target.detach()).square()
    return 0.5 * torch.maximum(raw_error, clipped_error).mean()


def sampled_reference_kl(
    current_logprob: Tensor,
    reference_logprob: Tensor,
    *,
    max_log_ratio: float = 20.0,
) -> Tensor:
    if max_log_ratio <= 0:
        raise ValueError("sampled reference KL max_log_ratio must be positive")
    log_ratio = (reference_logprob.detach() - current_logprob).clamp(
        -max_log_ratio, max_log_ratio
    )
    return (torch.expm1(log_ratio) - log_ratio).mean()


def recurrent_ppo_loss(
    current_speech_logprob: Tensor,
    old_speech_logprob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    old_value: Tensor,
    value_target: Tensor,
    reference_speech_logprob: Tensor,
    *,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    reference_kl_beta: float,
    entropy: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    speech = clipped_policy_loss(
        current_speech_logprob, old_speech_logprob, advantage, clip_epsilon
    )
    actor = speech
    value = clipped_value_loss(current_value, old_value, value_target, clip_epsilon)
    reference_kl = sampled_reference_kl(current_speech_logprob, reference_speech_logprob)
    entropy_value = entropy.mean() if entropy is not None else torch.zeros_like(actor)
    total = (
        actor
        + value_coef * value
        - entropy_coef * entropy_value
        + reference_kl_beta * reference_kl
    )
    return total, {
        "actor": actor,
        "speech_actor": speech,
        "value": value,
        "reference_kl": reference_kl,
        "entropy": entropy_value,
    }


__all__ = [
    "clipped_policy_loss",
    "clipped_value_loss",
    "generalized_advantage_estimate",
    "recurrent_ppo_loss",
    "sampled_reference_kl",
    "time_discounts",
]
