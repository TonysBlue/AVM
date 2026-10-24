from __future__ import annotations

import torch
from data import SyntheticEpisodeDataset
from model import SpeechSamplingConfig, StreamingLatentLoop
from training.ppo import (
    clipped_policy_loss,
    generalized_advantage_estimate,
    recurrent_ppo_loss,
    sampled_reference_kl,
    time_discounts,
)
from training.training import _policy_logprobs


def test_sampled_reference_kl_stays_finite_for_extreme_log_ratios() -> None:
    current = torch.tensor([-1_000.0, 1_000.0])
    reference = torch.tensor([1_000.0, -1_000.0])

    result = sampled_reference_kl(current, reference)

    assert torch.isfinite(result)


def test_time_discount_and_gae_respect_goal_boundary() -> None:
    discounts = time_discounts(torch.tensor([80, 80, 80]), 10_000)
    advantages, targets = generalized_advantage_estimate(
        rewards=torch.tensor([0.0, 1.0, 0.0]),
        values=torch.tensor([0.1, 0.2, 0.9]),
        next_value=torch.tensor(0.9),
        discounts=discounts,
        bootstrap_mask=torch.tensor([1.0, 0.0, 1.0]),
        gae_lambda=0.95,
    )
    assert advantages.shape == (3,)
    assert targets.shape == (3,)
    assert advantages[1] == torch.tensor(0.8)
    assert advantages[0] > 0


def test_ppo_advantage_is_stop_gradient() -> None:
    current = torch.tensor([0.1, -0.2], requires_grad=True)
    advantage = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = clipped_policy_loss(current, torch.zeros(2), advantage, 0.2)
    loss.backward()
    assert current.grad is not None
    assert advantage.grad is None


def test_recurrent_ppo_updates_speech_policy_and_value() -> None:
    speech = torch.tensor([0.1, 0.2], requires_grad=True)
    value = torch.tensor([0.3, 0.4], requires_grad=True)
    total, components = recurrent_ppo_loss(
        speech,
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        value,
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
        torch.zeros(2),
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        reference_kl_beta=0.02,
    )
    total.backward()
    assert speech.grad is not None and value.grad is not None
    assert set(components) == {
        "actor", "speech_actor", "value", "reference_kl", "entropy"
    }




def test_speech_policy_logprob_is_the_joint_mode_and_codec_probability(
    smoke_config,
) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    mode = torch.ones(1, dtype=torch.long)
    codes = unit.speech_codes
    sampling = SpeechSamplingConfig(temperature=0.8, top_k=0)
    output = model.forward_step(
        unit,
        model.initial_state(1, "cpu"),
        speech_teacher_codes=codes,
        speech_teacher_mode=mode,
    )

    speech = _policy_logprobs(output, mode, codes, sampling)
    mode_logprob = torch.log_softmax(
        output.speech_mode_logits / sampling.temperature, dim=-1
    ).gather(-1, mode[:, None]).squeeze(-1)
    codec_logprob = torch.log_softmax(
        output.speech_codec_logits / sampling.temperature, dim=-1
    ).gather(-1, codes[..., None]).squeeze(-1).sum(dim=(1, 2))

    assert torch.allclose(speech, mode_logprob + codec_logprob)
