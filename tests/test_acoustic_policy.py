from __future__ import annotations

import pytest
import torch
from model.acoustic_policy import (
    acoustic_action_logprob,
    acoustic_distribution_logits,
    acoustic_token_loss,
)


def test_five_token_probability_uses_the_constrained_sampling_law() -> None:
    logits = torch.zeros(2, 5, 8, requires_grad=True)
    codes = torch.zeros(2, 5, dtype=torch.long)
    probability = acoustic_action_logprob(logits, codes, forbidden_tokens=(7,))
    assert torch.allclose(probability, torch.full((2,), -5 * torch.log(torch.tensor(7.0))))
    (-probability.mean()).backward()
    assert torch.all(logits.grad[:, :, 0] < 0)
    assert torch.equal(logits.grad[:, :, 7], torch.zeros(2, 5))


def test_silence_codes_receive_all_five_supervised_gradients() -> None:
    logits = torch.zeros(1, 5, 8, requires_grad=True)
    # Zero is an ordinary fixture acoustic ID, not a privileged silence bypass.
    codes = torch.zeros(1, 5, dtype=torch.long)
    loss = acoustic_token_loss(logits, codes, torch.ones(1, 5, dtype=torch.bool),
                               forbidden_tokens=(7,))
    assert loss > 0
    loss.backward()
    assert torch.all(logits.grad[0, :, 0] < 0)


def test_missing_targets_are_not_trained_as_silence() -> None:
    logits = torch.randn(1, 5, 8, requires_grad=True)
    loss = acoustic_token_loss(logits, torch.full((1, 5), -1, dtype=torch.long),
                               torch.zeros(1, 5, dtype=torch.bool), forbidden_tokens=(7,))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_future_token_loss_reaches_an_earlier_continuous_condition() -> None:
    source = torch.randn(1, 4, requires_grad=True)
    recurrent = torch.nn.GRUCell(4, 4)
    projection = torch.nn.Linear(4, 8)
    first = recurrent(source, torch.zeros_like(source))
    second = recurrent(torch.zeros_like(source), first)
    logits = projection(second)[:, None].expand(-1, 5, -1)
    loss = acoustic_token_loss(logits, torch.zeros(1, 5, dtype=torch.long),
                               torch.ones(1, 5, dtype=torch.bool), forbidden_tokens=(7,))
    loss.backward()
    assert source.grad is not None and source.grad.abs().sum() > 0


def test_topk_has_exact_legal_support_even_when_logits_tie() -> None:
    scores = acoustic_distribution_logits(torch.zeros(2, 8), forbidden_tokens=(7,), top_k=3)
    assert torch.all(torch.isfinite(scores).sum(-1) == 3)
    assert torch.all(torch.isneginf(scores[:, 7]))


def test_eos_cannot_be_a_supervised_or_sampled_acoustic_action() -> None:
    logits = torch.zeros(1, 5, 8)
    codes = torch.full((1, 5), 7, dtype=torch.long)
    with pytest.raises(ValueError, match="EOS/control"):
        acoustic_action_logprob(logits, codes, forbidden_tokens=(7,))
    with pytest.raises(ValueError, match="EOS/control"):
        acoustic_token_loss(logits, codes, torch.ones(1, 5, dtype=torch.bool),
                             forbidden_tokens=(7,))
