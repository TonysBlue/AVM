from __future__ import annotations

import copy

import torch
from model import StreamingLatentLoop
from training.training import _apply_candidate_result, _CandidateResult, _ppo_guard_episodes


def test_rejected_candidate_does_not_change_serving_policy(smoke_config) -> None:
    policy = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = copy.deepcopy(policy.state_dict())

    applied = _apply_candidate_result(
        policy,
        optimizer,
        _CandidateResult(False, "gate", None, {}, {}),
    )

    assert not applied
    assert all(torch.equal(policy.state_dict()[name], value) for name, value in before.items())


def test_accepted_candidate_does_not_mutate_recurrent_state(smoke_config) -> None:
    policy = StreamingLatentLoop(smoke_config.model)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    recurrent_state = policy.initial_state(1, "cpu")
    recurrent_state.unit_index.fill_(37)
    before = recurrent_state.detach()
    candidate = copy.deepcopy(policy)

    applied = _apply_candidate_result(
        policy,
        optimizer,
        _CandidateResult(
            True,
            None,
            candidate,
            optimizer.state_dict(),
            {},
        ),
    )

    assert applied is candidate and applied is not policy
    assert recurrent_state.unit_index.item() == 37
    assert torch.equal(recurrent_state.semantic_memory, before.semantic_memory)


def test_synthetic_ppo_guard_episodes_are_independent(smoke_config) -> None:
    replay, preservation = _ppo_guard_episodes(smoke_config)
    assert replay.episode_id != preservation.episode_id
