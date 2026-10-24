from __future__ import annotations

import pytest
from contracts import GoalOutcome, RewardEvent, RewardStatus, RewardVector
from harness.reward import PerceptualRewardClient, SingleActiveGoalTracker


def _event(
    event_id: str,
    goal_id: str,
    status: RewardStatus,
    outcome: GoalOutcome,
) -> RewardEvent:
    return RewardEvent(
        event_id,
        "life",
        "session",
        goal_id,
        0,
        1,
        0,
        1,
        status,
        outcome,
        RewardVector(1, 1, 1, 0),
        "spec",
        "judge",
        "revision",
        "rubric",
        "chain",
    )


def test_single_active_goal_rejects_overlapping_goals() -> None:
    tracker = SingleActiveGoalTracker()
    tracker.accept(_event("one", "goal-1", RewardStatus.PROVISIONAL, GoalOutcome.SUCCESS))
    with pytest.raises(ValueError, match="multiple active goals"):
        tracker.accept(_event("two", "goal-2", RewardStatus.PROVISIONAL, GoalOutcome.SUCCESS))


def test_finalized_reward_event_is_immutable() -> None:
    tracker = SingleActiveGoalTracker()
    event = _event("one", "goal-1", RewardStatus.FINALIZED, GoalOutcome.SUCCESS)
    assert tracker.accept(event)
    assert not tracker.accept(event)
    with pytest.raises(ValueError, match="cannot be revised"):
        tracker.accept(_event("one", "goal-1", RewardStatus.FINALIZED, GoalOutcome.FAILURE))


def test_reward_client_rejects_future_events_and_invalid_watermark(monkeypatch) -> None:
    client = PerceptualRewardClient(
        "/tmp/reward.sock",
        spec_id="spec",
        judge_model_id="judge",
        judge_revision="revision",
        rubric_sha256="rubric",
    )
    event = _event("future", "goal", RewardStatus.FINALIZED, GoalOutcome.SUCCESS)
    value = {
        "event_id": event.event_id,
        "lineage_id": event.lineage_id,
        "session_id": event.session_id,
        "goal_id": event.goal_id,
        "goal_start_unit": event.goal_start_unit,
        "outcome_unit": 3,
        "evidence_start_unit": 0,
        "evidence_end_unit": 3,
        "status": event.status.value,
        "outcome": event.outcome.value,
        "reward": {
            "task": 1,
            "speech_quality": 1,
            "latency_quality": 1,
            "safety_quality": 0,
        },
        "spec_id": event.spec_id,
        "judge_model_id": event.judge_model_id,
        "judge_revision": event.judge_revision,
        "rubric_sha256": event.rubric_sha256,
        "observation_chain_end_sha256": "chain",
    }
    monkeypatch.setattr(
        client,
        "_request",
        lambda _: {"finalized_through_unit": 3, "events": [value]},
    )
    with pytest.raises(RuntimeError, match="beyond"):
        client.observe(
            lineage_id="life",
            session_id="session",
            unit_index=1,
            observation_payload=b"observation",
            observation_chain_sha256="chain",
        )


def test_reward_client_rejects_regressing_finalization_watermark(monkeypatch) -> None:
    client = PerceptualRewardClient(
        "/tmp/reward.sock",
        spec_id="spec",
        judge_model_id="judge",
        judge_revision="revision",
        rubric_sha256="rubric",
    )
    responses = iter(
        [
            {"finalized_through_unit": 1, "events": []},
            {"finalized_through_unit": 0, "events": []},
        ]
    )
    monkeypatch.setattr(client, "_request", lambda _: next(responses))
    client.observe(
        lineage_id="life",
        session_id="session",
        unit_index=1,
        observation_payload=b"one",
        observation_chain_sha256="chain",
    )
    with pytest.raises(RuntimeError, match="regressed"):
        client.observe(
            lineage_id="life",
            session_id="session",
            unit_index=2,
            observation_payload=b"two",
            observation_chain_sha256="chain",
        )
