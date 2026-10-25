from __future__ import annotations

import pytest
from contracts import (
    GoalOutcome,
    MicSignal,
    ObservationSignal,
    RewardEvent,
    RewardStatus,
    RewardVector,
    VideoSignal,
)
from data import ObservationTimeline


def _observation(unit: int) -> ObservationSignal:
    return ObservationSignal(
        "session", unit, unit * 200, 200,
        MicSignal(b"x" * 12800),
        VideoSignal(b"x" * (224 * 224 * 3), 224, 224, captured_at_ms=0, revision=0),
    )


def test_observation_timeline_is_canonical_hash_chain() -> None:
    timeline = ObservationTimeline("life", "session")
    first = timeline.append(_observation(0), "policy-1")
    second = timeline.append(_observation(1), "policy-1")
    assert second.previous_chain_sha256 == first.chain_sha256
    assert first.payload == timeline.records[0].payload
    with pytest.raises(ValueError, match="contiguous"):
        timeline.append(_observation(3), "policy-1")


def test_reward_event_only_finalized_outcomes_are_trainable() -> None:
    reward = RewardVector(1, 0.5, 0.5, 0)
    event = RewardEvent(
        "event", "life", "session", "goal", 0, 3, 1, 3,
        RewardStatus.FINALIZED, GoalOutcome.SUCCESS, reward, "spec", "judge", "rev",
        "rubric", "chain",
    )
    assert event.trainable
    assert event.reward.total == pytest.approx(1.1)
    with pytest.raises(ValueError, match="uncertain"):
        RewardEvent(
            "event", "life", "session", "goal", 0, 3, 1, 3,
            RewardStatus.FINALIZED, GoalOutcome.UNCERTAIN, reward, "spec", "judge", "rev",
            "rubric", "chain",
        )
