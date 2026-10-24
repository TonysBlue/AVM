from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RewardStatus(StrEnum):
    PROVISIONAL = "provisional"
    FINALIZED = "finalized"


class GoalOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABANDONED = "abandoned"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class RewardVector:
    task: float
    speech_quality: float
    latency_quality: float
    safety_quality: float

    def __post_init__(self) -> None:
        for name in ("task", "speech_quality", "latency_quality"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"reward {name} must be in [0, 1]")
        if not -1.0 <= self.safety_quality <= 0.0:
            raise ValueError("reward safety_quality must be in [-1, 0]")

    @property
    def interaction(self) -> float:
        return (
            0.5 * self.speech_quality
            + 0.5 * self.latency_quality
        )

    @property
    def total(self) -> float:
        return self.task + 0.2 * self.interaction + self.safety_quality


@dataclass(frozen=True, slots=True)
class RewardEvent:
    event_id: str
    lineage_id: str
    session_id: str
    goal_id: str
    goal_start_unit: int
    outcome_unit: int
    evidence_start_unit: int
    evidence_end_unit: int
    status: RewardStatus
    outcome: GoalOutcome
    reward: RewardVector
    spec_id: str
    judge_model_id: str
    judge_revision: str
    rubric_sha256: str
    observation_chain_end_sha256: str

    def __post_init__(self) -> None:
        identities = (
            self.event_id,
            self.lineage_id,
            self.session_id,
            self.goal_id,
            self.spec_id,
            self.judge_model_id,
            self.judge_revision,
            self.rubric_sha256,
            self.observation_chain_end_sha256,
        )
        if not all(identities):
            raise ValueError("reward event identity fields are required")
        if not (
            0 <= self.goal_start_unit
            <= self.evidence_start_unit
            <= self.evidence_end_unit
            <= self.outcome_unit
        ):
            raise ValueError("reward event units are out of order")
        if self.status is RewardStatus.FINALIZED and self.outcome is GoalOutcome.UNCERTAIN:
            raise ValueError("an uncertain reward event cannot be finalized")

    @property
    def trainable(self) -> bool:
        return self.status is RewardStatus.FINALIZED and self.outcome in {
            GoalOutcome.SUCCESS,
            GoalOutcome.FAILURE,
        }


__all__ = ["GoalOutcome", "RewardEvent", "RewardStatus", "RewardVector"]
