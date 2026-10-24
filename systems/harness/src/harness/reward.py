from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import GoalOutcome, RewardEvent, RewardStatus, RewardVector
from contracts.framing import frame, read_frame


def _read_exact(connection: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        block = connection.recv(size - len(output))
        if not block:
            raise ConnectionError("Reward Judge disconnected")
        output.extend(block)
    return bytes(output)


def _event_from_dict(value: dict[str, Any]) -> RewardEvent:
    reward = value.get("reward")
    if not isinstance(reward, dict):
        raise ValueError("Reward Judge event is missing reward components")
    return RewardEvent(
        event_id=str(value["event_id"]),
        lineage_id=str(value["lineage_id"]),
        session_id=str(value["session_id"]),
        goal_id=str(value["goal_id"]),
        goal_start_unit=int(value["goal_start_unit"]),
        outcome_unit=int(value["outcome_unit"]),
        evidence_start_unit=int(value["evidence_start_unit"]),
        evidence_end_unit=int(value["evidence_end_unit"]),
        status=RewardStatus(str(value["status"])),
        outcome=GoalOutcome(str(value["outcome"])),
        reward=RewardVector(**reward),
        spec_id=str(value["spec_id"]),
        judge_model_id=str(value["judge_model_id"]),
        judge_revision=str(value["judge_revision"]),
        rubric_sha256=str(value["rubric_sha256"]),
        observation_chain_end_sha256=str(value["observation_chain_end_sha256"]),
    )


@dataclass(frozen=True, slots=True)
class RewardObservationResult:
    finalized_through_unit: int
    events: tuple[RewardEvent, ...]


class PerceptualRewardClient:
    """Submit only canonical ObservationSignal bytes to a frozen Judge."""

    def __init__(
        self,
        socket_path: str,
        *,
        spec_id: str,
        judge_model_id: str,
        judge_revision: str,
        rubric_sha256: str,
        timeout: float = 120.0,
    ) -> None:
        identities = (socket_path, spec_id, judge_model_id, judge_revision, rubric_sha256)
        if not all(identities):
            raise ValueError("Reward Judge identity is incomplete")
        self.socket_path = str(Path(socket_path).expanduser())
        self.spec_id = spec_id
        self.judge_model_id = judge_model_id
        self.judge_revision = judge_revision
        self.rubric_sha256 = rubric_sha256
        self.timeout = timeout
        self._finalized_through: dict[tuple[str, str], int] = {}

    def identity(self) -> dict[str, str]:
        return self._request({"operation": "identity"})["identity"]

    def observe(
        self,
        *,
        lineage_id: str,
        session_id: str,
        unit_index: int,
        observation_payload: bytes,
        observation_chain_sha256: str,
    ) -> RewardObservationResult:
        response = self._request(
            {
                "operation": "observe",
                "lineage_id": lineage_id,
                "session_id": session_id,
                "unit_index": unit_index,
                "observation": base64.b64encode(observation_payload).decode(),
                "observation_chain_sha256": observation_chain_sha256,
            }
        )
        raw_events = response.get("events", [])
        if not isinstance(raw_events, list):
            raise RuntimeError("Reward Judge events must be a list")
        events = tuple(_event_from_dict(item) for item in raw_events)
        for event in events:
            if (
                event.lineage_id != lineage_id
                or event.session_id != session_id
                or event.spec_id != self.spec_id
                or event.judge_model_id != self.judge_model_id
                or event.judge_revision != self.judge_revision
                or event.rubric_sha256 != self.rubric_sha256
            ):
                raise RuntimeError("Reward Judge event identity does not match configuration")
            if event.outcome_unit > unit_index or event.evidence_end_unit > unit_index:
                raise RuntimeError("Reward Judge event points beyond the observed timeline")
        finalized_through_unit = int(response["finalized_through_unit"])
        if not -1 <= finalized_through_unit <= unit_index:
            raise RuntimeError("Reward Judge finalization watermark is invalid")
        key = (lineage_id, session_id)
        previous = self._finalized_through.get(key, -1)
        if finalized_through_unit < previous:
            raise RuntimeError("Reward Judge finalization watermark regressed")
        self._finalized_through[key] = finalized_through_unit
        return RewardObservationResult(finalized_through_unit, events)

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(frame(json.dumps(request, separators=(",", ":")).encode()))
            response = json.loads(read_frame(lambda size: _read_exact(connection, size)))
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Reward Judge request failed")))
        return response


class SingleActiveGoalTracker:
    """Validate one active perceptual goal and immutable finalized events."""

    def __init__(self) -> None:
        self.active_goal_id: str | None = None
        self._finalized: dict[str, RewardEvent] = {}

    def accept(self, event: RewardEvent) -> bool:
        existing = self._finalized.get(event.event_id)
        if existing is not None:
            if existing != event:
                raise ValueError("a finalized reward event cannot be revised")
            return False
        if event.status is RewardStatus.PROVISIONAL:
            if self.active_goal_id not in {None, event.goal_id}:
                raise ValueError("Reward Judge reported multiple active goals")
            self.active_goal_id = event.goal_id
            return True
        if self.active_goal_id not in {None, event.goal_id}:
            raise ValueError("finalized reward event does not match the active goal")
        self._finalized[event.event_id] = event
        self.active_goal_id = None
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "active_goal_id": self.active_goal_id,
            "finalized": [
                {
                    "event_id": event.event_id,
                    "lineage_id": event.lineage_id,
                    "session_id": event.session_id,
                    "goal_id": event.goal_id,
                    "goal_start_unit": event.goal_start_unit,
                    "outcome_unit": event.outcome_unit,
                    "evidence_start_unit": event.evidence_start_unit,
                    "evidence_end_unit": event.evidence_end_unit,
                    "status": event.status.value,
                    "outcome": event.outcome.value,
                    "reward": {
                        "task": event.reward.task,
                        "speech_quality": event.reward.speech_quality,
                        "latency_quality": event.reward.latency_quality,
                        "safety_quality": event.reward.safety_quality,
                    },
                    "spec_id": event.spec_id,
                    "judge_model_id": event.judge_model_id,
                    "judge_revision": event.judge_revision,
                    "rubric_sha256": event.rubric_sha256,
                    "observation_chain_end_sha256": event.observation_chain_end_sha256,
                }
                for event in self._finalized.values()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.active_goal_id = state.get("active_goal_id")
        self._finalized = {}
        for item in state.get("finalized", []):
            event = _event_from_dict(item)
            if event.status is not RewardStatus.FINALIZED:
                raise ValueError("stored reward tracker event is not finalized")
            self._finalized[event.event_id] = event


__all__ = [
    "PerceptualRewardClient",
    "RewardObservationResult",
    "SingleActiveGoalTracker",
]
