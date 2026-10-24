from __future__ import annotations

import pytest
from contracts import MicSignal, ObservationSignal, ProtocolIdentity, VideoSignal


def test_observation_cannot_contain_privileged_reward_fields() -> None:
    observation = ObservationSignal(
        "session", 0, 0, 80, MicSignal(b"x" * 7680),
        VideoSignal(b"x" * 48, 4, 4, captured_at_ms=0, revision=0),
    )
    assert not hasattr(observation, "task_success")


def test_environment_identity_requires_protocol() -> None:
    with pytest.raises(ValueError, match="protocol identity"):
        ProtocolIdentity(protocol_version="")
