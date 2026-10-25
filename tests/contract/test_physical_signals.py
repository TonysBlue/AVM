from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "packages/contracts/src")

from contracts import (  # noqa: E402
    ActuationSignal,
    EnvironmentReceipt,
    MicSignal,
    ObservationSignal,
    SpeechSignal,
    VideoSignal,
    payload_to_receipt,
    receipt_to_payload,
)
from contracts.protocol import (  # noqa: E402
    actuation_to_message,
    message_to_actuation,
    message_to_observation,
    observation_to_message,
)
from contracts.realtime_pb2 import VideoFrame as VideoMessage  # noqa: E402


def test_observation_physical_signal_round_trip() -> None:
    value = ObservationSignal(
        "session",
        0,
        0,
        200,
        MicSignal(b"x" * 12800),
        VideoSignal(b"rgb", 1, 1, captured_at_ms=0, revision=0),
    )
    restored = message_to_observation(observation_to_message(value))
    assert restored == value


def test_avm_video_has_capture_time_and_revision() -> None:
    assert "revision" in VideoMessage.DESCRIPTOR.fields_by_name
    assert "captured_at_ms" in VideoMessage.DESCRIPTOR.fields_by_name


def test_actuation_does_not_contain_raw_action_tokens() -> None:
    value = ActuationSignal(
        "session",
        0,
        SpeechSignal(b"x" * 19200),
    )
    restored = message_to_actuation(actuation_to_message(value))
    assert restored == value
    assert not hasattr(restored, "controls")
    assert not hasattr(restored, "action_tokens")


def test_physical_signal_rejects_wrong_audio_clock() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        MicSignal(b"x", sample_rate_hz=24_000)


def test_receipt_protobuf_round_trip() -> None:
    receipt = EnvironmentReceipt("session", 3, True, execution_latency_ms=1.5, terminated=True)
    assert payload_to_receipt(receipt_to_payload(receipt)) == receipt
