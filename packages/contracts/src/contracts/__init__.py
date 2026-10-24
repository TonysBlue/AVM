"""Public cross-system contracts for Model Service, Training, and Harness."""

from contracts.control import ActuationSignal, SpeechSignal
from contracts.identity import ProtocolIdentity
from contracts.observation import MicSignal, ObservationSignal, VideoSignal
from contracts.protocol import (
    payload_to_receipt,
    receipt_to_payload,
)
from contracts.receipt import EnvironmentReceipt
from contracts.reward import GoalOutcome, RewardEvent, RewardStatus, RewardVector

__all__ = [
    "ActuationSignal", "SpeechSignal", "ProtocolIdentity", "MicSignal",
    "VideoSignal", "ObservationSignal", "EnvironmentReceipt", "GoalOutcome",
    "RewardEvent", "RewardStatus", "RewardVector", "payload_to_receipt", "receipt_to_payload",
]
