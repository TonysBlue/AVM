"""Deployment adapters for the realtime audiovisual dialogue Harness boundary."""

from __future__ import annotations

from typing import Protocol

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal


class SensorAdapter(Protocol):
    """Captures only physical microphone/video/time signals."""

    def capture(self, session_id: str, unit_index: int) -> ObservationSignal: ...

    def close(self) -> None: ...


class ActuatorAdapter(Protocol):
    """Plays speech while capture continues."""

    def apply(self, output: ActuationSignal) -> EnvironmentReceipt: ...

    def close(self) -> None: ...
