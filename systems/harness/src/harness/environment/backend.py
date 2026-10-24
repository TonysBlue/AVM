from __future__ import annotations

from typing import Protocol

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal


class DialogueBackend(Protocol):
    environment_id: str
    environment_version: str

    def start_lifetime_session(
        self, scenario_id: str, seed: int, session_id: str | None = None
    ) -> ObservationSignal: ...
    def apply(self, output: ActuationSignal) -> tuple[ObservationSignal, EnvironmentReceipt]: ...
    def close(self) -> None: ...
