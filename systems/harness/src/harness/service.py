from __future__ import annotations

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal

from harness.session import Session


class HarnessService:
    """Coordinates sensors, Model Service client, actuators, and environment."""

    def __init__(self, environment, model_client) -> None:
        self.environment = environment
        self.model_client = model_client
        self.session: Session | None = None

    def start_lifetime_session(
        self, scenario_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        observation = self.environment.start_lifetime_session(
            scenario_id, seed, session_id
        )
        if observation.session_id != session_id:
            raise ValueError("environment observation has wrong session identity")
        self.session = Session(session_id)
        self.session.accept_observation(observation)
        if hasattr(self.model_client, "open_session"):
            self.model_client.open_session(session_id)
        return observation

    def step(
        self,
        observation: ObservationSignal,
    ) -> tuple[ObservationSignal, EnvironmentReceipt, ActuationSignal]:
        if self.session is None:
            raise RuntimeError("Harness session is not active")
        self.session.accept_observation(observation)
        output = self.model_client.infer(observation)
        next_observation, receipt = self.environment.apply(output)
        self.session.accept_actuation(output)
        return next_observation, receipt, output

    def close(self) -> None:
        if self.session is not None and hasattr(self.model_client, "close_session"):
            self.model_client.close_session(self.session.session_id)
        self.environment.close()
        self.session = None
