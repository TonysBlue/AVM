"""Drive one persistent AVM dialogue without importing model or training code."""

from __future__ import annotations

import time
from typing import Any

from harness.transport.control import HarnessControlClient
from harness.transport.model_client import UnixModelServiceClient


def converse(
    harness: HarnessControlClient,
    model: UnixModelServiceClient,
    *,
    scenario_id: str,
    session_id: str,
    seed: int = 17,
    max_units: int | None = None,
) -> dict[str, Any]:
    if max_units is not None and max_units < 1:
        raise ValueError("max_units must be positive")
    if model.identity().get("protocol_version") != "avm-realtime-v1":
        raise ValueError("model service does not implement the AVM protocol")
    units = 0
    inference_ms = 0.0
    opened = False
    try:
        model.open_session(session_id)
        opened = True
        observation = harness.start_lifetime_session(scenario_id, seed, session_id)
        while max_units is None or units < max_units:
            started = time.monotonic()
            output = model.infer(observation)
            inference_ms += (time.monotonic() - started) * 1000
            observation, receipt = harness.apply(output)
            if not receipt.accepted or receipt.infrastructure_failure:
                raise RuntimeError("dialogue playback or capture failed")
            units += 1
            if receipt.terminated:
                break
    finally:
        try:
            harness.close(session_id)
        finally:
            if opened:
                model.close_session(session_id)
    return {"units": units, "mean_inference_ms": inference_ms / max(units, 1)}
