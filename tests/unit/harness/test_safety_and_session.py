from __future__ import annotations

import pytest
from harness.session import Session


def test_session_rejects_out_of_order_unit() -> None:
    session = Session("s")
    observation = type(
        "Observation",
        (),
        {"session_id": "s", "unit_index": 1},
    )()
    with pytest.raises(ValueError, match="out of order"):
        session.accept_observation(observation)
