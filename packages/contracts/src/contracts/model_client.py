from __future__ import annotations

import json
import socket
from pathlib import Path

from contracts import ActuationSignal, ObservationSignal
from contracts.framing import frame, read_frame
from contracts.protocol import (
    message_to_actuation,
    observation_to_payload,
)


class UnixModelServiceClient:
    """Harness-facing physical signal client for Model Service."""

    def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.timeout = timeout

    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        request = observation_to_payload(observation)
        # The server's signal endpoint uses one framed request/response per connection.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(frame(b"P" + request))

            def read_exact(size: int) -> bytes:
                output = bytearray()
                while len(output) < size:
                    block = connection.recv(size - len(output))
                    if not block:
                        raise ConnectionError("Model Service closed the connection")
                    output.extend(block)
                return bytes(output)

            response = read_frame(read_exact)
            if response.startswith(b"J"):
                import json

                error = json.loads(response[1:])
                raise RuntimeError(str(error.get("error", "Model Service inference failed")))
            if not response.startswith(b"P"):
                raise RuntimeError("Model Service returned a non-signal response")
            return message_to_actuation(response[1:])

    def _control(self, operation: str, **fields: object) -> dict[str, object]:
        payload = json.dumps({"operation": operation, **fields}, separators=(",", ":")).encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall(frame(b"J" + payload))

            def read_exact(size: int) -> bytes:
                output = bytearray()
                while len(output) < size:
                    block = connection.recv(size - len(output))
                    if not block:
                        raise ConnectionError("Model Service closed the connection")
                    output.extend(block)
                return bytes(output)

            response = json.loads(read_frame(read_exact)[1:])
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Model Service control request failed")))
        return response

    def open_session(self, session_id: str) -> str:
        return str(self._control("open_session", session_id=session_id)["session_id"])

    def close_session(self, session_id: str) -> None:
        self._control("close_session", session_id=session_id)

    def identity(self) -> dict[str, str]:
        return self._control("identity")["identity"]  # type: ignore[return-value]
