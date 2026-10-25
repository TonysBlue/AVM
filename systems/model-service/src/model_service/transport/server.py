from __future__ import annotations

import base64
import json
import socket
import threading
from pathlib import Path
from typing import Any

from contracts.framing import frame, read_frame
from contracts.protocol import actuation_to_payload, message_to_observation
from model_service.service import ModelService


class UnixModelServer:
    def __init__(self, service: ModelService, socket_path: str) -> None:
        self.service = service
        self.socket_path = Path(socket_path).expanduser()
        self._server: socket.socket | None = None

    def _control_handler(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "identity":
            return {"ok": True, "identity": self.service.identity()}
        if operation == "open_session":
            return {
                "ok": True,
                "session_id": self.service.open_session(str(request.get("session_id") or "")),
            }
        if operation == "close_session":
            self.service.close_session(str(request["session_id"]))
            return {"ok": True}
        if operation == "finish_session":
            outputs = self.service.finish_session(str(request["session_id"]))
            return {
                "ok": True,
                "actuations": [
                    base64.b64encode(actuation_to_payload(value)).decode() for value in outputs
                ],
            }
        if operation == "cancel_speech":
            return {
                "ok": True,
                "generation_epoch": self.service.cancel_speech(str(request["session_id"])),
            }
        raise ValueError("binary signal operations must use the protobuf message boundary")

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        server.bind(str(self.socket_path))
        server.listen(64)
        while True:
            connection, _ = server.accept()
            threading.Thread(target=self._serve_connection, args=(connection,), daemon=True).start()

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            try:

                def read_exact(size: int) -> bytes:
                    output = bytearray()
                    while len(output) < size:
                        block = connection.recv(size - len(output))
                        if not block:
                            raise ConnectionError("client disconnected")
                        output.extend(block)
                    return bytes(output)

                payload = read_frame(read_exact)
                if not payload:
                    raise ValueError("empty Model Service request")
                if payload[:1] == b"J":
                    response = json.dumps(
                        self._control_handler(json.loads(payload[1:])), separators=(",", ":")
                    ).encode()
                    connection.sendall(frame(b"J" + response))
                elif payload[:1] == b"P":
                    observation = message_to_observation(payload[1:])
                    output = self.service.infer(observation)
                    connection.sendall(frame(b"P" + actuation_to_payload(output)))
                else:
                    raise ValueError("unknown Model Service request kind")
            except (
                ConnectionError,
                KeyError,
                ValueError,
                RuntimeError,
                OSError,
                FloatingPointError,
            ) as error:
                error_payload = json.dumps({"ok": False, "error": str(error)}).encode()
                connection.sendall(frame(b"J" + error_payload))
