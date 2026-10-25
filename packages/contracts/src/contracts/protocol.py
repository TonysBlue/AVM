"""Strict protobuf serialization for the physical signal boundary.

Signal encoders return protobuf payloads only.  ``framing`` is exclusively a
socket concern; keeping the two layers separate prevents accidental nested or
double length headers between Model Service and Harness.
"""

from __future__ import annotations

import socket
from collections.abc import Callable

from contracts.control import ActuationSignal, SpeechSignal
from contracts.observation import MicSignal, ObservationSignal, VideoSignal
from contracts.realtime_pb2 import (
    ActuationSignal as ActuationMessage,
)
from contracts.realtime_pb2 import (
    EnvironmentReceipt as ReceiptMessage,
)
from contracts.realtime_pb2 import (
    ObservationSignal as ObservationMessage,
)
from contracts.receipt import EnvironmentReceipt


def observation_to_payload(value: ObservationSignal) -> bytes:
    message = ObservationMessage(
        protocol_version="avm-realtime-v2",
        session_id=value.session_id,
        unit_index=value.unit_index,
        timestamp_ms=value.timestamp_ms,
        delta_ms=value.delta_ms,
    )
    message.mic.sample_rate_hz = value.mic.sample_rate_hz
    message.mic.channels = value.mic.channels
    message.mic.encoding = value.mic.encoding
    message.mic.samples = value.mic.samples
    message.video.width = value.video.width
    message.video.height = value.video.height
    message.video.pixel_format = value.video.pixel_format
    message.video.encoding = value.video.encoding
    message.video.image = value.video.image
    message.video.captured_at_ms = value.video.captured_at_ms
    message.video.revision = value.video.revision
    return message.SerializeToString()


def observation_to_message(value: ObservationSignal) -> bytes:
    """Return the strict protobuf payload (without transport framing)."""
    return observation_to_payload(value)


def message_to_observation(payload: bytes) -> ObservationSignal:
    message = ObservationMessage.FromString(payload)
    if message.protocol_version != "avm-realtime-v2":
        raise ValueError("incompatible AVM observation protocol")
    if not message.HasField("mic") or not message.HasField("video"):
        raise ValueError("ObservationSignal requires mic and video messages")
    return ObservationSignal(
        session_id=message.session_id,
        unit_index=message.unit_index,
        timestamp_ms=message.timestamp_ms,
        delta_ms=message.delta_ms,
        mic=MicSignal(
            message.mic.samples,
            sample_rate_hz=message.mic.sample_rate_hz,
            channels=message.mic.channels,
            encoding=message.mic.encoding,
        ),
        video=VideoSignal(
            message.video.image,
            captured_at_ms=message.video.captured_at_ms,
            revision=message.video.revision,
            width=message.video.width,
            height=message.video.height,
            pixel_format=message.video.pixel_format,
            encoding=message.video.encoding,
        ),
    )


def actuation_to_payload(value: ActuationSignal) -> bytes:
    message = ActuationMessage(
        session_id=value.session_id, unit_index=value.unit_index,
        protocol_version="avm-realtime-v2",
    )
    if value.speech is not None:
        message.speech.sample_rate_hz = value.speech.sample_rate_hz
        message.speech.channels = value.speech.channels
        message.speech.encoding = value.speech.encoding
        message.speech.pcm = value.speech.pcm
    if value.speech_unit_index is not None:
        message.speech_unit_index = value.speech_unit_index
    if value.play_at_ms is not None:
        message.play_at_ms = value.play_at_ms
    message.generation_epoch = value.generation_epoch
    return message.SerializeToString()


def actuation_to_message(value: ActuationSignal) -> bytes:
    return actuation_to_payload(value)


def message_to_actuation(payload: bytes) -> ActuationSignal:
    message = ActuationMessage.FromString(payload)
    if message.protocol_version != "avm-realtime-v2":
        raise ValueError("incompatible AVM actuation protocol")
    return ActuationSignal(
        session_id=message.session_id,
        unit_index=message.unit_index,
        speech=SpeechSignal(
            message.speech.pcm,
            sample_rate_hz=message.speech.sample_rate_hz,
            channels=message.speech.channels,
            encoding=message.speech.encoding,
        ) if message.HasField("speech") else None,
        speech_unit_index=(message.speech_unit_index if message.HasField("speech_unit_index")
                           else None),
        play_at_ms=message.play_at_ms if message.HasField("play_at_ms") else None,
        generation_epoch=message.generation_epoch,
    )


def receipt_to_payload(value: EnvironmentReceipt) -> bytes:
    return ReceiptMessage(
        session_id=value.session_id,
        unit_index=value.unit_index,
        accepted=value.accepted,
        execution_latency_ms=value.execution_latency_ms,
        safety_violation=value.safety_violation or "",
        terminated=value.terminated,
        infrastructure_failure=value.infrastructure_failure or "",
    ).SerializeToString()


def payload_to_receipt(payload: bytes) -> EnvironmentReceipt:
    message = ReceiptMessage.FromString(payload)
    return EnvironmentReceipt(
        session_id=message.session_id,
        unit_index=message.unit_index,
        accepted=message.accepted,
        execution_latency_ms=message.execution_latency_ms,
        safety_violation=message.safety_violation or None,
        terminated=message.terminated,
        infrastructure_failure=message.infrastructure_failure or None,
    )


def serve_socket(
    connection: socket.socket,
    handler: Callable[[bytes], bytes],
) -> None:
    """Serve one length-framed payload, invoking a strict bytes handler."""

    from contracts.framing import frame, read_frame

    def read_exact(size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            block = connection.recv(size - len(output))
            if not block:
                raise ConnectionError("client disconnected")
            output.extend(block)
        return bytes(output)

    connection.sendall(frame(handler(read_frame(read_exact))))
