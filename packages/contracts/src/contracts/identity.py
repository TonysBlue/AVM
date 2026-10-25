from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProtocolIdentity:
    protocol_version: str = "avm-realtime-v2"
    codec_id: str = "minicpm-o-4_5-s3-25hz"

    def __post_init__(self) -> None:
        if not all((self.protocol_version, self.codec_id)):
            raise ValueError("protocol identity fields are required")
