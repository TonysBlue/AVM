"""Harness-backed expert capture and append-only session ledgers.

The collector deliberately keeps the physical boundary in Harness.  It accepts
one expert 200 ms speech PCM frame and records the audiovisual
observation/speech/receipt transition.
No model or privileged environment state is involved.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, SpeechSignal
from contracts.protocol import (
    actuation_to_payload,
    message_to_actuation,
    message_to_observation,
    observation_to_payload,
    payload_to_receipt,
    receipt_to_payload,
)
from model.types import Episode, StreamUnit
from runtime.video import video_tensor


def speech_signal_from_base64(value: str) -> SpeechSignal:
    try:
        pcm = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("speech PCM base64 is invalid") from error
    return SpeechSignal(pcm)


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CaptureTransition:
    session_id: str
    unit_index: int
    observation_sha256: str
    next_observation_sha256: str
    actuation_sha256: str
    receipt_sha256: str
    previous_chain_sha256: str
    chain_sha256: str
    speech_present: bool
    receipt_accepted: bool


class CaptureLedger:
    """Crash-safe transition ledger for one expert lifetime session."""

    def __init__(self, root: str | Path, *, session_id: str, lineage_id: str) -> None:
        if not session_id or not lineage_id:
            raise ValueError("capture ledger identity is required")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.lineage_id = lineage_id
        self._manifest = self.root / "transitions.jsonl"
        self._initial: ObservationSignal | None = None
        self._current: ObservationSignal | None = None
        self._records: list[CaptureTransition] = []
        self._chain = "0" * 64
        self.seal_metadata: dict[str, Any] = {}
        if self._manifest.exists():
            raise ValueError("capture ledger resume requires an explicit recovery implementation")

    @classmethod
    def load_sealed(cls, root: str | Path) -> CaptureLedger:
        """Load and fully verify a sealed ledger for export or replay."""

        resolved = Path(root).expanduser().resolve()
        marker = resolved / "sealed.json"
        manifest = resolved / "transitions.jsonl"
        if not marker.is_file() or not manifest.is_file():
            raise ValueError("capture ledger is not sealed")
        seal = json.loads(marker.read_text(encoding="utf-8"))
        instance = cls.__new__(cls)
        instance.root = resolved
        instance.session_id = str(seal["session_id"])
        instance.lineage_id = str(seal["lineage_id"])
        instance._manifest = manifest
        instance._records = []
        instance._chain = "0" * 64
        instance._initial = None
        instance._current = None
        instance.seal_metadata = dict(seal.get("metadata", {}))
        instance._load_verified()
        if (
            int(seal["unit_count"]) != len(instance._records)
            or str(seal["chain_sha256"]) != instance._chain
        ):
            raise ValueError("capture seal does not match the verified ledger")
        return instance

    def _load_verified(self) -> None:
        initial_path = self.root / "observation-000000000000.pb"
        if not initial_path.is_file():
            raise ValueError("capture ledger initial observation is missing")
        initial = message_to_observation(initial_path.read_bytes())
        if initial.session_id != self.session_id or initial.unit_index != 0:
            raise ValueError("capture ledger initial observation identity is invalid")
        self._initial = initial
        self._current = initial
        for expected, line in enumerate(
            self._manifest.read_text(encoding="utf-8").splitlines()
        ):
            item = json.loads(line)
            if (
                str(item["lineage_id"]) != self.lineage_id
                or str(item["session_id"]) != self.session_id
                or int(item["unit_index"]) != expected
            ):
                raise ValueError("capture ledger transition identity is invalid")
            observation_payload = (
                self.root / f"observation-{expected:012d}.pb"
            ).read_bytes()
            next_observation_payload = (
                self.root / f"observation-{expected + 1:012d}.pb"
            ).read_bytes()
            actuation_payload = (self.root / f"actuation-{expected:012d}.pb").read_bytes()
            receipt_payload = (self.root / f"receipt-{expected:012d}.pb").read_bytes()
            hashes = {
                "observation_sha256": _payload_sha256(observation_payload),
                "next_observation_sha256": _payload_sha256(next_observation_payload),
                "actuation_sha256": _payload_sha256(actuation_payload),
                "receipt_sha256": _payload_sha256(receipt_payload),
            }
            if any(str(item[name]) != digest for name, digest in hashes.items()):
                raise ValueError("capture ledger payload hash is invalid")
            identity = {
                "lineage_id": self.lineage_id,
                "session_id": self.session_id,
                "unit_index": expected,
                **hashes,
                "speech_present": bool(item["speech_present"]),
                "receipt_accepted": bool(item["receipt_accepted"]),
            }
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            chain = hashlib.sha256(bytes.fromhex(self._chain) + encoded).hexdigest()
            if (
                str(item["previous_chain_sha256"]) != self._chain
                or str(item["chain_sha256"]) != chain
            ):
                raise ValueError("capture ledger hash chain is invalid")
            actuation = message_to_actuation(actuation_payload)
            receipt = payload_to_receipt(receipt_payload)
            next_observation = message_to_observation(next_observation_payload)
            if (
                actuation.session_id != self.session_id
                or actuation.unit_index != expected
                or receipt.session_id != self.session_id
                or receipt.unit_index != expected
                or next_observation.session_id != self.session_id
                or next_observation.unit_index != expected + 1
                or receipt.accepted != identity["receipt_accepted"]
            ):
                raise ValueError("capture ledger physical identity is invalid")
            record = CaptureTransition(
                session_id=self.session_id,
                unit_index=expected,
                observation_sha256=hashes["observation_sha256"],
                next_observation_sha256=hashes["next_observation_sha256"],
                actuation_sha256=hashes["actuation_sha256"],
                receipt_sha256=hashes["receipt_sha256"],
                previous_chain_sha256=self._chain,
                chain_sha256=chain,
                speech_present=bool(item["speech_present"]),
                receipt_accepted=receipt.accepted,
            )
            self._records.append(record)
            self._chain = chain
            self._current = next_observation

    @property
    def records(self) -> tuple[CaptureTransition, ...]:
        return tuple(self._records)

    @property
    def chain_sha256(self) -> str:
        return self._chain

    @property
    def initial_observation(self) -> ObservationSignal:
        if self._initial is None:
            raise RuntimeError("capture session has not started")
        return self._initial

    @property
    def current_observation(self) -> ObservationSignal:
        if self._current is None:
            raise RuntimeError("capture session has not started")
        return self._current

    def start(self, observation: ObservationSignal) -> None:
        if self._initial is not None:
            raise ValueError("capture session is already started")
        if observation.session_id != self.session_id or observation.unit_index != 0:
            raise ValueError("initial observation identity is invalid")
        if observation.delta_ms != 200:
            raise ValueError("capture observation must use the 200 ms physical clock")
        self._initial = observation
        self._current = observation
        self._write_payload("observation-000000000000.pb", observation_to_payload(observation))

    def append(
        self,
        actuation: ActuationSignal,
        next_observation: ObservationSignal,
        receipt: EnvironmentReceipt,
    ) -> CaptureTransition:
        if self._current is None:
            raise RuntimeError("capture session has not started")
        expected = len(self._records)
        if actuation.session_id != self.session_id or actuation.unit_index != expected:
            raise ValueError("actuation identity is not contiguous")
        if (
            next_observation.session_id != self.session_id
            or next_observation.unit_index != expected + 1
        ):
            raise ValueError("next observation identity is not contiguous")
        if (
            next_observation.delta_ms != 200
            or next_observation.timestamp_ms <= self._current.timestamp_ms
        ):
            raise ValueError("next observation physical clock is invalid")
        if receipt.session_id != self.session_id or receipt.unit_index != expected:
            raise ValueError("receipt identity is invalid")
        observation_payload = observation_to_payload(self._current)
        actuation_payload = actuation_to_payload(actuation)
        receipt_payload = receipt_to_payload(receipt)
        next_observation_payload = observation_to_payload(next_observation)
        identity = {
            "lineage_id": self.lineage_id,
            "session_id": self.session_id,
            "unit_index": expected,
            "observation_sha256": _payload_sha256(observation_payload),
            "next_observation_sha256": _payload_sha256(next_observation_payload),
            "actuation_sha256": _payload_sha256(actuation_payload),
            "receipt_sha256": _payload_sha256(receipt_payload),
            "speech_present": (actuation.speech is not None),
            "receipt_accepted": receipt.accepted,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        chain = hashlib.sha256(bytes.fromhex(self._chain) + encoded).hexdigest()
        record = CaptureTransition(
            self.session_id,
            expected,
            identity["observation_sha256"],
            identity["next_observation_sha256"],
            identity["actuation_sha256"],
            identity["receipt_sha256"],
            self._chain,
            chain,
            (actuation.speech is not None),
            receipt.accepted,
        )
        self._write_payload(
            f"observation-{expected + 1:012d}.pb", next_observation_payload
        )
        self._write_payload(f"actuation-{expected:012d}.pb", actuation_payload)
        self._write_payload(f"receipt-{expected:012d}.pb", receipt_payload)
        self._append_json({**identity, "previous_chain_sha256": self._chain, "chain_sha256": chain})
        self._records.append(record)
        self._chain = chain
        self._current = next_observation
        return record

    def seal(self, metadata: dict[str, Any] | None = None) -> Path:
        if self._initial is None or not self._records:
            raise ValueError("cannot seal an empty capture session")
        if any(not record.receipt_accepted for record in self._records):
            raise ValueError("cannot seal a capture session containing a rejected receipt")
        marker = self.root / "sealed.json"
        if marker.exists():
            raise ValueError("capture session is already sealed")
        self.seal_metadata = dict(metadata or {})
        payload = {
            "lineage_id": self.lineage_id,
            "session_id": self.session_id,
            "unit_count": len(self._records),
            "chain_sha256": self._chain,
            "metadata": self.seal_metadata,
        }
        self._atomic_json(marker, payload)
        return marker

    def _write_payload(self, name: str, payload: bytes) -> None:
        path = self.root / name
        if path.exists():
            raise ValueError(f"capture payload already exists: {name}")
        with path.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())

    def _append_json(self, value: dict[str, Any]) -> None:
        with self._manifest.open("a", encoding="utf-8") as target:
            target.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", delete=False
        ) as target:
            temporary = Path(target.name)
            json.dump(value, target, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)


class ExpertCaptureSession:
    """Drive one Harness session from expert-provided speech."""

    def __init__(self, harness: Any, ledger: CaptureLedger, *, seed: int, scenario_id: str) -> None:
        self.harness = harness
        self.ledger = ledger
        self.seed = seed
        self.scenario_id = scenario_id
        self._started = False

    def start(self) -> ObservationSignal:
        observation = self.harness.start_lifetime_session(
            self.scenario_id, self.seed, self.ledger.session_id
        )
        self.ledger.start(observation)
        self._started = True
        return observation

    def step(
        self, speech: SpeechSignal
    ) -> tuple[ObservationSignal, EnvironmentReceipt]:
        if not self._started:
            raise RuntimeError("capture session has not started")
        unit_index = len(self.ledger.records)
        output = ActuationSignal(
            self.ledger.session_id,
            unit_index,
            speech,
        )
        observation, receipt = self.harness.apply(output)
        self.ledger.append(output, observation, receipt)
        if not receipt.accepted:
            failure = receipt.safety_violation or receipt.infrastructure_failure or "rejected"
            raise RuntimeError(f"Harness rejected capture unit {unit_index}: {failure}")
        return observation, receipt

    def finish(self, metadata: dict[str, Any] | None = None) -> Episode:
        episode = episode_from_capture(self.ledger, metadata=metadata)
        episode.validate(
            audio_samples=3200,
            speech_frames=5,
            speech_codebooks=1,
            speech_codebook_size=6562,
        )
        self.ledger.seal(metadata)
        return episode

    def close(self) -> None:
        self.harness.close(self.ledger.session_id)


def _speech_waveform(signal: SpeechSignal) -> torch.Tensor:
    if signal.encoding == "pcm_f32le":
        return torch.from_numpy(np.frombuffer(signal.pcm, dtype="<f4").copy()).float()
    values = np.frombuffer(signal.pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(values)


def _mic_waveform(observation: ObservationSignal) -> torch.Tensor:
    if observation.mic.encoding == "pcm_f32le":
        values = np.frombuffer(observation.mic.samples, dtype="<f4").copy()
    else:
        values = np.frombuffer(observation.mic.samples, dtype="<i2").astype(np.float32)
        values /= 32768.0
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
        raise ValueError("capture microphone PCM is not normalized and finite")
    return torch.from_numpy(values).float().reshape(1, -1)


def episode_from_capture(
    ledger: CaptureLedger, *, metadata: dict[str, Any] | None = None
) -> Episode:
    initial = ledger.initial_observation
    observations = [initial]
    speech: dict[int, torch.Tensor] = {}
    for index, _record in enumerate(ledger.records):
        observation_path = ledger.root / f"observation-{index + 1:012d}.pb"
        next_observation = message_to_observation(observation_path.read_bytes())
        actuation = message_to_actuation((ledger.root / f"actuation-{index:012d}.pb").read_bytes())
        observations.append(next_observation)
        if actuation.speech is not None:
            source_index = actuation.speech_unit_index
            if source_index in speech:
                raise ValueError("capture contains duplicate scheduled speech units")
            speech[source_index] = _speech_waveform(actuation.speech)
    units: list[StreamUnit] = []
    for index, observation in enumerate(observations[:-1]):
        video = video_tensor(observation.video)
        units.append(
            StreamUnit(
                timestamp_ms=torch.tensor([observation.timestamp_ms]),
                delta_ms=torch.tensor([observation.delta_ms]),
                mic_audio=_mic_waveform(observation),
                video=video,
                speech_codes=torch.zeros(1, 5, 1, dtype=torch.long),
                speech_codec_mask=torch.full((1, 5), index in speech, dtype=torch.bool),
            )
        )
    from media.pcm import resample_pcm

    target_pcm = torch.zeros(len(units) * 4800)
    for index, samples in speech.items():
        if index is None or not 0 <= index < len(units):
            raise ValueError("capture speech unit lies outside its observation timeline")
        target_pcm[index * 4800:(index + 1) * 4800] = samples
    target_speech = torch.from_numpy(resample_pcm(target_pcm.numpy(), 24000, 16000))
    supplied = metadata or {}
    codec_id = str(supplied.get("codec_id", ""))
    codec_weight_hash = str(supplied.get("codec_weight_hash", ""))
    codec_revision = str(supplied.get("codec_revision", ""))
    if not all((codec_id, codec_weight_hash, codec_revision)):
        raise ValueError("capture export requires complete codec identity metadata")
    environment_id = str(supplied.get("environment_id", ""))
    environment_version = str(supplied.get("environment_version", ""))
    protocol_version = str(supplied.get("protocol_version", ""))
    if not all((environment_id, environment_version, protocol_version)):
        raise ValueError("capture export requires complete Harness identity metadata")
    if protocol_version != "avm-realtime-v2":
        raise ValueError("capture export protocol identity is incompatible")
    base = {
        "stage": "sft",
        "schema_id": "avm-episode-v2",
        "dataset_scale": "canary",
        "sample_kind": "expert_capture",
        "supervision_kind": "speech",
        "task_id": ledger.session_id,
        "environment_id": environment_id,
        "environment_version": environment_version,
        "protocol_version": protocol_version,
        "runtime_identity": {
            "protocol_version": protocol_version,
            "environment_id": environment_id,
            "environment_version": environment_version,
        },
        "sample_rate": 16_000,
        "unit_ms": 200,
        "codec_frame_rate": 25.0,
        "codec_id": codec_id,
        "codec_weight_hash": codec_weight_hash,
        "codec_revision": codec_revision,
        "speech_codes_encoded": False,
        "capture_lineage_id": ledger.lineage_id,
        "capture_chain_sha256": ledger.chain_sha256,
    }
    base.update(supplied)
    return Episode(ledger.session_id, units, base, target_speech=target_speech)


__all__ = [
    "CaptureLedger",
    "CaptureTransition",
    "ExpertCaptureSession",
    "episode_from_capture",
    "speech_signal_from_base64",
]
