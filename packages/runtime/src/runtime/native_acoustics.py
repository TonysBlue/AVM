"""Client for the isolated, asset-locked native acoustic qualification worker."""

from __future__ import annotations

import base64
import hashlib
import json
import selectors
import socket
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


class NativeAcousticBackend:
    def __init__(self, audit: dict[str, Any], *, cuda_device: int = 0) -> None:
        if audit.get("schema_id") != "avm-native-assets-audit-v1" or not audit.get("passed"):
            raise ValueError("native acoustic loading requires a successful local asset audit")
        worker = Path(__file__).resolve().parents[4] / "tools" / "native-speech" / "worker.py"
        self._process = subprocess.Popen(
            [audit["runtime_python"], str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            identity = self._request("initialize", audit=audit, cuda_device=cuda_device)
            identity["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
            identity["source_lock_sha256"] = audit["source_lock_sha256"]
            identity["vocab_size"] = audit["acoustic_vocab_size"]
            identity["acoustic_protocol_id"] = "avm-native-acoustics-v1"
            for key in ("codec_id", "codec_revision", "codec_weight_sha256"):
                identity[key] = audit[key]
            self.identity = identity
            self.lookahead_tokens = identity["lookahead_tokens"]
            self.tail_samples = identity["tail_samples"]
            self.vocab_size = audit["acoustic_vocab_size"]
            if any(
                type(value) is not int or value < 0
                for value in (self.lookahead_tokens, self.tail_samples)
            ):
                raise ValueError("native acoustic lookahead/tail must be non-negative integers")
        except Exception:
            self.close()
            raise

    def _request(self, operation: str, **values: Any) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("native acoustic worker pipes are unavailable")
        try:
            self._process.stdin.write(json.dumps({"operation": operation, **values}) + "\n")
            self._process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=120):
                    raise TimeoutError("native acoustic worker did not respond")
            line = self._process.stdout.readline(16 * 1024 * 1024)
            if not line or not line.endswith("\n"):
                raise RuntimeError("native acoustic worker disconnected or exceeded frame limit")
            result = json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError) as error:
            raise RuntimeError("native acoustic worker protocol failed") from error
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError(f"native acoustic worker failed: {result}")
        return result

    def synchronize(self) -> None:
        self._request("synchronize")

    def reset(self) -> None:
        self._request("reset")

    def encode(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 1 or not torch.isfinite(waveform).all():
            raise ValueError("native tokenizer input must be finite 16 kHz mono PCM")
        payload = waveform.detach().cpu().float().numpy().astype("<f4").tobytes()
        result = self._request("encode", pcm=base64.b64encode(payload).decode("ascii"))
        codes = torch.tensor(result["codes"], dtype=torch.long)
        if codes.ndim != 1 or codes.numel() * 16_000 != waveform.numel() * 25:
            raise ValueError("native tokenizer did not preserve the locked 25 Hz timeline")
        if torch.any((codes < 0) | (codes >= self.vocab_size - 1)):
            raise ValueError("native tokenizer returned a non-acoustic/EOS code")
        return codes

    def decode(self, tokens: list[int], *, final: bool) -> Tensor:
        result = self._request("decode", tokens=tokens, final=final)
        payload = base64.b64decode(result["pcm"], validate=True)
        if result.get("encoding") != "pcm_f32le" or len(payload) % 4:
            raise ValueError("native Token2Wav returned invalid float32 PCM bytes")
        waveform = np.frombuffer(payload, dtype="<f4").copy()
        return torch.from_numpy(waveform)

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._process.stdin:
            self._process.stdin.close()
        if self._process.stdout:
            self._process.stdout.close()

    def __enter__(self) -> NativeAcousticBackend:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class NativeAcousticClient:
    """Tensor RPC to the same native worker on a dedicated GPU."""

    def __init__(self, socket_path: str | Path, expected: dict[str, Any]) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.lease = uuid.uuid4().hex
        response, _ = self._request("health")
        self.identity = response["identity"]
        if self.identity.get("acoustic_protocol_id") != "avm-native-acoustics-v1":
            raise ValueError("native acoustic worker protocol differs from AVM")
        if any(self.identity.get(name) != value for name, value in expected.items()):
            raise ValueError("native acoustic worker identity differs from selected assets")
        self.lookahead_tokens = int(self.identity["lookahead_tokens"])
        self.tail_samples = int(self.identity["tail_samples"])
        self.vocab_size = int(self.identity["vocab_size"])
        self._closed = False

    def _request(self, operation: str, payload: bytes = b"", **fields):
        from runtime.codec_worker import receive_message, send_message

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(120)
            connection.connect(self.socket_path)
            send_message(
                connection,
                {
                    "operation": operation,
                    "lease": self.lease,
                    "acoustic_protocol_id": "avm-native-acoustics-v1",
                    **fields,
                },
                payload,
            )
            header, raw = receive_message(connection)
        if not header.get("ok"):
            raise RuntimeError(f"native acoustic worker failed: {header.get('error')}")
        return header, raw

    def reset(self) -> None:
        self._request("reset")

    def synchronize(self) -> None:
        self._request("synchronize")

    def encode(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 1 or not torch.isfinite(waveform).all():
            raise ValueError("native encode RPC requires finite mono PCM")
        raw = waveform.detach().cpu().float().numpy().astype("<f4").tobytes()
        header, response = self._request("encode", raw)
        if header.get("encoding") != "codes_i64le" or len(response) % 8:
            raise ValueError("native worker returned an invalid code frame")
        codes = torch.from_numpy(np.frombuffer(response, dtype="<i8").copy())
        if codes.numel() * 640 != waveform.numel() or torch.any(
            (codes < 0) | (codes >= self.vocab_size - 1)
        ):
            raise ValueError("native worker returned invalid acoustic codes or duration")
        return codes

    def decode(self, tokens: list[int], *, final: bool) -> Tensor:
        if not all(type(token) is int and 0 <= token < self.vocab_size - 1 for token in tokens):
            raise ValueError("native decode RPC excludes EOS/control/out-of-range codes")
        header, raw = self._request("decode", tokens=tokens, final=final)
        if header.get("encoding") != "pcm_f32le" or len(raw) % 4:
            raise ValueError("native worker returned an invalid PCM frame")
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())

    def close(self) -> None:
        if not self._closed:
            self._request("release")
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class NativeAcousticServer:
    def __init__(self, backend, socket_path: str | Path) -> None:
        self.backend, self.socket_path = backend, Path(socket_path).expanduser()
        self._lease: str | None = None
        self._decoder_ready = False
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._server: socket.socket | None = None

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server = server
        server.bind(str(self.socket_path))
        server.listen(16)
        server.settimeout(0.5)
        try:
            while not self._closed.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._closed.is_set():
                        break
                    raise
                threading.Thread(target=self._serve, args=(connection,), daemon=True).start()
        finally:
            self.close()

    def _serve(self, connection: socket.socket) -> None:
        from runtime.codec_worker import receive_message, send_message

        with connection:
            try:
                header, raw = receive_message(connection)
                if header.get("acoustic_protocol_id") != "avm-native-acoustics-v1":
                    raise ValueError("incompatible native acoustic worker protocol")
                operation, lease = header["operation"], str(header["lease"])
                with self._lock:
                    if operation == "health":
                        response, payload = {"identity": self.backend.identity}, b""
                    elif operation == "release":
                        if self._lease == lease:
                            self._lease = None
                            self._decoder_ready = False
                        response, payload = {}, b""
                    else:
                        if self._lease is not None and self._lease != lease:
                            raise ValueError(
                                "native worker already owns another active stream lease"
                            )
                        self._lease = lease
                        response, payload = {}, b""
                        if operation == "reset":
                            self._decoder_ready = False
                            self.backend.reset()
                            self._decoder_ready = True
                        elif operation == "synchronize":
                            self.backend.synchronize()
                        elif operation == "encode":
                            wave = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
                            codes = self.backend.encode(wave)
                            response = {"encoding": "codes_i64le"}
                            payload = codes.cpu().numpy().astype("<i8").tobytes()
                        elif operation == "decode":
                            if not self._decoder_ready:
                                raise ValueError(
                                    "native acoustic stream requires reset before decode"
                                )
                            tokens = header["tokens"]
                            if (
                                not isinstance(tokens, list)
                                or type(header["final"]) is not bool
                                or not all(
                                    type(token) is int and 0 <= token < self.backend.vocab_size - 1
                                    for token in tokens
                                )
                            ):
                                raise ValueError(
                                    "native acoustic decode requires valid acoustic codes"
                                )
                            waveform = self.backend.decode(tokens, final=header["final"])
                            response = {"encoding": "pcm_f32le"}
                            payload = waveform.cpu().float().numpy().astype("<f4").tobytes()
                        else:
                            raise ValueError("unknown native acoustic worker operation")
                send_message(connection, {"ok": True, **response}, payload)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                send_message(connection, {"ok": False, "error": str(error)})

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._server is not None:
            self._server.close()
        self.backend.close()
        self.socket_path.unlink(missing_ok=True)


def build_native_decoder(config):
    from runtime.native_assets import audit_native_assets, require_native_qualification

    expected = {
        "codec_id": config.data.codec_id,
        "codec_revision": config.data.codec_revision,
        "codec_weight_sha256": config.data.codec_weight_hash,
    }
    audit = None
    if config.data.dataset != "synthetic":
        if not config.model.native_source_lock:
            raise ValueError("formal native execution requires locked model assets")
        audit = audit_native_assets(config.model.native_source_lock, model_weights=False)
        qualification = require_native_qualification(audit, config.model.native_qualification)
        expected["source_lock_sha256"] = audit["source_lock_sha256"]
        expected["lookahead_tokens"] = qualification["lookahead_tokens"]
        expected["tail_samples"] = qualification["vocoder_tail_samples"]
        expected["vocab_size"] = config.model.speech_codebook_size
    if config.runtime.native_acoustic_socket:
        return NativeAcousticClient(config.runtime.native_acoustic_socket, expected)
    if audit is not None:
        decoder = NativeAcousticBackend(audit, cuda_device=config.runtime.acoustic_cuda_device)
        if any(decoder.identity.get(name) != value for name, value in expected.items()):
            decoder.close()
            raise ValueError("native decoder differs from the qualified acoustic timing identity")
        return decoder
    raise ValueError("synthetic physical tests require an explicit acoustic protocol service")
