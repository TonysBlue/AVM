from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

HEADER = struct.Struct("!I")
TARGET_LUFS = -23.0


def read_exact(connection: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise ConnectionError("client disconnected")
        value.extend(block)
    return bytes(value)


def normalize(waveform: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    if rms <= 1e-7:
        raise ValueError("CosyVoice returned silent audio")
    scaled = waveform * (10 ** (TARGET_LUFS / 20) / rms)
    peak = float(np.max(np.abs(scaled), initial=0.0))
    limit = 10 ** (-1 / 20)
    if peak > limit:
        scaled *= limit / peak
    return scaled.astype(np.float32)


def synthesize(
    model: Any, value: dict[str, Any], prompt_cache: dict[str, str]
) -> None:
    request = value["request"]
    output = Path(value["output"])
    voice_id = str(request["voice_id"])
    prompt_hash = str(request["voice_prompt_sha256"])
    cached_prompt = prompt_cache.get(voice_id)
    if cached_prompt is not None and cached_prompt != prompt_hash:
        raise ValueError(f"voice prompt changed while worker is running: {voice_id}")
    if cached_prompt is None:
        model.add_zero_shot_spk(
            str(request["voice_prompt_text"]),
            str(request["voice_prompt_audio"]),
            voice_id,
        )
        prompt_cache[voice_id] = prompt_hash
    torch.manual_seed(1986 + int(request.get("attempt", 1)) - 1)
    results = list(
        model.inference_zero_shot(
            str(request["text"]),
            str(request["voice_prompt_text"]),
            str(request["voice_prompt_audio"]),
            zero_shot_spk_id=voice_id,
            stream=False,
        )
    )
    if not results:
        raise RuntimeError("CosyVoice produced no audio")
    waveform = torch.cat([item["tts_speech"].detach().cpu() for item in results], dim=1)
    waveform = torchaudio.functional.resample(waveform, model.sample_rate, 16000)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output,
        normalize(waveform.squeeze(0).numpy()),
        16000,
        format="FLAC",
        subtype="PCM_16",
    )
    output.with_suffix(".metrics.json").write_text(
        json.dumps({"integrated_lufs": TARGET_LUFS}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / "third_party" / "Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model = CosyVoice2(str(args.model), load_jit=False, load_trt=False, fp16=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    os.chmod(args.socket, 0o600)
    server.listen(4)
    server.settimeout(0.5)
    stopping = False
    prompt_cache: dict[str, str] = {}

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopping:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    header_size = HEADER.unpack(read_exact(connection, 4))[0]
                    request = json.loads(read_exact(connection, header_size))
                    synthesize(model, request, prompt_cache)
                    response = {"ok": True}
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                encoded = json.dumps(response).encode()
                connection.sendall(HEADER.pack(len(encoded)) + encoded)
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
