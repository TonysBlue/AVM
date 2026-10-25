"""JSONL worker. Run with this subproject's pinned CUDA/audio dependencies."""

from __future__ import annotations

import base64
import contextlib
import importlib.metadata
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any


def prepare_native_cache(model: Any, prompt: str) -> None:
    import soundfile
    import torch
    import torchaudio

    original = torchaudio.load
    original_setup = None
    if getattr(model, "float16", False):
        original_setup = model.flow.setup_cache

        def setup_half_cache(*args, **kwargs):
            with torch.autocast("cuda", dtype=torch.float16):
                return original_setup(*args, **kwargs)

        model.flow.setup_cache = setup_half_cache

    def read_fixed_prompt(path, *args, **kwargs):
        if Path(path).resolve() != Path(prompt).resolve():
            raise ValueError("native soundfile adapter may only read the locked voice asset")
        audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T.copy()), rate

    # The native source requests backend='soundfile'; torchaudio 2.9 ignores
    # that backend and otherwise requires TorchCodec even for a PCM WAV.
    try:
        torchaudio.load = read_fixed_prompt
        caches = model.set_stream_cache(prompt)
        if not isinstance(caches, tuple) or len(caches) != 2:
            raise ValueError("locked native source must return flow and vocoder caches")
        model.stream_cache, model.hift_cache_dict = caches
    finally:
        torchaudio.load = original
        if original_setup is not None:
            model.flow.setup_cache = original_setup


def main() -> int:
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch

    model = None
    prompt = None
    tokenizer = None
    reference_pcm = None

    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                if operation == "initialize":
                    if model is not None:
                        raise ValueError("native worker is already initialized")
                    if not torch.cuda.is_available():
                        raise RuntimeError("native Token2Wav qualification requires a CUDA GPU")
                    torch.cuda.set_device(int(request.get("cuda_device", 0)))
                    audit = request["audit"]
                    sys.path.insert(0, audit["runtime_root"])
                    import s3tokenizer
                    from stepaudio2 import Token2wav

                    if (
                        torch.__version__.split("+")[0] != "2.9.1"
                        or importlib.metadata.version("torchaudio") != "2.9.1"
                        or importlib.metadata.version("stepaudio2-minicpmo") != "0.1.1"
                    ):
                        raise ValueError("native worker dependencies differ from pinned runtime")
                    if (
                        Path(inspect.getfile(Token2wav)).resolve()
                        != Path(audit["runtime_entrypoint"]).resolve()
                    ):
                        raise ValueError("imported Token2Wav differs from audited implementation")
                    precision = audit.get("acoustic_precision", "float32")
                    random.seed(17)
                    np.random.seed(17)
                    torch.manual_seed(17)
                    torch.cuda.manual_seed_all(17)
                    model = Token2wav(
                        str(Path(audit["root"]) / "assets" / "token2wav"),
                        float16=precision == "float16",
                    )
                    prompt = audit["speaker_reference"]
                    tokenizer = s3tokenizer
                    if audit.get("target_encoding_context") is not None:
                        import soundfile

                        samples, rate = soundfile.read(prompt, dtype="float32", always_2d=True)
                        if rate != 16000 or samples.shape != (32000, 1):
                            raise ValueError(
                                "locked encoding reference must be two-second 16 kHz mono"
                            )
                        reference_pcm = torch.from_numpy(samples[:, 0].copy())
                    torch.manual_seed(17)
                    torch.cuda.manual_seed_all(17)
                    prepare_native_cache(model, prompt)
                    response = {
                        "lookahead_tokens": int(model.flow.pre_lookahead_len),
                        "tail_samples": int(model.source_cache_len),
                        "torch_version": torch.__version__,
                        "gpu_name": torch.cuda.get_device_name(),
                        "gpu_memory_bytes": torch.cuda.get_device_properties(
                            torch.cuda.current_device()
                        ).total_memory,
                        "precision": precision,
                        "seed": 17,
                        "target_encoding_context": audit.get("target_encoding_context"),
                        "prompt_io": "soundfile",
                    }
                else:
                    if model is None:
                        raise ValueError("native worker requires initialization")
                    if operation == "reset":
                        torch.manual_seed(17)
                        torch.cuda.manual_seed_all(17)
                        prepare_native_cache(model, prompt)
                        response = {}
                    elif operation == "synchronize":
                        torch.cuda.synchronize()
                        response = {}
                    elif operation == "encode":
                        raw = base64.b64decode(request["pcm"], validate=True)
                        pcm = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
                        if reference_pcm is None:
                            raise ValueError(
                                "native target encoding requires locked speaker context"
                            )
                        pcm = torch.cat((reference_pcm, pcm))
                        mels = tokenizer.log_mel_spectrogram(pcm)
                        mels, lengths = tokenizer.padding([mels])
                        codes, code_lengths = model.audio_tokenizer.quantize(
                            mels.cuda(), lengths.cuda()
                        )
                        count = int(code_lengths.reshape(-1)[0].item())
                        if count * 640 != pcm.numel():
                            raise ValueError(
                                "native tokenizer changed the prefixed target timeline"
                            )
                        response = {"codes": codes[0, 50:count].cpu().long().tolist()}
                    elif operation == "decode":
                        value = model.stream(
                            request["tokens"],
                            prompt_wav=prompt,
                            last_chunk=request["final"],
                            return_waveform=True,
                        )
                        if (
                            not isinstance(value, np.ndarray)
                            or value.ndim != 2
                            or value.shape[0] != 1
                            or not np.isfinite(value).all()
                        ):
                            raise ValueError("native decoder must return finite mono PCM")
                        raw = value[0].astype("<f4").tobytes()
                        response = {
                            "pcm": base64.b64encode(raw).decode("ascii"),
                            "encoding": "pcm_f32le",
                        }
                    else:
                        raise ValueError("unknown native acoustic worker operation")
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
            return 2
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
