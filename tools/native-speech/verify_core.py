"""Probe the complete native Core with the main workspace's Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import model as model_package
import torch
from model import StreamingLatentLoop
from model.types import AudioVideoInput, GenerationOutput, SpeechSamplingConfig
from runtime.config import ModelConfig, load_config


def inspect_step(result: GenerationOutput, config: ModelConfig, unit: int) -> dict:
    output, codes = result.output, result.speech_codes
    state = output.state
    shapes = {
        "hidden": (1, config.perceiver_slots, config.model_dim),
        "perceiver_slots": (1, config.perceiver_slots, config.model_dim),
        "jepa_prediction": (1, config.perceiver_slots, config.model_dim),
        "speech_codec_logits": (1, 5, 1, config.speech_codebook_size),
        "value": (1,),
    }
    for name, expected in shapes.items():
        value = getattr(output, name)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"invalid full-Core output: {name}")
    if state.semantic_memory.shape != (1, config.semantic_memory_slots, config.model_dim):
        raise ValueError("semantic memory shape changed")
    if codes.shape != (1, 5, 1) or codes.dtype != torch.long:
        raise ValueError("full Core must generate exactly five int64 codes")
    if torch.any(codes < 0) or torch.any(codes >= config.speech_codebook_size - 1):
        raise ValueError("full Core generated EOS or an invalid acoustic code")
    if state.unit_index.item() != unit or state.audio_cache.position.item() != unit * 10:
        raise ValueError("recurrent unit or audio position did not advance")
    caches = (
        (state.layer_kv, config.num_layers, min(unit, config.kv_units) * config.perceiver_slots),
        (state.audio_cache.layer_kv, config.audio_layers,
         min(unit * 10, config.audio_cache_frames)),
        (state.speech_local.layer_kv, config.speech_depth_layers,
         min(unit, config.speech_window_units) * (config.perceiver_slots + 5)),
    )
    for layers, count, positions in caches:
        if len(layers) != count or any(
            cache.key.shape[2] != positions or cache.value.shape[2] != positions
            or not torch.isfinite(cache.key).all() or not torch.isfinite(cache.value).all()
            for cache in layers
        ):
            raise ValueError("full-Core cache continuity or finite-value check failed")
    return {
        "unit": unit,
        "hidden_shape": list(output.hidden.shape),
        "semantic_shape": list(state.semantic_memory.shape),
        "logit_shape": list(output.speech_codec_logits.shape),
        "codes": codes.tolist(),
        "backbone_kv_shape": list(state.layer_kv[0].key.shape),
        "audio_kv_shape": list(state.audio_cache.layer_kv[0].key.shape),
        "speech_kv_shape": list(state.speech_local.layer_kv[0].key.shape),
    }


def verify(config_path: str, steps: int, cpu_threads: int) -> dict:
    if steps < 2 or cpu_threads < 1:
        raise ValueError("full-Core probe requires at least two steps and one CPU thread")
    config = load_config(config_path)
    if not config.model.native_source_lock:
        raise ValueError("full-Core probe requires a locked native model")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(config.data.seed)
    model_root = Path(model_package.__file__).resolve().parent
    source_files = [*sorted(model_root.glob("*.py")), Path(__file__).resolve()]
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    model_config_sha256 = hashlib.sha256(
        json.dumps(asdict(config.model), sort_keys=True).encode()
    ).hexdigest()
    started = time.perf_counter()
    model = StreamingLatentLoop(config.model).eval()
    identity = model.native_identity
    if identity is None or model.parameter_count() < 8_000_000_000:
        raise ValueError("full native initialization was not performed")
    print("Strict full native import completed", flush=True)
    state = model.initial_state(1, "cpu")
    observation = AudioVideoInput(
        torch.zeros(1, 3200),
        torch.zeros(1, 3, config.data.video_height, config.data.video_width),
    )
    reports = []
    audio_shapes = []
    handle = model.audio_encoder.register_forward_hook(
        lambda module, inputs, output: audio_shapes.append(list(output[0].shape))
    )
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            for unit in range(1, steps + 1):
                step_started = time.perf_counter()
                result = model.generate_step(observation, state, SpeechSamplingConfig(greedy=True))
                report = inspect_step(result, config.model, unit)
                report["elapsed_seconds"] = time.perf_counter() - step_started
                reports.append(report)
                state = result.output.state
                print(json.dumps(report), flush=True)
    finally:
        handle.remove()
    if audio_shapes != [[1, 2, config.model.model_dim]] * steps:
        raise ValueError("native audio interface did not emit two tokens per unit")
    dtypes = Counter()
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
    return {
        "schema_id": "avm-native-core-probe-v1",
        "scope": "complete native Core, CPU BF16 calibration inference",
        "input_kind": "zero microphone and black RGB calibration signals",
        "source_lock_sha256": identity["source_lock_sha256"],
        "model_config_sha256": model_config_sha256,
        "avm_source_sha256": source_hashes,
        "revision": identity["revision"],
        "model_weights_audited": identity["model_weights_audited"],
        "parameters": model.parameter_count(),
        "parameter_dtypes": dict(dtypes),
        "imported_tensors": identity["imported_tensor_count"],
        "imported_parameters": identity["imported_parameters"],
        "unused_native_parameters": sum(
            math.prod(shape) for shape in identity["unused_native_parameters"].values()
        ),
        "audio_output_shapes": audio_shapes,
        "steps": reports,
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "seed": config.data.seed,
        "cpu_threads": cpu_threads,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "optimizer_run": False,
        "physical_session_run": False,
        "speech_qualified": False,
        "realtime_qualified": False,
        "core_probe_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canary.yaml")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.config, args.steps, args.cpu_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
