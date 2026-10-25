"""Fail-closed audit of the native MiniCPM model and acoustic source lock."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

LOCK_SCHEMA = "avm-native-source-lock-v1"
REPOSITORY = "openbmb/MiniCPM-o-4_5"
ACOUSTIC_FILES = (
    "flow.yaml",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2_25hz.onnx",
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("locked asset paths must be non-empty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or ".." in Path(relative).parts:
        raise ValueError(f"locked asset escapes root: {relative}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check_dimensions(config: dict[str, Any]) -> None:
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "query_num": 64,
        "audio_pool_step": 5,
    }
    nested = {
        "audio_config": {
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "num_mel_bins": 80,
        },
        "vision_config": {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 14,
        },
        "tts_config": {
            "hidden_size": 768,
            "num_hidden_layers": 20,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "intermediate_size": 3072,
            "num_vq": 1,
        },
    }
    if str(config.get("version")) != "4.5" or config.get("quantization_config"):
        raise ValueError("native full-precision MiniCPM-o 4.5 source is required")
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"native model dimension mismatch: {name} must be {value}")
    for section, fields in nested.items():
        values = config.get(section, {})
        for name, value in fields.items():
            if not isinstance(values, dict) or values.get(name) != value:
                raise ValueError(f"native model dimension mismatch: {section}.{name}")


def _audit_runtime(lock_path: Path, runtime: Any) -> tuple[Path, Path, Path]:
    if not isinstance(runtime, dict):
        raise ValueError("native acoustic runtime requires an immutable source revision")
    revision = str(runtime.get("revision", ""))
    package_release = revision == "stepaudio2-minicpmo==0.1.1"
    if not re.fullmatch(r"[0-9a-f]{40}", revision) and not package_release:
        raise ValueError("native acoustic runtime requires an immutable source revision")
    if not runtime.get("license") or not isinstance(runtime.get("root"), str):
        raise ValueError("native acoustic runtime root/license are required")
    interpreter = runtime.get("python")
    if not isinstance(interpreter, str) or not Path(interpreter).is_absolute():
        raise ValueError("native acoustic runtime requires an absolute isolated Python path")
    interpreter_path = Path(interpreter).expanduser()
    if not interpreter_path.is_file():
        raise FileNotFoundError(f"native acoustic Python is missing: {interpreter_path}")
    root = Path(runtime["root"]).expanduser()
    if not root.is_absolute():
        root = lock_path.parent / root
    root = root.resolve()
    files = runtime.get("files")
    entrypoint = runtime.get("entrypoint")
    if not isinstance(files, dict) or not isinstance(entrypoint, str) or entrypoint not in files:
        raise ValueError("native acoustic runtime entrypoint must be hash-locked")
    if package_release:
        archive = runtime.get("package_archive")
        published = "7bf176cdadedabca30f4abb1fa49b7514981fbc8f763314fa811c2e4160a4523"
        if not isinstance(archive, str) or files.get(archive) != published:
            raise ValueError("native acoustic runtime release archive must match published SHA-256")
    for name, expected in files.items():
        path = locked_path(root, name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid native runtime SHA-256: {name}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"missing or changed native acoustic runtime source: {name}")
    sources = {str(path.relative_to(root)) for path in root.rglob("*.py")}
    if not sources <= files.keys():
        raise ValueError("all native acoustic runtime Python sources must be hash-locked")
    return locked_path(root, entrypoint), interpreter_path, root


def _tensor_inventory(path: Path) -> dict[str, tuple[int, str]]:
    """Inspect safetensors metadata without loading the model into RAM or a GPU."""
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path.name}")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    data_bytes = path.stat().st_size - 8 - length
    result = {}
    intervals = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid tensor metadata: {name}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if dtype not in _DTYPE_BYTES or not isinstance(shape, list):
            raise ValueError(f"unsupported tensor metadata: {name}")
        count = 1
        for dimension in shape:
            if type(dimension) is not int or dimension < 0:
                raise ValueError(f"invalid tensor shape: {name}")
            count *= dimension
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_bytes
            or offsets[1] - offsets[0] != count * _DTYPE_BYTES[dtype]
        ):
            raise ValueError(f"invalid tensor byte offsets: {name}")
        intervals.append((offsets[0], offsets[1]))
        result[name] = (count, dtype)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"safetensors data has overlapping or missing bytes: {path.name}")
        cursor = end
    if cursor != data_bytes:
        raise ValueError(f"safetensors data length mismatch: {path.name}")
    return result


def _check_native_weights(inventory: dict[str, tuple[int, str]], vocab_size: int) -> None:
    required = {"resampler.query": 64 * 4096, "tts.emb_code.0.weight": vocab_size * 768}
    specifications = (
        ("llm.model.layers", 36, 4096, 1024, 12288, "mlp.gate_proj.weight"),
        ("tts.model.layers", 20, 768, 768, 3072, "mlp.gate_proj.weight"),
        ("apm.layers", 24, 1024, 1024, 4096, "fc1.weight"),
        ("vpm.encoder.layers", 27, 1152, 1152, 4304, "mlp.fc1.weight"),
    )
    for prefix, layers, width, kv_width, ffn_width, ffn_name in specifications:
        for layer in range(layers):
            required[f"{prefix}.{layer}.self_attn.q_proj.weight"] = width * width
            required[f"{prefix}.{layer}.self_attn.k_proj.weight"] = kv_width * width
            required[f"{prefix}.{layer}.{ffn_name}"] = width * ffn_width
    for name, elements in required.items():
        count, dtype = inventory.get(name, (0, ""))
        if count != elements or dtype not in {"F16", "BF16", "F32"}:
            raise ValueError(f"missing/incompatible native tensor: {name}")


def audit_native_assets(lock_path: str | Path, *, model_weights: bool = True) -> dict[str, Any]:
    """Require exact local sources; never download assets or accept fixture fallbacks."""
    lock_path = Path(lock_path).expanduser().resolve()
    lock = _read_json(lock_path)
    if lock.get("schema_id") != LOCK_SCHEMA or lock.get("repository") != REPOSITORY:
        raise ValueError("native source lock schema/repository is incompatible")
    if not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("revision", ""))):
        raise ValueError("native source revision must be an immutable 40-character commit")
    if not isinstance(lock.get("license"), str) or not lock["license"].strip():
        raise ValueError("native source license is required")
    runtime_entrypoint, runtime_python, runtime_root = _audit_runtime(
        lock_path, lock.get("acoustic_runtime")
    )
    root_value = lock.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("native source root is required")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = lock_path.parent / root
    root = root.resolve()
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("native source lock must contain per-file SHA-256 hashes")
    required = {"config.json"}
    if model_weights:
        required.add("model.safetensors.index.json")
    required.update(f"assets/token2wav/{name}" for name in ACOUSTIC_FILES)
    reference = lock.get("speaker_reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError("a locked fixed speaker reference is required")
    required.add(reference)
    if not required <= files.keys():
        missing = sorted(required - files.keys())
        raise ValueError(f"native lock is missing required files: {missing}")
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid SHA-256 for native file: {name}")
        path = locked_path(root, name)
        if not model_weights and name.endswith(".safetensors"):
            continue
        if not path.is_file():
            raise FileNotFoundError(f"missing locked native asset: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"native asset hash mismatch: {name}")
    config = _read_json(root / "config.json")
    _check_dimensions(config)
    context = lock.get("target_encoding_context")
    if context is not None and context != {
        "kind": "fixed-speaker-prefix-v1",
        "asset": reference,
        "samples": 32000,
        "sample_rate": 16000,
        "trim_tokens": 50,
    }:
        raise ValueError(
            "native target encoding context must be the locked two-second voice prefix"
        )
    precision = lock.get("acoustic_precision", "float32")
    if precision not in {"float16", "float32"}:
        raise ValueError("native flow precision must be float16 or float32")
    result = {
        "schema_id": "avm-native-assets-audit-v1",
        "passed": True,
        "source_lock_sha256": sha256_file(lock_path),
        "root": str(root),
        "revision": lock["revision"],
        "files_verified": sum(model_weights or not name.endswith(".safetensors") for name in files),
        "acoustic_vocab_size": config["tts_config"]["num_audio_tokens"],
        "speaker_reference": str(locked_path(root, reference)),
        "runtime_entrypoint": str(runtime_entrypoint),
        "runtime_python": str(runtime_python),
        "runtime_root": str(runtime_root),
        "model_weights_audited": model_weights,
        "speech_qualified": False,
        "target_encoding_context": context,
        "acoustic_precision": precision,
        "codec_id": "minicpm-o-4_5-s3-25hz",
        "codec_revision": lock["revision"],
        "codec_weight_sha256": files["assets/token2wav/speech_tokenizer_v2_25hz.onnx"],
    }
    if not model_weights:
        return result
    index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("native weight index is empty")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise ValueError("native weight shard names must be strings")
    shards = set(weight_map.values())
    if not shards <= files.keys():
        raise ValueError("every indexed native weight shard must be hash-locked")
    inventory = {}
    total_bytes = 0
    for shard in sorted(shards):
        tensors = _tensor_inventory(locked_path(root, shard))
        if inventory.keys() & tensors.keys():
            raise ValueError("duplicate tensor names across native weight shards")
        for name in tensors:
            if weight_map.get(name) != shard:
                raise ValueError(f"native weight index/shard mismatch: {name}")
        inventory.update(tensors)
        total_bytes += sum(count * _DTYPE_BYTES[dtype] for count, dtype in tensors.values())
    if inventory.keys() != weight_map.keys():
        raise ValueError("native weight index refers to missing tensors")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != total_bytes:
        raise ValueError("native weight total_size does not match tensor metadata")
    modules: dict[str, int] = defaultdict(int)
    for name, (count, _) in inventory.items():
        modules[name.split(".", maxsplit=1)[0]] += count
    required_modules = {"llm", "vpm", "resampler", "apm", "audio_projection_layer", "tts"}
    if not required_modules <= modules.keys():
        raise ValueError("native checkpoint is missing required model modules")
    _check_native_weights(inventory, config["tts_config"]["num_audio_tokens"])
    return {
        **result,
        "tensor_elements_by_module": dict(sorted(modules.items())),
        "tensor_elements": sum(modules.values()),
        "tensor_bytes": total_bytes,
    }


def require_native_qualification(
    audit: dict[str, Any], report_path: str | Path | None
) -> dict[str, Any]:
    if not report_path:
        raise ValueError("formal native execution requires a real speech qualification report")
    report = _read_json(Path(report_path).expanduser())
    if report.get("schema_id") != "avm-native-speech-qualification-v2" or not report.get("passed"):
        raise ValueError("native speech qualification is missing, incompatible, or failed")
    if report.get("source_lock_sha256") != audit["source_lock_sha256"]:
        raise ValueError("native speech qualification source lock differs from selected assets")
    runtime = report.get("runtime", {})
    if (
        not report.get("acoustic_passed")
        or not report.get("timing_passed")
        or runtime.get("target_encoding_context") != audit["target_encoding_context"]
        or runtime.get("precision") != audit["acoustic_precision"]
    ):
        raise ValueError(
            "native acoustic/timing qualification contract differs from selected assets"
        )
    worker = Path(__file__).resolve().parents[4] / "tools" / "native-speech" / "worker.py"
    if runtime.get("worker_sha256") != sha256_file(worker):
        raise ValueError("native qualification used a different AVM acoustic worker")
    if runtime.get("lookahead_tokens") != report.get("lookahead_tokens") or runtime.get(
        "tail_samples"
    ) != report.get("vocoder_tail_samples"):
        raise ValueError("native qualification has inconsistent acoustic timing identity")
    limits = {
        "silence_rms_max": 0.001,
        "boundary_jump_max": 0.15,
        "decode_p95_ms_max": 200,
        "rtf_max": 0.8,
    }
    thresholds = report.get("thresholds", {})
    if any(
        not isinstance(thresholds.get(name), (int, float)) or not 0 < thresholds[name] <= limit
        for name, limit in limits.items()
    ):
        raise ValueError("native qualification acceptance thresholds are missing or weakened")
    clips = report.get("clips", [])
    if report.get("error") or {clip.get("kind") for clip in clips} != {
        "speech",
        "silence",
        "transition",
    }:
        raise ValueError(
            "native qualification requires complete speech/silence/transition evidence"
        )
    for clip in clips:
        units = clip.get("units", 0)
        if (
            type(units) is not int
            or units <= 0
            or clip.get("token_count") != units * 5
            or clip.get("samples") != units * 4800
            or clip.get("duration_ms") != units * 200
            or not all(
                clip.get(name) is True for name in ("passed", "acoustic_passed", "timing_passed")
            )
        ):
            raise ValueError("native qualification duration or measurement evidence is invalid")
        measurements = {
            "decode_p95_ms_max": clip.get("decode_p95_ms"),
            "rtf_max": clip.get("real_time_factor"),
            "boundary_jump_max": clip.get("maximum_block_boundary_jump"),
        }
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= thresholds[name]
            for name, value in measurements.items()
        ):
            raise ValueError("native qualification measurements fail acceptance thresholds")
        silence = clip.get("silence_interval_rms", [])
        if clip["kind"] != "speech" and not silence:
            raise ValueError("native qualification is missing silence measurements")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= thresholds["silence_rms_max"]
            for value in silence
        ):
            raise ValueError("native qualification silence measurements fail acceptance")
    return report


def validate_native_checkpoint_identity(
    metadata: dict[str, Any],
    source_lock: str | Path | None,
    qualification: str | Path | None,
) -> None:
    if not source_lock:
        return
    expected = {
        "native_source_lock_sha256": sha256_file(Path(source_lock).expanduser()),
        "native_qualification_sha256": (
            sha256_file(Path(qualification).expanduser())
            if qualification and Path(qualification).expanduser().is_file()
            else None
        ),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"checkpoint {name} differs from the selected native identity")
