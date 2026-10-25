"""Compare AVM equations against immutable official modules and real weights.

Run with this isolated subproject's Python; no production import of remote code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open

REPOSITORY = Path(__file__).resolve().parents[2]
for package in ("model", "runtime", "contracts"):
    sys.path.insert(0, str(REPOSITORY / "packages" / package / "src"))

from model.acoustic_policy import acoustic_token_loss  # noqa: E402
from model.encoders import NativeVisionEncoder, NativeVisionResampler, empty_kv  # noqa: E402
from model.native_layers import (  # noqa: E402
    AudioEncoderLayer,
    NativeDecoderLayer,
    VisionEncoderLayer,
)
from model.native_weights import native_destination  # noqa: E402
from model.speech import NativeSpeechHead  # noqa: E402
from model.types import SpeechSamplingConfig  # noqa: E402
from runtime.config import ModelConfig  # noqa: E402
from runtime.native_assets import audit_native_assets  # noqa: E402
from transformers import LlamaConfig, Qwen3Config, WhisperConfig  # noqa: E402
from transformers.models.llama.modeling_llama import (  # noqa: E402
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)


def eager_vision_attention(attention, hidden):
    batch, count, dim = hidden.shape
    query, key, value = (
        projection(hidden).view(batch, count, attention.heads, -1).transpose(1, 2)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    scores = (query @ key.transpose(2, 3)) * ((dim // attention.heads) ** -0.5)
    probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
    output = probabilities @ value
    return attention.out_proj(output.transpose(1, 2).reshape(batch, count, dim)), None


def verify(root: Path, training_device: str = "cpu") -> dict:
    audit = audit_native_assets(root / "source-lock.json")
    package = types.ModuleType("locked_minicpm")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    official = importlib.import_module("locked_minicpm.modeling_minicpmo")
    vision = importlib.import_module("locked_minicpm.modeling_navit_siglip")
    results = {}
    torch.manual_seed(17)
    audio_config = WhisperConfig(**json.loads((root / "config.json").read_text())["audio_config"])
    audio_config._attn_implementation = "eager"
    with safe_open(root / "model-00001-of-00004.safetensors", framework="pt") as source:
        expected = actual = torch.randn(1, 10, 1024)
        with torch.no_grad():
            for index in range(24):
                prefix = f"apm.layers.{index}."
                state = {
                    name[len(prefix):]: source.get_tensor(name).float()
                    for name in source.keys() if name.startswith(prefix)
                }
                reference = official.MiniCPMWhisperEncoderLayer(audio_config, index).eval()
                candidate = AudioEncoderLayer(1024, 16, 4096).eval()
                reference.load_state_dict(state, assign=True)
                candidate.load_state_dict(state, assign=True)
                expected = reference(expected, attention_mask=None, layer_head_mask=None)[0]
                actual, _ = candidate(
                    actual, empty_kv(1, 16, 64, "cpu", torch.float32), 1500
                )
                del reference, candidate, state
            results["audio_encoder_24_layer_stack"] = float((expected - actual).abs().max())
    with safe_open(root / "model-00004-of-00004.safetensors", framework="pt") as source:

        def weights(prefix):
            return {
                key[len(prefix) :]: source.get_tensor(key).float()
                for key in source.keys()
                if key.startswith(prefix)
            }

        config = LlamaConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_hidden_layers=20,
            rms_norm_eps=1e-6,
        )
        config._attn_implementation = "eager"
        reference = LlamaDecoderLayer(config, 0).eval()
        candidate = NativeDecoderLayer(768, 12, 12, 3072, qk_norm=False, theta=10000.0).eval()
        state = weights("tts.model.layers.0.")
        reference.load_state_dict(state)
        candidate.load_state_dict(state)
        hidden = torch.randn(1, 9, 768)
        positions = torch.arange(9)[None]
        rotary = LlamaRotaryEmbedding(config)(hidden, positions)
        mask = torch.full((9, 9), -torch.inf).triu(1)[None, None]
        with torch.no_grad():
            expected = reference(
                hidden, attention_mask=mask, position_ids=positions, position_embeddings=rotary
            )[0]
            actual, _, _ = candidate(
                hidden, empty_kv(1, 12, 64, "cpu", torch.float32), max_tokens=100, causal=True
            )
        results["speech_layer"] = float((expected - actual).abs().max())
        del reference, candidate

        config = Qwen3Config(
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=36,
            rope_theta=1_000_000.0,
        )
        config._attn_implementation = "eager"
        reference = Qwen3DecoderLayer(config, 8).eval()
        candidate = NativeDecoderLayer(4096, 32, 8, 12288, qk_norm=True, theta=1_000_000.0).eval()
        state = weights("llm.model.layers.8.")
        reference.load_state_dict(state)
        candidate.load_state_dict(state)
        hidden = torch.randn(1, 9, 4096)
        rotary = Qwen3RotaryEmbedding(config)(hidden, positions)
        with torch.no_grad():
            expected = reference(
                hidden, attention_mask=mask, position_ids=positions, position_embeddings=rotary
            )[0]
            actual, _, _ = candidate(
                hidden, empty_kv(1, 8, 128, "cpu", torch.float32), max_tokens=100, causal=True
            )
        results["qwen3_layer"] = float((expected - actual).abs().max())
        del reference, candidate, state

        cfg = vision.SiglipVisionConfig(
            hidden_size=1152, intermediate_size=4304, num_attention_heads=16, num_hidden_layers=27
        )
        cfg._attn_implementation = "eager"
        reference = vision.SiglipEncoderLayer(cfg).eval()
        candidate = VisionEncoderLayer(1152, 16, 4304).eval()
        state = weights("vpm.encoder.layers.0.")
        reference.load_state_dict(state)
        candidate.load_state_dict(state)
        hidden = torch.randn(1, 16, 1152)
        with torch.no_grad():
            expected = reference(hidden, attention_mask=None)[0]
            actual = candidate(hidden)
        results["vision_layer"] = float((expected - actual).abs().max())
        del reference, candidate

        native_config = json.loads((root / "config.json").read_text())["vision_config"]
        cfg = vision.SiglipVisionConfig(**native_config)
        cfg._attn_implementation = "eager"
        reference = vision.SiglipVisionTransformer(cfg).eval()
        candidate = NativeVisionEncoder(ModelConfig()).eval()
        state = weights("vpm.")
        reference.load_state_dict(state, assign=True)
        candidate.load_state_dict(state, assign=True)
        del state
        reference.to(training_device)
        candidate.to(training_device)
        shapes = [(56, 56), (42, 70)]
        if training_device.startswith("cuda"):
            shapes.extend([(448, 448), (336, 602)])
        with torch.no_grad():
            for height, width in shapes:
                pixels = torch.randn(1, 3, height, width, device=training_device)
                expected = reference(pixels).last_hidden_state
                actual, shape = candidate(pixels)
                assert shape == (height // 14, width // 14)
                results[f"vision_encoder_{height}x{width}"] = float(
                    (expected - actual).abs().max()
                )
            for layer in candidate.encoder.layers:
                layer.self_attn.forward = types.MethodType(
                    eager_vision_attention, layer.self_attn
                )
            controlled, _ = candidate(pixels)
            results["vision_eager_order_control"] = float(
                (expected - controlled).abs().max()
            )
        del reference, candidate

        reference = official.Resampler(64, 4096, 32, kv_dim=1152).eval()
        candidate = NativeVisionResampler(ModelConfig()).eval()
        state = weights("resampler.")
        reference.load_state_dict(state)
        candidate.load_state_dict(state)
        hidden = torch.randn(1, 16, 1152)
        with torch.no_grad():
            expected = reference(hidden, tgt_sizes=torch.tensor([[4, 4]]))
            actual = candidate(hidden, (4, 4))
        results["vision_resampler"] = float((expected - actual).abs().max())
        del reference, candidate, state

        head = NativeSpeechHead(ModelConfig()).to(training_device)
        state = {}
        for key in source.keys():
            if key.startswith("tts."):
                destination = native_destination(key)
                if destination is not None:
                    state[destination.removeprefix("speech_head.")] = source.get_tensor(key)
        head.load_state_dict(state, strict=True)
        del state
        condition = torch.randn(1, 64, 4096, device=training_device, requires_grad=True)
        initial = head.initial_state(1, training_device, torch.float32)
        with torch.no_grad():
            generated, codes, _ = head(
                condition, initial, teacher_codes=None, sampling=SpeechSamplingConfig(greedy=True)
            )
        teacher, _, local = head(
            condition, initial, teacher_codes=codes, sampling=SpeechSamplingConfig(greedy=True)
        )
        error = float((teacher.detach() - generated).abs().max())
        torch.testing.assert_close(teacher.detach(), generated, atol=5e-5, rtol=5e-5)
        loss = acoustic_token_loss(
            teacher.squeeze(-2),
            codes.squeeze(-1),
            torch.ones(1, 5, dtype=torch.bool, device=training_device),
            forbidden_tokens=(6561,),
        )
        loss.backward()
        if not torch.isfinite(condition.grad).all() or condition.grad.norm() <= 0:
            raise ValueError("real native speech loss did not reach H conditions")
        speech_report = {
            "device": training_device,
            "parameters": sum(p.numel() for p in head.parameters()),
            "logit_shape": list(teacher.shape),
            "tokens": codes.cpu().tolist(),
            "teacher_sampling_maximum_error": error,
            "loss": float(loss.detach()),
            "condition_gradient_norm": float(condition.grad.norm()),
            "cache_shape_per_layer": list(local.layer_kv[0].key.shape),
            "all_parameter_gradients_finite": all(
                p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in head.parameters()
            ),
        }
        if not speech_report["all_parameter_gradients_finite"]:
            raise ValueError("real native speech head has missing/nonfinite parameter gradients")

    return {
        "revision": root.name,
        "dtype": "float32",
        "real_weights": True,
        "source_lock_sha256": audit["source_lock_sha256"],
        "model_weights_audited": audit["model_weights_audited"],
        "maximum_absolute_errors": results,
        "speech_forward_backward": speech_report,
        "fp32_absolute_tolerance": 3e-5,
        "passed": all(value <= 3e-5 for value in results.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-device", default="cpu")
    args = parser.parse_args()
    report = verify(args.root, args.training_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)
