from __future__ import annotations

import io

import pytest
import torch
from model.encoders import empty_kv
from model.native_layers import EncoderAttention, NativeAttention


def test_fp32_vision_preserves_native_attention_order(monkeypatch) -> None:
    attention = EncoderAttention(32, 4, key_bias=True)
    hidden = torch.randn(1, 17, 32, requires_grad=True)
    query, key, value = (
        projection(hidden).view(1, 17, 4, 8).transpose(1, 2)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    probabilities = ((query @ key.transpose(2, 3)) * (8 ** -0.5)).softmax(-1)
    expected = attention.out_proj((probabilities @ value).transpose(1, 2).reshape(1, 17, 32))
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention",
        lambda *args, **kwargs: pytest.fail("FP32 vision must preserve native eager order"),
    )
    actual, _ = attention(hidden)
    assert torch.equal(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().mean(), (hidden, *attention.parameters()))
    expected_grads = torch.autograd.grad(
        expected.square().mean(), (hidden, *attention.parameters())
    )
    assert all(torch.equal(actual, expected)
               for actual, expected in zip(actual_grads, expected_grads, strict=True))


def test_bf16_vision_keeps_sdpa_backend(monkeypatch) -> None:
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def tracked(*args, **kwargs):
        calls.append(args[0].dtype)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", tracked)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        EncoderAttention(32, 4, key_bias=True)(torch.randn(1, 17, 32))
    assert calls == [torch.bfloat16]


@pytest.mark.parametrize("qk_norm", [False, True])
def test_autocast_kv_remains_bounded_and_preserves_cross_unit_gradients(qk_norm) -> None:
    attention = NativeAttention(64, 4, 2, qk_norm=qk_norm)
    first = torch.randn(1, 7, 64, requires_grad=True)
    second = torch.randn(1, 7, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cache, _, _ = attention(
            first, empty_kv(1, 2, 16, "cpu", torch.float32), max_tokens=10
        )
        assert cache.key.dtype == cache.value.dtype == torch.bfloat16
        output, updated, _, _ = attention(second, cache, max_tokens=10)
    assert updated.key.shape[2] == updated.value.shape[2] == 10
    output.square().mean().backward()
    assert first.grad is not None and first.grad.norm() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in attention.parameters())
    buffer = io.BytesIO()
    torch.save(cache, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, restored_cache, _, _ = attention(second, restored, max_tokens=10)
    assert torch.equal(actual, output)
    assert restored_cache.key.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in attention.parameters())


def test_audio_encoder_cache_follows_projection_precision() -> None:
    attention = EncoderAttention(32, 4, key_bias=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, cache = attention(
            torch.randn(1, 10, 32), empty_kv(1, 4, 8, "cpu", torch.float32)
        )
    assert cache.key.dtype == cache.value.dtype == torch.bfloat16
