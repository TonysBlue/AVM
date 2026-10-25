"""Weight-compatible native Qwen3/Llama and Whisper/SigLIP tensor layers.

Cache keys are stored before RoPE so bounded recurrent windows can be reindexed.
The native layer equations and parameter names are retained.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.types import LayerKV


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, True) + self.eps)
        return self.weight * normalized.to(value.dtype)


def rotary(value: Tensor, positions: Tensor, theta: float) -> Tensor:
    dim = value.shape[-1]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=value.device).float() / dim)
    phase = positions.float()[:, None] * frequencies[None]
    phase = torch.cat((phase, phase), -1)[None, None]
    first, second = value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), -1)
    return value * phase.cos().to(value.dtype) + rotated * phase.sin().to(value.dtype)


class NativeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        *,
        qk_norm: bool,
        theta: float = 1_000_000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.theta = theta
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: Tensor,
        cache: LayerKV,
        *,
        max_tokens: int,
        cache_tokens: int | None = None,
        causal: bool = False,
        slow_context: Tensor | None = None,
    ) -> tuple[Tensor, LayerKV, LayerKV, Tensor]:
        batch, count, dim = hidden.shape
        query = self.q_norm(self.q_proj(hidden).view(batch, count, self.heads, -1)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden).view(batch, count, self.kv_heads, -1)).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, count, self.kv_heads, -1).transpose(1, 2)
        if torch.is_autocast_enabled(hidden.device.type):
            dtype = torch.get_autocast_dtype(hidden.device.type)
            query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
        retained = count if cache_tokens is None else cache_tokens
        combined_key = torch.cat((cache.key.to(key.dtype), key), 2)
        combined_value = torch.cat((cache.value.to(value.dtype), value), 2)
        prefix = cache.key.shape[2]
        keys = rotary(combined_key, torch.arange(prefix + count, device=hidden.device), self.theta)
        queries = rotary(
            query, torch.arange(prefix, prefix + count, device=hidden.device), self.theta
        )
        mask = None
        if causal:
            mask = torch.arange(prefix + count, device=hidden.device)[None] <= (
                prefix + torch.arange(count, device=hidden.device)[:, None]
            )
        attended = F.scaled_dot_product_attention(
            queries, keys, combined_value, attn_mask=mask,
            enable_gqa=self.heads != self.kv_heads,
        )
        if slow_context is not None:
            attended = attended + slow_context
        output = self.o_proj(attended.transpose(1, 2).reshape(batch, count, dim))
        cached_key = combined_key[:, :, : prefix + retained]
        cached_value = combined_value[:, :, : prefix + retained]
        excess = max(0, cached_key.shape[2] - max_tokens)
        updated = LayerKV(cached_key[:, :, excess:], cached_value[:, :, excess:])
        expired = LayerKV(cached_key[:, :, :excess], cached_value[:, :, :excess])
        return output, updated, expired, query


class GatedMLP(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class NativeDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        ffn: int,
        *,
        qk_norm: bool,
        theta: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.self_attn = NativeAttention(
            dim, heads, kv_heads, qk_norm=qk_norm, theta=theta, eps=eps
        )
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.mlp = GatedMLP(dim, ffn)

    def forward(
        self,
        hidden: Tensor,
        cache: LayerKV,
        *,
        max_tokens: int,
        cache_tokens: int | None = None,
        causal: bool = False,
        slow_memory=None,
        slow_state=None,
    ):
        normalized = self.input_layernorm(hidden)
        slow_context = None
        if slow_memory is not None:
            query = self.self_attn.q_norm(
                self.self_attn.q_proj(normalized).view(
                    *normalized.shape[:2], self.self_attn.heads, self.self_attn.head_dim
                )
            ).transpose(1, 2)
            repeat = self.self_attn.heads // self.self_attn.kv_heads
            from model.types import SlowMemoryState

            expanded = SlowMemoryState(
                slow_state.matrix.repeat_interleave(repeat, 1),
                slow_state.normalizer.repeat_interleave(repeat, 1),
            )
            slow_context = slow_memory.read(query, expanded)
        attended, updated, expired, _ = self.self_attn(
            normalized,
            cache,
            max_tokens=max_tokens,
            cache_tokens=cache_tokens,
            causal=causal,
            slow_context=slow_context,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        memory = slow_memory.write(slow_state, expired) if slow_memory is not None else None
        return hidden, updated, memory


class EncoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, key_bias: bool) -> None:
        super().__init__()
        self.heads, self.dim = heads, dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=key_bias)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden: Tensor, cache: LayerKV | None = None, max_tokens: int = 1500):
        batch, count, dim = hidden.shape
        query, key, value = (
            projection(hidden).view(batch, count, self.heads, -1).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        if cache is not None:
            key, value = (
                torch.cat((cache.key.to(key.dtype), key), 2),
                torch.cat((cache.value.to(value.dtype), value), 2),
            )
        if cache is None and query.dtype == torch.float32:
            scores = (query @ key.transpose(2, 3)) * ((dim // self.heads) ** -0.5)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output = probabilities @ value
        else:
            output = F.scaled_dot_product_attention(query, key, value)
        updated = LayerKV(key[:, :, -max_tokens:], value[:, :, -max_tokens:])
        return self.out_proj(output.transpose(1, 2).reshape(batch, count, dim)), updated


class AudioEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(dim, heads, key_bias=False)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, ffn), nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden: Tensor, cache: LayerKV, max_tokens: int):
        attended, updated = self.self_attn(self.self_attn_layer_norm(hidden), cache, max_tokens)
        hidden = hidden + attended
        hidden = hidden + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))
        return hidden, updated


class VisionMLP(nn.Module):
    def __init__(self, dim: int, ffn: int) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, ffn), nn.Linear(ffn, dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden), approximate="tanh"))


class VisionEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(dim, heads, key_bias=True)
        self.layer_norm1, self.layer_norm2 = (
            nn.LayerNorm(dim, eps=1e-6),
            nn.LayerNorm(dim, eps=1e-6),
        )
        self.mlp = VisionMLP(dim, ffn)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.self_attn(self.layer_norm1(hidden))[0]
        return hidden + self.mlp(self.layer_norm2(hidden))
