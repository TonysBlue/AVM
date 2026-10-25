from __future__ import annotations

import math

import numpy as np
import torch
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.native_layers import AudioEncoderLayer, VisionEncoderLayer
from model.types import AudioEncoderState, LayerKV


def empty_kv(batch: int, heads: int, dim: int, device, dtype) -> LayerKV:
    return LayerKV(
        torch.empty(batch, heads, 0, dim, device=device, dtype=dtype),
        torch.empty(batch, heads, 0, dim, device=device, dtype=dtype),
    )


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(in_dim, out_dim), nn.Linear(out_dim, out_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(features)))


def whisper_positions(positions: Tensor, dim: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(dim // 2, device=positions.device, dtype=torch.float64)
        / (dim // 2 - 1)
    )
    phase = positions.double()[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1).float()


def whisper_mel_filters() -> Tensor:
    frequencies = np.linspace(0, 8000, 201)
    log_step = np.log(6.4) / 27
    mel_max = 15 + np.log(8000 / 1000) / log_step
    mel_points = np.linspace(0, mel_max, 82)
    hz = mel_points * (200 / 3)
    logarithmic = mel_points >= 15
    hz[logarithmic] = 1000 * np.exp(log_step * (mel_points[logarithmic] - 15))
    ramps = hz[:, None] - frequencies[None]
    differences = np.diff(hz)
    filters = np.maximum(
        0, np.minimum(-ramps[:-2] / differences[:-1, None], ramps[2:] / differences[1:, None])
    )
    filters *= (2 / (hz[2:] - hz[:-2]))[:, None]
    return torch.from_numpy(filters.astype(np.float32))


class NativeAudioEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.audio_dim
        self.conv1 = nn.Conv1d(80, dim, 3)
        self.conv2 = nn.Conv1d(dim, dim, 3, stride=2)
        self.embed_positions = nn.Embedding(1500, dim)
        self.embed_positions.weight.requires_grad_(False)
        self.layers = nn.ModuleList(
            AudioEncoderLayer(dim, config.audio_heads, config.audio_ffn_dim)
            for _ in range(config.audio_layers)
        )
        self.layer_norm = nn.LayerNorm(dim)
        if not self.embed_positions.weight.is_meta:
            with torch.no_grad():
                self.embed_positions.weight.copy_(whisper_positions(torch.arange(1500), dim))


class StreamingAudioEncoder(nn.Module):
    """Native audio weights with causal, continuous convolution and bounded KV."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = NativeAudioEncoder(config)
        self.projection = MultiModalProjector(config.audio_dim, config.model_dim)
        self.register_buffer("mel_filters", whisper_mel_filters(), persistent=False)

    def initial_state(self, batch: int, device, dtype) -> AudioEncoderState:
        config = self.config
        return AudioEncoderState(
            pcm=torch.zeros(batch, 240, device=device, dtype=torch.float32),
            conv1=torch.zeros(batch, 80, 2, device=device, dtype=dtype),
            conv2=torch.zeros(batch, config.audio_dim, 2, device=device, dtype=dtype),
            layer_kv=tuple(
                empty_kv(
                    batch, config.audio_heads, config.audio_dim // config.audio_heads, device, dtype
                )
                for _ in self.encoder.layers
            ),
            max_logmel=torch.full((batch, 1, 1), -10.0, device=device),
            position=torch.zeros(batch, device=device, dtype=torch.long),
        )

    def forward(self, audio: Tensor, cache: AudioEncoderState):
        if audio.shape[-1] != 3200:
            raise ValueError("native input requires exactly 3200 samples per 200 ms")
        combined = torch.cat((cache.pcm.float(), audio.float()), -1)
        spectrum = (
            torch.stft(
                combined,
                400,
                160,
                window=torch.hann_window(400, device=audio.device),
                center=False,
                return_complex=True,
            )
            .abs()
            .square()
        )
        raw_log = (self.mel_filters.float() @ spectrum).clamp_min(1e-10).log10()
        maxima = raw_log.amax(1, keepdim=True).cummax(-1).values
        maxima = torch.maximum(maxima, cache.max_logmel.float())
        mel = (torch.maximum(raw_log, maxima - 8.0) + 4.0) / 4.0
        mel = mel.to(self.encoder.conv1.weight.dtype)
        conv1_input = torch.cat((cache.conv1, mel), -1)
        first = F.gelu(self.encoder.conv1(conv1_input))
        conv2_input = torch.cat((cache.conv2, first), -1)
        hidden = F.gelu(self.encoder.conv2(conv2_input)).transpose(1, 2)
        count = hidden.shape[1]
        positions = cache.position[:, None] + torch.arange(count, device=audio.device)[None]
        position_features = self.encoder.embed_positions(positions.clamp_max(1499))
        extended = whisper_positions(positions, self.config.audio_dim).to(hidden.dtype)
        hidden = hidden + torch.where((positions < 1500)[..., None], position_features, extended)
        updated = []
        for layer, previous in zip(self.encoder.layers, cache.layer_kv, strict=True):
            if self.training and self.config.activation_checkpointing:

                def run(value, key, encoded_value, layer=layer):
                    result, new = layer(
                        value, LayerKV(key, encoded_value), self.config.audio_cache_frames
                    )
                    return result, new.key, new.value

                hidden, key, value = checkpoint(
                    run, hidden, previous.key, previous.value, use_reentrant=False
                )
                current = LayerKV(key, value)
            else:
                hidden, current = layer(hidden, previous, self.config.audio_cache_frames)
            updated.append(current)
        projected = self.projection(self.encoder.layer_norm(hidden))
        pooled = F.avg_pool1d(projected.transpose(1, 2), 5, 5).transpose(1, 2)
        if pooled.shape[1] != 2:
            raise RuntimeError("native audio must produce exactly two new tokens")
        return pooled, AudioEncoderState(
            combined[:, -240:],
            conv1_input[:, :, -2:],
            conv2_input[:, :, -2:],
            tuple(updated),
            maxima[:, :, -1:],
            cache.position + count,
        )


class VisionEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, config.vision_dim, config.vision_patch_size, stride=config.vision_patch_size
        )
        self.position_embedding = nn.Embedding(config.vision_position_grid**2, config.vision_dim)
        self.grid = config.vision_position_grid

    def forward(self, video: Tensor) -> tuple[Tensor, tuple[int, int]]:
        patches = self.patch_embedding(video)
        height, width = patches.shape[-2:]
        boundaries = torch.arange(1 / self.grid, 1.0, 1 / self.grid, device="cpu")
        rows = torch.bucketize(torch.arange(height).float() / height, boundaries, right=True)
        columns = torch.bucketize(torch.arange(width).float() / width, boundaries, right=True)
        positions = (rows[:, None] * self.grid + columns[None]).flatten().to(video.device)
        return patches.flatten(2).transpose(1, 2) + self.position_embedding(positions), (
            height,
            width,
        )


class NativeVisionEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            VisionEncoderLayer(config.vision_dim, config.vision_heads, config.vision_ffn_dim)
            for _ in range(config.vision_layers)
        )
        self.post_layernorm = nn.LayerNorm(config.vision_dim, eps=1e-6)
        self.activation_checkpointing = config.activation_checkpointing

    def forward(self, video: Tensor) -> tuple[Tensor, tuple[int, int]]:
        hidden, shape = self.embeddings(video)
        for layer in self.encoder.layers:
            hidden = (
                checkpoint(layer, hidden, use_reentrant=False)
                if self.training and self.activation_checkpointing
                else layer(hidden)
            )
        return self.post_layernorm(hidden), shape


def spatial_positions(height: int, width: int, dim: int, device) -> Tensor:
    omega = np.arange(dim // 4, dtype=np.float32) / (dim / 4.0)
    omega = 1.0 / 10000**omega
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    horizontal = np.einsum("hw,d->hwd", columns, omega)
    vertical = np.einsum("hw,d->hwd", rows, omega)
    positions = np.concatenate(
        (np.sin(horizontal), np.cos(horizontal), np.sin(vertical), np.cos(vertical)), axis=-1
    )
    return torch.from_numpy(positions).to(device)


class NativeVisionResampler(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.query = nn.Parameter(torch.randn(config.vision_tokens, dim) * 0.02)
        self.kv_proj = nn.Linear(config.vision_dim, dim, bias=False)
        self.attn = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)
        self.ln_q, self.ln_kv, self.ln_post = (nn.LayerNorm(dim, eps=1e-6) for _ in range(3))
        self.proj = nn.Parameter(torch.randn(dim, dim) * dim**-0.5)

    def forward(self, features: Tensor, shape: tuple[int, int]) -> Tensor:
        height, width = shape
        positions = spatial_positions(height, width, self.proj.shape[0], features.device)
        positions = positions.reshape(height * width, -1).to(features.dtype)
        features = self.ln_kv(self.kv_proj(features))
        query = self.ln_q(self.query)[None].expand(features.shape[0], -1, -1)
        hidden = self.attn(query, features + positions, features, need_weights=True)[0]
        return self.ln_post(hidden) @ self.proj


class VisionEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = NativeVisionEncoder(config)
        self.resampler = NativeVisionResampler(config)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim == 4:
            video = video[:, None]
        batch, slices, channels, height, width = video.shape
        flattened = video.reshape(batch * slices, channels, height, width)
        features, shape = self.encoder(flattened * 2.0 - 1.0)
        return self.resampler(features, shape).reshape(batch, -1, self.resampler.proj.shape[0])
