from __future__ import annotations

import torch
from torch import Tensor, nn


class StreamingAudioEncoder(nn.Module):
    def __init__(self, dim: int, output_tokens: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.cache_samples = kernel - stride
        self.conv = nn.Conv1d(1, dim, kernel_size=kernel, stride=stride)
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        combined = torch.cat((cache, audio), dim=1)
        encoded = self.conv(combined.unsqueeze(1))
        batch, channels, frames = encoded.shape
        if frames % self.output_tokens:
            raise ValueError("audio convolution frames must divide into output tokens")
        encoded = encoded.reshape(
            batch, channels, self.output_tokens, frames // self.output_tokens
        ).mean(dim=-1).transpose(1, 2)
        next_cache = combined[:, -self.cache_samples :] if self.cache_samples else combined[:, :0]
        return self.norm(encoded), next_cache


class VisionEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner = max(dim // 8, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, inner, 3, stride=2, padding=1),
            nn.GroupNorm(4, inner),
            nn.SiLU(),
            nn.Conv2d(inner, inner * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, inner * 2),
            nn.SiLU(),
            nn.Conv2d(inner * 2, dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 16, dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, video: Tensor) -> Tensor:
        features = self.encoder(video)
        batch, channels, height, width = features.shape
        if height % 4 or width % 4:
            raise ValueError("vision feature grid must be divisible into 4x4 tokens")
        features = features.reshape(
            batch, channels, 4, height // 4, 4, width // 4
        ).mean(dim=(3, 5))
        return features.flatten(2).transpose(1, 2) + self.position
