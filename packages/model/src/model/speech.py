from __future__ import annotations

import torch
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

from model.acoustic_policy import acoustic_distribution_logits
from model.encoders import MultiModalProjector, empty_kv
from model.native_layers import NativeDecoderLayer, RMSNorm
from model.types import LayerKV, SpeechLocalState, SpeechSamplingConfig


class NativeSpeechHead(nn.Module):
    """Native semantic projector/Llama/code head, five steps including silence."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.speech_dim
        self.projector_semantic = MultiModalProjector(config.model_dim, dim)
        self.emb_code = nn.ModuleList([nn.Embedding(config.speech_codebook_size, dim)])
        self.head_code = nn.ModuleList(
            [weight_norm(nn.Linear(dim, config.speech_codebook_size, bias=False), name="weight")]
        )
        self.layers = nn.ModuleList(
            NativeDecoderLayer(
                dim,
                config.speech_depth_heads,
                config.speech_depth_heads,
                config.speech_depth_ffn_dim,
                qk_norm=False,
                theta=10000.0,
            )
            for _ in range(config.speech_depth_layers)
        )
        self.norm = RMSNorm(dim)

    def initial_state(self, batch: int, device, dtype) -> SpeechLocalState:
        return SpeechLocalState(
            tuple(
                empty_kv(
                    batch,
                    self.config.speech_depth_heads,
                    self.config.speech_dim // self.config.speech_depth_heads,
                    device,
                    dtype,
                )
                for _ in self.layers
            ),
            torch.zeros(batch, 5, 1, device=device, dtype=torch.long),
        )

    def _run(self, hidden: Tensor, caches: tuple[LayerKV, ...]):
        updated = []
        limit = self.config.speech_window_units * (self.config.perceiver_slots + 5)
        for layer, cache in zip(self.layers, caches, strict=True):
            if self.training and self.config.activation_checkpointing:

                def run(value, key, encoded_value, layer=layer):
                    result, new, _ = layer(
                        value, LayerKV(key, encoded_value), max_tokens=limit, causal=True
                    )
                    return result, new.key, new.value

                hidden, key, value = checkpoint(
                    run, hidden, cache.key, cache.value, use_reentrant=False
                )
                new = LayerKV(key, value)
            else:
                hidden, new, _ = layer(hidden, cache, max_tokens=limit, causal=True)
            updated.append(new)
        return self.norm(hidden), tuple(updated)

    def forward(
        self,
        hidden: Tensor,
        state: SpeechLocalState,
        *,
        teacher_codes: Tensor | None,
        sampling: SpeechSamplingConfig,
    ) -> tuple[Tensor, Tensor, SpeechLocalState]:
        batch = hidden.shape[0]
        if teacher_codes is not None:
            if teacher_codes.shape != (batch, 5, 1):
                raise ValueError("speech teacher codes must be [B,5,1]")
            if torch.any(
                (teacher_codes < 0) | (teacher_codes >= self.config.speech_codebook_size - 1)
            ):
                raise ValueError("speech teacher codes must exclude EOS/control tokens")
        group_size = self.config.perceiver_slots + 5
        maximum_prefix = (self.config.speech_window_units - 1) * group_size
        caches = tuple(
            LayerKV(
                cache.key[:, :, -maximum_prefix:] if maximum_prefix else cache.key[:, :, :0],
                cache.value[:, :, -maximum_prefix:] if maximum_prefix else cache.value[:, :, :0],
            )
            for cache in state.layer_kv
        )
        condition = F.normalize(self.projector_semantic(hidden), dim=-1)
        if teacher_codes is not None:
            sequence = torch.cat((condition, self.emb_code[0](teacher_codes[:, :, 0])), dim=1)
            encoded, caches = self._run(sequence, caches)
            start = condition.shape[1] - 1
            logits = self.head_code[0](encoded[:, start : start + 5])[:, :, None]
            return logits, teacher_codes, SpeechLocalState(caches, teacher_codes)
        current, caches = self._run(condition, caches)
        logits, selected = [], []
        for index in range(5):
            raw = self.head_code[0](current[:, -1])
            logits.append(raw)
            code = (
                teacher_codes[:, index, 0]
                if teacher_codes is not None
                else self._sample(raw, sampling)
            )
            selected.append(code)
            # Consume the fifth token too. The next condition must follow it.
            current, caches = self._run(self.emb_code[0](code)[:, None], caches)
        codes = torch.stack(selected, 1)[..., None]
        return torch.stack(logits, 1)[:, :, None], codes, SpeechLocalState(caches, codes)

    @staticmethod
    def _sample(logits: Tensor, sampling: SpeechSamplingConfig) -> Tensor:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite acoustic policy logits")
        if sampling.greedy or sampling.temperature <= 0:
            return acoustic_distribution_logits(
                logits, forbidden_tokens=(logits.shape[-1] - 1,)
            ).argmax(-1)
        constrained = acoustic_distribution_logits(
            logits,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            forbidden_tokens=(logits.shape[-1] - 1,),
        )
        return torch.multinomial(constrained.softmax(-1), 1).squeeze(-1)
