from __future__ import annotations

from contextlib import nullcontext

import torch
from runtime.config import ModelConfig
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model.attention import SlowMemory
from model.encoders import StreamingAudioEncoder, VisionEncoder, empty_kv
from model.native_layers import NativeDecoderLayer, RMSNorm
from model.perceiver import JEPAHead, Perceiver
from model.speech import NativeSpeechHead
from model.types import (
    AudioEncoderState,
    AudioVideoInput,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SlowMemoryState,
    SpeechSamplingConfig,
    StepOutput,
)
from model.value import ValueHead


class StreamingLatentLoop(nn.Module):
    """Native MiniCPM dimensions with AVM H/C, bounded KV and SlowMemory."""

    def __init__(self, config: ModelConfig, *, load_native: bool = True) -> None:
        super().__init__()
        self.config = config
        self.native_identity = None
        dim = config.model_dim
        with torch.device("meta") if config.native_source_lock else nullcontext():
            self.audio_encoder = StreamingAudioEncoder(config)
            self.vision_encoder = VisionEncoder(config)
            self.type_embedding = nn.Embedding(2, dim)
            self.perceiver = Perceiver(
                config.perceiver_dim,
                config.perceiver_heads,
                config.perceiver_ffn_dim,
                config.perceiver_slots,
                config.perceiver_layers,
                config.dropout,
                input_dim=dim,
            )
            self.jepa_head = JEPAHead(
                config.jepa_dim,
                config.jepa_heads,
                config.jepa_ffn_dim,
                config.perceiver_slots,
                config.jepa_layers,
                config.dropout,
                input_dim=dim,
            )
            self.semantic_slot_identity = nn.Parameter(
                torch.randn(config.semantic_memory_slots, dim) * 0.02
            )
            self.layers = nn.ModuleList(
                NativeDecoderLayer(
                    dim,
                    config.num_heads,
                    config.kv_heads,
                    config.ffn_dim,
                    qk_norm=True,
                    theta=config.rope_theta,
                )
                for _ in range(config.num_layers)
            )
            self.final_norm = RMSNorm(dim)
            self.slow_memories = nn.ModuleList(
                SlowMemory(config.kv_heads, dim // config.num_heads) for _ in self.layers
            )
            self.speech_head = NativeSpeechHead(config)
            self.value_head = ValueHead(dim, config.num_heads)
        if config.native_source_lock and load_native:
            from model.native_weights import initialize_native_model

            self.native_identity = initialize_native_model(self, config.native_source_lock)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        dtype = next(self.parameters()).dtype
        device = torch.device(device)
        head_dim = self.config.model_dim // self.config.num_heads
        return RecurrentState(
            layer_kv=tuple(
                empty_kv(batch_size, self.config.kv_heads, head_dim, device, dtype)
                for _ in self.layers
            ),
            semantic_memory=self.semantic_slot_identity.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            slow_memory=tuple(
                SlowMemoryState(
                    matrix=torch.zeros(
                        batch_size,
                        self.config.kv_heads,
                        head_dim,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    normalizer=torch.zeros(
                        batch_size, self.config.kv_heads, head_dim, device=device, dtype=dtype
                    ),
                )
                for _ in self.layers
            ),
            audio_cache=self.audio_encoder.initial_state(batch_size, device, dtype),
            hidden=torch.zeros(
                batch_size,
                self.config.perceiver_slots,
                self.config.model_dim,
                device=device,
                dtype=dtype,
            ),
            speech_local=self.speech_head.initial_state(batch_size, device, dtype),
            unit_index=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def place(self, devices: list[str], *, dtype: torch.dtype | None = None) -> None:
        """Place contiguous module groups by parameter count in one autograd graph."""
        if not devices:
            raise ValueError("model placement requires at least one device")
        groups = [
            self.audio_encoder,
            self.vision_encoder,
            self.type_embedding,
            self.perceiver,
            *self.layers,
            self.final_norm,
            self.speech_head,
            self.jepa_head,
            self.value_head,
        ]
        total = sum(parameter.numel() for module in groups for parameter in module.parameters())
        consumed = 0
        for module in groups:
            count = sum(parameter.numel() for parameter in module.parameters())
            index = min(len(devices) - 1, int((consumed + count / 2) * len(devices) / total))
            module.to(device=devices[index], dtype=dtype)
            consumed += count
        for layer, memory in zip(self.layers, self.slow_memories, strict=True):
            memory.to(device=next(layer.parameters()).device, dtype=dtype)
        self.semantic_slot_identity.data = self.semantic_slot_identity.data.to(
            device=devices[0], dtype=dtype
        )

    @property
    def input_device(self) -> torch.device:
        return self.perceiver.queries.device

    def encode_observation(
        self,
        unit: AudioVideoInput,
        audio_cache: AudioEncoderState,
    ) -> tuple[Tensor, AudioEncoderState]:
        audio_device = next(self.audio_encoder.parameters()).device
        audio_cache = torch.utils._pytree.tree_map(
            lambda value: value.to(audio_device) if isinstance(value, Tensor) else value,
            audio_cache,
        )
        audio, next_audio_cache = self.audio_encoder(unit.mic_audio.to(audio_device), audio_cache)
        vision = self.vision_encoder(unit.video.to(next(self.vision_encoder.parameters()).device))
        audio, vision = audio.to(self.input_device), vision.to(self.input_device)
        embedding = self.type_embedding.weight.to(self.input_device)
        inputs = torch.cat((audio + embedding[0], vision + embedding[1]), dim=1)
        return self.perceiver(inputs), next_audio_cache

    def forward_step(
        self,
        unit: AudioVideoInput,
        state: RecurrentState,
        *,
        speech_teacher_codes: Tensor | None = None,
        sampling: SpeechSamplingConfig | None = None,
    ) -> StepOutput:
        perceived, audio_cache = self.encode_observation(unit, state.audio_cache)
        sequence = torch.cat(
            (
                perceived + state.hidden.to(perceived.device),
                state.semantic_memory.to(perceived.device),
            ),
            dim=1,
        )
        hidden_count = self.config.perceiver_slots
        caches, memories = [], []
        for layer, previous, slow_module, slow_state in zip(
            self.layers,
            state.layer_kv,
            self.slow_memories,
            state.slow_memory,
            strict=True,
        ):
            layer_device = next(layer.parameters()).device
            sequence = sequence.to(layer_device)
            previous = LayerKV(previous.key.to(layer_device), previous.value.to(layer_device))
            slow_state = SlowMemoryState(
                slow_state.matrix.to(layer_device), slow_state.normalizer.to(layer_device)
            )

            def run(current, key, value, matrix, normalizer, layer=layer, slow_module=slow_module):
                output, updated, memory = layer(
                    current,
                    LayerKV(key, value),
                    max_tokens=self.config.kv_units * hidden_count,
                    cache_tokens=hidden_count,
                    slow_memory=slow_module,
                    slow_state=SlowMemoryState(matrix, normalizer),
                )
                return output, updated.key, updated.value, memory.matrix, memory.normalizer

            arguments = (
                sequence,
                previous.key,
                previous.value,
                slow_state.matrix,
                slow_state.normalizer,
            )
            if self.training and self.config.activation_checkpointing:
                sequence, key, value, matrix, normalizer = checkpoint(
                    run,
                    *arguments,
                    use_reentrant=False,
                )
            else:
                sequence, key, value, matrix, normalizer = run(*arguments)
            caches.append(LayerKV(key, value))
            memories.append(SlowMemoryState(matrix, normalizer))
        sequence = self.final_norm(sequence.to(self.final_norm.weight.device))
        hidden, semantic = sequence[:, :hidden_count], sequence[:, hidden_count:]
        speech_device = self.speech_head.emb_code[0].weight.device
        local = torch.utils._pytree.tree_map(
            lambda value: value.to(speech_device) if isinstance(value, Tensor) else value,
            state.speech_local,
        )
        logits, codes, local = self.speech_head(
            hidden.to(speech_device),
            local,
            teacher_codes=speech_teacher_codes.to(speech_device)
            if speech_teacher_codes is not None
            else None,
            sampling=sampling or SpeechSamplingConfig(greedy=True),
        )
        output_device = self.input_device
        prediction = self.jepa_head(hidden.to(next(self.jepa_head.parameters()).device))
        value = self.value_head(hidden.to(next(self.value_head.parameters()).device))
        hidden, semantic = hidden.to(output_device), semantic.to(output_device)
        return StepOutput(
            state=RecurrentState(
                tuple(caches),
                semantic,
                tuple(memories),
                audio_cache,
                hidden,
                local,
                state.unit_index + 1,
            ),
            speech_codec_logits=logits.to(output_device),
            hidden=hidden,
            perceiver_slots=perceived,
            jepa_prediction=prediction.to(output_device),
            value=value.to(output_device),
        )

    def forward(
        self,
        unit: AudioVideoInput,
        state: RecurrentState,
        speech_teacher_codes: Tensor | None = None,
    ) -> StepOutput:
        return self.forward_step(unit, state, speech_teacher_codes=speech_teacher_codes)

    @torch.no_grad()
    def generate_step(
        self,
        unit: AudioVideoInput,
        state: RecurrentState,
        sampling: SpeechSamplingConfig | None = None,
    ) -> GenerationOutput:
        output = self.forward_step(unit, state, sampling=sampling)
        return GenerationOutput(output, output.state.speech_local.previous_codes)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
