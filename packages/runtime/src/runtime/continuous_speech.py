"""Continuous acoustic-token scheduling without a speech-mode or silence bypass."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import torch
from contracts import ActuationSignal, ObservationSignal, SpeechSignal
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ContinuousSpeechClock:
    lookahead_tokens: int
    vocoder_tail_samples: int
    playback_delay_ms: int
    compute_reserve_ms: int = 200
    unit_ms: int = 200
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    token_rate: int = 25
    tokens_per_unit: int = 5

    def __post_init__(self) -> None:
        fixed = (
            self.unit_ms,
            self.input_sample_rate,
            self.output_sample_rate,
            self.token_rate,
            self.tokens_per_unit,
        )
        if fixed != (200, 16_000, 24_000, 25, 5):
            raise ValueError("continuous speech requires 200 ms, 16/24 kHz and five 25 Hz tokens")
        integers = (
            self.lookahead_tokens,
            self.vocoder_tail_samples,
            self.playback_delay_ms,
            self.compute_reserve_ms,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("speech timing values must be non-negative integers")
        if self.compute_reserve_ms < 1 or self.playback_delay_ms % self.unit_ms:
            raise ValueError("playback delay must use full units and reserve compute time")
        if self.playback_delay_ms < self.minimum_delay_ms:
            raise ValueError("playback delay does not cover lookahead, vocoder tail and compute")

    @property
    def input_samples(self) -> int:
        return self.input_sample_rate * self.unit_ms // 1000

    @property
    def output_samples(self) -> int:
        return self.output_sample_rate * self.unit_ms // 1000

    @property
    def samples_per_token(self) -> int:
        return self.output_sample_rate // self.token_rate

    @property
    def minimum_delay_ms(self) -> int:
        lookahead_units = math.ceil(self.lookahead_tokens / self.tokens_per_unit)
        tail_units = math.ceil(self.vocoder_tail_samples / self.output_samples)
        compute_units = math.ceil(self.compute_reserve_ms / self.unit_ms)
        return (lookahead_units + tail_units + compute_units) * self.unit_ms


class ContinuousAcousticDecoder(Protocol):
    """An already initialized, asset-locked decoder for one stream epoch."""

    def decode(self, tokens: list[int], *, final: bool) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class ScheduledSpeechPCM:
    epoch: int
    unit_index: int
    scheduled_at_ms: int
    samples: Tensor


class ContinuousSpeechStream:
    def __init__(
        self,
        decoder: ContinuousAcousticDecoder,
        clock: ContinuousSpeechClock,
        *,
        acoustic_vocab_size: int,
        forbidden_tokens: frozenset[int],
        epoch: int,
    ) -> None:
        if acoustic_vocab_size < 2 or epoch < 0:
            raise ValueError("invalid acoustic vocabulary or epoch")
        if not forbidden_tokens or not all(
            0 <= token < acoustic_vocab_size for token in forbidden_tokens
        ):
            raise ValueError("locked EOS/control token exclusions are required")
        if len(forbidden_tokens) >= acoustic_vocab_size:
            raise ValueError("acoustic vocabulary must contain valid codes")
        self.decoder = decoder
        self.clock = clock
        self.acoustic_vocab_size = acoustic_vocab_size
        self.forbidden_tokens = forbidden_tokens
        self.epoch = epoch
        self.next_input_unit = 0
        self.next_output_unit = 0
        self.tokens_consumed = 0
        self.samples_decoded = 0
        self._tokens: list[int] = []
        self._pcm = torch.empty(0, dtype=torch.float32)
        self._origin_ms: int | None = None
        self._closed = False
        self._failed = False
        self._finished = False

    @property
    def buffered_tokens(self) -> int:
        return len(self._tokens)

    @property
    def buffered_samples(self) -> int:
        return self._pcm.numel()

    def _check_active(self, epoch: int) -> bool:
        if epoch != self.epoch or self._closed:
            return False
        if self._failed:
            raise RuntimeError("speech stream failed; a new decoder epoch is required")
        return True

    def _decode(self, count: int, *, final: bool) -> None:
        needed = count if final else count + self.clock.lookahead_tokens
        try:
            waveform = self.decoder.decode(self._tokens[:needed], final=final)
            if not isinstance(waveform, Tensor) or waveform.ndim != 1:
                raise ValueError("native decoder must return one-dimensional PCM")
            waveform = waveform.detach().cpu().to(torch.float32)
            if not torch.isfinite(waveform).all() or torch.any(waveform.abs() > 1):
                peak = float(waveform.abs().max()) if waveform.numel() else 0.0
                raise ValueError(f"native decoder PCM must be finite and normalized; peak={peak}")
            consumed = self.tokens_consumed + count
            decoded = self.samples_decoded + waveform.numel()
            expected = consumed * self.clock.samples_per_token
            retained = expected - decoded
            if not 0 <= retained <= self.clock.vocoder_tail_samples:
                raise ValueError("native decoder sample count violates locked tail/time alignment")
            if final and retained:
                raise ValueError("native final flush did not return the complete waveform tail")
            queue_limit = (
                self.clock.playback_delay_ms // self.clock.unit_ms + 2
            ) * self.clock.output_samples
            if self._pcm.numel() + waveform.numel() > queue_limit:
                raise BufferError("continuous speech PCM queue is full")
        except Exception:
            # A native decoder may already have mutated its flow/vocoder caches.
            self._failed = True
            raise
        self._pcm = torch.cat((self._pcm, waveform))
        del self._tokens[:count]
        self.tokens_consumed = consumed
        self.samples_decoded = decoded

    def push(self, unit_index: int, timestamp_ms: int, codes: Tensor, *, epoch: int) -> bool:
        if not self._check_active(epoch):
            return False
        if self._finished:
            raise ValueError("cannot append acoustic tokens after session-end flush")
        if unit_index != self.next_input_unit:
            raise ValueError("continuous speech input unit is out of order")
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError("continuous speech timestamp must be non-negative milliseconds")
        if self._origin_ms is not None:
            if timestamp_ms != self._origin_ms + unit_index * self.clock.unit_ms:
                raise ValueError("continuous speech timestamps must follow the fixed clock")
        if codes.shape != (self.clock.tokens_per_unit,) or codes.dtype != torch.long:
            raise ValueError("each unit must contain exactly five int64 acoustic tokens")
        values = codes.detach().cpu().tolist()
        if any(
            not 0 <= token < self.acoustic_vocab_size or token in self.forbidden_tokens
            for token in values
        ):
            raise ValueError("EOS/control/out-of-vocabulary tokens are not acoustic output")
        if self._origin_ms is None:
            self._origin_ms = timestamp_ms
        self._tokens.extend(values)
        while len(self._tokens) >= self.clock.tokens_per_unit + self.clock.lookahead_tokens:
            self._decode(self.clock.tokens_per_unit, final=False)
        self.next_input_unit += 1
        return True

    def take(self, unit_index: int, *, now_ms: int, epoch: int) -> ScheduledSpeechPCM | None:
        if not self._check_active(epoch):
            return None
        if self._origin_ms is None or unit_index != self.next_output_unit:
            raise ValueError("continuous speech output unit is out of order")
        if unit_index >= self.next_input_unit:
            raise ValueError("cannot play speech before its condition was generated")
        scheduled = self._origin_ms + unit_index * self.clock.unit_ms + self.clock.playback_delay_ms
        if self._pcm.numel() < self.clock.output_samples:
            if now_ms < scheduled:
                return None
            raise TimeoutError("continuous speech missed its playback deadline")
        samples = self._pcm[: self.clock.output_samples].clone()
        self._pcm = self._pcm[self.clock.output_samples :]
        self.next_output_unit += 1
        return ScheduledSpeechPCM(self.epoch, unit_index, scheduled, samples)

    def finish(self, *, epoch: int) -> bool:
        if not self._check_active(epoch):
            return False
        if self._finished:
            return False
        self._decode(len(self._tokens), final=True)
        if self.samples_decoded != self.next_input_unit * self.clock.output_samples:
            self._failed = True
            raise ValueError("native stream duration differs from the complete token timeline")
        self._finished = True
        return True

    def cancel(self) -> None:
        self._closed = True
        self._tokens.clear()
        self._pcm = torch.empty(0, dtype=torch.float32)


class ContinuousSpeechExecutor:
    """Shared physical speech boundary for Model Service and Online PPO."""

    def __init__(self, decoder, clock: ContinuousSpeechClock, *, vocab_size: int) -> None:
        self.decoder, self.clock, self.vocab_size = decoder, clock, vocab_size
        self.epoch = 0
        self.unit_offset = 0
        self.session_id: str | None = None
        self.stream: ContinuousSpeechStream | None = None

    def reset(self, session_id: str, *, unit_offset: int = 0) -> None:
        if self.stream is not None:
            self.stream.cancel()
            self.epoch += 1
        self.decoder.reset()
        self.session_id = session_id
        self.unit_offset = unit_offset
        self.stream = ContinuousSpeechStream(
            self.decoder,
            self.clock,
            acoustic_vocab_size=self.vocab_size,
            forbidden_tokens=frozenset({self.vocab_size - 1}),
            epoch=self.epoch,
        )

    def _actuation(self, request_unit: int, packet: ScheduledSpeechPCM | None) -> ActuationSignal:
        if self.session_id is None:
            raise RuntimeError("speech executor has no active session")
        if packet is None:
            return ActuationSignal(self.session_id, request_unit, None, generation_epoch=self.epoch)
        return ActuationSignal(
            self.session_id,
            request_unit,
            SpeechSignal(packet.samples.numpy().astype("<f4").tobytes()),
            packet.unit_index + self.unit_offset,
            packet.scheduled_at_ms,
            packet.epoch,
        )

    def step(self, observation: ObservationSignal, codes: Tensor, *, elapsed_ms: float = 0.0):
        if observation.session_id != self.session_id or self.stream is None:
            raise ValueError("speech executor session identity mismatch")
        started = time.monotonic()
        self.stream.push(
            observation.unit_index - self.unit_offset,
            observation.timestamp_ms,
            codes.detach().cpu().reshape(-1),
            epoch=self.epoch,
        )
        now = observation.timestamp_ms + math.ceil(elapsed_ms + (time.monotonic() - started) * 1000)
        packet = self.stream.take(self.stream.next_output_unit, now_ms=now, epoch=self.epoch)
        if packet is not None and now > packet.scheduled_at_ms:
            self.stream.cancel()
            raise TimeoutError("continuous speech PCM became available after its playback deadline")
        return self._actuation(observation.unit_index, packet)

    def finish(self) -> list[ActuationSignal]:
        if self.stream is None or self.stream.next_input_unit == 0:
            return []
        if not self.stream.finish(epoch=self.epoch):
            return []
        outputs = []
        while self.stream.next_output_unit < self.stream.next_input_unit:
            index = self.stream.next_output_unit
            scheduled = self.stream._origin_ms + index * 200 + self.clock.playback_delay_ms
            packet = self.stream.take(index, now_ms=scheduled, epoch=self.epoch)
            if packet is None:
                raise RuntimeError("native speech end-of-session flush is incomplete")
            outputs.append(self._actuation(self.unit_offset + self.stream.next_input_unit, packet))
        return outputs

    def close(self) -> None:
        if self.stream is not None:
            self.stream.cancel()
        self.decoder.close()
