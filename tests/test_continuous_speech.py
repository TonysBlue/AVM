from __future__ import annotations

import pytest
import torch
from runtime.continuous_speech import ContinuousSpeechClock, ContinuousSpeechStream


class ProtocolTestDecoder:
    """In-process protocol fixture, never a native qualification backend."""

    def __init__(self, lookahead: int, tail: int) -> None:
        self.lookahead = lookahead
        self.tail = tail
        self.pending = torch.empty(0)
        self.calls: list[tuple[list[int], bool]] = []

    def decode(self, tokens: list[int], *, final: bool) -> torch.Tensor:
        self.calls.append((tokens.copy(), final))
        actual = tokens if final or not self.lookahead else tokens[:-self.lookahead]
        new = torch.tensor(actual, dtype=torch.float32).repeat_interleave(960) / 100
        all_samples = torch.cat((self.pending, new))
        if final:
            self.pending = torch.empty(0)
            return all_samples
        count = max(all_samples.numel() - self.tail, 0)
        result = all_samples[:count].clone()
        self.pending = all_samples[count:].clone()
        return result


def make_stream(lookahead: int = 3, tail: int = 3840):
    clock = ContinuousSpeechClock(lookahead, tail, 2000)
    clock = ContinuousSpeechClock(lookahead, tail, clock.minimum_delay_ms)
    decoder = ProtocolTestDecoder(lookahead, tail)
    return ContinuousSpeechStream(
        decoder, clock, acoustic_vocab_size=100,
        forbidden_tokens=frozenset({99}), epoch=7,
    ), decoder


@pytest.mark.parametrize("lookahead", [0, 3, 5, 6, 11])
@pytest.mark.parametrize("tail", [0, 3840])
def test_continuous_tokens_lookahead_tail_and_fixed_playback(lookahead, tail) -> None:
    stream, decoder = make_stream(lookahead, tail)
    original = torch.arange(40, dtype=torch.long)
    emitted = []
    for unit in range(8):
        stream.push(unit, unit * 200, original[unit * 5 : unit * 5 + 5], epoch=7)
        while stream.next_output_unit < stream.next_input_unit:
            item = stream.take(stream.next_output_unit, now_ms=unit * 200, epoch=7)
            if item is None:
                break
            assert item.scheduled_at_ms == item.unit_index * 200 + stream.clock.playback_delay_ms
            emitted.append(item.samples)
        assert stream.buffered_tokens < 5 + lookahead
    stream.finish(epoch=7)
    while stream.next_output_unit < 8:
        emitted.append(stream.take(stream.next_output_unit, now_ms=4000, epoch=7).samples)
    expected = original.float().repeat_interleave(960) / 100
    assert torch.equal(torch.cat(emitted), expected)
    assert stream.buffered_samples == stream.buffered_tokens == 0
    assert decoder.calls[-1][1]
    if lookahead:
        assert decoder.calls[0][0] == list(range(5 + lookahead))
        assert decoder.calls[1][0][0] == 5
    assert not stream.finish(epoch=7)
    with pytest.raises(ValueError, match="after session-end"):
        stream.push(8, 1600, torch.zeros(5, dtype=torch.long), epoch=7)


def test_first_small_block_retains_real_vocoder_tail_without_padding() -> None:
    stream, _ = make_stream()
    stream.push(0, 0, torch.zeros(5, dtype=torch.long), epoch=7)
    assert stream.buffered_samples == 0
    stream.push(1, 200, torch.zeros(5, dtype=torch.long), epoch=7)
    assert stream.buffered_samples == 960
    assert stream.samples_decoded == 960
    assert stream.clock.minimum_delay_ms == 600


def test_deadline_failure_does_not_manufacture_silence() -> None:
    stream, _ = make_stream()
    stream.push(0, 0, torch.zeros(5, dtype=torch.long), epoch=7)
    with pytest.raises(TimeoutError, match="deadline"):
        stream.take(0, now_ms=stream.clock.playback_delay_ms, epoch=7)
    assert stream.next_output_unit == 0


@pytest.mark.parametrize("late", [False, True])
def test_physical_executor_delivers_early_and_rejects_late_pcm(late) -> None:
    from contracts import MicSignal, ObservationSignal, VideoSignal
    from runtime.codec import CodecIdentity
    from runtime.continuous_speech import ContinuousSpeechExecutor

    from tests.support.physical_workers import ProtocolAcousticDecoder

    decoder = ProtocolAcousticDecoder(CodecIdentity("test", "hash", "rev", codebook_size=32))
    executor = ContinuousSpeechExecutor(decoder, ContinuousSpeechClock(3, 3840, 600), vocab_size=32)
    executor.reset("test")
    codes = torch.ones(5, dtype=torch.long)
    for index in range(3):
        observation = ObservationSignal(
            "test", index, index * 200, 200,
            MicSignal(torch.zeros(3200).numpy().tobytes()),
            VideoSignal(b"rgb", 1, 1, index * 200, index),
        )
        if index == 2 and late:
            with pytest.raises(TimeoutError, match="after its playback deadline"):
                executor.step(observation, codes, elapsed_ms=500)
        else:
            output = executor.step(observation, codes)
            if index < 2:
                assert output.speech is None
            else:
                assert output.speech_unit_index == 0 and output.play_at_ms == 600
                assert len(output.speech.pcm) == 4800 * 4


def test_cancel_rejects_late_epoch_results() -> None:
    stream, _ = make_stream()
    assert not stream.push(0, 0, torch.zeros(5, dtype=torch.long), epoch=6)
    stream.push(0, 0, torch.zeros(5, dtype=torch.long), epoch=7)
    stream.cancel()
    assert not stream.push(1, 200, torch.zeros(5, dtype=torch.long), epoch=7)
    assert stream.take(0, now_ms=2000, epoch=7) is None
    assert stream.buffered_samples == stream.buffered_tokens == 0


@pytest.mark.parametrize("codes", [torch.zeros(4, dtype=torch.long),
                                   torch.zeros(5), torch.full((5,), 99, dtype=torch.long)])
def test_invalid_acoustic_unit_has_no_state_effect(codes) -> None:
    stream, decoder = make_stream()
    with pytest.raises(ValueError):
        stream.push(0, 0, codes, epoch=7)
    assert stream.next_input_unit == stream.buffered_tokens == 0
    assert not decoder.calls


def test_input_order_and_clock_are_checked() -> None:
    stream, _ = make_stream()
    with pytest.raises(ValueError, match="order"):
        stream.push(1, 200, torch.zeros(5, dtype=torch.long), epoch=7)
    stream.push(0, 1000, torch.zeros(5, dtype=torch.long), epoch=7)
    with pytest.raises(ValueError, match="fixed clock"):
        stream.push(1, 1300, torch.zeros(5, dtype=torch.long), epoch=7)


def test_bad_native_sample_count_poisons_stream() -> None:
    stream, decoder = make_stream()
    decoder.decode = lambda tokens, final: torch.zeros(5000)
    stream.push(0, 0, torch.zeros(5, dtype=torch.long), epoch=7)
    with pytest.raises(ValueError, match="sample count"):
        stream.push(1, 200, torch.zeros(5, dtype=torch.long), epoch=7)
    with pytest.raises(RuntimeError, match="new decoder epoch"):
        stream.push(1, 200, torch.zeros(5, dtype=torch.long), epoch=7)


def test_unconsumed_pcm_cannot_grow_without_bound() -> None:
    stream, _ = make_stream()
    with pytest.raises(BufferError, match="queue is full"):
        for unit in range(30):
            stream.push(unit, unit * 200, torch.zeros(5, dtype=torch.long), epoch=7)


def test_clock_rejects_unbudgeted_lookahead_and_tail() -> None:
    with pytest.raises(ValueError, match="lookahead"):
        ContinuousSpeechClock(3, 3840, 400)
