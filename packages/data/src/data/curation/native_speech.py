"""Real-asset qualification of five-token native continuous speech decoding."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from runtime.continuous_speech import ContinuousSpeechClock, ContinuousSpeechStream
from runtime.native_acoustics import NativeAcousticBackend
from torch import Tensor

from data.curation.native_assets import audit_native_assets, locked_path, sha256_file


def _rms(values: Tensor) -> float:
    return float(values.double().square().mean().sqrt().item()) if values.numel() else 0.0


def _evaluate_clip(
    backend: Any,
    clock: ContinuousSpeechClock,
    waveform: Tensor,
    *,
    kind: str,
    silence_intervals: list[list[int]],
) -> tuple[dict[str, Any], Tensor]:
    backend.reset()
    backend.synchronize()
    started = time.perf_counter()
    codes = backend.encode(waveform)
    backend.synchronize()
    encode_ms = (time.perf_counter() - started) * 1000
    if codes.ndim != 1 or codes.numel() % 5 or codes.numel() * 640 != waveform.numel():
        raise ValueError("qualification corpus must encode to exactly five tokens per 200 ms")
    stream = ContinuousSpeechStream(
        backend, clock, acoustic_vocab_size=backend.vocab_size,
        forbidden_tokens=frozenset({backend.vocab_size - 1}), epoch=0,
    )
    returned: list[Tensor] = []
    times = []
    units = codes.numel() // 5
    for index in range(units):
        backend.synchronize()
        started = time.perf_counter()
        stream.push(index, index * 200, codes[index * 5 : (index + 1) * 5], epoch=0)
        backend.synchronize()
        times.append((time.perf_counter() - started) * 1000)
        while stream.next_output_unit < units:
            item = stream.take(stream.next_output_unit, now_ms=index * 200, epoch=0)
            if item is None:
                break
            returned.append(item.samples)
    backend.synchronize()
    started = time.perf_counter()
    stream.finish(epoch=0)
    backend.synchronize()
    flush_ms = (time.perf_counter() - started) * 1000
    while stream.next_output_unit < units:
        item = stream.take(
            stream.next_output_unit,
            now_ms=units * 200 + clock.playback_delay_ms,
            epoch=0,
        )
        if item is None:
            raise RuntimeError("native end-of-session output is incomplete")
        returned.append(item.samples)
    decoded = torch.cat(returned)
    if stream.buffered_samples or stream.buffered_tokens:
        raise ValueError("native final flush left unassigned PCM or acoustic tokens")
    if decoded.numel() * 2 != waveform.numel() * 3:
        raise ValueError("native decoded duration is not equal to the 16 kHz input duration")
    intervals = []
    for start_ms, end_ms in silence_intervals:
        intervals.append(_rms(decoded[start_ms * 24 : end_ms * 24]))
    edges = torch.arange(4800, decoded.numel(), 4800)
    boundary_jump = float((decoded[edges] - decoded[edges - 1]).abs().max().item())
    sorted_times = sorted(times)
    return {
        "kind": kind, "units": units, "token_count": codes.numel(),
        "samples": decoded.numel(), "duration_ms": units * 200,
        "encode_ms": encode_ms, "decode_ms": sum(times), "flush_ms": flush_ms,
        "decode_p95_ms": sorted_times[math.ceil(len(times) * 0.95) - 1],
        "decode_max_ms": max(times),
        "real_time_factor": (sum(times) + flush_ms) / (units * 200),
        "output_rms": _rms(decoded), "silence_interval_rms": intervals,
        "maximum_block_boundary_jump": boundary_jump,
    }, decoded


def qualify_native_speech(
    lock_path: str | Path,
    corpus_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Only the real CUDA backend can produce this acoustic qualification report."""
    audit = audit_native_assets(lock_path, model_weights=False)
    corpus_path = Path(corpus_path).expanduser().resolve()
    with corpus_path.open(encoding="utf-8") as source:
        corpus = json.load(source)
    if not isinstance(corpus, dict) or corpus.get("schema_id") != "avm-native-speech-corpus-v1":
        raise ValueError("native speech qualification requires a locked corpus manifest")
    entries = corpus.get("clips")
    if (not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries)
            or {entry.get("kind") for entry in entries} != {"speech", "silence", "transition"}):
        raise ValueError("qualification requires speech, long silence and transition recordings")
    thresholds = corpus.get("thresholds")
    names = ("silence_rms_max", "boundary_jump_max", "decode_p95_ms_max", "rtf_max")
    if not isinstance(thresholds, dict) or any(
        type(thresholds.get(name)) not in (float, int)
        or not math.isfinite(thresholds[name]) or thresholds[name] <= 0
        for name in names
    ):
        raise ValueError("corpus must declare finite positive acoustic acceptance thresholds")
    if thresholds["decode_p95_ms_max"] > 200 or thresholds["rtf_max"] >= 1:
        raise ValueError("acoustic qualification must sustain the 200 ms clock")
    clips = []
    for entry in entries:
        if not isinstance(entry.get("path"), str):
            raise ValueError("qualification recording path is required")
        path = locked_path(corpus_path.parent, entry["path"])
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"qualification audio hash mismatch: {path.name}")
        if not entry.get("license"):
            raise ValueError("qualification recording license is required")
        values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != 16_000 or values.shape[1] != 1 or values.shape[0] % 3200:
            raise ValueError("qualification recordings must be 16 kHz mono, full 200 ms units")
        waveform = torch.from_numpy(values[:, 0].copy())
        if not torch.isfinite(waveform).all() or torch.any(waveform.abs() > 1):
            raise ValueError("qualification input PCM must be finite and normalized")
        minimum_ms = {"speech": 2000, "silence": 10_000, "transition": 4000}[entry["kind"]]
        duration_ms = waveform.numel() // 16
        if duration_ms < minimum_ms:
            raise ValueError(f"qualification {entry['kind']} recording is too short")
        spans = entry.get("silence_intervals_ms", [])
        if entry["kind"] == "silence":
            spans = [[0, duration_ms]]
        if not isinstance(spans, list) or any(
            not isinstance(span, list) or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= duration_ms
            for span in spans
        ):
            raise ValueError("invalid qualification silence interval")
        if entry["kind"] == "transition" and not spans:
            raise ValueError("transition recording must annotate its silence intervals")
        for start_ms, end_ms in spans:
            if _rms(waveform[start_ms * 16 : end_ms * 16]) > thresholds["silence_rms_max"]:
                raise ValueError("annotated silence is not quiet in the source recording")
        clips.append((entry, waveform, spans))
    output = Path(output_root).expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[5]):
        raise ValueError("qualification audio/reports must be outside the Git repository")
    with NativeAcousticBackend(audit) as backend:
        minimum_clock = ContinuousSpeechClock(
            lookahead_tokens=backend.lookahead_tokens,
            vocoder_tail_samples=backend.tail_samples,
            playback_delay_ms=60_000,
        )
        clock = ContinuousSpeechClock(
            lookahead_tokens=backend.lookahead_tokens,
            vocoder_tail_samples=backend.tail_samples,
            playback_delay_ms=minimum_clock.minimum_delay_ms,
        )
        output.mkdir(parents=True, exist_ok=True)
        reports = []
        failure = None
        for index, (entry, waveform, spans) in enumerate(clips):
            try:
                report, decoded = _evaluate_clip(
                    backend, clock, waveform, kind=entry["kind"], silence_intervals=spans
                )
                if (entry["kind"] != "silence"
                        and report["output_rms"] <= thresholds["silence_rms_max"]):
                    raise ValueError("native speech qualification decoded speech as silence")
            except (ValueError, RuntimeError, FloatingPointError) as error:
                failure = f"{type(error).__name__}: {error}"
                reports.append({"kind": entry["kind"], "passed": False,
                                "acoustic_passed": False, "timing_passed": False,
                                "input_sha256": entry["sha256"], "error": failure})
                continue
            report["acoustic_passed"] = (
                report["maximum_block_boundary_jump"] <= thresholds["boundary_jump_max"]
                and all(value <= thresholds["silence_rms_max"]
                        for value in report["silence_interval_rms"])
            )
            report["timing_passed"] = (report["decode_p95_ms"] <= thresholds["decode_p95_ms_max"]
                                       and report["real_time_factor"] <= thresholds["rtf_max"])
            report["passed"] = report["acoustic_passed"] and report["timing_passed"]
            audio_path = output / f"{index:03d}-{entry['kind']}.wav"
            sf.write(audio_path, decoded.numpy(), 24_000, subtype="FLOAT")
            report.update({"input_sha256": entry["sha256"],
                           "output_sha256": sha256_file(audio_path)})
            reports.append(report)
        result = {
            "schema_id": "avm-native-speech-qualification-v2",
            "passed": all(report["passed"] for report in reports),
            "source_lock_sha256": audit["source_lock_sha256"],
            "corpus_sha256": sha256_file(corpus_path), "lookahead_tokens": backend.lookahead_tokens,
            "vocoder_tail_samples": backend.tail_samples,
            "playback_delay_ms": clock.playback_delay_ms,
            "thresholds": thresholds, "clips": reports,
            "runtime": backend.identity,
            "error": failure,
            "acoustic_passed": (failure is None
                                and all(report["acoustic_passed"] for report in reports)),
            "timing_passed": (failure is None
                              and all(report["timing_passed"] for report in reports)),
            "scope": "real native tokenizer and five-token Token2Wav; not an AVM end-to-end gate",
        }
        report_path = output / "qualification.json"
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
