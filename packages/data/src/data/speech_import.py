from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from model.types import Episode, SpeechMode, StreamUnit
from runtime.config import DataConfig, ModelConfig


def _read_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    values, actual_rate = sf.read(Path(path).expanduser(), dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(f"{path} must be pre-resampled to {sample_rate} Hz")
    if values.shape[1] != 1:
        raise ValueError(f"{path} must be mono")
    waveform = torch.from_numpy(values[:, 0].copy())
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{path} contains a non-finite audio sample")
    if waveform.abs().max() > 1.0:
        raise ValueError(f"{path} contains audio outside [-1, 1]")
    return waveform


def _speech_activity(target: torch.Tensor, frame_samples: int) -> list[bool]:
    frames = target.reshape(-1, frame_samples)
    return [bool(active) for active in (frames.square().mean(dim=1) > 1e-5).tolist()]


def _segment_activity(segments: list[dict[str, Any]], ticks: int, frame_samples: int) -> list[bool]:
    active = [False] * ticks
    previous_end = -1
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"target_segments[{index}] must be an object")
        try:
            start = int(segment["start_sample"])
            end = int(segment["end_sample"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"target_segments[{index}] requires integer start_sample and end_sample"
            ) from error
        if not isinstance(segment.get("turn_id"), str) or not segment["turn_id"].strip():
            raise ValueError(f"target_segments[{index}].turn_id must be a non-empty string")
        if start < 0 or start % frame_samples:
            raise ValueError("assistant segment starts must be on an 80 ms boundary")
        if end <= start:
            raise ValueError("assistant segment end must be greater than its start")
        start_frame = start // frame_samples
        final_frame = (end - 1) // frame_samples
        if final_frame >= ticks:
            raise ValueError("assistant segment is outside the episode timeline")
        if start_frame <= previous_end:
            raise ValueError("assistant segments must not overlap")
        for frame in range(start_frame, final_frame + 1):
            active[frame] = True
        previous_end = final_frame
    return active


def _load_videos(
    path: str | None, model: ModelConfig, data: DataConfig
) -> dict[int, torch.Tensor]:
    if path is None:
        return {}
    loaded = np.load(Path(path).expanduser(), allow_pickle=False)
    try:
        ticks = loaded["ticks"]
        frames = loaded["frames"]
    finally:
        loaded.close()
    if len(ticks) != len(frames):
        raise ValueError("video ticks and frames must have identical lengths")
    expected = (3, data.video_height, data.video_width)
    result: dict[int, torch.Tensor] = {}
    for tick, frame in zip(ticks.tolist(), frames, strict=True):
        if tuple(frame.shape) != expected:
            raise ValueError(f"video frame must have shape {expected}")
        if int(tick) in result:
            raise ValueError("video tick is duplicated")
        result[int(tick)] = torch.from_numpy(frame.copy()).float()
    return result


def import_speech_manifest(
    path: str | Path, data: DataConfig, model: ModelConfig
) -> Iterator[Episode]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                yield _build_episode(record, manifest_path.parent, data, model)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid speech source manifest line {line_number}: {error}"
                ) from error


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _build_episode(
    record: dict[str, Any], base: Path, data: DataConfig, model: ModelConfig
) -> Episode:
    required = (
        "episode_id",
        "mic_audio",
        "target_speech",
        "source",
        "source_license",
        "redistribution_allowed",
        "language",
        "split",
        "session_id_hash",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for name in ("episode_id", "source", "source_license", "language", "split"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(record["redistribution_allowed"], bool):
        raise ValueError("redistribution_allowed must be a boolean")
    mic = _read_audio(_resolve(base, record["mic_audio"]), data.audio_sample_rate)
    target = _read_audio(_resolve(base, record["target_speech"]), data.audio_sample_rate)
    explicit_segments = record.get("target_segments")
    if explicit_segments is not None and not isinstance(explicit_segments, list):
        raise ValueError("target_segments must be a list")
    segment_timeline = 0
    if explicit_segments:
        try:
            segment_timeline = (
                max(int(segment["end_sample"]) for segment in explicit_segments)
                + data.unit_audio_samples
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("target_segments contain an invalid end_sample") from error
    samples = max(mic.numel(), target.numel(), segment_timeline)
    ticks = -(-samples // data.unit_audio_samples)
    timeline_samples = ticks * data.unit_audio_samples
    mic = torch.nn.functional.pad(mic, (0, timeline_samples - mic.numel()))
    target = torch.nn.functional.pad(target, (0, timeline_samples - target.numel()))
    active = (
        _segment_activity(explicit_segments, ticks, data.unit_audio_samples)
        if explicit_segments is not None
        else _speech_activity(target, data.unit_audio_samples)
    )
    video_path = record.get("videos")
    videos = _load_videos(str(_resolve(base, video_path)) if video_path else None, model, data)
    if any(tick < 0 or tick >= ticks for tick in videos):
        raise ValueError("video tick is outside the episode timeline")
    units: list[StreamUnit] = []
    if any(key == "actions" or key.startswith("action_") for key in record):
        raise ValueError("legacy action labels are not supported by AVM")
    if 0 not in videos:
        raise ValueError("AVM episode requires an initial video frame")
    current_video = videos[0]
    for tick in range(ticks):
        speaking = active[tick]
        current_video = videos.get(tick, current_video)
        units.append(
            StreamUnit(
                timestamp_ms=torch.tensor([tick * data.unit_ms]),
                delta_ms=torch.tensor([data.unit_ms]),
                mic_audio=mic[
                    None, tick * data.unit_audio_samples : (tick + 1) * data.unit_audio_samples
                ],
                video=current_video[None],
                speech_mode=torch.tensor(
                    [int(SpeechMode.SPEECH if speaking else SpeechMode.SILENCE)]
                ),
                speech_mode_mask=torch.tensor([True]),
                speech_codes=torch.zeros(
                    1, model.speech_frames_per_unit, model.speech_codebooks, dtype=torch.long
                ),
                speech_codec_mask=torch.full(
                    (1, model.speech_frames_per_unit), speaking, dtype=torch.bool
                ),
            )
        )
    metadata = {
        **record,
        "sample_rate": data.audio_sample_rate,
        "unit_ms": data.unit_ms,
        "codec_frame_rate": data.codec_frame_rate,
        "codec_id": data.codec_id,
        "codec_weight_hash": data.codec_weight_hash,
        "codec_revision": data.codec_revision,
        "speech_codes_encoded": False,
        "scenario": record.get("scenario", "spoken-response"),
        "device_id_hash": record.get("device_id_hash", "speech-import"),
        "turns": record.get("turns", []),
        "stage": record.get("stage", "pretrain"),
        "schema_id": "avm-episode-v1",
        "dataset_scale": data.dataset,
        "sample_kind": record.get("sample_kind", "supervised_episode"),
        "supervision_kind": record.get("supervision_kind", "speech"),
        "task_id": record.get("task_id", record["episode_id"]),
        "environment_id": record.get("environment_id", "recorded"),
        "environment_version": record.get("environment_version", "1"),
        "protocol_version": record.get("protocol_version", "avm-realtime-v1"),
    }
    episode = Episode(str(record["episode_id"]), units, metadata, target_speech=target)
    episode.validate(
        audio_samples=data.unit_audio_samples,
        speech_frames=model.speech_frames_per_unit,
        speech_codebooks=model.speech_codebooks,
        speech_codebook_size=model.speech_codebook_size,
    )
    return episode
