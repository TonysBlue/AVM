from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from runtime.config import ProjectConfig
from runtime.native_acoustics import NativeAcousticBackend
from runtime.native_assets import audit_native_assets

from data.codec_targets import encode_target_speech
from data.curation.audit import audit_canary_data
from data.curation.common import (
    SPLITS,
    dataset_path,
    ensure_tree,
    read_json,
    read_jsonl,
    registry_root,
    sha256_file,
    write_json,
)
from data.curation.fetch import fetch_canary_data
from data.curation.manifest import build_canary_manifest, build_source_inventory
from data.curation.synthesis import synthesize_canary
from data.curation.text import build_canary_text
from data.curation.voices import select_canary_voices
from data.speech_import import import_speech_manifest
from data.webdataset import EpisodeShardReader, write_episode_shards


def codec_client(
    config: ProjectConfig, source_lock: str | Path | None = None
) -> NativeAcousticBackend:
    if config.data.dataset == "synthetic" and config.runtime.native_acoustic_socket:
        from runtime.native_acoustics import build_native_decoder

        return build_native_decoder(config)
    selected = source_lock or config.model.native_source_lock
    if not selected:
        raise ValueError("native target encoding requires a locked source asset")
    audit = audit_native_assets(selected, model_weights=False)
    if audit["codec_weight_sha256"] != config.data.codec_weight_hash:
        raise ValueError("native tokenizer weight differs from the selected data identity")
    return NativeAcousticBackend(audit, cuda_device=config.runtime.acoustic_cuda_device)


def check_native_decode(
    root: str | Path,
    *,
    dataset: str,
    client: NativeAcousticBackend,
    fixture: bool = False,
    minimum_segments: int = 100,
) -> dict[str, Any]:
    """Check whole native target timelines; chunks never reset at speech boundaries."""
    from runtime.continuous_speech import ContinuousSpeechClock

    from data.curation.native_speech import _evaluate_clip

    root = Path(root).expanduser().resolve()
    manifest_path = dataset_path(root, dataset, "manifests", "episodes.jsonl")
    clock = ContinuousSpeechClock(
        lookahead_tokens=client.lookahead_tokens,
        vocoder_tail_samples=client.tail_samples,
        playback_delay_ms=600,
    )
    checked = failed = 0
    for record in read_jsonl(manifest_path):
        target_path = Path(record["target_speech"])
        if not target_path.is_absolute():
            target_path = root / target_path
        values, rate = sf.read(target_path, dtype="float32", always_2d=True)
        segments = max(len(record.get("target_segments", [])), 1)
        try:
            if rate != 16000 or values.shape[1] != 1 or values.shape[0] % 3200:
                raise ValueError("native decode audit requires full 16 kHz / 200 ms timelines")
            waveform = torch.from_numpy(values[:, 0].copy())
            measurement, _ = _evaluate_clip(
                client, clock, waveform, kind="speech", silence_intervals=[]
            )
            if measurement["maximum_block_boundary_jump"] > 0.15:
                raise ValueError("native decoded timeline exceeds the boundary discontinuity gate")
            checked += segments
        except (RuntimeError, ValueError, OSError):
            failed += segments
    if not fixture and checked < minimum_segments:
        raise ValueError(
            f"Native decode-check requires at least {minimum_segments} successful "
            f"target segments; got {checked}"
        )
    report = {
        "dataset": dataset,
        "manifest_sha256": sha256_file(manifest_path),
        "checked_segments": checked,
        "failed_segments": failed,
        "native_weight_sha256": client.identity["codec_weight_sha256"],
        "codec_id": client.identity["codec_id"],
        "codec_revision": client.identity["codec_revision"],
        "fixture": fixture,
        "native_source_lock_sha256": client.identity["source_lock_sha256"],
        "target_encoding_context": client.identity["target_encoding_context"],
    }
    path = dataset_path(root, dataset, "reports", "native-decode.json")
    write_json(path, report)
    return {**report, "path": str(path)}


def encode_canary_shards(
    root: str | Path,
    *,
    config: ProjectConfig,
    client: NativeAcousticBackend,
) -> dict[str, Any]:
    dataset = "canary"
    root = Path(root).expanduser().resolve()
    results: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = dataset_path(root, dataset, "manifests", f"{split}.jsonl")
        staging = dataset_path(root, dataset, "shards", "staging", split, f"{split}-%06d.tar")
        processed = dataset_path(root, dataset, "shards", "processed", split, f"{split}-%06d.tar")
        write_episode_shards(import_speech_manifest(manifest, config.data, config.model), staging)
        staged = EpisodeShardReader(
            str(staging).replace("%06d", "*"),
            config.data,
            config.model,
            require_encoded_speech=False,
            validate_manifest=False,
        )
        write_episode_shards(encode_target_speech(staged, client), processed)
        encoded = EpisodeShardReader(
            str(processed).replace("%06d", "*"),
            config.data,
            config.model,
            require_encoded_speech=True,
            validate_manifest=False,
        )
        episodes = units = 0
        for episode in encoded:
            episodes += 1
            units += len(episode.units)
        results.append(
            {
                "split": split,
                "episodes": episodes,
                "units": units,
                "staging": str(staging),
                "processed": str(processed),
            }
        )
    report = {"dataset": dataset, "splits": results, "codec_id": config.data.codec_id}
    write_json(dataset_path(root, dataset, "reports", "encoded.json"), report)
    return report


def prepare_canary_data(
    root: str | Path,
    *,
    config: ProjectConfig,
    fixture: bool = False,
    lock_path: str | Path | None = None,
    download: bool = False,
    extract: bool = False,
    library: str | Path | None = None,
    synth_command: str | None = None,
    asr_command: str | None = None,
    model_sha256: str | None = None,
    normalize_command: str | None = None,
    video_command: str | None = None,
    native_source_lock: str | Path | None = None,
    encode: bool = False,
    native_report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all deterministic Canary preparation stages in dependency order."""
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    fetch = fetch_canary_data(
        root, fixture=fixture, lock_path=lock_path, download=download, extract=extract
    )
    inventory = None
    if not fixture:
        if not normalize_command:
            raise ValueError("formal Canary preparation requires --normalize-command")
        inventory = build_source_inventory(normalize_command, root)
    voices = select_canary_voices(root, library=library, fixture=fixture)
    datasets: dict[str, Any] = {}
    client = codec_client(config, native_source_lock) if (encode or native_source_lock) else None
    if encode and client is None:
        raise ValueError("--encode requires locked native model assets")
    try:
        dataset_names = ("canary",)
        for dataset_name in dataset_names:
            text = build_canary_text(root, fixture=fixture, seed=config.data.seed)
            synthesis = synthesize_canary(
                root,
                fixture=fixture,
                synth_command=synth_command,
                asr_command=asr_command,
                model_sha256=model_sha256,
            )
            manifest = build_canary_manifest(
                root,
                fixture=fixture,
                normalize_command=normalize_command,
                video_command=video_command,
            )
            native_path: str | None = None
            if client:
                native = check_native_decode(
                    root, dataset=dataset_name, client=client, fixture=fixture
                )
                native_path = str(native["path"])
            else:
                report_dir = (
                    Path(native_report_dir).expanduser()
                    if native_report_dir
                    else dataset_path(root, dataset_name, "reports")
                )
                candidate = report_dir / "native-decode.json"
                manifest_path = dataset_path(root, dataset_name, "manifests", "episodes.jsonl")
                current_manifest_hash = sha256_file(manifest_path)
                provided = False
                if candidate.is_file():
                    provided = read_json(candidate).get("manifest_sha256") == current_manifest_hash
                native = {"path": str(candidate), "provided": provided}
                native_path = str(candidate) if provided else None
            audit = (
                audit_canary_data(
                    root,
                    fixture=fixture,
                    native_report=native_path,
                )
                if fixture or native_path
                else None
            )
            encoded = encode_canary_shards(root, config=config, client=client) if encode else None
            datasets[dataset_name] = {
                "text": text,
                "synthesis": synthesis,
                "manifest": manifest,
                "native": native,
                "audit": audit,
                "encoded": encoded,
            }
        result = {
            "root": str(root),
            "fetch": fetch,
            "voices": voices,
            "inventory": inventory,
            "datasets": datasets,
        }
        write_json(registry_root(root) / "prepare.json", result)
        return result
    finally:
        if client is not None:
            client.close()
