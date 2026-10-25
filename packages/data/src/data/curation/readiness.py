from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import Any

import torch
from runtime.config import ProjectConfig
from runtime.native_assets import audit_native_assets, require_native_qualification

from data.curation.common import SPLITS, dataset_path, read_json, read_jsonl, write_json


def _required(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(f"{label}: {path}")


def _resolve_config_path(config: ProjectConfig, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.runtime.data_path() / path


def _path_is_dataset_member(path: Path, dataset_root: Path) -> bool:
    try:
        path.resolve().relative_to(dataset_root.resolve())
        return True
    except ValueError:
        return False


def check_readiness(
    root: str | Path,
    *,
    config: ProjectConfig,
    dataset: str | None = None,
    require_checkpoint: str | Path | None = None,
    require_encoded: bool = True,
) -> dict[str, Any]:
    """Fail-closed machine check before any real-data training run."""
    dataset = dataset or config.data.dataset
    if dataset not in {"canary", "direct-speech-overfit"}:
        raise ValueError(f"readiness is not required for dataset={dataset!r}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"training readiness failed: data root: {root}")
    dataset_root = (
        root / "gates" / "direct-speech-overfit"
        if dataset == "direct-speech-overfit"
        else dataset_path(root, dataset)
    )
    missing: list[str] = []
    invalid: list[str] = []
    formal_dataset = dataset == "canary"
    native_audit = None
    native_qualification = None
    if formal_dataset:
        lock = (
            Path(config.model.native_source_lock).expanduser()
            if config.model.native_source_lock
            else None
        )
        qualification = (
            Path(config.model.native_qualification).expanduser()
            if config.model.native_qualification
            else None
        )
        if lock is None or not lock.is_file():
            missing.append("locked native MiniCPM model assets")
        if qualification is None or not qualification.is_file():
            missing.append("passed native acoustic/timing qualification")
        if (
            lock is not None
            and lock.is_file()
            and qualification is not None
            and qualification.is_file()
        ):
            try:
                native_audit = audit_native_assets(lock, model_weights=True)
                native_qualification = require_native_qualification(native_audit, qualification)
            except (ValueError, OSError, KeyError) as error:
                invalid.append(f"native assets/qualification: {error}")

    if formal_dataset and config.training.stage == "rl":
        rl = config.training.rl
        for label, value in (
            ("dialogue Harness", rl.environment_socket),
            ("perceptual Judge", rl.reward_socket),
        ):
            endpoint = Path(value).expanduser() if value else None
            if endpoint is None or not endpoint.is_socket():
                missing.append(f"{label} socket: {value or 'unconfigured'}")
        session_manifest = Path(rl.session_manifest).expanduser() if rl.session_manifest else None
        if session_manifest is None or not session_manifest.is_file():
            missing.append("AVM dialogue lifetime session manifest")
        else:
            sessions = read_jsonl(session_manifest)
            if len(sessions) != 1 or not all(
                isinstance(sessions[0].get(key), str) and sessions[0][key].strip()
                for key in ("session_id", "scenario_id")
            ):
                invalid.append("AVM requires one identified dialogue lifetime session")
        guard_assets = (
            ("SFT replay manifest", config.training.rl.sft_replay_manifest, False),
            ("SFT replay shards", config.training.rl.sft_replay_shards, True),
            (
                "SFT preservation manifest",
                config.training.rl.sft_preservation_manifest,
                False,
            ),
            (
                "SFT preservation shards",
                config.training.rl.sft_preservation_shards,
                True,
            ),
        )
        for label, value, is_pattern in guard_assets:
            resolved = _resolve_config_path(config, value)
            if resolved is None:
                missing.append(f"{label}: unconfigured")
            elif is_pattern and not glob.glob(str(resolved)):
                missing.append(f"{label}: {resolved}")
            elif not is_pattern:
                _required(resolved, label, missing)

    def path(*parts: str) -> Path:
        return dataset_root.joinpath(*parts)

    audit_path = path("reports", "audit.json")
    if formal_dataset:
        _required(audit_path, "audit report", missing)
        if audit_path.is_file() and not read_json(audit_path).get("passed"):
            invalid.append(f"audit report is not passed: {audit_path}")
    manifest_paths: dict[str, Path] = {}
    shard_paths: dict[str, list[Path]] = {}
    splits = ("train",) if dataset == "direct-speech-overfit" else SPLITS
    for split in splits:
        manifest = path("manifests", f"{split}.jsonl")
        if dataset == "direct-speech-overfit" and not manifest.is_file():
            # The deterministic gate has no separate source manifest; its
            # processed manifest is the immutable source of episode identity.
            manifest = path("shards", "processed", split, f"{split}-manifest.jsonl")
        manifest_paths[split] = manifest
        _required(manifest, f"{split} manifest", missing)
        processed = path("shards", "processed", split)
        shards = sorted(processed.glob(f"{split}-*.tar"))
        shard_paths[split] = shards
        if split == "train":
            configured_manifest = _resolve_config_path(config, config.data.manifest)
            configured_shards = _resolve_config_path(config, config.data.shards)
            expected_processed_manifest = processed / "train-manifest.jsonl"
            if configured_manifest and not _path_is_dataset_member(
                configured_manifest, dataset_root
            ):
                invalid.append(
                    f"config data.manifest is outside {dataset} dataset root: {configured_manifest}"
                )
            if configured_manifest and configured_manifest.name != "train-manifest.jsonl":
                invalid.append(
                    "config data.manifest must be the processed train manifest: "
                    f"{configured_manifest}"
                )
            elif configured_manifest and (
                configured_manifest.resolve() != expected_processed_manifest.resolve()
            ):
                invalid.append(
                    "config data.manifest does not point to the processed train manifest: "
                    f"{configured_manifest}"
                )
            if configured_shards and not _path_is_dataset_member(configured_shards, dataset_root):
                invalid.append(
                    f"config data.shards is outside {dataset} dataset root: {configured_shards}"
                )
            expected_shard_root = dataset_root / "shards" / "processed" / "train"
            if configured_shards and not _path_is_dataset_member(
                configured_shards, expected_shard_root
            ):
                invalid.append(
                    f"config data.shards must target processed/train shards: {configured_shards}"
                )
        if require_encoded:
            if not shards:
                missing.append(f"{split} processed shards: {processed / (split + '-*.tar')}")
            _required(
                processed / f"{split}-manifest.jsonl",
                f"{split} processed manifest",
                missing,
            )
    if require_encoded:
        for split in splits:
            if not manifest_paths[split].is_file():
                continue
            source_entries = read_jsonl(manifest_paths[split])
            processed_manifest = path("shards", "processed", split, f"{split}-manifest.jsonl")
            if processed_manifest.is_file():
                encoded_entries = read_jsonl(processed_manifest)
                required_metadata = {
                    "protocol_version",
                    "schema_id",
                    "runtime_identity",
                    "environment_id",
                    "environment_version",
                }
                if any(required_metadata - set(entry) for entry in encoded_entries):
                    invalid.append(f"{split} processed manifest is missing runtime identity")
                if any(
                    entry.get("schema_id") != "avm-episode-v2"
                    or entry.get("protocol_version") != "avm-realtime-v2"
                    for entry in encoded_entries
                ):
                    invalid.append(f"{split} processed manifest has incompatible AVM identity")
                if {entry["episode_id"] for entry in source_entries} != {
                    entry["episode_id"] for entry in encoded_entries
                }:
                    invalid.append(f"{split} processed manifest does not match source manifest")
                if any(not entry.get("speech_codes_encoded") for entry in encoded_entries):
                    invalid.append(f"{split} processed manifest contains unencoded speech")
                if formal_dataset:
                    if any(
                        not entry.get("native_source_lock_sha256")
                        or not entry.get("target_encoding_context")
                        for entry in encoded_entries
                    ):
                        invalid.append(f"{split} processed manifest lacks native encoding identity")
                    if native_audit is not None and any(
                        entry.get("native_source_lock_sha256") != native_audit["source_lock_sha256"]
                        or entry.get("target_encoding_context")
                        != native_audit["target_encoding_context"]
                        for entry in encoded_entries
                    ):
                        invalid.append(
                            f"{split} target encoding identity differs from native assets"
                        )
    native_reports = {}
    if formal_dataset:
        report = path("reports", "native-decode.json")
        _required(report, f"{dataset} Native decode report", missing)
        if report.is_file():
            native = read_json(report)
            native_reports[dataset] = native
            source_manifest = path("manifests", "episodes.jsonl")
            from data.curation.common import sha256_file

            if native.get("manifest_sha256") != (
                sha256_file(source_manifest) if source_manifest.is_file() else None
            ):
                invalid.append(
                    f"{dataset} Native report manifest hash does not match source manifest"
                )
            if native.get("failed_segments") != 0:
                invalid.append(f"{dataset} Native report contains failed segments")
            for field, expected in (
                ("codec_id", config.data.codec_id),
                ("codec_revision", config.data.codec_revision),
                ("native_weight_sha256", config.data.codec_weight_hash),
            ):
                if native.get(field) != expected:
                    invalid.append(f"{dataset} Native report {field} differs from config")
    if require_checkpoint:
        checkpoint = Path(require_checkpoint).expanduser()
        _required(checkpoint, "initial checkpoint", missing)
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {})
                if metadata.get("model_schema_id") != "avm-model-v3":
                    invalid.append("initial checkpoint is not an AVM model")
                if metadata.get("codec_id") != config.data.codec_id:
                    invalid.append("initial checkpoint codec_id differs from Canary config")
                if metadata.get("codec_revision") != config.data.codec_revision:
                    invalid.append("initial checkpoint codec_revision differs from Canary config")
                if metadata.get("codec_weight_hash") != config.data.codec_weight_hash:
                    invalid.append(
                        "initial checkpoint codec weight hash differs from Canary config"
                    )
                state = payload.get("model", {})
                if not isinstance(state, dict) or not state:
                    invalid.append("initial checkpoint has no model state")
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                invalid.append(f"initial checkpoint cannot be inspected: {error}")
    disk = shutil.disk_usage(root)
    result = {
        "dataset": dataset,
        "root": str(root),
        "passed": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "manifests": {split: str(path) for split, path in manifest_paths.items()},
        "shards": {split: [str(path) for path in paths] for split, paths in shard_paths.items()},
        "native_reports": native_reports,
        "native_assets": native_audit,
        "native_qualification": native_qualification,
        "free_disk_bytes": disk.free,
    }
    write_json(path("reports", "readiness.json"), result)
    if not result["passed"]:
        problems = "; ".join(missing + invalid)
        raise ValueError(f"training readiness failed: {problems}")
    return result
