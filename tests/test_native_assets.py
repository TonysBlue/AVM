from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
import torch
from data.cli import main
from data.curation.native_assets import (
    ACOUSTIC_FILES,
    LOCK_SCHEMA,
    REPOSITORY,
    _check_dimensions,
    _check_native_weights,
    _tensor_inventory,
    audit_native_assets,
    locked_path,
    sha256_file,
)
from data.curation.native_speech import _evaluate_clip, qualify_native_speech
from runtime.continuous_speech import ContinuousSpeechClock
from runtime.native_acoustics import NativeAcousticBackend

from tests.test_continuous_speech import ProtocolTestDecoder


def dimension_fixture() -> dict:
    return {
        "version": "4.5", "hidden_size": 4096, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 12288, "query_num": 64, "audio_pool_step": 5,
        "audio_config": {"d_model": 1024, "encoder_layers": 24,
                         "encoder_attention_heads": 16, "encoder_ffn_dim": 4096,
                         "num_mel_bins": 80},
        "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27,
                          "num_attention_heads": 16, "intermediate_size": 4304,
                          "patch_size": 14},
        "tts_config": {"hidden_size": 768, "num_hidden_layers": 20,
                       "num_attention_heads": 12, "num_key_value_heads": 12,
                       "intermediate_size": 3072, "num_vq": 1, "num_audio_tokens": 6562},
    }


def make_lock_fixture(tmp_path: Path) -> tuple[Path, dict]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    entry = runtime / "token2wav.py"
    entry.write_text("# In-process asset-audit fixture; not executable native speech.\n")
    root = tmp_path / "model"
    root.mkdir()
    files = {"config.json": json.dumps(dimension_fixture()).encode(),
             "model.safetensors.index.json": json.dumps({
                 "weight_map": {"llm.weight": "missing.safetensors"}
             }).encode(), "speaker.wav": b"fixture-not-a-recording"}
    files.update({f"assets/token2wav/{name}": b"fixture-not-native-weights"
                  for name in ACOUSTIC_FILES})
    hashes = {}
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        hashes[name] = sha256_file(path)
    payload = {
        "schema_id": LOCK_SCHEMA, "repository": REPOSITORY, "revision": "a" * 40,
        "license": "test-only", "root": "model", "files": hashes,
        "speaker_reference": "speaker.wav",
        "acoustic_runtime": {"root": "runtime", "revision": "b" * 40,
                             "python": sys.executable,
                             "license": "test-only", "entrypoint": "token2wav.py",
                             "files": {"token2wav.py": sha256_file(entry)}},
    }
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(payload))
    return lock, payload


def test_native_audit_rejects_a_mutable_source_revision(tmp_path) -> None:
    lock, payload = make_lock_fixture(tmp_path)
    payload["revision"] = "main"
    lock.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="immutable"):
        audit_native_assets(lock)


def test_native_audit_rejects_changed_runtime_source(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    (tmp_path / "runtime" / "token2wav.py").write_text("changed\n")
    with pytest.raises(ValueError, match="runtime source"):
        audit_native_assets(lock)


def test_native_audit_rejects_unlocked_runtime_dependencies(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    (tmp_path / "runtime" / "flow.py").write_text("unlocked\n")
    with pytest.raises(ValueError, match="all native acoustic"):
        audit_native_assets(lock)


def test_native_audit_rejects_wrong_weight_hash(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    (tmp_path / "model" / "assets" / "token2wav" / "flow.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_native_assets(lock)


def test_native_audit_rejects_a_missing_tokenizer(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    (tmp_path / "model" / "assets" / "token2wav" /
     "speech_tokenizer_v2_25hz.onnx").unlink()
    with pytest.raises(FileNotFoundError, match="missing locked"):
        audit_native_assets(lock)


def test_native_audit_rejects_unlocked_indexed_weights(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    with pytest.raises(ValueError, match="every indexed"):
        audit_native_assets(lock)


def test_acoustic_audit_cannot_claim_full_model_or_speech_qualification(tmp_path) -> None:
    lock, _ = make_lock_fixture(tmp_path)
    result = audit_native_assets(lock, model_weights=False)
    assert result["passed"]
    assert result["model_weights_audited"] is False
    assert result["speech_qualified"] is False
    assert "tensor_elements" not in result


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "sub/../../outside"])
def test_native_paths_cannot_escape_the_asset_root(tmp_path, relative) -> None:
    with pytest.raises(ValueError):
        locked_path(tmp_path, relative)


def test_native_paths_cannot_escape_through_a_symlink(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        locked_path(root, "link/outside")


def test_native_dimensions_are_not_scaled_down() -> None:
    config = dimension_fixture()
    _check_dimensions(config)
    config["hidden_size"] = 192
    with pytest.raises(ValueError, match="dimension mismatch"):
        _check_dimensions(config)


def test_native_checkpoint_cannot_use_smaller_weights_under_a_full_size_config() -> None:
    with pytest.raises(ValueError, match="native tensor"):
        _check_native_weights({"resampler.query": (16 * 192, "F32")}, 6562)


def test_tensor_audit_counts_without_loading_pytorch_weights(tmp_path) -> None:
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2, 3],
                                     "data_offsets": [0, 24]}}).encode()
    path = tmp_path / "fixture.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
    assert _tensor_inventory(path) == {"weight": (6, "F32")}
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(20))
    with pytest.raises(ValueError, match="offsets"):
        _tensor_inventory(path)


def test_native_cli_reports_missing_assets_as_failure(tmp_path, capsys) -> None:
    assert main(["inspect-native-assets", "--source-lock", str(tmp_path / "absent.json")]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["passed"] is False
    assert "absent.json" in report["error"]


def test_real_qualification_does_not_replace_a_missing_asset_with_fixture(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        qualify_native_speech(tmp_path / "missing-lock.json", tmp_path / "corpus.json",
                              tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_client_requires_audited_assets() -> None:
    with pytest.raises(ValueError, match="asset audit"):
        NativeAcousticBackend({"passed": True})


@pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the missing-CUDA failure path")
def test_native_worker_fails_closed_without_cuda() -> None:
    # Protocol fixture only; no asset qualification report is produced.
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        NativeAcousticBackend({"schema_id": "avm-native-assets-audit-v1", "passed": True,
                               "runtime_python": sys.executable})


def test_acoustic_qualification_measurement_keeps_real_tail_and_silence() -> None:
    class QualificationProtocolFixture(ProtocolTestDecoder):
        vocab_size = 100

        def reset(self) -> None:
            self.pending = torch.empty(0)

        def synchronize(self) -> None:
            pass

        def encode(self, waveform):
            return torch.zeros(waveform.numel() // 640, dtype=torch.long)

    report, waveform = _evaluate_clip(
        QualificationProtocolFixture(3, 3840), ContinuousSpeechClock(3, 3840, 600),
        torch.zeros(3200 * 10), kind="silence", silence_intervals=[[0, 2000]],
    )
    assert report["samples"] == 48_000
    assert report["token_count"] == 50
    assert report["silence_interval_rms"] == [0.0]
    assert torch.equal(waveform, torch.zeros(48_000))
    assert "passed" not in report  # A fixture never produces native qualification.
