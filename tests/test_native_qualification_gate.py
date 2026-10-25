from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from runtime.native_assets import require_native_qualification


def report_fixture() -> tuple[dict, dict]:
    worker = Path("tools/native-speech/worker.py")
    audit = {"source_lock_sha256": "test-only-audit", "target_encoding_context": None,
             "acoustic_precision": "float32"}
    clip = {"units": 50, "token_count": 250, "samples": 240000, "duration_ms": 10000,
            "passed": True, "acoustic_passed": True, "timing_passed": True,
            "decode_p95_ms": 100, "real_time_factor": 0.5,
            "maximum_block_boundary_jump": 0.01, "silence_interval_rms": [0.0001]}
    report = {"schema_id": "avm-native-speech-qualification-v2", "passed": True,
              "acoustic_passed": True, "timing_passed": True,
              "source_lock_sha256": audit["source_lock_sha256"],
              "lookahead_tokens": 3, "vocoder_tail_samples": 3840,
              "runtime": {"target_encoding_context": None, "precision": "float32",
                          "lookahead_tokens": 3, "tail_samples": 3840,
                          "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest()},
              "thresholds": {"silence_rms_max": 0.001, "boundary_jump_max": 0.15,
                             "decode_p95_ms_max": 200, "rtf_max": 0.8},
              "clips": [{**copy.deepcopy(clip), "kind": kind}
                        for kind in ("speech", "silence", "transition")]}
    return audit, report


@pytest.mark.parametrize("mutation", ["weakened", "incomplete", "timing", "duration", "silence"])
def test_qualification_flags_cannot_override_failed_measurements(tmp_path, mutation) -> None:
    audit, report = report_fixture()
    if mutation == "weakened":
        report["thresholds"]["decode_p95_ms_max"] = 400
    elif mutation == "incomplete":
        report["clips"].pop()
    elif mutation == "timing":
        report["clips"][0]["decode_p95_ms"] = 201
    elif mutation == "duration":
        report["clips"][0]["samples"] -= 1
    else:
        report["clips"][1]["silence_interval_rms"] = [0.01]
    path = tmp_path / "test-only-qualification.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification"):
        require_native_qualification(audit, path)


def test_qualification_rejects_a_different_worker_or_source_lock(tmp_path) -> None:
    audit, report = report_fixture()
    path = tmp_path / "test-only-qualification.json"
    path.write_text(json.dumps(report))
    assert require_native_qualification(audit, path)["passed"]
    report["runtime"]["worker_sha256"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different AVM acoustic worker"):
        require_native_qualification(audit, path)
    report["source_lock_sha256"] = "other"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source lock"):
        require_native_qualification(audit, path)
