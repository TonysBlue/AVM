from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SUMMARY = Path(__file__).parents[1] / "tools" / "curation" / "summarize_canary.py"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def run_summary(*args: str) -> str:
    return subprocess.check_output([sys.executable, str(SUMMARY), *args], text=True)


def test_prepare_summary_reports_key_counts(tmp_path: Path) -> None:
    reports = tmp_path / "canary" / "reports"
    registry = tmp_path / "registry"
    write_json(
        registry / "prepare.json",
        {
            "fetch": {
                "sources": [
                    {"source_id": "speech-a"},
                    {"source_id": "speech-a"},
                    {"source_id": "dialogue-b"},
                ]
            }
        },
    )
    write_json(
        reports / "synthesis.json",
        {"utterances": 12, "rejected": 1},
    )
    write_json(
        reports / "manifest.json",
        {"episodes": 4, "quotas": [{"actual_seconds": 90.0}]},
    )

    output = run_summary("prepare", "--root", str(tmp_path))

    assert "Sources: 2 datasets, 3 archives, hashes verified" in output
    assert "Synthesis: 12 utterances, 1 rejected" in output
    assert "Manifest: 4 episodes, 1.5 minutes" in output


def test_train_and_evaluation_summaries_explain_smoke_quality(tmp_path: Path) -> None:
    training = tmp_path / "runs" / "training.json"
    checkpoint = tmp_path / "checkpoints" / "step-00000005.pt"
    write_json(
        training,
        {
            "train_state": {"update": 5},
            "metrics": {
                "runtime/elapsed_seconds": 12.5,
                "runtime/units_per_second": 8.0,
                "runtime/peak_memory_allocated_bytes": 2 * 1024**3,
                "train/loss_total": 0.25,
                "speech/codec_accuracy_q0": 0.1,
                "speech/codec_accuracy_q1": 0.3,
            },
            "tracking": {
                "effective_mode": "online",
                "run_url": "http://127.0.0.1:8080/run/abc",
            },
        },
    )
    for split in ("validation", "test"):
        write_json(
            tmp_path / "runs" / f"{split}-evaluation.json",
            {
                "episodes": 3,
                "speech_mode_accuracy": 0.8,
                "speech_codec_accuracy": [0.0, 0.1],
            },
        )

    train_output = run_summary("train", "--report", str(training), "--checkpoint", str(checkpoint))
    evaluation_output = run_summary("evaluate", "--run-root", str(tmp_path), "--max-updates", "5")

    assert "Peak GPU memory: 2.00 GiB" in train_output
    assert "mean codec accuracy 0.2000" in train_output
    assert "W&B: online, http://127.0.0.1:8080/run/abc" in train_output
    assert "Validation: 3 episodes, mode accuracy 0.800" in evaluation_output
    assert "Quality: pipeline smoke test only; model is not converged" in evaluation_output
