from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.curation.common import dataset_path, registry_path


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def line(label: str, value: str) -> None:
    print(f"  {label}: {value}")


def gibibytes(value: float | int) -> str:
    return f"{float(value) / 1024**3:.2f} GiB"


def prepare_summary(root: Path) -> None:
    reports = dataset_path(root, "canary", "reports")
    prepared = read_json(registry_path(root, "prepare.json"))
    synthesis = read_json(reports / "synthesis.json")
    manifest = read_json(reports / "manifest.json")
    sources = prepared.get("fetch", {}).get("sources", [])
    source_ids = {item.get("source_id") for item in sources if item.get("source_id")}
    duration = sum(float(item.get("actual_seconds", 0.0)) for item in manifest["quotas"])
    line("Sources", f"{len(source_ids)} datasets, {len(sources)} archives, hashes verified")
    line(
        "Synthesis",
        f"{synthesis['utterances']} utterances, {synthesis['rejected']} rejected",
    )
    line("Manifest", f"{manifest['episodes']} episodes, {duration / 60:.1f} minutes")


def encode_summary(root: Path) -> None:
    reports = dataset_path(root, "canary", "reports")
    benchmark = read_json(reports / "codec-benchmark.json")
    native = read_json(reports / "native-decode.json")
    audit = read_json(reports / "audit.json")
    encoded = read_json(reports / "encoded.json")
    readiness = read_json(reports / "readiness.json")
    health = benchmark["health"]
    metrics = benchmark["benchmark"]
    identity = health["identity"]
    splits = ", ".join(f"{item['split']} {item['episodes']}" for item in encoded["splits"])
    line("Codec", f"{identity['codec_id']} on {health['device']}")
    line("Decode speed", f"RTF {metrics['rtf']:.3f}, p95 {metrics['p95_frame_ms']:.2f} ms")
    line(
        "Decode checks",
        f"{native['checked_segments'] - native['failed_segments']} passed, "
        f"{native['failed_segments']} failed",
    )
    line("Dataset", f"{audit['episodes']} episodes, {audit['duration_seconds'] / 60:.1f} minutes")
    line("Splits", splits)
    line("Readiness", "passed" if readiness["passed"] else "failed")


def train_summary(report_path: Path, checkpoint: Path) -> None:
    report = read_json(report_path)
    state = report["train_state"]
    metrics = report["metrics"]
    tracking = report.get("tracking", {})
    codec_values = [
        float(value) for key, value in metrics.items() if key.startswith("speech/codec_accuracy_q")
    ]
    line("Updates", str(state["update"]))
    line(
        "Runtime",
        f"{metrics['runtime/elapsed_seconds']:.1f} s, "
        f"{metrics['runtime/units_per_second']:.2f} units/s",
    )
    if "runtime/peak_memory_allocated_bytes" in metrics:
        line("Peak GPU memory", gibibytes(metrics["runtime/peak_memory_allocated_bytes"]))
    line(
        "Last metrics",
        f"loss {metrics.get('train/loss_total', 0.0):.4f}, "
        f"mean codec accuracy {sum(codec_values) / max(len(codec_values), 1):.4f}",
    )
    line(
        "Supervision",
        f"{metrics.get('speech/valid_frames', 0.0):.0f} speech frames, "
        f"{metrics.get('speech/active_unit_fraction', 0.0):.1%} active units, "
        f"{metrics.get('speech/no_speech_chunks', 0.0):.0f} silent chunks",
    )
    line("Speech mode accuracy", f"{metrics.get('speech/mode_accuracy', 0.0):.4f}")
    line("Checkpoint", str(checkpoint))
    tracking_mode = tracking.get("effective_mode")
    run_url = tracking.get("run_url")
    if run_url:
        line("W&B", f"{tracking_mode}, {run_url}")
    elif tracking_mode:
        line("W&B", tracking_mode)


def evaluate_summary(run_root: Path, max_updates: int) -> None:
    reports = {
        split: read_json(run_root / "runs" / f"{split}-evaluation.json")
        for split in ("validation", "test")
    }
    for split, report in reports.items():
        codec = report["speech_codec_accuracy"]
        line(
            split.capitalize(),
            f"{report['episodes']} episodes, mode accuracy "
            f"{report['speech_mode_accuracy']:.3f}, mean codec accuracy "
            f"{sum(codec) / max(len(codec), 1):.4f}",
        )
    if max_updates <= 5:
        line("Quality", "pipeline smoke test only; model is not converged")
    else:
        line("Quality", "metrics recorded; Canary does not enforce a convergence threshold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print concise Canary report summaries")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)

    encode = subparsers.add_parser("encode")
    encode.add_argument("--root", type=Path, required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-root", type=Path, required=True)
    evaluate.add_argument("--max-updates", type=int, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_summary(args.root.expanduser())
    elif args.command == "encode":
        encode_summary(args.root.expanduser())
    elif args.command == "train":
        train_summary(args.report.expanduser(), args.checkpoint.expanduser())
    else:
        evaluate_summary(args.run_root.expanduser(), args.max_updates)


if __name__ == "__main__":
    main()
