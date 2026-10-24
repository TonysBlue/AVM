"""Prepare episode-aligned video from an explicitly locked audiovisual source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("operation") != "prepare-dialogue-video":
        raise ValueError("unsupported video adapter operation")
    sources = args.sources.expanduser().resolve()
    entries = [json.loads(line) for line in sources.read_text().splitlines() if line.strip()]
    matching = [item for item in entries if item["episode_id"] == request["episode_id"]]
    if len(matching) != 1:
        raise ValueError("episode requires exactly one aligned real video source")
    item = matching[0]
    for key in ("source_license", "source_version", "source_sha256", "alignment_id"):
        if not item.get(key):
            raise ValueError(f"video source requires {key}")
    source = Path(item["path"]).expanduser()
    if not source.is_absolute():
        source = sources.parent / source
    if hashlib.sha256(source.read_bytes()).hexdigest() != item["source_sha256"]:
        raise ValueError("video source SHA-256 mismatch")
    ticks = int(request["ticks"])
    if ticks < 1:
        raise ValueError("video timeline must contain at least one unit")
    start = float(item.get("start_seconds", 0))
    if start < 0:
        raise ValueError("video start_seconds must be non-negative")
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start), "-i", str(source),
         "-an", "-vf", "fps=25/2,scale=224:224", "-frames:v", str(ticks),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True,
    )
    values = np.frombuffer(completed.stdout, dtype=np.uint8)
    if values.size != ticks * 224 * 224 * 3:
        raise ValueError("real video does not cover the complete episode")
    frames = values.reshape(ticks, 224, 224, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ticks=np.arange(ticks, dtype=np.int64), frames=frames)
    args.output.with_suffix(".source.json").write_text(
        json.dumps(
            {**item, "source_manifest_sha256": hashlib.sha256(sources.read_bytes()).hexdigest()},
            sort_keys=True,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
