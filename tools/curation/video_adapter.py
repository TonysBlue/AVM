"""Prepare episode-aligned video from an explicitly locked audiovisual source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


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
    if request.get("tick_ms") != 200:
        raise ValueError("video preparation requires the 200 ms native clock")
    resolution = int(request["scale_resolution"])
    patch = int(request["patch_size"])
    frames = []
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start), "-i", str(source),
         "-an", "-vf", "fps=5", "-frames:v", str(ticks),
         "-pix_fmt", "rgb24", "-vcodec", "ppm", "-f", "image2pipe", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        for _ in range(ticks):
            if process.stdout.readline(8) != b"P6\n":
                raise ValueError("real video does not cover the complete episode")
            dimensions = process.stdout.readline(64).split()
            if len(dimensions) != 2 or process.stdout.readline(8) != b"255\n":
                raise ValueError("video preparation requires valid binary RGB PPM frames")
            width, height = (int(value) for value in dimensions)
            if min(width, height) <= 0 or width * height > 8_388_608:
                raise ValueError("video source frame geometry exceeds the physical signal budget")
            raw = process.stdout.read(width * height * 3)
            if len(raw) != width * height * 3:
                raise ValueError("real video does not cover the complete episode")
            values = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            target_height = int(resolution / math.sqrt(width / height))
            target_width = int(target_height * width / height)
            target_size = (max(round(target_width / patch) * patch, patch),
                           max(round(target_height / patch) * patch, patch))
            image = Image.fromarray(values).resize(target_size, Image.Resampling.BICUBIC)
            frames.append(np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255)
        if process.wait(timeout=120) != 0:
            raise RuntimeError("aligned video decoding failed")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ticks=np.arange(ticks, dtype=np.int64),
                        **{f"frame_{index}": frame for index, frame in enumerate(frames)})
    args.output.with_suffix(".source.json").write_text(
        json.dumps(
            {**item, "source_manifest_sha256": hashlib.sha256(sources.read_bytes()).hexdigest()},
            sort_keys=True,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
