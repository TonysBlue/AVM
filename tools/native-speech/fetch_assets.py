"""Fetch immutable official assets and write a complete AVM source lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfApi, hf_hub_download

REPOSITORY = "openbmb/MiniCPM-o-4_5"
REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"
WHEEL_SHA256 = "7bf176cdadedabca30f4abb1fa49b7514981fbc8f763314fa811c2e4160a4523"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_matches(path: Path, item) -> bool:
    if not path.is_file() or path.stat().st_size != item.size:
        return False
    if item.lfs:
        return digest(path) == item.lfs.sha256
    blob = hashlib.sha1(f"blob {item.size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            blob.update(block)
    return blob.hexdigest() == item.blob_id


def fetch(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.card_data.get("license") != "apache-2.0":
        raise ValueError("official revision or license differs from AVM source contract")
    selected = [item for item in info.siblings if (
        "/" not in item.rfilename and not item.rfilename.startswith(".")
    ) or item.rfilename.startswith("assets/token2wav/")
        or item.rfilename == "assets/system_ref_audio.wav"]
    files = {}
    for item in selected:
        path = root / item.rfilename
        if not source_matches(path, item):
            print(f"Downloading {item.rfilename}", flush=True)
            hf_hub_download(REPOSITORY, item.rfilename, revision=REVISION,
                            local_dir=root, force_download=path.is_file())
        actual = digest(path)
        if not source_matches(path, item):
            raise ValueError(f"official source hash mismatch: {item.rfilename}")
        files[item.rfilename] = actual
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    with urllib.request.urlopen("https://pypi.org/pypi/stepaudio2-minicpmo/0.1.1/json") as response:
        release = json.load(response)
    wheels = [item for item in release["urls"] if item["filename"].endswith(".whl")]
    if len(wheels) != 1 or wheels[0]["digests"]["sha256"] != WHEEL_SHA256:
        raise ValueError("published acoustic runtime wheel identity changed")
    archive = runtime / wheels[0]["filename"]
    if not archive.is_file() or digest(archive) != WHEEL_SHA256:
        urllib.request.urlretrieve(wheels[0]["url"], archive)
    if digest(archive) != WHEEL_SHA256:
        raise ValueError("downloaded acoustic runtime wheel hash mismatch")
    with zipfile.ZipFile(archive) as wheel:
        for member in wheel.namelist():
            if not member.startswith("stepaudio2/") or member.endswith("/"):
                continue
            destination = runtime / member
            if not destination.resolve().is_relative_to(runtime.resolve()):
                raise ValueError("wheel member escapes acoustic runtime root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(wheel.read(member))
    samples, rate = sf.read(root / "assets/system_ref_audio.wav", dtype="float32",
                           always_2d=True)
    waveform = torch.from_numpy(samples.mean(axis=1))
    if rate != 16000:
        waveform = torchaudio.functional.resample(waveform, rate, 16000)
    reference = waveform[160000:192000]
    if reference.numel() != 32000:
        raise ValueError("official speaker reference cannot cover the locked two-second crop")
    sf.write(root / "reference-voice-2s.wav", reference.numpy(), 16000, subtype="FLOAT")
    files["reference-voice-2s.wav"] = digest(root / "reference-voice-2s.wav")
    runtime_files = {str(path.relative_to(runtime)): digest(path)
                     for path in runtime.rglob("*.py")}
    runtime_files[archive.name] = WHEEL_SHA256
    lock = {
        "schema_id": "avm-native-source-lock-v1", "repository": REPOSITORY,
        "revision": REVISION, "license": "Apache-2.0", "root": ".", "files": files,
        "speaker_reference": "reference-voice-2s.wav",
        "speaker_reference_provenance": {
            "source_sha256": files["assets/system_ref_audio.wav"],
            "start_sample_16khz": 160000, "samples": 32000,
        },
        "target_encoding_context": {"kind": "fixed-speaker-prefix-v1",
                                    "asset": "reference-voice-2s.wav", "samples": 32000,
                                    "sample_rate": 16000, "trim_tokens": 50},
        "acoustic_precision": "float32",
        "acoustic_runtime": {
            "root": "runtime", "license": "Apache-2.0",
            "revision": "stepaudio2-minicpmo==0.1.1", "python": sys.executable,
            "entrypoint": "stepaudio2/token2wav.py", "package_archive": archive.name,
            "files": runtime_files,
        },
    }
    output = root / "source-lock.json"
    temporary = root / "source-lock.json.partial"
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(fetch(args.root.expanduser().absolute()), flush=True)
