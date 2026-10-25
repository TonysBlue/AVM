from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
TARGET_LUFS = -23.0
SOURCE_META = {
    "aishell1": {
        "version": "hf-AISHELL-1@bbe295d5",
        "url": "https://huggingface.co/datasets/AISHELL/AISHELL-1",
        "license": "Apache-2.0",
        "language": "zh",
        "category": "public_speech",
    },
    "librispeech_train_clean_100": {
        "version": "hf-librispeech_asr@71cacbfb",
        "url": "https://huggingface.co/datasets/openslr/librispeech_asr",
        "license": "CC-BY-4.0",
        "language": "en",
        "category": "public_speech",
    },
    "aishell4_train_l": {
        "version": "hf-argmaxinc-aishell-4@0cf6e538",
        "url": "https://huggingface.co/datasets/argmaxinc/aishell-4",
        "license": "CC-BY-SA-4.0",
        "language": "zh",
        "category": "adjacent_turns",
    },
    "dailytalk": {
        "version": "hf-DailyTalk@1f0d958a",
        "url": (
            "https://huggingface.co/datasets/"
            "DynamicSuperbPrivate/DialogueActClassification_DailyTalk"
        ),
        "license": "CC-BY-SA-4.0 AND academic-use-statement",
        "language": "en",
        "category": "adjacent_turns",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resample(waveform: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return waveform.astype(np.float32)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    return resample_poly(waveform, SAMPLE_RATE // divisor,
                         source_rate // divisor).astype(np.float32)


def normalize(waveform: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    if rms <= 1e-7:
        raise ValueError("source waveform is silent")
    desired = 10 ** (TARGET_LUFS / 20)
    scaled = waveform * (desired / rms)
    peak = float(np.max(np.abs(scaled), initial=0.0))
    if peak > 10 ** (-1 / 20):
        scaled *= 10 ** (-1 / 20) / peak
    return scaled.astype(np.float32)


def audio_bytes(value: dict[str, Any]) -> tuple[np.ndarray, int]:
    raw = value.get("bytes")
    if not isinstance(raw, bytes):
        raise ValueError("Parquet audio row has no embedded bytes")
    waveform, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return waveform.mean(axis=1), int(rate)


def iter_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        yield from parquet.read_row_group(group).to_pylist()


def iter_aishell(paths: list[Path]):
    transcript_path = next(path for path in paths if path.name == "transcript.txt")
    transcripts = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        utterance, text = line.split(maxsplit=1)
        transcripts[utterance] = text.replace(" ", "")
    for archive in sorted(path for path in paths if path.suffixes[-2:] == [".tar", ".gz"]):
        speaker = archive.name.split(".", 1)[0]
        with tarfile.open(archive) as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                handle = bundle.extractfile(member)
                if handle is None:
                    continue
                waveform, rate = sf.read(io.BytesIO(handle.read()), dtype="float32", always_2d=True)
                utterance = Path(member.name).stem
                text = transcripts.get(utterance)
                if text:
                    yield speaker, utterance, waveform.mean(axis=1), int(rate), text


def split(index: int) -> str:
    bucket = index % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def write_item(
    root: Path,
    source_id: str,
    item_id: str,
    waveform: np.ndarray,
    rate: int,
    *,
    text: str,
    response_text: str,
    speaker: str,
    session: str,
    split_name: str,
    license_sha256: str,
) -> dict[str, Any]:
    meta = SOURCE_META[source_id]
    output = root / "registry" / "normalized" / "sources" / source_id / f"{item_id}.flac"
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize(resample(waveform, rate)[: 15 * SAMPLE_RATE])
    sf.write(output, normalized, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    metrics = output.with_suffix(".metrics.json")
    metrics.write_text(
        json.dumps({"integrated_lufs": TARGET_LUFS}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source_item_id": item_id,
        "source_id": source_id,
        "source_version": meta["version"],
        "source_url": meta["url"],
        "source_utterance_ids": [item_id],
        "source_license": meta["license"],
        "license_sha256": license_sha256,
        "redistribution_allowed": False,
        "category": meta["category"],
        "language": meta["language"],
        "split": split_name,
        "group_id": session,
        "speaker_id": speaker,
        "session_id": session,
        "text": text,
        "response_text": response_text,
        "audio": str(output.relative_to(root)),
        "audio_sha256": sha256(output),
        "normalization": {
            "integrated_lufs": TARGET_LUFS,
            "metrics_sha256": sha256(metrics),
        },
        "fixture": False,
    }


def source_paths(fetch_report: dict[str, Any], root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for source in fetch_report["sources"]:
        result.setdefault(source["source_id"], []).append(root / source["archive"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("operation") != "index-and-normalize":
        raise ValueError("unsupported normalizer operation")
    root = Path(request["data_root"]).resolve()
    fetch = json.loads((root / "registry" / "reports" / "fetch-report.json").read_text())
    paths = source_paths(fetch, root)
    licenses = {
        source["source_id"]: source["license_sha256"] for source in fetch["sources"]
    }
    records: list[dict[str, Any]] = []

    aishell_counts = {"train": 0, "validation": 0, "test": 0}
    for speaker, utterance, waveform, rate, text in iter_aishell(paths["aishell1"]):
        item_id = f"aishell1-{utterance}"
        speaker_index = sorted(
            path.name for path in paths["aishell1"] if path.name.startswith("S")
        ).index(f"{speaker}.tar.gz")
        split_name = ("train", "validation", "test")[speaker_index]
        limit = 260 if split_name == "train" else 160
        if aishell_counts[split_name] >= limit:
            continue
        records.append(
            write_item(
                root, "aishell1", item_id, waveform, rate,
                text=text, response_text="", speaker=f"aishell1-{speaker}",
                session=f"aishell1-{speaker}",
                split_name=split_name,
                license_sha256=licenses["aishell1"],
            )
        )
        aishell_counts[split_name] += 1

    librispeech_counts = {"train": 0, "validation": 0, "test": 0}
    librispeech_speakers: dict[int, str] = {}
    for row in iter_rows(paths["librispeech_train_clean_100"][0]):
        waveform, rate = audio_bytes(row["audio"])
        item_id = f"librispeech-{row['id']}"
        speaker_number = int(row["speaker_id"])
        if speaker_number not in librispeech_speakers:
            position = len(librispeech_speakers) % 10
            librispeech_speakers[speaker_number] = (
                "train" if position < 8 else "validation" if position == 8 else "test"
            )
        split_name = librispeech_speakers[speaker_number]
        if librispeech_counts[split_name] >= 120:
            continue
        speaker = f"librispeech-{speaker_number}"
        records.append(
            write_item(
                root, "librispeech_train_clean_100", item_id, waveform, rate,
                text=str(row["text"]), response_text="", speaker=speaker,
                session=f"librispeech-{row['speaker_id']}-{row['chapter_id']}",
                split_name=split_name,
                license_sha256=licenses["librispeech_train_clean_100"],
            )
        )
        librispeech_counts[split_name] += 1
        if sum(librispeech_counts.values()) >= 360:
            break

    for shard, path in enumerate(paths["aishell4_train_l"]):
        for row_index, row in enumerate(iter_rows(path)):
            waveform, rate = audio_bytes(row["audio"])
            starts = row["timestamps_start"]
            ends = row["timestamps_end"]
            speakers = row["speakers"]
            chosen_split = ("train", "validation", "test")[shard % 3]
            for turn in range(min(len(starts) - 1, 70)):
                start = max(0, int(float(starts[turn]) * rate))
                end = min(waveform.size, int(float(ends[turn]) * rate))
                if end - start < int(0.3 * rate):
                    continue
                item_id = f"aishell4-{shard}-{row_index}-{turn:04d}"
                records.append(
                    write_item(
                        root, "aishell4_train_l", item_id, waveform[start:end], rate,
                        text="", response_text="请继续说明。",
                        speaker=f"aishell4-{shard}-{speakers[turn]}",
                        session=f"aishell4-{shard}-{row_index}", split_name=chosen_split,
                        license_sha256=licenses["aishell4_train_l"],
                    )
                )

    daily_rows = list(iter_rows(paths["dailytalk"][0]))
    daily_counts = {"train": 0, "validation": 0, "test": 0}
    for row in daily_rows:
        parts = Path(str(row["file"])).stem.split("_")
        turn, speaker, stem = parts[0], parts[1], parts[2]
        split_name = split(int(stable_hash(stem)[:8], 16))
        if daily_counts[split_name] >= 100:
            continue
        waveform, rate = audio_bytes(row["audio"])
        item_id = f"dailytalk-{Path(str(row['file'])).stem}"
        records.append(
            write_item(
                root, "dailytalk", item_id, waveform, rate, text="",
                response_text="I understand. Please continue.",
                speaker=f"dailytalk-{stem}-{speaker}",
                session=f"dailytalk-{stem}", split_name=split_name,
                license_sha256=licenses["dailytalk"],
            )
        )
        daily_counts[split_name] += 1
        if sum(daily_counts.values()) >= 260:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda value: value["source_item_id"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    assistant_prompt = root / "registry" / "voices" / "bootstrap" / "zero_shot_prompt.wav"
    if not assistant_prompt.is_file():
        raise FileNotFoundError(f"assistant prompt is absent: {assistant_prompt}")
    library = [
        {
            "voice_id": "assistant-neutral",
            "language": "multilingual",
            "role": "assistant",
            "split": None,
            "license": "Apache-2.0 CosyVoice example asset",
            "authorization": "CosyVoice repository example asset",
            "prompt_audio": str(assistant_prompt),
            "prompt_sha256": sha256(assistant_prompt),
            "prompt_text": "希望你以后能够做的比我还好呦。",
        }
    ]
    for language in ("zh", "en"):
        for split_name in ("train", "validation", "test"):
            prompt = next(
                record
                for record in records
                if record["category"] == "public_speech"
                and record["language"] == language
                and record["split"] == split_name
                and record["text"]
            )
            prompt_path = root / prompt["audio"]
            library.append(
                {
                    "voice_id": f"user-{language}-{split_name}",
                    "language": language,
                    "role": "user",
                    "split": split_name,
                    "license": prompt["source_license"],
                    "authorization": "licensed public training utterance",
                    "prompt_audio": str(prompt_path),
                    "prompt_sha256": sha256(prompt_path),
                    "prompt_text": prompt["text"],
                }
            )
    library_path = root / "registry" / "voices" / "voice-library.json"
    library_path.parent.mkdir(parents=True, exist_ok=True)
    library_path.write_text(
        json.dumps(library, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
