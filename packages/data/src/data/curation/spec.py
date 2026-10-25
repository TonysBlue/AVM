from __future__ import annotations

from dataclasses import dataclass

from data.curation.common import CATEGORIES, LANGUAGES, SPLITS

CATEGORY_FRACTIONS = {
    "public_speech": 0.30,
    "synthetic_dialogue": 0.50,
    "adjacent_turns": 0.15,
    "video_task": 0.05,
}
LANGUAGE_FRACTIONS = {"zh": 0.80, "en": 0.20}
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
DATASET_HOURS = {"canary": 1.0}
NATIVE_CODEC_ID = "minicpm-o-4_5-s3-25hz"
NATIVE_REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"
NATIVE_WEIGHT_SHA256 = "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71"

SOURCE_CATALOG = {
    "aishell1": {
        "source_version": "hf-AISHELL-1@bbe295d5",
        "source_url": "https://huggingface.co/datasets/AISHELL/AISHELL-1",
        "license": "Apache-2.0",
        "license_url": "https://www.openslr.org/33/",
        "language": "zh",
        "category": "public_speech",
    },
    "librispeech_train_clean_100": {
        "source_version": "hf-librispeech_asr@71cacbfb",
        "source_url": "https://huggingface.co/datasets/openslr/librispeech_asr",
        "license": "CC-BY-4.0",
        "license_url": "https://www.openslr.org/12/",
        "language": "en",
        "category": "public_speech",
    },
    "aishell4_train_l": {
        "source_version": "hf-argmaxinc-aishell-4@0cf6e538",
        "source_url": "https://huggingface.co/datasets/argmaxinc/aishell-4",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://www.openslr.org/111/",
        "language": "zh",
        "category": "adjacent_turns",
    },
    "dailytalk": {
        "source_version": "hf-DailyTalk@1f0d958a",
        "source_url": (
            "https://huggingface.co/datasets/"
            "DynamicSuperbPrivate/DialogueActClassification_DailyTalk"
        ),
        "license": "CC-BY-SA-4.0 AND academic-use-statement",
        "license_url": "https://github.com/keonlee9420/DailyTalk",
        "language": "en",
        "category": "adjacent_turns",
    },
}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    duration_seconds: float

    def target_seconds(self, category: str, language: str, split: str) -> float:
        return (
            self.duration_seconds
            * CATEGORY_FRACTIONS[category]
            * LANGUAGE_FRACTIONS[language]
            * SPLIT_FRACTIONS[split]
        )

    def quota_rows(self) -> list[dict[str, object]]:
        return [
            {
                "category": category,
                "language": language,
                "split": split,
                "target_seconds": self.target_seconds(category, language, split),
            }
            for category in CATEGORIES
            for language in LANGUAGES
            for split in SPLITS
        ]


def dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_HOURS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_HOURS)}")
    return DatasetSpec(name, DATASET_HOURS[name] * 3600)
