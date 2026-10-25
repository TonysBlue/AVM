from __future__ import annotations

from collections.abc import Iterable, Iterator

from model.types import Episode
from runtime.native_acoustics import NativeAcousticBackend


def encode_target_speech(
    episodes: Iterable[Episode], client: NativeAcousticBackend,
) -> Iterator[Episode]:
    """Encode complete target timelines once, including all supervised silence."""
    for episode in episodes:
        if episode.target_speech is None:
            raise ValueError(f"episode {episode.episode_id} has no target_speech waveform")
        waveform = episode.target_speech.flatten()
        if waveform.numel() != len(episode.units) * 3200:
            raise ValueError("native target speech must exactly cover the 16 kHz episode")
        codes = client.encode(waveform)
        if codes.numel() != len(episode.units) * 5:
            raise ValueError("native target tokenizer changed episode duration")
        grouped = codes.reshape(len(episode.units), 1, 5, 1)
        for unit, target in zip(episode.units, grouped, strict=True):
            unit.speech_codes.copy_(target)
        episode.metadata.update(
            speech_codes_encoded=True,
            native_source_lock_sha256=client.identity["source_lock_sha256"],
            target_encoding_context=client.identity["target_encoding_context"],
        )
        yield episode
