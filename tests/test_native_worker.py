from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch


def test_native_prompt_io_preserves_pcm_and_attaches_returned_caches(tmp_path, monkeypatch) -> None:
    prepare = runpy.run_path("tools/native-speech/worker.py")["prepare_native_cache"]
    reference = tmp_path / "reference.wav"
    values = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    sf.write(reference, values, 16_000, subtype="FLOAT")
    original = object()
    audio_api = SimpleNamespace(load=original)
    monkeypatch.setitem(sys.modules, "torchaudio", audio_api)
    flow, vocoder = {"flow": 1}, {"vocoder": 2}

    class ProtocolFixture:
        def set_stream_cache(self, path):
            samples, rate = audio_api.load(path, backend="soundfile")
            assert rate == 16_000
            assert torch.equal(samples, torch.from_numpy(values)[None])
            return flow, vocoder

    model = ProtocolFixture()
    prepare(model, str(reference))
    assert model.stream_cache is flow and model.hift_cache_dict is vocoder
    assert audio_api.load is original


def test_native_prompt_io_is_restored_when_initialization_fails(monkeypatch) -> None:
    prepare = runpy.run_path("tools/native-speech/worker.py")["prepare_native_cache"]
    original = object()
    audio_api = SimpleNamespace(load=original)
    monkeypatch.setitem(sys.modules, "torchaudio", audio_api)

    class ProtocolFixture:
        def set_stream_cache(self, path):
            raise RuntimeError("native cache failed")

    with pytest.raises(RuntimeError, match="native cache failed"):
        prepare(ProtocolFixture(), str(Path("locked.wav")))
    assert audio_api.load is original
