from __future__ import annotations

import runpy

import pytest
import torch
from model import StreamingLatentLoop
from model.types import AudioVideoInput, GenerationOutput, SpeechSamplingConfig


def test_full_core_probe_checks_output_and_continuous_state(smoke_config) -> None:
    inspect = runpy.run_path("tools/native-speech/verify_core.py")["inspect_step"]
    model = StreamingLatentLoop(smoke_config.model).eval()
    observation = AudioVideoInput(torch.zeros(1, 3200), torch.zeros(1, 3, 224, 224))
    state = model.initial_state(1, "cpu")
    with torch.inference_mode():
        for unit in (1, 2):
            result = model.generate_step(observation, state, SpeechSamplingConfig(greedy=True))
            report = inspect(result, smoke_config.model, unit)
            assert report["backbone_kv_shape"][2] == unit * smoke_config.model.perceiver_slots
            state = result.output.state
        invalid = GenerationOutput(
            result.output,
            torch.full_like(result.speech_codes, smoke_config.model.speech_codebook_size - 1),
        )
        with pytest.raises(ValueError, match="EOS"):
            inspect(invalid, smoke_config.model, 2)
        with pytest.raises(ValueError, match="advance"):
            inspect(result, smoke_config.model, 3)
