from __future__ import annotations

import copy

import torch
from data import SyntheticEpisodeDataset
from model import StreamingLatentLoop
from runtime.config import load_config


def test_partitioning_preserves_recurrent_outputs_and_gradients(smoke_config) -> None:
    baseline = StreamingLatentLoop(smoke_config.model).eval()
    partitioned = copy.deepcopy(baseline)
    partitioned.place(["cpu", "cpu", "cpu"])
    units = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units
    states = [model.initial_state(1, "cpu") for model in (baseline, partitioned)]
    losses = [[], []]
    for unit in units[:3]:
        for index, model in enumerate((baseline, partitioned)):
            output = model(unit.observation, states[index], unit.speech_codes)
            states[index] = output.state
            losses[index].append(output.speech_codec_logits.square().mean())
    for values in losses:
        torch.stack(values).sum().backward()
    for (name, expected), (other_name, actual) in zip(
        baseline.named_parameters(), partitioned.named_parameters(), strict=True
    ):
        assert name == other_name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(states[0].hidden, states[1].hidden)
    assert states[1].semantic_memory.grad_fn is not None


def test_explicit_device_configuration_is_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="explicit cuda:N"):
        load_config("configs/smoke.yaml", ["training.devices=[cuda]"])


def test_cuda_partition_preserves_output_when_available(smoke_config) -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline = StreamingLatentLoop(smoke_config.model).eval().cuda()
    partitioned = copy.deepcopy(baseline)
    partitioned.place(["cpu", "cuda:0"])
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    first = baseline(unit.to("cuda:0").observation, baseline.initial_state(1, "cuda:0"),
                     unit.speech_codes.cuda())
    second = partitioned(unit.observation, partitioned.initial_state(1, "cpu"), unit.speech_codes)
    torch.testing.assert_close(second.speech_codec_logits.cpu(), first.speech_codec_logits.cpu(),
                               atol=2e-5, rtol=2e-5)
    first.speech_codec_logits.square().mean().backward()
    second.speech_codec_logits.square().mean().backward()
    for (_, expected), (_, actual) in zip(baseline.named_parameters(),
                                         partitioned.named_parameters(), strict=True):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad.cpu(), expected.grad.cpu(),
                                       atol=2e-5, rtol=2e-5)
