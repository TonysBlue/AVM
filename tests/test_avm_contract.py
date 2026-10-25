from __future__ import annotations

import threading
from dataclasses import fields

import numpy as np
import pytest
import torch
from contracts import ActuationSignal, ObservationSignal, RewardVector, SpeechSignal, VideoSignal
from contracts.protocol import message_to_actuation, message_to_observation
from contracts.realtime_pb2 import ActuationSignal as ActuationMessage
from contracts.realtime_pb2 import ObservationSignal as ObservationMessage
from data import SyntheticEpisodeDataset
from harness.deployment.dialogue import create_backend
from harness.environment.dialogue import BoundedPlayback, VideoTimeline
from model import StreamingLatentLoop
from model.types import RecurrentState, StreamUnit
from runtime.config import load_config
from training.training import _policy_logprobs


def frame(timestamp: int = 0, revision: int = 0) -> VideoSignal:
    return VideoSignal(b"rgb", 1, 1, timestamp, revision)


def test_avm_has_no_action_parameters_states_targets_or_outputs(smoke_config) -> None:
    model = StreamingLatentLoop(smoke_config.model)
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    generated = model.generate_step(unit.observation, model.initial_state(1, "cpu"))
    assert not any("action" in name for name, _ in model.named_parameters())
    for cls in (RecurrentState, StreamUnit, ActuationSignal):
        assert not any("action" in field.name or field.name == "controls" for field in fields(cls))
    assert not hasattr(generated, "action_frame")
    assert not hasattr(generated.output, "action")


def test_silent_policy_probability_excludes_codec(smoke_config) -> None:
    from model import SpeechSamplingConfig

    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    output = model(unit.observation, model.initial_state(1, "cpu"), unit.speech_codes)
    mode = torch.zeros(1, dtype=torch.long)
    sampling = SpeechSamplingConfig(temperature=0.8, top_k=0)
    actual = _policy_logprobs(output, mode, unit.speech_codes, sampling)
    expected = torch.log_softmax(output.speech_mode_logits / 0.8, -1)[:, 0]
    assert torch.allclose(actual, expected)


def test_video_timeline_selects_causal_frames_and_reuses_revision() -> None:
    timeline = VideoTimeline(max_age_ms=200)
    with pytest.raises(RuntimeError, match="initial"):
        timeline.at(0)
    timeline.append(frame(10, 0))
    timeline.append(frame(100, 1))
    assert timeline.at(80).revision == 0
    assert timeline.at(160).revision == 1
    assert timeline.at(240).revision == 1
    with pytest.raises(RuntimeError, match="stale"):
        timeline.at(301)
    with pytest.raises(ValueError, match="backwards"):
        timeline.append(frame(90, 2))
    with pytest.raises(ValueError, match="backwards"):
        timeline.append(frame(110, 1))


def test_video_timeline_rejects_future_only_frames() -> None:
    timeline = VideoTimeline()
    timeline.append(frame(100))
    with pytest.raises(RuntimeError, match="causal"):
        timeline.at(80)


@pytest.mark.parametrize("decoder,message", [
    (message_to_observation, ObservationMessage), (message_to_actuation, ActuationMessage),
])
def test_legacy_physical_payload_is_rejected(decoder, message) -> None:
    with pytest.raises(ValueError, match="protocol"):
        decoder(message(session_id="legacy").SerializeToString())


def test_reward_has_no_action_efficiency() -> None:
    reward = RewardVector(task=1, speech_quality=0.8, latency_quality=0.2, safety_quality=-0.1)
    assert reward.total == pytest.approx(1.0)
    assert not hasattr(reward, "action_efficiency")


def test_formal_harness_has_no_fixture_device_fallback() -> None:
    with pytest.raises(ValueError, match="real dialogue"):
        create_backend({"protocol_version": "avm-realtime-v1"})


def test_legacy_action_configuration_is_rejected() -> None:
    with pytest.raises(Exception, match="action_schema_id"):
        load_config("configs/smoke.yaml", ["model.action_schema_id=structured-action-v1"])


def test_video_changes_fused_hidden_for_identical_audio(smoke_config) -> None:
    from dataclasses import replace

    model = StreamingLatentLoop(smoke_config.model).eval()
    unit = SyntheticEpisodeDataset(smoke_config.data, smoke_config.model).make_episode(0).units[0]
    alternate = replace(unit, video=torch.ones_like(unit.video))
    state = model.initial_state(1, "cpu")
    with torch.no_grad():
        first = model(unit.observation, state, unit.speech_codes)
        second = model(alternate.observation, state, alternate.speech_codes)
    assert not torch.allclose(first.hidden, second.hidden)
    assert not torch.allclose(first.speech_mode_logits, second.speech_mode_logits)


def test_silence_discards_queued_speech_and_allows_resuming() -> None:
    class Sink:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.played = []
            self.interrupts = 0

        def play(self, output):
            self.entered.set()
            assert self.release.wait(2)
            self.played.append(output.unit_index)

        def interrupt(self):
            self.interrupts += 1

        def close(self):
            self.release.set()

    sink = Sink()
    playback = BoundedPlayback(sink, capacity=2)
    pcm = np.zeros(1920, dtype=np.float32).tobytes()
    try:
        playback.submit(ActuationSignal("s", 0, SpeechSignal(pcm)))
        assert sink.entered.wait(2)
        playback.submit(ActuationSignal("s", 1, SpeechSignal(pcm)))
        playback.submit(ActuationSignal("s", 2, SpeechSignal(pcm)))
        with pytest.raises(RuntimeError, match="budget"):
            playback.submit(ActuationSignal("s", 3, SpeechSignal(pcm)))
        sink.release.set()
        playback.submit(ActuationSignal("s", 3, SpeechSignal(pcm, silent=True)))
        assert sink.interrupts == 1
        assert not playback._pending
        playback.submit(ActuationSignal("s", 4, SpeechSignal(pcm)))
    finally:
        playback.close()


def test_observation_has_no_text_or_gui_channels() -> None:
    assert {field.name for field in fields(ObservationSignal)} == {
        "session_id", "unit_index", "timestamp_ms", "delta_ms", "mic", "video"
    }


def test_shared_video_normalization_supports_rgba_and_arbitrary_resolution() -> None:
    from runtime.video import video_tensor

    image = np.zeros((16, 32, 4), dtype=np.uint8)
    image[..., 0] = 255
    result = video_tensor(VideoSignal(image.tobytes(), 32, 16, 0, 0, pixel_format="rgba32"))
    assert result.shape == (1, 3, 224, 224)
    assert torch.allclose(result[:, 0], torch.ones_like(result[:, 0]))
    assert result[:, 1:].count_nonzero() == 0


def test_explicit_weight_migration_starts_new_lineage(tmp_path, smoke_config) -> None:
    from training.checkpoint import file_sha256
    from training.migration import migrate_latentloop_weights
    from training.training import initialize_exact_weights

    model = StreamingLatentLoop(smoke_config.model)
    weights = {**model.state_dict(), "action_head.legacy": torch.ones(2)}
    source = tmp_path / "latentloop.pt"
    torch.save({
        "model": weights,
        "metadata": {
            "action_schema_id": "structured-action-v1",
            "codec_id": smoke_config.data.codec_id,
            "codec_revision": smoke_config.data.codec_revision,
            "codec_weight_hash": smoke_config.data.codec_weight_hash,
        },
        "optimizer": {"legacy": True},
    }, source)
    with pytest.raises(ValueError, match="schema"):
        initialize_exact_weights(model, source)
    output = migrate_latentloop_weights(smoke_config, source, tmp_path / "avm.pt")
    imported = torch.load(output, weights_only=False)
    assert imported["metadata"]["model_schema_id"] == "avm-model-v2"
    assert imported["metadata"]["parent_sha256"] == file_sha256(source)
    assert imported["recurrent_state"] is None
    assert imported["train_state"]["update"] == 0
    assert not imported["optimizer"]["state"]
    assert not any(key.startswith("action_head.") for key in imported["model"])
    initialize_exact_weights(StreamingLatentLoop(smoke_config.model), output)
    with pytest.raises(ValueError, match="new .pt"):
        migrate_latentloop_weights(smoke_config, source, output)


def test_live_conversation_closes_sessions_on_inference_failure() -> None:
    from harness.conversation import converse

    closed = []

    class Model:
        def identity(self):
            return {"protocol_version": "avm-realtime-v1"}

        def open_session(self, session_id):
            return session_id

        def infer(self, observation):
            raise RuntimeError("inference failed")

        def close_session(self, session_id):
            closed.append(("model", session_id))

    class Harness:
        def start_lifetime_session(self, *args):
            return object()

        def close(self, session_id):
            closed.append(("harness", session_id))

    with pytest.raises(RuntimeError, match="inference failed"):
        converse(Harness(), Model(), scenario_id="camera", session_id="life")
    assert closed == [("harness", "life"), ("model", "life")]
