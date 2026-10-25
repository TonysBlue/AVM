from __future__ import annotations

import io
import threading

import numpy as np
import pytest
from contracts import ActuationSignal, MicSignal, ObservationSignal, SpeechSignal, VideoSignal
from harness.deployment.dialogue import FFmpegSensor
from harness.environment.dialogue import BoundedPlayback, DialogueBackend


@pytest.mark.parametrize(
    "kind,source,expected",
    [
        ("camera", "/dev/video0", "v4l2"),
        ("screen", ":0", "x11grab"),
        ("video", "/tmp/scene.mp4", "-re"),
    ],
)
def test_persistent_capture_uses_configured_source_and_cleans_up(
    monkeypatch, kind, source, expected
) -> None:
    monkeypatch.setattr("harness.deployment.dialogue.shutil.which", lambda value: value)
    sensor = FFmpegSensor(
        {
            "scenarios": {"talk": {"video_kind": kind, "video_source": source}},
            "mixed_audio_source": "test-mix",
            "device_timeout_s": 1,
        }
    )
    processes = []
    commands = []

    class Pipe:
        def __init__(self, data):
            self.data = io.BytesIO(data)
            self.done = False
            self.closed = threading.Event()

        def read(self, size):
            value = self.data.read(size)
            if value:
                return value
            assert self.closed.wait(2)
            return b""

        def readline(self, size):
            value = self.data.readline(size)
            if value:
                return value
            assert self.closed.wait(2)
            return b""

    class Process:
        def __init__(self, data):
            self.stdout = Pipe(data)
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True
            self.stdout.closed.set()

        def wait(self, timeout):
            assert self.stopped

    def spawn(args):
        commands.append(args)
        data = (
            b"P6\n224 224\n255\n" + b"x" * (224 * 224 * 3)
            if "image2pipe" in args
            else np.zeros(3200, dtype=np.float32).tobytes()
        )
        process = Process(data)
        processes.append(process)
        sensor._processes.append(process)
        return process

    monkeypatch.setattr(sensor, "_spawn", spawn)
    try:
        sensor.start("life", "talk", 17)
        observed = sensor.capture("life", 0)
        assert observed.video.revision == 0
        assert observed.video.captured_at_ms <= observed.timestamp_ms
        assert expected in commands[0]
        assert source in commands[0]
        assert "test-mix" in commands[1]
        assert len(processes) == 2
    finally:
        sensor.close()
    assert all(process.stopped for process in processes)
    assert all(not thread.is_alive() for thread in sensor._threads)


def test_backend_captures_next_unit_while_speech_is_playing() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Sink:
        def play(self, output):
            entered.set()
            assert release.wait(2)

        def interrupt(self):
            release.set()

        def close(self):
            release.set()

    class Sensor:
        def start(self, *args):
            return None

        def capture(self, session_id, unit_index):
            return ObservationSignal(
                session_id,
                unit_index,
                unit_index * 200,
                200,
                MicSignal(np.zeros(3200, dtype=np.float32).tobytes()),
                VideoSignal(b"rgb", 1, 1, unit_index * 200, unit_index),
            )

        def close(self):
            return None

    backend = DialogueBackend(Sensor(), BoundedPlayback(Sink()))
    pcm = np.zeros(4800, dtype=np.float32).tobytes()
    try:
        backend.start_lifetime_session("camera", 17, "life")
        first, receipt = backend.apply(ActuationSignal("life", 0, SpeechSignal(pcm)))
        assert entered.wait(2)
        assert first.unit_index == 1 and receipt.accepted
        second, _ = backend.apply(ActuationSignal("life", 1, SpeechSignal(pcm)))
        assert second.unit_index == 2
        assert not release.is_set()
    finally:
        backend.close()


def test_end_flush_waits_for_space_without_dropping_tail() -> None:
    entered, release = threading.Event(), threading.Event()
    played = []

    class Sink:
        def play(self, output):
            if output.unit_index == 0:
                entered.set()
                assert release.wait(2)
            played.append(output.unit_index)

        def interrupt(self):
            release.set()

        def close(self):
            release.set()

    playback = BoundedPlayback(Sink(), capacity=1)
    pcm = SpeechSignal(np.zeros(4800, dtype=np.float32).tobytes())
    try:
        playback.submit(ActuationSignal("s", 0, pcm))
        assert entered.wait(2)
        playback.submit(ActuationSignal("s", 1, pcm))
        completed = threading.Event()

        def flush():
            playback.submit(ActuationSignal("s", 2, pcm), wait_for_space=True)
            completed.set()

        thread = threading.Thread(target=flush)
        thread.start()
        assert not completed.wait(0.02)
        release.set()
        thread.join(timeout=2)
        assert completed.is_set()
        playback.drain()
        assert played == [0, 1, 2]
    finally:
        playback.close()


def test_capture_cannot_use_unbounded_queue() -> None:
    with pytest.raises(ValueError, match="positive"):
        FFmpegSensor({"capture_backlog_units": 0})
