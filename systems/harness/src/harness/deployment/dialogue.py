"""Persistent FFmpeg/PulseAudio adapters for camera, video and shared-screen dialogue."""

from __future__ import annotations

import os
import queue
import select
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contracts import ActuationSignal, MicSignal, ObservationSignal, VideoSignal

from harness.environment.dialogue import (
    BoundedPlayback,
    DialogueBackend,
    VideoTimeline,
    enqueue_observation,
)


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _read_exact(process: subprocess.Popen[bytes], size: int) -> bytes:
    assert process.stdout is not None
    result = bytearray()
    while len(result) < size:
        block = process.stdout.read(size - len(result))
        if not block:
            raise RuntimeError("audiovisual source disconnected")
        result.extend(block)
    return bytes(result)


class FFmpegSensor:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self._processes: list[subprocess.Popen[bytes]] = []
        self._threads: list[threading.Thread] = []
        self._closed = threading.Event()
        self._error: BaseException | None = None
        self._video_ready = threading.Event()
        self._video = VideoTimeline(int(config.get("video_max_age_ms", 1000)))
        backlog = int(config.get("capture_backlog_units", 3))
        if backlog < 1:
            raise ValueError("capture_backlog_units must be positive")
        self._observations: queue.Queue[ObservationSignal] = queue.Queue(
            maxsize=backlog
        )
        self.timeout = float(config.get("device_timeout_s", 5))
        if self.timeout <= 0:
            raise ValueError("device_timeout_s must be positive")

    def _spawn(self, args: list[str]) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._processes.append(process)
        return process

    def start(self, session_id: str, scenario_id: str, seed: int) -> None:
        scenarios = self.config.get("scenarios", {})
        if scenario_id not in scenarios:
            raise ValueError("dialogue scenario is not configured")
        scenario = scenarios[scenario_id]
        source = str(scenario["video_source"])
        kind = str(scenario["video_kind"])
        ffmpeg = str(self.config.get("ffmpeg_binary", "ffmpeg"))
        if not shutil.which(ffmpeg):
            raise FileNotFoundError("FFmpeg is required for realtime audiovisual capture")
        args = [ffmpeg, "-nostdin", "-loglevel", "error"]
        if kind == "camera":
            args += ["-f", "v4l2", "-i", source]
        elif kind == "screen":
            args += ["-f", "x11grab", "-i", source]
        elif kind == "video":
            args += ["-re", "-i", str(Path(source).expanduser()) if "://" not in source else source]
        else:
            raise ValueError("video_kind must be camera, video or screen")
        video = self._spawn(args + ["-an", "-vf", "scale=224:224", "-pix_fmt", "rgb24",
                                    "-f", "rawvideo", "pipe:1"])
        self._origin = time.monotonic()

        def capture_video() -> None:
            revision = 0
            try:
                while not self._closed.is_set():
                    image = _read_exact(video, 224 * 224 * 3)
                    self._video.append(VideoSignal(image, 224, 224, self._now(), revision))
                    revision += 1
                    self._video_ready.set()
            except BaseException as error:
                if not self._closed.is_set():
                    self._error = error
                self._video_ready.set()

        thread = threading.Thread(target=capture_video, daemon=True)
        self._threads.append(thread)
        thread.start()
        if not self._video_ready.wait(self.timeout):
            raise RuntimeError("initial video frame timed out")
        self._check()
        audio_source = str(self.config.get("mixed_audio_source", "")).strip()
        if not audio_source:
            raise ValueError("mixed_audio_source must include peer audio and playback feedback")
        audio = self._spawn([ffmpeg, "-nostdin", "-loglevel", "error", "-f", "pulse",
                             "-i", audio_source, "-ar", "24000", "-ac", "1",
                             "-f", "f32le", "pipe:1"])

        def capture_audio() -> None:
            unit = 0
            anchor: int | None = None
            try:
                while not self._closed.is_set():
                    pcm = _read_exact(audio, 1920 * 4)
                    if anchor is None:
                        anchor = self._now()
                    timestamp = anchor + unit * 80
                    delay = (timestamp - self._now()) / 1000
                    if delay > 0 and self._closed.wait(delay):
                        return
                    observation = ObservationSignal(
                        session_id, unit, timestamp, 80, MicSignal(pcm), self._video.at(timestamp)
                    )
                    enqueue_observation(self._observations, observation)
                    unit += 1
            except BaseException as error:
                if not self._closed.is_set():
                    self._error = error

        thread = threading.Thread(target=capture_audio, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _now(self) -> int:
        return int((time.monotonic() - self._origin) * 1000)

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("realtime audiovisual capture failed") from self._error

    def capture(self, session_id: str, unit_index: int) -> ObservationSignal:
        self._check()
        try:
            observation = self._observations.get(timeout=self.timeout)
        except queue.Empty as error:
            self._check()
            raise RuntimeError("realtime audio capture timed out") from error
        self._check()
        if observation.session_id != session_id or observation.unit_index != unit_index:
            raise ValueError("capture cursor does not match dialogue session")
        return observation

    def close(self) -> None:
        self._closed.set()
        for process in self._processes:
            _stop(process)
        for thread in self._threads:
            thread.join(timeout=2)


class PulsePlayback:
    def __init__(self, sink: str, binary: str = "pacat") -> None:
        if not sink or not shutil.which(binary):
            raise ValueError("PulseAudio playback sink and pacat binary are required")
        self.sink = sink
        self.binary = binary
        self.process: subprocess.Popen[bytes] | None = None

    def play(self, output: ActuationSignal) -> None:
        if output.speech.encoding != "pcm_f32le":
            raise ValueError("Pulse playback requires float32 speech")
        if self.process is None:
            self.process = subprocess.Popen(
                [self.binary, "--playback", "--raw", "--format=float32le", "--rate=24000",
                 "--channels=1", "--latency-msec=80", f"--device={self.sink}"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            assert self.process.stdin is not None
            os.set_blocking(self.process.stdin.fileno(), False)
        if self.process.poll() is not None:
            raise RuntimeError("PulseAudio playback disconnected")
        assert self.process.stdin is not None
        descriptor = self.process.stdin.fileno()
        pending = memoryview(output.speech.pcm)
        deadline = time.monotonic() + 0.16
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                raise RuntimeError("audio device exceeded the playback latency budget")
            try:
                pending = pending[os.write(descriptor, pending):]
            except BlockingIOError:
                continue

    def interrupt(self) -> None:
        process, self.process = self.process, None
        _stop(process)

    def close(self) -> None:
        self.interrupt()


def create_backend(config: Mapping[str, Any]) -> DialogueBackend:
    if config.get("protocol_version") != "avm-realtime-v1":
        raise ValueError("Harness requires the AVM physical protocol")
    if not config.get("scenarios") or not config.get("mixed_audio_source"):
        raise ValueError("real dialogue scenarios and mixed_audio_source must be configured")
    sink = PulsePlayback(str(config.get("playback_sink", "")))
    return DialogueBackend(
        FFmpegSensor(config), BoundedPlayback(sink, int(config.get("playback_queue_units", 2))),
        str(config["environment_id"]), str(config["environment_version"]),
    )
