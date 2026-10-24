"""Continuous audiovisual sessions with independent capture and speech playback."""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, VideoSignal


class VideoTimeline:
    """Select the newest causal frame without synthesizing missing video."""

    def __init__(self, max_age_ms: int = 1000, capacity: int = 64) -> None:
        if max_age_ms < 1 or capacity < 2:
            raise ValueError("invalid video timeline bounds")
        self.max_age_ms = max_age_ms
        self.frames: deque[VideoSignal] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, frame: VideoSignal) -> None:
        with self._lock:
            if self.frames and (
                frame.captured_at_ms < self.frames[-1].captured_at_ms
                or frame.revision <= self.frames[-1].revision
            ):
                raise ValueError("video timestamp or revision moved backwards")
            self.frames.append(frame)

    def at(self, timestamp_ms: int) -> VideoSignal:
        with self._lock:
            frames = [frame for frame in self.frames if frame.captured_at_ms <= timestamp_ms]
            if not frames:
                raise RuntimeError("no initial causal video frame is available")
            selected = frames[-1]
            if timestamp_ms - selected.captured_at_ms > self.max_age_ms:
                raise RuntimeError("video source is stale or disconnected")
            return selected


class PlaybackSink(Protocol):
    def play(self, output: ActuationSignal) -> None: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...


class BoundedPlayback:
    """Playback never blocks capture; silence cancels all pending speech."""

    def __init__(self, sink: PlaybackSink, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("playback capacity must be positive")
        self.sink = sink
        self._pending: deque[ActuationSignal] = deque()
        self._capacity = capacity
        self._condition = threading.Condition()
        self._sink_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, output: ActuationSignal) -> None:
        with self._condition:
            self.check()
            if output.speech.silent:
                self._generation += 1
                self._pending.clear()
                with self._sink_lock:
                    self.sink.interrupt()
            else:
                if len(self._pending) >= self._capacity:
                    raise RuntimeError("speech playback queue exceeded its latency budget")
                self._pending.append(output)
                self._condition.notify()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("speech playback failed") from self._error
        if self._closed:
            raise RuntimeError("speech playback is closed")

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    output = self._pending.popleft()
                    generation = self._generation
                with self._sink_lock:
                    if generation == self._generation:
                        self.sink.play(output)
        except BaseException as error:
            self._error = error

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        self.sink.close()
        self._thread.join(timeout=2)


class DialogueSensor(Protocol):
    def start(self, session_id: str, scenario_id: str, seed: int) -> None: ...
    def capture(self, session_id: str, unit_index: int) -> ObservationSignal: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class DialogueBackend:
    sensor: DialogueSensor
    playback: BoundedPlayback
    environment_id: str = "avm-dialogue"
    environment_version: str = "1"
    session_id: str | None = None
    next_unit: int = 0

    def start_lifetime_session(
        self, scenario_id: str, seed: int, session_id: str
    ) -> ObservationSignal:
        if self.session_id is not None:
            raise ValueError("dialogue lifetime session already started")
        if not scenario_id or not session_id:
            raise ValueError("scenario and session identity are required")
        self.sensor.start(session_id, scenario_id, seed)
        self.session_id = session_id
        self.next_unit = 0
        return self.sensor.capture(session_id, 0)

    def apply(self, output: ActuationSignal) -> tuple[ObservationSignal, EnvironmentReceipt]:
        if output.session_id != self.session_id or output.unit_index != self.next_unit:
            raise ValueError("dialogue output identity is not contiguous")
        self.playback.submit(output)
        observation = self.sensor.capture(output.session_id, self.next_unit + 1)
        self.playback.check()
        self.next_unit += 1
        return observation, EnvironmentReceipt(output.session_id, output.unit_index, True)

    def close(self) -> None:
        try:
            self.sensor.close()
        finally:
            self.playback.close()
            self.session_id = None


def enqueue_observation(
    target: queue.Queue[ObservationSignal], observation: ObservationSignal
) -> None:
    try:
        target.put_nowait(observation)
    except queue.Full as error:
        raise RuntimeError("capture exceeded its realtime backlog budget") from error
