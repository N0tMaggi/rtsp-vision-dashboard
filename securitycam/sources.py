"""Video sources: uploaded files and URL streams as virtual cameras.

A virtual camera is an ordinary ``state.CameraRuntime`` (``source_kind`` =
"file" or "url") registered in ``state.camera_states``, so the whole pipeline
(YOLO worker, ESP overlay, targets/lock, zones, alarms, recognition, heatmap,
snapshots, grid) runs on it unchanged. The only hook is in
``streaming._acquire_feed``: when ``camera_state.source`` is set, the reader
comes from ``source.open_stream()`` instead of an RTSP ``VideoStream``.

File playback is paced to the clip's native FPS (wall clock; frames are
dropped when decoding falls behind), loops at the end (setting) and can be
paused, seeked, stepped and sped up. Pausing keeps re-publishing the last
frame at a low rate so the ESP still renders and the detector still sees the
picture. Loops and seeks are cuts: per-track state (motion model, tracer
history, tracker, re-entry ids) is reset so ids and tracers don't jump across.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import cv2

from . import config, state
from .streaming import VideoStream

SPEED_MIN = 0.25
SPEED_MAX = 4.0
PAUSED_REPUBLISH_S = 0.1        # paused / ended: re-publish the still frame at ~10 fps
SEEK_INSTEAD_OF_DROP_S = 2.0    # further behind than this -> seek instead of grabbing frames
CUT_THRESHOLD_S = 1.0           # seeks farther than this reset per-track state

URL_SCHEMES = ("http", "https", "rtsp")


class SourceError(Exception):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------- track-state reset
def reset_track_state(camera_state) -> None:
    """Forget everything keyed by track ids / positions (called at a playback cut)."""
    camera_state.source_epoch += 1          # in-flight inference of the old position is dropped
    lock = getattr(camera_state, "esp_lock", None) or threading.RLock()
    with lock:
        camera_state.track_history.clear()
        camera_state.esp_motion = None
        camera_state.esp_last_results = None
        camera_state.results_meta = None
        camera_state.esp_locked = None
        camera_state.last_faces = []
        camera_state.last_hands = []
        stats = getattr(camera_state, "esp_trackstats", None)
        if isinstance(stats, dict):
            stats.clear()
    with camera_state.frame_lock:
        camera_state.latest_results = None
        camera_state.latest_results_ts = None
    # A changed tracker config makes recognition.track_kwargs() rebuild the
    # tracker (fresh Kalman state, new ids) and clear the re-entry memory.
    try:
        from .recognition.common import peek_runtime

        rt = peek_runtime(camera_state.camera_id)
        if rt is not None:
            rt.tracker_cfg = None
    except Exception:
        pass
    model = camera_state.yolo_model
    predictor = getattr(model, "predictor", None) if model is not None else None
    if predictor is not None and hasattr(predictor, "trackers"):
        try:
            predictor.trackers = []
        except Exception:
            pass


# ---------------------------------------------------------------------------- playback control
class Playback:
    """Playback state of one file source. HTTP threads change it, the reader applies it.

    Positions are frame indices; ``frame`` is the index of the frame currently shown.
    """

    def __init__(self, fps: float, frame_count: int, loop: bool = True):
        self.cond = threading.Condition()
        self.fps = fps if fps and fps > 0 else 25.0
        self.frame_count = max(0, int(frame_count))
        self.loop = bool(loop)
        self.playing = True
        self.ended = False
        self.speed = 1.0
        self.frame = 0
        self.cuts = 0                    # number of loop/seek cuts (diagnostics / tests)
        self.pending: tuple | None = None    # ("seek", frame) | ("step", n)

    # -- derived
    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.frame_count else 0.0

    def _clamp(self, frame: int) -> int:
        top = self.frame_count - 1 if self.frame_count else frame
        return max(0, min(int(frame), max(0, top)))

    def payload(self) -> dict[str, Any]:
        with self.cond:
            return {
                "kind": "file",
                "seekable": True,
                "playing": self.playing,
                "ended": self.ended,
                "speed": self.speed,
                "loop": self.loop,
                "fps": round(self.fps, 3),
                "frame": self.frame,
                "frames": self.frame_count,
                "position": round(self.frame / self.fps, 3),
                "duration": round(self.duration, 3),
                "server_time": time.time(),
            }

    # -- commands (any thread)
    def _notify(self) -> None:
        self.cond.notify_all()

    def play(self) -> None:
        with self.cond:
            if self.ended:
                self.pending = ("seek", 0)
                self.frame = 0
                self.ended = False
            self.playing = True
            self._notify()

    def pause(self) -> None:
        with self.cond:
            self.playing = False
            self._notify()

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek_frame(self, frame: int) -> None:
        with self.cond:
            target = self._clamp(frame)
            self.pending = ("seek", target)
            self.frame = target
            self.ended = False
            self._notify()

    def seek_seconds(self, seconds: float) -> None:
        self.seek_frame(round(float(seconds) * self.fps))

    def seek_fraction(self, fraction: float) -> None:
        f = max(0.0, min(1.0, float(fraction)))
        self.seek_frame(round(f * max(0, self.frame_count - 1)))

    def step(self, frames: int) -> None:
        with self.cond:
            self.playing = False
            n = max(-100, min(100, int(frames)))
            if n:
                if self.pending and self.pending[0] == "step":
                    n += self.pending[1]
                self.pending = ("step", n)
            self._notify()

    def restart(self) -> None:
        with self.cond:
            self.pending = ("seek", 0)
            self.frame = 0
            self.ended = False
            self.playing = True
            self._notify()

    def set_speed(self, speed: float) -> None:
        with self.cond:
            self.speed = max(SPEED_MIN, min(SPEED_MAX, float(speed)))
            self._notify()

    def set_loop(self, loop: bool) -> None:
        with self.cond:
            self.loop = bool(loop)
            self._notify()


# ---------------------------------------------------------------------------- readers
def _base_stream_attrs(stream, camera_state) -> None:
    stream.camera_state = camera_state
    stream.lock = threading.Lock()
    stream.cond = threading.Condition(stream.lock)
    stream.grabbed = False
    stream.frame = None
    stream.frame_ts = time.perf_counter()
    stream.seq = 0
    stream.stopped = False
    stream.thread = None


class FileVideoStream(VideoStream):
    """Reader for a local video file, paced to its FPS and driven by ``Playback``.

    Same interface as ``streaming.VideoStream`` (seq / cond / wait_newer / release),
    so ``CameraFeed`` and every client path work unchanged.
    """

    def __init__(self, path: str, camera_state, playback: Playback,
                 clock: Callable[[], float] = time.perf_counter):
        _base_stream_attrs(self, camera_state)
        self.playback = playback
        self.clock = clock
        self.published = 0                # frames published while playing (pacing diagnostics)
        self.stream = cv2.VideoCapture(path)
        if not self.stream.isOpened():
            camera_state.stream_error = "Could not open the video file."
            return
        with playback.cond:
            start = playback.frame
            playback.pending = None
        if start > 0:
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, frame = self.stream.read()
        if not ok and start > 0:            # stale position beyond the end -> from the start
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start = 0
            ok, frame = self.stream.read()
        if not ok:
            camera_state.stream_error = "The video file has no readable frames."
            return
        self.cur = start
        with playback.cond:
            playback.frame = start
        reset_track_state(camera_state)     # (re)opened: nothing carries over from the last session
        self._emit(frame)
        self._rebase()

    # -- helpers (reader thread only, except during __init__)
    def _emit(self, frame) -> None:
        ts = self.clock()
        with self.cond:
            self.grabbed = True
            self.frame = frame
            self.frame_ts = ts
            self.seq += 1
            self.cond.notify_all()
        self._publish(frame, ts)

    def _rebase(self) -> None:
        self.origin_wall = self.clock()
        self.origin_frame = self.cur

    def _read_at(self, target: int):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
        ok, frame = self.stream.read()
        return ok, frame

    def _cut(self) -> None:
        self.playback.cuts += 1
        reset_track_state(self.camera_state)

    def _apply_pending(self, pending) -> None:
        kind, value = pending
        if kind == "seek":
            target = int(value)
            if target == self.cur + 1:
                ok, frame = self.stream.read()
            else:
                ok, frame = self._read_at(target)
            if not ok:
                return
            if abs(target - self.cur) > CUT_THRESHOLD_S * self.playback.fps or target < self.cur - 2:
                self._cut()
            self.cur = target
        else:  # step
            n = int(value)
            if n == 1:
                ok, frame = self.stream.read()
                target = self.cur + 1
            else:
                target = max(0, self.cur + n)
                if self.playback.frame_count:
                    target = min(target, self.playback.frame_count - 1)
                ok, frame = self._read_at(target)
            if not ok:
                return
            self.cur = target
        with self.playback.cond:
            self.playback.frame = self.cur
            self.playback.ended = False
        self._emit(frame)
        self._rebase()

    def _end_of_file(self) -> None:
        pb = self.playback
        with pb.cond:
            loop = pb.loop
            if self.cur + 1 > pb.frame_count:
                pb.frame_count = self.cur + 1       # metadata under-reported the length
        if loop:
            ok, frame = self._read_at(0)
            if ok:
                self._cut()
                self.cur = 0
                with pb.cond:
                    pb.frame = 0
                self._emit(frame)
                self._rebase()
                return
        with pb.cond:
            pb.playing = False
            pb.ended = True

    def update(self):
        pb = self.playback
        last_emit = self.clock()
        was_playing = True
        speed = pb.speed
        while not self.stopped:
            with pb.cond:
                pending, pb.pending = pb.pending, None
                playing = pb.playing
                if pb.speed != speed or playing != was_playing:
                    speed = pb.speed
                    self._rebase()              # speed change / resume: continue from here
                was_playing = playing
            if pending is not None:
                self._apply_pending(pending)
                last_emit = self.clock()
                continue

            if not playing:
                # Paused / ended: keep the still frame alive (ESP renders, detector sees it).
                now = self.clock()
                if now - last_emit >= PAUSED_REPUBLISH_S and self.frame is not None:
                    self._emit(self.frame.copy())
                    last_emit = now
                with pb.cond:
                    if pb.pending is None and not self.stopped:
                        pb.cond.wait(PAUSED_REPUBLISH_S)
                continue

            now = self.clock()
            due = self.origin_frame + (now - self.origin_wall) * pb.fps * speed
            nxt = self.cur + 1
            if nxt > due:
                wait = (nxt - due) / (pb.fps * speed)
                with pb.cond:
                    if pb.pending is None and pb.playing and not self.stopped:
                        pb.cond.wait(min(wait, 0.05))
                continue

            behind = int(due) - nxt                 # whole frames we are late
            if behind > SEEK_INSTEAD_OF_DROP_S * pb.fps:
                target = int(due)
                if pb.frame_count and target >= pb.frame_count:
                    self.cur = pb.frame_count - 1
                    self._end_of_file()
                    continue
                ok, frame = self._read_at(target)
                nxt = target
            else:
                ok = True
                for _ in range(behind):             # drop late frames without decoding them
                    if not self.stream.grab():
                        ok = False
                        break
                    nxt += 1
                frame = None
                if ok:
                    ok, frame = self.stream.read()
            if not ok:
                self.cur = max(self.cur, nxt - 1)
                self._end_of_file()
                last_emit = self.clock()
                continue
            self.cur = nxt
            with pb.cond:
                pb.frame = nxt
            self._emit(frame)
            self.published += 1
            last_emit = self.clock()


class UrlVideoStream(VideoStream):
    """Reader for a URL source (HTTP file, MJPEG, HLS, RTSP) - a live source, no seeking.

    Finite remote files (frame count known) are paced to their FPS like local
    files; live streams are read as fast as they arrive.
    """

    def __init__(self, url: str, camera_state, clock: Callable[[], float] = time.perf_counter):
        _base_stream_attrs(self, camera_state)
        self.clock = clock
        self.stream = cv2.VideoCapture(url)
        try:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.stream.isOpened():
            camera_state.stream_error = "Could not open the stream URL."
            return
        fps = self.stream.get(cv2.CAP_PROP_FPS) or 0.0
        frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        self.pace_fps = fps if 0 < fps <= 240 and frames > 0 else 0.0
        ok, frame = self.stream.read()
        if not ok:
            camera_state.stream_error = "The stream URL delivered no frames."
            return
        self.grabbed, self.frame, self.frame_ts, self.seq = True, frame, self.clock(), 1
        self._publish(frame, self.frame_ts)

    def update(self):
        next_due = self.clock()
        failures = 0
        while not self.stopped:
            if self.pace_fps:
                next_due += 1.0 / self.pace_fps
                delay = next_due - self.clock()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_due = self.clock()
            grabbed, frame = self.stream.read()
            ts = self.clock()
            if grabbed:
                failures = 0
                with self.cond:
                    self.grabbed, self.frame, self.frame_ts = True, frame, ts
                    self.seq += 1
                    self.cond.notify_all()
                self._publish(frame, ts)
                continue
            failures += 1
            if failures >= 50:                    # ~5 s without frames -> report, clients reconnect
                with self.cond:
                    self.grabbed = False
                    self.cond.notify_all()
                self.camera_state.stream_error = "The stream URL stopped delivering frames."
                return
            time.sleep(0.1)


# ---------------------------------------------------------------------------- sources & registry
@dataclass
class VideoSource:
    kind: str                     # "file" | "url"
    camera_id: str
    label: str
    path: str = ""                # file sources
    url: str = ""                 # url sources (never sent to the browser with credentials)
    video_id: str = ""
    playback: Playback | None = None
    opened_at: float = 0.0

    def open_stream(self, camera_state):
        if self.kind == "file":
            return FileVideoStream(self.path, camera_state, self.playback)
        return UrlVideoStream(self.url, camera_state)

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "camera_id": self.camera_id,
            "kind": self.kind,
            "label": self.label,
            "video_id": self.video_id or None,
            "opened_at": self.opened_at,
        }
        if self.kind == "url":
            out["url"] = redact_url(self.url)
        return out

    def playback_payload(self) -> dict[str, Any]:
        if self.playback is not None:
            return self.playback.payload()
        return {"kind": "url", "seekable": False, "playing": True, "ended": False, "speed": 1.0,
                "loop": False, "fps": 0.0, "frame": 0, "frames": 0, "position": 0.0,
                "duration": 0.0, "server_time": time.time()}


_lock = threading.RLock()
_url_seq = itertools.count(1)


def redact_url(url: str) -> str:
    """Strip user:password from a URL for display / logs."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username:
        netloc = f"{parts.username}:***@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def validate_url(raw: Any) -> str:
    if not isinstance(raw, str):
        raise SourceError("invalid_url")
    url = raw.strip()
    if not url or len(url) > 2048 or any(ord(c) < 32 or c.isspace() for c in url):
        raise SourceError("invalid_url")
    try:
        parts = urlsplit(url)
        _ = parts.port            # raises on a malformed port
    except ValueError:
        raise SourceError("invalid_url") from None
    if parts.scheme.lower() not in URL_SCHEMES:
        raise SourceError("unsupported_scheme")
    if not parts.hostname:
        raise SourceError("invalid_url")
    return url


def _copy_settings(cs: state.CameraRuntime, copy_from: str | None) -> None:
    src = state.get_camera_state(copy_from) if copy_from else None
    if src is None:
        cs.yolo_enabled = True        # video analysis is for testing detection: AI on by default
        return
    cs.esp_settings.clear()
    cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
    cs.esp_settings.update(src.esp_settings)
    cs.yolo_enabled = bool(src.yolo_enabled)
    cs.yolo_model_name = src.yolo_model_name


def _register(source: VideoSource, ip: str, stream_path: str, rtsp_url: str, copy_from: str | None) -> state.CameraRuntime:
    cs = state.CameraRuntime(
        camera_id=source.camera_id,
        label=source.label,
        ip=ip,
        stream_path=stream_path,
        rtsp_url=rtsp_url,
    )
    cs.source_kind = source.kind
    cs.source = source
    _copy_settings(cs, copy_from)
    source.opened_at = time.time()
    state.camera_states[source.camera_id] = cs
    return cs


def find_by_video(video_id: str) -> state.CameraRuntime | None:
    for cs in state.list_camera_states():
        src = cs.source
        if src is not None and src.kind == "file" and src.video_id == video_id:
            return cs
    return None


def open_file(video_id: str, path: str, label: str, fps: float, frame_count: int,
              loop: bool = True, copy_from: str | None = None) -> state.CameraRuntime:
    """Open a library video as virtual camera ``video-<id>`` (idempotent)."""
    with _lock:
        existing = find_by_video(video_id)
        if existing is not None:
            return existing
        source = VideoSource(kind="file", camera_id=f"video-{video_id}", label=label, path=path,
                             video_id=video_id, playback=Playback(fps, frame_count, loop))
        # rtsp_url only marks "has a stream" for the shared code paths; it is never opened.
        return _register(source, "", label, f"video://{video_id}", copy_from)


def open_url(raw_url: Any, label: Any = None, copy_from: str | None = None) -> state.CameraRuntime:
    url = validate_url(raw_url)
    with _lock:
        for cs in state.list_camera_states():
            if cs.source is not None and cs.source.kind == "url" and cs.source.url == url:
                return cs
        host = urlsplit(url).hostname or "stream"
        name = label.strip()[:60] if isinstance(label, str) and label.strip() else host
        source = VideoSource(kind="url", camera_id=f"url-{next(_url_seq)}", label=name, url=url)
        return _register(source, host, redact_url(url), url, copy_from)


def list_sources() -> list[dict[str, Any]]:
    return [cs.source.payload() for cs in state.list_camera_states() if cs.source is not None]


def get_source(camera_id: str) -> VideoSource | None:
    cs = state.get_camera_state(camera_id)
    return cs.source if cs is not None else None


def _drop_runtime_caches(camera_id: str) -> None:
    """Free per-camera caches of other packages (pre-roll ring, alarm engine, recognition)."""
    try:
        from .alerts import clips

        with clips._rings_lock:
            clips._rings.pop(camera_id, None)
    except Exception:
        pass
    try:
        from .alerts import engine

        getattr(engine, "_engines", {}).pop(camera_id, None)
    except Exception:
        pass
    try:
        from .recognition import common

        with common._runtimes_lock:
            common._runtimes.pop(camera_id, None)
    except Exception:
        pass
    try:
        from . import scaled_feed

        with scaled_feed._cache_lock:
            for key in [k for k in scaled_feed._cache if k[0] == camera_id]:
                scaled_feed._cache.pop(key, None)
    except Exception:
        pass


def close(camera_id: str) -> bool:
    """Close a virtual camera: stop its reader and worker, forget its runtime state."""
    from . import streaming

    with _lock:
        cs = state.get_camera_state(camera_id)
        if cs is None or cs.source is None:
            return False
        state.camera_states.pop(camera_id, None)
    cs.closed = True
    cs.yolo_enabled = False
    with streaming._feeds_lock:
        feed = streaming._feeds.pop(camera_id, None)
        streaming._connect_failures.pop(camera_id, None)
    if feed is not None:
        feed.stream.release()          # MJPEG clients see the stream stop and disconnect
    with cs.frame_lock:
        cs.latest_frame = None
        cs.latest_results = None
    _drop_runtime_caches(camera_id)
    cs.yolo_model = None
    return True


def close_all() -> None:
    for cs in list(state.list_camera_states()):
        if cs.source is not None:
            close(cs.camera_id)
