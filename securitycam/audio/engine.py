"""Per-camera Sound ESP engine + the supervisor that starts and stops it.

A camera's engine exists only while its ``sound`` setting is on AND its video
is on screen (an MJPEG client holds the shared feed). The supervisor thread
checks that every ``TICK_S`` and owns all process starts/stops, so request
threads never wait for ffmpeg.

RTSP cameras: ffmpeg only starts once the video feed is connected (the camera
accepted the credentials), never while the video connect is backing off, and
with its own backoff (5 s doubling to 60 s, >= 30 s after an auth error, 5 min
after "no audio track"), so a wrong password can't lock the Tapo account.

Video files: ffmpeg plays the file's audio from the current playback position
at the playback speed; pause stops it, seek / loop / speed changes or drift
over ``FILE_MAX_DRIFT_S`` restart it at the new position. URL sources are not
supported (their address passed the SSRF preflight for OpenCV only).
"""
from __future__ import annotations

import atexit
import base64
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import capture, classes
from .classifier import CLASSIFIER, WINDOW_SAMPLES
from .levels import (CHUNK_SAMPLES, SAMPLE_RATE, LoudDetector, MelSpectrum, Meter, NoiseFloor,
                     peak_dbfs, rms_dbfs, to_float)
from .rules import RULES, emit

TICK_S = 0.25
CLASSIFY_HOP = SAMPLE_RATE // 2          # one inference per 0.5 s of audio
COLS_KEPT = 6                            # spectrogram columns per SSE payload (seq-tagged)
LABELS_STALE_S = 3.0
EVENTS_KEPT = 8
EVENT_SHOW_S = 30.0                      # radar pings / HUD list keep events this long
NOTABLE_MIN = 0.45
BACKOFF_MIN_S = 5.0
BACKOFF_MAX_S = 60.0
AUTH_BACKOFF_S = 30.0
NO_AUDIO_RETRY_S = 300.0
STABLE_RUN_S = 30.0
FILE_MAX_DRIFT_S = 1.5
FLOOR_FREEZE_MS = 10000.0

MESSAGES = {
    "idle": "Starts when the camera is on screen.",
    "starting": "Connecting to the audio track…",
    "live": "",
    "no_audio": "No audio track on this camera.",
    "no_audio_file": "This video has no audio track.",
    "error": "Audio stream failed - retrying.",
    "auth": "The camera refused the audio stream (login) - retrying later.",
    "waiting_video": "Waiting for the video connection.",
    "paused": "Paused.",
    "unsupported": "Sound is not available for URL sources.",
    "no_ffmpeg": "ffmpeg was not found on PATH.",
}


class SoundEngine:
    def __init__(self, cs, popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic, classifier=None):
        self.cs = cs
        self.camera_id = cs.camera_id
        self.popen = popen
        self.clock = clock
        self.classifier = classifier if classifier is not None else CLASSIFIER
        self.lock = threading.Lock()
        self.meter = Meter()
        self.floor = NoiseFloor()
        self.loud = LoudDetector()
        self.mel = MelSpectrum()
        self.ring = np.zeros(WINDOW_SAMPLES, np.float32)
        self.ring_fill = 0
        self.since_classify = 0
        self.cols: deque = deque(maxlen=COLS_KEPT)
        self.seq = 0
        self.level = self.peak = -96.0
        self.labels: List[Dict[str, Any]] = []
        self.labels_at = 0.0
        self.heard: Optional[tuple] = None
        self.prev_scores: Dict[str, float] = {}
        self.events: deque = deque(maxlen=EVENTS_KEPT)
        self.event_seq = 0
        self.state = "idle"
        self.reader: Optional[capture.FfmpegReader] = None
        self.failures = 0
        self.retry_at = 0.0
        # file sync
        self.file_start = 0.0
        self.file_speed = 1.0
        self.file_cuts = -1
        self.file_samples = 0
        self.closed = False

    # ------------------------------------------------------------------ analysis (reader thread)
    def feed(self, chunk: bytes) -> None:
        x = to_float(chunk)
        cfg = RULES.config()
        loud_cfg = cfg["loud"]
        self.loud.threshold_db = float(loud_cfg["db_above_floor"])
        self.loud.min_ms = float(loud_cfg["min_ms"])
        lvl = rms_dbfs(x)
        pk = peak_dbfs(x)
        col = self.mel.column(x)
        with self.lock:
            self.level, self.peak = lvl, pk
            self.meter.update(lvl)
            # the floor ignores a loud episode for its first FLOOR_FREEZE_MS, then slowly accepts it
            floor = self.floor.update(lvl, frozen=self.loud.active and self.loud.above_ms < FLOOR_FREEZE_MS)
            ev = self.loud.update(lvl, floor, 1000.0 * x.size / SAMPLE_RATE)
            self.seq += 1
            self.cols.append(col.tobytes())
            n = x.size
            if n >= WINDOW_SAMPLES:
                self.ring[:] = x[-WINDOW_SAMPLES:]
            else:
                self.ring[:-n] = self.ring[n:]
                self.ring[-n:] = x
            self.ring_fill = min(WINDOW_SAMPLES, self.ring_fill + n)
            self.since_classify += n
            self.file_samples += n
            if self.state == "starting":
                self.state = "live"
            window = None
            if self.ring_fill >= WINDOW_SAMPLES and self.since_classify >= CLASSIFY_HOP:
                self.since_classify = 0
                window = self.ring.copy()
        if ev is not None:
            self._on_loud(ev)
        if window is not None:
            self.classifier.submit(self.camera_id, window, self.on_scores)

    def _add_event(self, kind: str, label: str, score: Optional[float], alarm: bool) -> None:
        with self.lock:
            self.event_seq += 1
            self.events.append({"id": self.event_seq, "ts": round(time.time(), 2), "kind": kind,
                                "label": label, "score": None if score is None else round(score, 2),
                                "alarm": alarm})

    def _on_loud(self, ev) -> None:
        notify = RULES.check_loud(self.camera_id)
        alarm = notify is not None
        self._add_event("loud", "Loud sound", None, alarm)
        if alarm:
            emit(self.cs, "loud", "Loud sound",
                 f"Loud sound ({ev.level_db:.0f} dBFS, {ev.above_db:.0f} dB above background)", notify,
                 {"level_db": round(ev.level_db, 1), "above_db": round(ev.above_db, 1)})

    def on_scores(self, scores: np.ndarray, ms: float, device: str) -> None:
        if self.closed:
            return
        curated = classes.curate(scores)
        disp = float(RULES.config()["display_threshold"])
        top = classes.top(curated, disp, 3)
        heard = classes.top_raw(scores, self.classifier.labels, 1)
        fired = RULES.check_classes(self.camera_id, curated)
        fired_ids = {cid for cid, _, _ in fired}
        with self.lock:
            self.labels = [{"id": cid, "label": classes.BY_ID[cid].label, "icon": classes.BY_ID[cid].icon,
                            "score": round(sc, 2)} for cid, sc in top]
            self.labels_at = self.clock()
            self.heard = heard[0] if heard else None
            prev = self.prev_scores
            self.prev_scores = curated
        for cid, sc, notify in fired:
            cat = classes.BY_ID[cid]
            self._add_event(cid, cat.label, sc, True)
            emit(self.cs, cid, cat.label, f"{cat.label} heard ({sc * 100:.0f}%)", notify,
                 {"score": round(sc, 3)})
        for cid, sc in curated.items():
            cat = classes.BY_ID[cid]
            if cid in fired_ids or not cat.notable:
                continue
            if sc >= max(NOTABLE_MIN, disp) and prev.get(cid, 0.0) < max(NOTABLE_MIN, disp):
                self._add_event(cid, cat.label, sc, False)

    # ------------------------------------------------------------------ lifecycle (supervisor thread)
    def _start(self, cmd: List[str]) -> None:
        with self.lock:
            self.state = "starting"
        self.reader = capture.FfmpegReader(cmd, self.feed, self._on_exit, popen=self.popen,
                                           name=self.camera_id).start()

    def _stop_reader(self) -> None:
        r, self.reader = self.reader, None
        if r is not None:
            r.stop()

    def _on_exit(self, reader: capture.FfmpegReader) -> None:
        if reader.stopped or self.closed:
            return
        kind = capture.classify_exit(reader.stderr_text(), reader.got_audio)
        now = self.clock()
        ran = now - reader.started_at
        with self.lock:
            if kind == "no_audio":
                self.state = "no_audio_file" if self.cs.source_kind == "file" else "no_audio"
                self.retry_at = now + NO_AUDIO_RETRY_S
                return
            if self.cs.source_kind == "file" and reader.got_audio:
                self.state = "paused" if self.state == "paused" else "live"
                self.retry_at = now + 5.0                  # end of file: a loop / seek cut restarts it sooner
                return
            if reader.got_audio and ran >= STABLE_RUN_S:
                self.failures = 0
            self.failures += 1
            delay = min(BACKOFF_MAX_S, BACKOFF_MIN_S * 2 ** (self.failures - 1))
            if kind == "auth":
                delay = max(delay, AUTH_BACKOFF_S)
            self.state = "auth" if kind == "auth" else "error"
            self.retry_at = now + delay
        tail = reader.stderr_text().strip().splitlines()[-1:] or [""]
        print(f"[Sound] Audio for {self.camera_id} ended ({kind}, {self.failures}x), retry in {delay:.0f} s. "
              f"{capture.redact(tail[0])[:160]}")

    def tick(self) -> None:
        """Start / stop / resync ffmpeg. Called by the supervisor only."""
        if self.closed:
            return
        kind = self.cs.source_kind
        if kind == "url":
            self.state = "unsupported"
            return
        ffmpeg = capture.env_ffmpeg()
        if not ffmpeg:
            self.state = "no_ffmpeg"
            return
        if kind == "file":
            self._tick_file(ffmpeg)
        else:
            self._tick_rtsp(ffmpeg)

    def _running(self) -> bool:
        return self.reader is not None and self.reader.exit_code is None and not self.reader.stopped

    def _tick_rtsp(self, ffmpeg: str) -> None:
        if self._running():
            return
        self.reader = None
        now = self.clock()
        if now < self.retry_at:
            return
        from .. import streaming

        if not self.cs.rtsp_url:
            self.state = "error"
            return
        _, video_retry = streaming._connect_failures.get(self.camera_id, (0, 0.0))
        feed = streaming._feeds.get(self.camera_id)
        if video_retry > time.monotonic() or feed is None or not feed.stream.grabbed:
            self.state = "waiting_video"
            return
        print(f"[Sound] Starting audio for {self.camera_id}: {capture.redact(self.cs.rtsp_url)}")
        self._start(capture.rtsp_command(self.cs.rtsp_url, ffmpeg))

    def _tick_file(self, ffmpeg: str) -> None:
        src = self.cs.source
        pb = getattr(src, "playback", None)
        if src is None or pb is None or not getattr(src, "path", ""):
            self.state = "unsupported"
            return
        with pb.cond:
            playing, ended, speed = pb.playing, pb.ended, float(pb.speed or 1.0)
            cuts, pos = pb.cuts, pb.frame / (pb.fps or 25.0)
        if self.state in ("no_audio_file",) and self.clock() < self.retry_at:
            return
        if not playing or ended:
            self._stop_reader()
            self.state = "paused"
            return
        if self._running():
            audio_pos = self.file_start + self.file_samples / SAMPLE_RATE * self.file_speed
            drift = abs(audio_pos - pos)
            if cuts == self.file_cuts and abs(speed - self.file_speed) < 1e-3 and \
                    (drift <= FILE_MAX_DRIFT_S or self.file_samples < SAMPLE_RATE):
                return
            self._stop_reader()
        elif self.clock() < self.retry_at and cuts == self.file_cuts:
            return
        self.reader = None
        self.file_start, self.file_speed, self.file_cuts, self.file_samples = pos, speed, cuts, 0
        self._start(capture.file_command(src.path, pos, speed, ffmpeg))

    def close(self) -> None:
        self.closed = True
        self._stop_reader()
        self.classifier.drop(self.camera_id)

    # ------------------------------------------------------------------ payload
    def payload(self) -> Dict[str, Any]:
        now = self.clock()
        wall = time.time()
        with self.lock:
            labels = self.labels if now - self.labels_at < LABELS_STALE_S else []
            heard = self.heard if now - self.labels_at < LABELS_STALE_S else None
            out = {
                "camera": self.camera_id,
                "on": True,
                "state": self.state,
                "message": MESSAGES.get(self.state, ""),
                "source": self.cs.source_kind,
                "level": round(self.level, 1),
                "peak": round(self.peak, 1),
                "meter": round(self.meter.value, 1),
                "floor": round(self.floor.value, 1) if self.floor.value is not None else None,
                "loud": self.loud.active,
                "seq": self.seq,
                "bands": self.mel.n_bands,
                "rate": SAMPLE_RATE // CHUNK_SAMPLES,
                "cols": base64.b64encode(b"".join(self.cols)).decode("ascii"),
                "labels": labels,
                "heard": {"label": heard[0], "score": round(heard[1], 2)} if heard else None,
                "events": [e for e in self.events if wall - e["ts"] < EVENT_SHOW_S],
            }
        out["classifier"] = self.classifier.info()
        return out


# ----------------------------------------------------------------------------- supervisor
_engines: Dict[str, SoundEngine] = {}
_engines_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
POPEN: Callable[..., subprocess.Popen] = subprocess.Popen        # injectable for tests


def sound_on(cs) -> bool:
    return bool(cs.esp_settings.get("sound", False)) and not cs.closed


def viewed(cs) -> bool:
    from .. import streaming

    feed = streaming._feeds.get(cs.camera_id)
    return feed is not None and feed.clients > 0 and not feed.stream.stopped


def get(camera_id: str) -> Optional[SoundEngine]:
    return _engines.get(camera_id)


def tick_all() -> None:
    """One supervisor pass (also called directly by tests)."""
    from .. import state

    live = set()
    for cs in state.list_camera_states():
        if not (sound_on(cs) and viewed(cs)):
            continue
        live.add(cs.camera_id)
        with _engines_lock:
            eng = _engines.get(cs.camera_id)
            if eng is None or eng.cs is not cs:
                if eng is not None:
                    eng.close()
                eng = _engines[cs.camera_id] = SoundEngine(cs, popen=POPEN)
        try:
            eng.tick()
        except Exception as exc:
            print(f"[Sound] tick for {cs.camera_id} failed: {exc}")
    with _engines_lock:
        doomed = [cid for cid in _engines if cid not in live]
        dropped = [_engines.pop(cid) for cid in doomed]
    for eng in dropped:
        eng.close()


def _loop() -> None:
    while not _stop.wait(TICK_S):
        try:
            tick_all()
        except Exception as exc:
            print(f"[Sound] supervisor error: {exc}")


def ensure_started() -> None:
    global _thread
    with _engines_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, daemon=True, name="SoundSupervisor")
        _thread.start()


def shutdown() -> None:
    """Stop every ffmpeg (atexit and tests)."""
    _stop.set()
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for eng in engines:
        eng.close()


atexit.register(shutdown)


def payload(cs) -> Optional[Dict[str, Any]]:
    """Live payload for the SSE ``sound`` topic: None while sound is off for this camera."""
    if not sound_on(cs):
        return None
    eng = _engines.get(cs.camera_id)
    if eng is None:
        return {"camera": cs.camera_id, "on": True, "state": "idle", "message": MESSAGES["idle"],
                "source": cs.source_kind, "level": -96.0, "peak": -96.0, "meter": -96.0, "floor": None,
                "loud": False, "seq": 0, "bands": 32, "rate": SAMPLE_RATE // CHUNK_SAMPLES, "cols": "",
                "labels": [], "heard": None, "events": [], "classifier": CLASSIFIER.info()}
    return eng.payload()


def wants_buffer(camera_id: str) -> bool:
    """Keep the alarm pre-roll buffer filled while sound alarms can fire on this camera."""
    return camera_id in _engines and RULES.any_enabled()
