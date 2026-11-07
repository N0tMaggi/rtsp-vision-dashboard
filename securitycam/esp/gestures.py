"""Pose pack: gesture control (hold detection, cooldowns, actions).

``GestureMachine`` turns per-observation gesture conditions into one-shot
triggers: a gesture fires once its condition held for ``HOLDS[g]`` seconds
(short dropouts up to ``GAP_S`` are bridged), then not again until the
condition was released for ``RELEASE_S``; on top of that every gesture has a
camera-wide cooldown. So holding a thumbs up for a minute takes one snapshot.

Actions run on a small background worker (PTZ / auto-track calls may take the
ESP lock and talk to the camera); the render path only enqueues.

* ``wave``      (>= 2 s)   -> PTZ auto-track follows that person (only with a PTZ camera)
* ``palm``      (1.5 s)    -> stop auto-track
* ``thumbs_up`` (1 s)      -> snapshot event

Every fired gesture is logged as an event (kind ``gesture``, no notification)
and kept for a few seconds in ``recent`` so the UI can show a confirmation.
"""
from __future__ import annotations

import collections
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

GESTURES = ("wave", "palm", "thumbs_up")
HOLDS = {"wave": 2.0, "palm": 1.5, "thumbs_up": 1.0}
LABELS = {"wave": "Wave", "palm": "Open palm", "thumbs_up": "Thumbs up"}
GAP_S = 0.45                 # dropouts shorter than this do not restart a hold
RELEASE_S = 0.8              # a fired gesture must be released this long before it can fire again
COOLDOWN_S = 6.0             # per gesture, per camera
RECENT_S = 5.0               # how long a fired gesture stays in the targets payload


class GestureMachine:
    def __init__(self, holds: Optional[Dict[str, float]] = None, cooldown_s: float = COOLDOWN_S,
                 gap_s: float = GAP_S, release_s: float = RELEASE_S):
        self.holds = dict(HOLDS if holds is None else holds)
        self.cooldown_s = cooldown_s
        self.gap_s = gap_s
        self.release_s = release_s
        self._hold: Dict[Tuple[Any, str], List[float]] = {}     # (key, g) -> [since, last, fired]
        self._last_fire: Dict[str, float] = {}

    def progress(self, key: Any, gesture: str, t: float) -> float:
        """0..1 of the current hold (for UI feedback)."""
        h = self._hold.get((key, gesture))
        if h is None or h[2] or t - h[1] > self.gap_s:
            return 0.0
        return min(1.0, (t - h[0]) / max(1e-6, self.holds.get(gesture, 1.0)))

    def update(self, t: float, active: Dict[Any, Iterable[str]]) -> List[Tuple[Any, str]]:
        """``active``: track key -> gestures whose condition holds now. Returns fired (key, gesture)."""
        fired: List[Tuple[Any, str]] = []
        for key, gs in active.items():
            for g in gs:
                if g not in self.holds:
                    continue
                h = self._hold.get((key, g))
                if h is None or t - h[1] > (self.release_s if h[2] else self.gap_s):
                    h = self._hold[(key, g)] = [t, t, 0.0]
                h[1] = t
                if h[2]:
                    continue
                if t - h[0] >= self.holds[g] and t - self._last_fire.get(g, -math.inf) >= self.cooldown_s:
                    h[2] = 1.0
                    self._last_fire[g] = t
                    fired.append((key, g))
        # forget released holds (and holds of vanished tracks)
        for k in [k for k, h in self._hold.items() if t - h[1] > max(self.release_s, self.gap_s)]:
            del self._hold[k]
        return fired


# ----------------------------------------------------------------------------- actions
_queue: "queue.Queue[Callable[[], None]]" = queue.Queue(maxsize=32)
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
RUN_ASYNC = True             # tests run actions inline


def _run(job: Callable[[], None]) -> None:
    if not RUN_ASYNC:
        job()
        return
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="PoseGestures", daemon=True)
            _worker.start()
    try:
        _queue.put_nowait(job)
    except queue.Full:
        pass


def _loop() -> None:
    while True:
        job = _queue.get()
        try:
            job()
        except Exception as exc:  # never kill the worker
            print(f"[Gestures] action failed: {exc.__class__.__name__}: {exc}")


def _ptz_ready(camera_id: str) -> bool:
    try:
        from .. import ptz
        return bool(ptz.available() and ptz.configured(camera_id))
    except Exception:
        return False


def _autotrack():
    from ..ptz import autotrack
    return autotrack


def plan(camera_state, key: Any, gesture: str, enabled: Set[str]) -> Tuple[str, bool, Optional[Callable[[], None]]]:
    """(message, ok, job) for a fired gesture; ``job`` runs off the render path."""
    cam = getattr(camera_state, "camera_id", "default")
    if gesture not in enabled:
        return f"{LABELS[gesture]} (disabled)", False, None
    if gesture == "wave":
        if not _ptz_ready(cam):
            return "No PTZ camera: auto-track unavailable", False, None
        at = _autotrack()
        if at.enabled(cam) and getattr(camera_state, "esp_locked", None) == key:
            return "Already tracking", True, None

        def job():
            try:
                at.set_target(cam, int(key))
            except Exception as exc:
                print(f"[Gestures] auto-track start failed: {exc.__class__.__name__}")
        return "Auto-track started", True, job
    if gesture == "palm":
        at = _autotrack()
        if not at.enabled(cam):
            return "Auto-track is off", False, None
        return "Auto-track stopped", True, lambda: at.update_config(cam, {"enabled": False})
    if gesture == "thumbs_up":
        return "Snapshot saved", True, None          # the gesture event itself carries the snapshot
    return LABELS.get(gesture, gesture), False, None


def log_event(camera_state, key: Any, gesture: str, message: str) -> bool:
    """Event entry (with snapshot + clip) for a fired gesture; never notifies."""
    try:
        from ..alerts.events import events
    except Exception:
        return False
    cam = getattr(camera_state, "camera_id", "default")
    try:
        tid = int(key)
    except (TypeError, ValueError):
        tid = None
    return events.trigger({
        "camera": cam,
        "camera_label": getattr(camera_state, "label", cam),
        "ts": time.time(),
        "kind": "gesture",
        "source_id": f"gesture-{gesture}",
        "source_label": LABELS.get(gesture, gesture),
        "cls": "person",
        "track_id": tid,
        "message": f"{LABELS.get(gesture, gesture)} by person #{tid}: {message}" if tid is not None else message,
        "notify": False,
        "gesture": gesture,
    })


def fire(camera_state, key: Any, gesture: str, enabled: Set[str], recent: "collections.deque", seq: int) -> dict:
    """Run one fired gesture: plan, queue its action, log the event, remember it for the UI."""
    message, ok, job = plan(camera_state, key, gesture, enabled)
    if job is not None:
        _run(job)
    log_event(camera_state, key, gesture, message)
    rec = {"seq": seq, "id": int(key) if isinstance(key, int) else key, "gesture": gesture,
           "label": LABELS.get(gesture, gesture), "message": message, "ok": ok, "ts": time.time()}
    recent.append(rec)
    return rec


def recent_payload(recent: Optional["collections.deque"], now: Optional[float] = None) -> List[dict]:
    if not recent:
        return []
    now = time.time() if now is None else now
    return [dict(r, age=round(now - r["ts"], 2)) for r in list(recent) if now - r["ts"] <= RECENT_S]
