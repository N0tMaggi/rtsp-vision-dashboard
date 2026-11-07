"""Server-Sent Events push channel: one ``GET /api/stream`` per dashboard tab.

Replaces the per-tab polling loops (status, targets, playback, events, plate
badge, grid overview, PTZ). Data flows server -> browser only; commands stay
ordinary JSON POST/PATCH/DELETE requests behind the app-wide CSRF guard.

``GET /api/stream?camera=<id>[&overview=1]`` answers ``text/event-stream`` with
named events, each carrying exactly the JSON of the matching GET endpoint:

=========  ==============================================  ==========================
event      payload (same as)                                when
=========  ==============================================  ==========================
status     ``/api/cameras/<id>/status``                     ~1 Hz + right after a change
targets    ``/api/cameras/<id>/targets``                    per rendered frame, <= 12 Hz
playback   ``/api/sources/<id>/playback`` (video sources)    on change, 2 Hz while playing
events     ``/api/events?limit=300``                        on change
plates     ``/api/recognition/platelog/summary`` + recent   on change
overview   ``/api/overview`` (only with ``overview=1``)     ~1 Hz
ptz        ``/api/cameras/<id>/ptz``                        on change, 2.5 Hz while moving
=========  ==============================================  ==========================

A topic is only sent when its payload changed (volatile fields such as
``ts`` / ``now`` / ``server_time`` are ignored for the comparison); a comment
line goes out every ``HEARTBEAT_S`` so proxies keep the connection and dead
clients are noticed. Switching camera = the browser opens a new stream.

Every client is one generator (one thread on the dev server) that sleeps on a
condition variable. Payloads are computed at most once per short TTL for all
clients together (``_SNAPSHOTS``), so many tabs cost little more than one.
Any successful mutating request bumps a generation counter and wakes every
stream, so a toggle shows up immediately instead of on the next tick.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from flask import Blueprint, Response, jsonify, request

from . import state, targets
from .status import status_payload

blueprint = Blueprint("live", __name__)

HEARTBEAT_S = 15.0
RETRY_MS = 2000               # EventSource reconnect delay hint (the frontend adds backoff)
MAX_CLIENTS = 48              # bounded threads / memory; beyond that the frontend polls
MAX_IDLE_WAIT_S = 1.0         # upper bound of one sleep (keeps shutdown / disconnects prompt)
EVENTS_LIMIT = 300
PLATES_RECENT = 200

STATUS_S = 1.0
TARGETS_S = 1.0 / 12.0
TARGETS_IDLE_S = 0.5          # YOLO off: just watch for it being switched on
PLAYBACK_PLAYING_S = 0.5
PLAYBACK_IDLE_S = 1.0
EVENTS_S = 1.0
PLATES_S = 1.5
OVERVIEW_S = 1.0
PTZ_MOVING_S = 0.4
PTZ_IDLE_S = 1.5

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

# ---------------------------------------------------------------------------- change notification
_change = threading.Condition()
_generation = 0


def notify_change() -> None:
    """Wake every stream and invalidate cached snapshots (called after successful writes)."""
    global _generation
    with _change:
        _generation += 1
        _change.notify_all()


def generation() -> int:
    return _generation


@blueprint.after_app_request
def _wake_streams_after_write(response):
    if request.method in _MUTATING and response.status_code < 400:
        notify_change()
    return response


# ---------------------------------------------------------------------------- shared snapshots
@dataclass
class Snapshot:
    data: str                 # serialized JSON sent to the browser
    digest: str               # hash of the payload without volatile fields
    payload: Any = None


class _SnapshotCache:
    """(topic, args) -> serialized payload, shared by all clients for ``ttl`` seconds."""

    def __init__(self, max_entries: int = 256):
        self._lock = threading.Lock()
        self._items: dict[tuple, tuple[float, int, Optional[Snapshot]]] = {}
        self.max_entries = max_entries
        self.computes = 0

    def get(self, key: tuple, ttl: float, fn: Callable[[], Any], volatile: tuple[str, ...] = ()) -> Optional[Snapshot]:
        now = time.monotonic()
        gen = _generation
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and hit[0] > now and hit[1] == gen:
                return hit[2]
        payload = fn()        # outside the lock: a slow topic never blocks the others
        snap = None if payload is None else _snapshot(payload, volatile)
        with self._lock:
            self.computes += 1
            if len(self._items) >= self.max_entries:
                for k in [k for k, v in self._items.items() if v[0] <= now] or list(self._items)[: self.max_entries // 2]:
                    self._items.pop(k, None)
            self._items[key] = (now + ttl, gen, snap)
        return snap

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_SNAPSHOTS = _SnapshotCache()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def _snapshot(payload: Any, volatile: tuple[str, ...]) -> Snapshot:
    data = _dumps(payload)
    if volatile and isinstance(payload, dict):
        stable = _dumps({k: v for k, v in payload.items() if k not in volatile})
    else:
        stable = data
    return Snapshot(data, hashlib.blake2b(stable.encode("utf-8"), digest_size=12).hexdigest(), payload)


def format_event(name: str, data: str) -> str:
    """One SSE frame. ``data`` is single-line JSON; newlines are split defensively."""
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {name}\n{lines}\n"


# ---------------------------------------------------------------------------- topics
@dataclass
class Topic:
    name: str
    interval: Callable[[], float]
    poll: Callable[[], Optional[Snapshot]]
    last_digest: Optional[str] = None
    due: float = 0.0
    extra: dict = field(default_factory=dict)


def _status_topic(cs) -> Topic:
    def poll():
        return _SNAPSHOTS.get(("status", cs.camera_id), 0.25, lambda: status_payload(cs))
    return Topic("status", lambda: STATUS_S, poll)


def _targets_topic(cs) -> Topic:
    topic = Topic("targets", lambda: TARGETS_S if cs.yolo_enabled else TARGETS_IDLE_S, lambda: None)

    def poll():
        # A new rendered frame (or a YOLO / generation change) is the only reason the boxes move.
        mark = (cs.stats.get("total_frames", 0), cs.yolo_enabled, cs.esp_locked, _generation)
        if topic.extra.get("mark") == mark:
            return None
        topic.extra["mark"] = mark
        return _SNAPSHOTS.get(("targets",) + (cs.camera_id,) + mark, 1.0,
                              lambda: targets.targets_payload(cs), volatile=("ts",))

    topic.poll = poll
    return topic


def _playback_topic(cs) -> Topic:
    from . import sources

    def payload():
        src = sources.get_source(cs.camera_id)
        return src.playback_payload() if src is not None else None

    def poll():
        return _SNAPSHOTS.get(("playback", cs.camera_id), 0.2, payload, volatile=("server_time",))

    def interval():
        snap = _SNAPSHOTS.get(("playback", cs.camera_id), 0.2, payload, volatile=("server_time",))
        playing = bool(snap and isinstance(snap.payload, dict) and snap.payload.get("playing"))
        return PLAYBACK_PLAYING_S if playing else PLAYBACK_IDLE_S

    return Topic("playback", interval, poll)


def _events_topic() -> Topic:
    def payload():
        from .alerts.routes import events_payload
        return events_payload(None, None, EVENTS_LIMIT)

    return Topic("events", lambda: EVENTS_S,
                 lambda: _SNAPSHOTS.get(("events",), 0.5, payload, volatile=("now",)))


def _plates_topic() -> Topic:
    def payload():
        from .recognition.platelog_api import summary_payload
        return summary_payload(0.0, PLATES_RECENT)

    return Topic("plates", lambda: PLATES_S, lambda: _SNAPSHOTS.get(("plates",), 0.75, payload))


def _overview_topic() -> Topic:
    def payload():
        from .ops_api import overview_payload
        return overview_payload()

    return Topic("overview", lambda: OVERVIEW_S,
                 lambda: _SNAPSHOTS.get(("overview",), 0.5, payload, volatile=("ts",)))


def _ptz_topic(cs) -> Topic:
    from . import ptz

    def poll():
        return _SNAPSHOTS.get(("ptz", cs.camera_id), 0.2, lambda: ptz.status(cs.camera_id))

    def interval():
        snap = _SNAPSHOTS.get(("ptz", cs.camera_id), 0.2, lambda: ptz.status(cs.camera_id))
        p = snap.payload if snap is not None and isinstance(snap.payload, dict) else {}
        pose = p.get("pose") or {}
        fast = bool(p.get("moving") or pose.get("moving") or (pose.get("calibration") or {}).get("state") == "running")
        return PTZ_MOVING_S if fast else PTZ_IDLE_S

    return Topic("ptz", interval, poll)


def build_topics(cs, overview: bool) -> list[Topic]:
    topics = [_status_topic(cs), _targets_topic(cs), _playback_topic(cs), _events_topic(), _plates_topic(), _ptz_topic(cs)]
    if overview:
        topics.append(_overview_topic())
    return topics


# ---------------------------------------------------------------------------- clients
_clients_lock = threading.Lock()
_clients = 0


def client_count() -> int:
    return _clients


def _acquire_slot() -> bool:
    global _clients
    with _clients_lock:
        if _clients >= MAX_CLIENTS:
            return False
        _clients += 1
        return True


def _release_slot() -> None:
    global _clients
    with _clients_lock:
        _clients = max(0, _clients - 1)


def client_stream(cs, overview: bool = False) -> Iterator[str]:
    """Generator for one browser tab. Ends (and frees its slot) on GeneratorExit / close().

    The slot is taken on the first ``next()``, so a response that is never
    iterated (client gone before the first byte) cannot leak one.
    """
    if not _acquire_slot():
        return
    topics = build_topics(cs, overview)
    try:
        yield f"retry: {RETRY_MS}\n: connected\n\n"
        last_sent = time.monotonic()
        seen_gen = _generation
        while True:
            if cs.closed:                      # virtual camera closed: let the browser reconnect elsewhere
                yield format_event("closed", _dumps({"camera": cs.camera_id}))
                return
            now = time.monotonic()
            if _generation != seen_gen:        # a write happened: re-check everything now
                seen_gen = _generation
                for t in topics:
                    t.due = now
            out = []
            for t in topics:
                if now < t.due:
                    continue
                try:
                    snap = t.poll()
                    t.due = now + t.interval()
                except Exception as exc:       # one broken topic must not end the stream
                    print(f"[Live] {t.name} for {cs.camera_id} failed: {exc}")
                    t.due = now + 2.0
                    continue
                if snap is None or snap.digest == t.last_digest:
                    continue
                t.last_digest = snap.digest
                out.append(format_event(t.name, snap.data))
            now = time.monotonic()
            if out:
                last_sent = now
                yield "".join(out)
            elif now - last_sent >= HEARTBEAT_S:
                last_sent = now
                yield ": ping\n\n"
            wait = min(min(t.due for t in topics), last_sent + HEARTBEAT_S) - time.monotonic()
            wait = max(0.005, min(MAX_IDLE_WAIT_S, wait))
            with _change:
                if _generation == seen_gen:
                    _change.wait(wait)
    finally:
        _release_slot()


@blueprint.route("/api/stream")
def stream():
    camera_id = request.args.get("camera") or ""
    cs = state.get_camera_state(camera_id)
    if cs is None:
        return jsonify({"ok": False, "error": "unknown_camera"}), 404
    overview = request.args.get("overview") in ("1", "true")
    if client_count() >= MAX_CLIENTS:
        resp = jsonify({"ok": False, "error": "too_many_streams"})
        resp.status_code = 503
        resp.headers["Retry-After"] = "30"
        return resp
    resp = Response(client_stream(cs, overview), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache, no-store"
    resp.headers["X-Accel-Buffering"] = "no"     # nginx: don't buffer the stream
    return resp
