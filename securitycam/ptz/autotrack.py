"""PTZ auto-track: the camera follows a person so a chosen body point stays framed.

One shared, lightweight loop (``LOOP_HZ``) steps every camera with auto-track
enabled. A step reads the live targets (``targets.targets_payload``: brief ESP
lock, the same call the UI polls), picks the target and sends a raw velocity
through ``PtzController.track_velocity`` (latest-velocity slot + watchdog, so
it never blocks on the camera). It only acts while the controller is ``ready``
with pan/tilt; it never connects on its own (enabling it from the UI requests
one backoff-guarded connect).

Control (per axis, normalized frame coordinates)::

    err  = aim - 0.5
    pred = err - K * (commanded motion of the last LATENCY_S)   # still "in flight"
    start moving when |pred| > deadzone, stop when |pred| < deadzone / 2
    v    = gain * 2 * pred, clamped to +-max_speed, at least MIN_SPEED while moving

The picture lags the camera (ONVIF + RTSP + detection, ``LATENCY_S`` assumed),
so the error is predicted with the commands that have not shown up in the
picture yet (Smith-predictor style). ``K`` (frame widths per second at
velocity 1.0) is learned online from how fast the target moves in the picture
while the camera turns; after a stop the loop waits ``SETTLE_S``.

Pan positive = right, tilt positive = up. Velocities are re-sent only when they
change noticeably or every ``REFRESH_S`` (keeps the controller watchdog fed);
a stop is sent once when the camera should rest (holding / lost / off).

Modes: ``locked`` follows ``camera_state.esp_locked`` (re-locks the same
recognized identity if it reappears within ``lost_timeout_s``); ``auto`` keeps
its target while visible and otherwise picks a recognized person first, then
the largest person, and locks it so the UI highlights it.

Manual PTZ input (move / center / preset / home) pauses auto-track for
``manual_pause_s``. Config is persisted in the ``ptz`` section (``autotrack``).
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Any, Callable, Optional

AIM_KEYS = ("head", "face", "neck", "chest", "belly", "groin", "feet", "center")
MODES = ("locked", "auto")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "locked",
    "aim": "chest",
    "deadzone": 0.12,
    "gain": 1.0,
    "max_speed": 0.6,
    "invert_pan": False,
    "invert_tilt": False,
    "auto_zoom": False,
    "zoom_target": 0.5,
    "lost_timeout_s": 4.0,
    "manual_pause_s": 3.0,
}
FLOAT_LIMITS: dict[str, tuple[float, float]] = {
    "deadzone": (0.03, 0.45),
    "gain": (0.2, 3.0),
    "max_speed": (0.05, 1.0),
    "zoom_target": (0.2, 0.9),
    "lost_timeout_s": (1.0, 30.0),
    "manual_pause_s": (0.0, 30.0),
}
BOOL_KEYS = ("enabled", "invert_pan", "invert_tilt", "auto_zoom")

LOOP_HZ = 10.0
MIN_SPEED = 0.08             # slow cameras ignore tiny velocities
RESEND_DELTA = 0.04          # re-send when a component changed by more than this
REFRESH_S = 0.7              # ... or this often (controller watchdog is 1.5 s)
LATENCY_S = 0.5              # assumed command -> picture latency (ONVIF + RTSP + detection)
K_INIT = 0.6                 # initial guess: frame widths per second at velocity 1.0
K_LIMITS = (0.05, 5.0)
K_LEARN = 0.15               # EMA gain of the online estimate of K
K_MIN_CMD = 0.15             # learn K only from commands at least this fast
DERIV_TAU_S = 0.15           # smoothing of the error derivative
DERIV_MAX_GAP_S = 0.5        # longer gaps reset the derivative
SETTLE_S = 0.3               # after a stop, wait this long before moving again
ZOOM_SPEED = 0.15
ZOOM_IN_BELOW, ZOOM_OUT_ABOVE = 0.8, 1.25   # x zoom_target


class TargetNotVisible(LookupError):
    pass


def clean_config(data: Any) -> dict[str, Any]:
    """Validated (partial) config patch; raises ValueError("invalid_autotrack")."""
    if not isinstance(data, dict):
        raise ValueError("invalid_autotrack")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError("invalid_autotrack")
            out[key] = value
        elif key == "mode":
            if not isinstance(value, str) or value not in MODES:
                raise ValueError("invalid_autotrack")
            out[key] = value
        elif key == "aim":
            if not isinstance(value, str) or value not in AIM_KEYS:
                raise ValueError("invalid_autotrack")
            out[key] = value
        elif key in FLOAT_LIMITS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
                    or value in (float("inf"), float("-inf")):
                raise ValueError("invalid_autotrack")
            lo, hi = FLOAT_LIMITS[key]
            out[key] = round(max(lo, min(hi, float(value))), 3)
        else:
            raise ValueError("invalid_autotrack")
    return out


def _identity_keys(target: dict) -> set:
    ident = target.get("identity") or {}
    keys = set()
    if ident.get("person_id"):
        keys.add(("p", ident["person_id"]))
    if ident.get("face_id") is not None:
        keys.add(("f", ident["face_id"]))
    return keys


def _target_label(target: dict) -> str:
    ident = target.get("identity") or {}
    if ident.get("name"):
        return str(ident["name"])
    if ident.get("face_id") is not None:
        return f"Face #{ident['face_id']}"
    return f"{str(target.get('label') or 'target').capitalize()} #{target['id']}"


def _box_height(target: dict) -> float:
    b = target["box"]
    return float(b[3] - b[1])


def _box_area(target: dict) -> float:
    b = target["box"]
    return float((b[2] - b[0]) * (b[3] - b[1]))


class AutoTracker:
    def __init__(self, camera_id: str, clock: Callable[[], float], wall: Callable[[], float]):
        self.camera_id = camera_id
        self.clock = clock
        self.wall = wall
        self._mu = threading.RLock()
        self.state = "off"
        self.reason = "Auto-track is off"
        self.since = wall()
        self.target_id: Optional[int] = None
        self.target_label: Optional[str] = None
        self.aim_point: Optional[list] = None
        self.error: Optional[list] = None
        self.velocity = (0.0, 0.0, 0.0)
        self._ident: set = set()
        self._lost_since: Optional[float] = None
        self._manual_until = -1e9
        self._last_sent: Optional[tuple] = None
        self._last_send_t = -1e9
        self._stopped = True
        self._stop_t = -1e9
        self._cmds: collections.deque = collections.deque()
        self._inv = (False, False)
        self._k = [K_INIT, K_INIT]
        self._reset_control()
        self._snapshot: dict = {}
        self._publish()

    # ------------------------------------------------------------------ status
    def _reset_control(self) -> None:
        self._moving = [False, False]
        self._prev_err: Optional[tuple] = None
        self._prev_t: Optional[float] = None
        self._derr = [0.0, 0.0]

    # error-space command history: (t, ux, uy); positive = the move that shrinks a positive error
    def _record_cmd(self, now: float, sent: tuple) -> None:
        inv_x, inv_y = self._inv
        ux = -sent[0] if inv_x else sent[0]
        uy = sent[1] if inv_y else -sent[1]
        self._cmds.append((now, ux, uy))
        while len(self._cmds) > 2 and self._cmds[1][0] < now - LATENCY_S - 2.0:
            self._cmds.popleft()

    def _cmd_at(self, t: float) -> tuple:
        cur = (0.0, 0.0)
        for ts, ux, uy in self._cmds:
            if ts > t:
                break
            cur = (ux, uy)
        return cur

    def _inflight(self, now: float) -> list:
        """Integral of the error-space command over the last LATENCY_S seconds."""
        start = now - LATENCY_S
        out = [0.0, 0.0]
        cur, cur_t = self._cmd_at(start), start
        for ts, ux, uy in self._cmds:
            if ts <= start:
                continue
            if ts > now:
                break
            out[0] += cur[0] * (ts - cur_t)
            out[1] += cur[1] * (ts - cur_t)
            cur, cur_t = (ux, uy), ts
        out[0] += cur[0] * (now - cur_t)
        out[1] += cur[1] * (now - cur_t)
        return out

    def _set(self, state: str, reason: str) -> None:
        if state != self.state:
            self.since = self.wall()
        self.state, self.reason = state, reason

    def _publish(self) -> None:
        self._snapshot = {
            "state": self.state,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "aim_point": list(self.aim_point) if self.aim_point is not None else None,
            "error": list(self.error) if self.error is not None else None,
            "velocity": [round(v, 3) for v in self.velocity],
            "since": round(self.since, 3),
            "reason": self.reason,
        }

    def snapshot(self) -> dict:
        return dict(self._snapshot)

    def forget_target(self) -> None:
        with self._mu:
            self.target_id = self.target_label = None
            self.aim_point = self.error = None
            self._ident = set()
            self._lost_since = None
            self._reset_control()
            self._publish()

    # ------------------------------------------------------------------ PTZ access
    def _controller(self):
        """(controller, None) when the camera can be driven, else (None, reason)."""
        from .. import ptz, state

        cs = state.get_camera_state(self.camera_id)
        if cs is None:
            return None, "Unknown camera"
        if getattr(cs, "source", None) is not None or getattr(cs, "source_kind", "rtsp") in ("file", "url"):
            return None, "This source has no PTZ"
        if not ptz.available():
            return None, "PTZ support is not available"
        if not ptz.configured(self.camera_id):
            return None, "PTZ not configured for this camera"
        ctl = ptz.peek_controller(self.camera_id)
        if ctl is None:
            return None, "PTZ not connected"
        if ctl.state != "ready":
            return None, {
                "connecting": "Connecting to the PTZ camera",
                "backoff": "PTZ connection failed - waiting to retry",
                "unsupported": "Camera has no PTZ",
            }.get(ctl.state, "PTZ not connected")
        if not ctl.caps.get("pan_tilt"):
            return None, "Camera cannot pan / tilt"
        return ctl, None

    def _send(self, ctl, v: tuple, now: float) -> None:
        try:
            sent = ctl.track_velocity(*v)
        except Exception:
            return          # not ready any more: the next step reports waiting_ptz
        self._last_sent = tuple(v)
        self._last_send_t = now
        self._stopped = not any(sent)
        self.velocity = tuple(sent)
        self._record_cmd(now, sent)

    def _send_stop(self, ctl, now: float) -> None:
        self._moving = [False, False]
        if not self._stopped and ctl is not None:
            self._send(ctl, (0.0, 0.0, 0.0), now)
            self._stop_t = now
        self._stopped = True
        self.velocity = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ external events
    def note_manual(self, pause_s: float) -> None:
        with self._mu:
            if pause_s <= 0:
                return
            self._manual_until = self.clock() + pause_s
            self._last_sent = None
            self._stopped = True          # the manual command owns the camera now
            self.velocity = (0.0, 0.0, 0.0)
            self._reset_control()
            self._cmds.clear()
            self._set("paused", "Manual PTZ control - auto-track paused")
            self._publish()

    def disable(self) -> None:
        with self._mu:
            ctl, _ = self._controller()
            self._send_stop(ctl, self.clock())
            self._set("off", "Auto-track is off")
            self.forget_target()

    def precheck(self) -> None:
        """Cheap status refresh (no target read) right after enabling."""
        with self._mu:
            ctl, why = self._controller()
            if ctl is None:
                self._set("waiting_ptz", why)
            elif self.state in ("off", "waiting_ptz"):
                self._set("searching", "Looking for the target")
            self._publish()

    # ------------------------------------------------------------------ one control iteration
    def step(self) -> None:
        with self._mu:
            try:
                self._step()
            finally:
                self._publish()

    def _step(self) -> None:
        cfg = config_for(self.camera_id)
        now = self.clock()
        if not cfg["enabled"]:
            self._set("off", "Auto-track is off")
            return
        self._inv = (bool(cfg["invert_pan"]), bool(cfg["invert_tilt"]))
        ctl, why = self._controller()
        if ctl is None:
            self._stopped = True
            self._last_sent = None
            self.velocity = (0.0, 0.0, 0.0)
            self._reset_control()
            self._set("waiting_ptz", why)
            return
        if now < self._manual_until:
            self._set("paused", "Manual PTZ control - auto-track paused")
            return

        from .. import state, targets

        cs = state.get_camera_state(self.camera_id)
        if not cs.yolo_enabled:
            self.aim_point = self.error = None
            self._send_stop(ctl, now)
            self._set("searching", "Detection is off - turn it on to track")
            return
        payload = targets.targets_payload(cs)
        target = self._select(cs, cfg, payload, now)
        if target is None:
            self.aim_point = self.error = None
            self._send_stop(ctl, now)
            return
        self._control(ctl, cfg, target, now)

    # ------------------------------------------------------------------ target selection
    def _adopt(self, target: dict) -> dict:
        if target["id"] != self.target_id:
            self._reset_control()
            self._ident = set()
        self.target_id = int(target["id"])
        self.target_label = _target_label(target)
        keys = _identity_keys(target)
        if keys:
            self._ident = keys
        self._lost_since = None
        return target

    def _match_lost(self, visible: dict) -> Optional[dict]:
        if self.target_id in visible:
            return visible[self.target_id]
        if self._ident:
            for t in visible.values():
                if _identity_keys(t) & self._ident:
                    return t
        return None

    def _lost(self, cs, cfg, visible: dict, now: float, relock: bool) -> Optional[dict]:
        from .. import targets

        if self._lost_since is None:
            self._lost_since = now
        within = now - self._lost_since <= cfg["lost_timeout_s"]
        if within and relock:
            match = self._match_lost(visible)
            if match is not None and targets.set_lock(cs, int(match["id"])):
                return self._adopt(match)
        if within:
            self._set("lost", "Target lost - holding position")
        elif cfg["mode"] == "auto":
            self._set("lost", "Target lost - no person in view")
        else:
            self._set("lost", "Target lost - lock a target to resume")
        return None

    def _select(self, cs, cfg, payload: dict, now: float) -> Optional[dict]:
        from .. import targets

        visible = {int(t["id"]): t for t in payload.get("targets", [])}
        locked = payload.get("locked")
        if cfg["mode"] == "locked":
            if locked is not None and locked in visible:
                return self._adopt(visible[locked])
            if self._lost_since is not None:
                return self._lost(cs, cfg, visible, now, relock=True)
            if self.target_id is not None and locked is None and self.target_id in visible:
                # the user unlocked the target on purpose: stop, don't chase it
                self.forget_target()
                self._set("searching", "No locked target - lock a person to track")
                return None
            if self.target_id is None:
                self._set("searching", "No locked target - lock a person to track")
                return None
            return self._lost(cs, cfg, visible, now, relock=True)

        # auto mode
        if locked is not None and locked in visible:
            return self._adopt(visible[locked])     # keeps ours, or follows a lock the user set
        if self.target_id in visible:
            t = visible[self.target_id]
            targets.set_lock(cs, int(t["id"]))
            return self._adopt(t)
        cand = self._pick(visible)
        if cand is not None and targets.set_lock(cs, int(cand["id"])):
            return self._adopt(cand)
        if self.target_id is not None or self._lost_since is not None:
            return self._lost(cs, cfg, visible, now, relock=False)
        self._set("searching", "No person in view")
        return None

    def _pick(self, visible: dict) -> Optional[dict]:
        people = [t for t in visible.values() if str(t.get("label", "")).lower() == "person" and not t.get("hidden")]
        if not people:
            return None
        if self._ident:
            same = [t for t in people if _identity_keys(t) & self._ident]
            if same:
                return max(same, key=_box_area)
        known = [t for t in people if (t.get("identity") or {}).get("name")]
        return max(known or people, key=_box_area)

    # ------------------------------------------------------------------ control law
    def _axis(self, cfg, e: float) -> float:
        gain, vmax = cfg["gain"], cfg["max_speed"]
        m = min(max(abs(gain * 2.0 * e), min(MIN_SPEED, vmax)), vmax)
        return math.copysign(m, e)

    def _control(self, ctl, cfg, target: dict, now: float) -> None:
        aim = (target.get("aim") or {})
        point = aim.get(cfg["aim"]) or aim.get("center")
        if point is None:
            b = target["box"]
            point = [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5]
        u, v = float(point[0]), float(point[1])
        err = (u - 0.5, v - 0.5)
        self.aim_point = [round(u, 4), round(v, 4)]
        self.error = [round(err[0], 4), round(err[1], 4)]

        if self._prev_err is not None and self._prev_t is not None and 0 < now - self._prev_t < DERIV_MAX_GAP_S:
            dt = now - self._prev_t
            a = 1.0 - math.exp(-dt / DERIV_TAU_S)
            for i in range(2):
                self._derr[i] += ((err[i] - self._prev_err[i]) / dt - self._derr[i]) * a
            # learn K: picture speed of the target vs. the command that is visible by now
            delayed = self._cmd_at(now - LATENCY_S)
            for i in range(2):
                if abs(delayed[i]) >= K_MIN_CMD:
                    k_obs = -self._derr[i] / delayed[i]
                    if K_LIMITS[0] <= k_obs <= K_LIMITS[1]:
                        self._k[i] += (k_obs - self._k[i]) * K_LEARN
        else:
            self._derr = [0.0, 0.0]
        self._prev_err, self._prev_t = err, now
        inflight = self._inflight(now)
        pred = (err[0] - self._k[0] * inflight[0], err[1] - self._k[1] * inflight[1])

        dz = cfg["deadzone"]
        if self._stopped and now - self._stop_t < SETTLE_S:
            self._set("holding", "Camera stopped - letting the picture settle")
            return
        for i in range(2):
            if self._moving[i]:
                if abs(pred[i]) < 0.5 * dz:
                    self._moving[i] = False
            elif abs(pred[i]) > dz:
                self._moving[i] = True

        vx = self._axis(cfg, pred[0]) if self._moving[0] else 0.0
        vy = -self._axis(cfg, pred[1]) if self._moving[1] else 0.0
        if cfg["invert_pan"]:
            vx = -vx
        if cfg["invert_tilt"]:
            vy = -vy
        vz = 0.0
        if cfg["auto_zoom"] and ctl.caps.get("zoom") and abs(err[0]) < 2 * dz and abs(err[1]) < 2 * dz:
            h, zt = _box_height(target), cfg["zoom_target"]
            if h < ZOOM_IN_BELOW * zt:
                vz = ZOOM_SPEED
            elif h > ZOOM_OUT_ABOVE * zt:
                vz = -ZOOM_SPEED

        vel = (round(vx, 4), round(vy, 4), round(vz, 4))
        if not any(vel):
            self._send_stop(ctl, now)
            self._set("holding", "Target framed - holding position")
            return
        if now < self._manual_until:       # manual input arrived during this step
            return
        last = self._last_sent
        if last is None or self._stopped or max(abs(a - b) for a, b in zip(vel, last)) > RESEND_DELTA \
                or now - self._last_send_t >= REFRESH_S:
            self._send(ctl, vel, now)
        self._set("tracking", "Following target")


# ---------------------------------------------------------------------------- registry / loop
_lock = threading.RLock()
_configs: dict[str, dict[str, Any]] = {}
_trackers: dict[str, AutoTracker] = {}
_thread: Optional[threading.Thread] = None
_generation = 0
clock: Callable[[], float] = time.monotonic     # tests replace these
wall: Callable[[], float] = time.time
RUN_LOOP = True                                 # tests drive ``step`` themselves


def config_for(camera_id: str) -> dict[str, Any]:
    with _lock:
        c = _configs.get(camera_id)
        return dict(c) if c is not None else dict(DEFAULT_CONFIG)


def tracker(camera_id: str) -> AutoTracker:
    with _lock:
        t = _trackers.get(camera_id)
        if t is None:
            t = _trackers[camera_id] = AutoTracker(camera_id, lambda: clock(), lambda: wall())
        return t


def peek(camera_id: str) -> Optional[AutoTracker]:
    with _lock:
        return _trackers.get(camera_id)


def status(camera_id: str) -> dict[str, Any]:
    t = peek(camera_id)
    if t is not None:
        return t.snapshot()
    return {"state": "off", "target_id": None, "target_label": None, "aim_point": None, "error": None,
            "velocity": [0.0, 0.0, 0.0], "since": 0.0, "reason": "Auto-track is off"}


def payload(camera_id: str) -> dict[str, Any]:
    return {"config": config_for(camera_id), "status": status(camera_id)}


def enabled(camera_id: str) -> bool:
    with _lock:
        return bool((_configs.get(camera_id) or {}).get("enabled"))


def _request_connect(camera_id: str) -> Optional[str]:
    from .. import ptz

    if not ptz.available() or not ptz.configured(camera_id):
        return None
    ctl = ptz.get_controller(camera_id)
    return ctl.request_connect() if ctl is not None else None


def update_config(camera_id: str, data: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Apply a config patch. Returns (payload, connect outcome or None). Raises ValueError."""
    patch = clean_config(data)
    with _lock:
        old = dict(_configs.get(camera_id) or DEFAULT_CONFIG)
        new = {**old, **patch}
        _configs[camera_id] = new
    _mark_dirty()
    t = tracker(camera_id)
    outcome = None
    if new["enabled"] and (not old["enabled"] or new["mode"] != old["mode"]):
        t.forget_target()
    if new["enabled"] and not old["enabled"]:
        outcome = _request_connect(camera_id)       # explicit user action
        _ensure_loop()
    if not new["enabled"] and old["enabled"]:
        t.disable()
    elif new["enabled"]:
        t.precheck()
    return payload(camera_id), outcome


def set_target(camera_id: str, target_id: Optional[int]) -> tuple[dict[str, Any], Optional[str]]:
    """Lock ``target_id`` and follow it (mode locked, enabled); None disables and stops."""
    from .. import state, targets

    if target_id is None:
        return update_config(camera_id, {"enabled": False})
    cs = state.get_camera_state(camera_id)
    if cs is None or not targets.set_lock(cs, int(target_id)):
        raise TargetNotVisible(target_id)
    t = tracker(camera_id)
    t.forget_target()
    return update_config(camera_id, {"enabled": True, "mode": "locked"})


def note_manual(camera_id: str) -> None:
    """Manual PTZ input: pause auto-track for ``manual_pause_s``."""
    cfg = config_for(camera_id)
    t = peek(camera_id)
    if t is not None and cfg["enabled"]:
        t.note_manual(float(cfg["manual_pause_s"]))


def _ensure_loop() -> None:
    global _thread
    if not RUN_LOOP:
        return
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, args=(_generation,), name="PTZ-autotrack", daemon=True)
        _thread.start()


def _loop(generation: int) -> None:
    global _thread
    period = 1.0 / LOOP_HZ
    errors: set = set()
    while True:
        with _lock:
            active = [t for cam, t in _trackers.items() if (_configs.get(cam) or {}).get("enabled")]
            if not active or generation != _generation:
                if _thread is threading.current_thread():
                    _thread = None
                return
        start = time.monotonic()
        for t in active:
            try:
                t.step()
            except Exception as exc:     # a bug must not kill the loop for other cameras
                key = (t.camera_id, type(exc).__name__)
                if key not in errors:
                    errors.add(key)
                    print(f"[PTZ] {t.camera_id}: auto-track error {type(exc).__name__}: {exc}")
        time.sleep(max(0.02, period - (time.monotonic() - start)))


def reset() -> None:
    """Forget all configs and trackers; a running loop exits (tests)."""
    global _generation, _thread
    with _lock:
        _configs.clear()
        _trackers.clear()
        _generation += 1
        _thread = None


def _mark_dirty() -> None:
    try:
        from .. import persistence

        persistence.mark_dirty()
    except Exception:
        pass


# ---------------------------------------------------------------------------- persistence (inside "ptz")
def dump() -> dict[str, dict[str, Any]]:
    with _lock:
        out = {}
        for cam, cfg in _configs.items():
            diff = {k: v for k, v in cfg.items() if DEFAULT_CONFIG.get(k) != v}
            if diff:
                out[cam] = diff
        return out


def clean_dump(raw: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    out: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if raw is None:
        return out, warnings
    if not isinstance(raw, dict):
        return out, ["PTZ auto-track settings dropped."]
    for cam, cfg in raw.items():
        try:
            out[str(cam)] = clean_config(cfg)
        except ValueError:
            warnings.append(f"PTZ auto-track settings for '{cam}' dropped.")
    return out, warnings


def load(clean: dict[str, dict[str, Any]]) -> None:
    start = False
    for cam, cfg in clean.items():
        with _lock:
            _configs[cam] = {**DEFAULT_CONFIG, **cfg}
        if _configs[cam]["enabled"]:
            start = True
            try:
                tracker(cam).precheck()
            except Exception:
                pass
    if start:
        _ensure_loop()
