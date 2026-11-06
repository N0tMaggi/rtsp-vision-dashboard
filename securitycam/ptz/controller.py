"""Per-camera PTZ controller: one worker thread, non-blocking API.

Flask handlers only *enqueue* work and read cached state, so a slow or
hanging camera never blocks a request thread or the video render.

* Continuous moves use a single "latest velocity" slot (newest wins, no
  backlog). A watchdog stops the camera if the client stops refreshing the
  move (lost key-up, closed tab), see ``MOVE_WATCHDOG_S``.
* Connecting happens only on explicit request (UI opened the PTZ panel or
  sent a command) - never in the background - and failed attempts back off.
  Authentication failures back off much longer: Tapo (and others) lock the
  account for ONVIF *and* RTSP after a few failed logins.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable

from .onvif_backend import PtzUnsupported, is_auth_error

MOVE_WATCHDOG_S = 1.5
CONNECT_BACKOFF_MIN_S = 5.0
CONNECT_BACKOFF_MAX_S = 120.0
AUTH_BACKOFF_MIN_S = 60.0
AUTH_BACKOFF_MAX_S = 1800.0
MAX_JOBS = 16


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    return max(lo, min(hi, f))


class PtzController:
    def __init__(
        self,
        camera_id: str,
        backend_factory: Callable[[], Any],
        settings: dict[str, float],
        clock: Callable[[], float] = time.monotonic,
        connect_guard: Callable[[], str | None] | None = None,
    ):
        self.camera_id = camera_id
        self.backend_factory = backend_factory
        self.settings = settings            # fov_h, fov_v, pan_range, tilt_range, speed
        self.clock = clock
        self.connect_guard = connect_guard
        self.backend = None
        self.state = "idle"                 # idle | connecting | ready | unsupported | backoff
        self.error = ""
        self.caps: dict[str, bool] = {}
        self.presets: list[dict[str, str]] = []
        self.busy_since: float | None = None
        self.failures = 0
        self.auth_failures = 0
        self.retry_at = 0.0
        self.last_error_kind = ""
        self._cond = threading.Condition()
        self._jobs: collections.deque = collections.deque()
        self._motion: tuple[float, float, float] | None = None
        self._moving = False
        self._last_move_at = 0.0
        self._thread: threading.Thread | None = None
        self._closed = False

    # ------------------------------------------------------------------ status
    def retry_in(self) -> float:
        return max(0.0, self.retry_at - self.clock())

    def status(self) -> dict[str, Any]:
        with self._cond:
            state = self.state
            if state == "backoff" and self.retry_in() <= 0:
                state = "idle"
            return {
                "state": state,
                "error": self.error,
                "error_kind": self.last_error_kind,
                "retry_in": round(self.retry_in(), 1),
                "caps": dict(self.caps),
                "presets": [dict(p) for p in self.presets],
                "busy": self.busy_since is not None,
                "moving": self._moving,
            }

    # ------------------------------------------------------------------ public API (non-blocking)
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=f"PTZ-{self.camera_id}", daemon=True)
            self._thread.start()

    def _enqueue(self, job: tuple) -> None:
        with self._cond:
            if len(self._jobs) >= MAX_JOBS:
                raise PtzBusy()
            self._jobs.append(job)
            self._ensure_thread()
            self._cond.notify_all()

    def request_connect(self) -> str:
        """Returns 'ready', 'queued', 'connecting' or 'backoff'."""
        with self._cond:
            if self.state == "ready":
                return "ready"
            if self.state == "connecting" or any(j[0] == "connect" for j in self._jobs):
                return "connecting"
            if self.retry_in() > 0:
                return "backoff"
            self.state = "connecting"
            self._jobs.append(("connect",))
            self._ensure_thread()
            self._cond.notify_all()
            return "queued"

    def _require_ready(self) -> None:
        if self.state != "ready":
            raise PtzNotReady(self.state)

    def move(self, x: Any, y: Any, zoom: Any = 0.0) -> None:
        self._require_ready()
        speed = _clamp(self.settings.get("speed", 0.5), 0.05, 1.0)
        vx, vy, vz = _clamp(x, -1, 1) * speed, _clamp(y, -1, 1) * speed, _clamp(zoom, -1, 1) * speed
        if not self.caps.get("zoom"):
            vz = 0.0
        with self._cond:
            self._motion = (vx, vy, vz)
            self._last_move_at = self.clock()
            self._ensure_thread()
            self._cond.notify_all()

    def stop(self) -> None:
        self._require_ready()
        with self._cond:
            self._motion = (0.0, 0.0, 0.0)
            self._cond.notify_all()

    def center_on(self, u: Any, v: Any) -> tuple[float, float]:
        """Relative move that brings normalized frame point (u, v) to the image center."""
        self._require_ready()
        if not self.caps.get("relative"):
            raise PtzUnsupportedOp("relative")
        u, v = _clamp(u, 0, 1), _clamp(v, 0, 1)
        fov_h = _clamp(self.settings.get("fov_h", 100), 1, 360)
        fov_v = _clamp(self.settings.get("fov_v", 56), 1, 180)
        pan_range = _clamp(self.settings.get("pan_range", 360), 1, 720)
        tilt_range = _clamp(self.settings.get("tilt_range", 114), 1, 360)
        pan_deg = (u - 0.5) * fov_h
        tilt_deg = (0.5 - v) * fov_v
        # Generic translation space: -1..1 spans the full mechanical range.
        x = _clamp(pan_deg / (pan_range / 2.0), -1, 1)
        y = _clamp(tilt_deg / (tilt_range / 2.0), -1, 1)
        self._enqueue(("relative", x, y))
        return x, y

    def goto_preset(self, token: str) -> None:
        self._require_ready()
        self._enqueue(("goto", str(token)))

    def save_preset(self, name: str) -> None:
        self._require_ready()
        self._enqueue(("set_preset", str(name)))

    def remove_preset(self, token: str) -> None:
        self._require_ready()
        self._enqueue(("remove_preset", str(token)))

    def home(self) -> None:
        self._require_ready()
        if not self.caps.get("home"):
            raise PtzUnsupportedOp("home")
        self._enqueue(("home",))

    def disconnect(self) -> None:
        with self._cond:
            self._jobs.clear()
            self._jobs.append(("disconnect",))
            self._ensure_thread()
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------------ worker
    def _next_action(self):
        """Wait for work; returns an action tuple or None to exit."""
        with self._cond:
            while True:
                if self._closed:
                    return None
                if self._motion is not None:
                    motion, self._motion = self._motion, None
                    return ("motion",) + motion
                if self._jobs:
                    return self._jobs.popleft()
                if self._moving and self.clock() - self._last_move_at > MOVE_WATCHDOG_S:
                    return ("watchdog",)
                self._cond.wait(0.25 if self._moving else 1.0)

    def _run(self) -> None:
        while True:
            action = self._next_action()
            if action is None:
                return
            self.busy_since = self.clock()
            try:
                self._execute(action)
            finally:
                self.busy_since = None

    def run_pending(self) -> None:
        """Process everything queued right now in the calling thread (tests)."""
        while True:
            with self._cond:
                if self._motion is not None:
                    motion, self._motion = self._motion, None
                    action = ("motion",) + motion
                elif self._jobs:
                    action = self._jobs.popleft()
                elif self._moving and self.clock() - self._last_move_at > MOVE_WATCHDOG_S:
                    action = ("watchdog",)
                else:
                    return
            self._execute(action)

    def _execute(self, action: tuple) -> None:
        kind = action[0]
        if kind == "connect":
            self._do_connect()
            return
        if kind == "disconnect":
            with self._cond:
                self.backend = None
                self.state = "idle"
                self._moving = False
            return
        backend = self.backend
        if backend is None or self.state != "ready":
            return
        try:
            if kind == "motion":
                _, x, y, z = action
                if x or y or z:
                    backend.continuous_move(x, y, z)
                    self._moving = True
                else:
                    backend.stop()
                    self._moving = False
            elif kind == "watchdog":
                self._moving = False
                backend.stop()
            elif kind == "relative":
                backend.relative_move(action[1], action[2])
            elif kind == "goto":
                backend.goto_preset(action[1])
            elif kind == "home":
                backend.goto_home()
            elif kind == "set_preset":
                backend.set_preset(action[1])
                self.presets = backend.get_presets()
            elif kind == "remove_preset":
                backend.remove_preset(action[1])
                self.presets = backend.get_presets()
            self.error = ""
        except Exception as exc:
            self._command_failed(exc)

    def _do_connect(self) -> None:
        guard = self.connect_guard() if self.connect_guard else None
        if guard:
            with self._cond:
                self.state = "backoff"
                self.error = guard
                self.last_error_kind = "rtsp"
                self.retry_at = self.clock() + CONNECT_BACKOFF_MIN_S
            return
        try:
            backend = self.backend_factory()
            caps = backend.connect()
            presets = backend.get_presets() if caps.get("presets") else []
        except PtzUnsupported as exc:
            with self._cond:
                self.backend = None
                self.state = "unsupported"
                self.error = f"Camera has no PTZ ({exc})."
                self.last_error_kind = "unsupported"
            return
        except Exception as exc:
            self._connect_failed(exc)
            return
        with self._cond:
            self.backend = backend
            self.caps = caps
            self.presets = presets
            self.state = "ready"
            self.error = ""
            self.last_error_kind = ""
            self.failures = 0
            self.auth_failures = 0
            self.retry_at = 0.0
        print(f"[PTZ] {self.camera_id}: connected ({', '.join(k for k, v in caps.items() if v) or 'no features'}).")

    def _connect_failed(self, exc: BaseException) -> None:
        auth = is_auth_error(exc)
        with self._cond:
            self.backend = None
            self._moving = False
            self.state = "backoff"
            if auth:
                self.auth_failures += 1
                delay = min(AUTH_BACKOFF_MAX_S, AUTH_BACKOFF_MIN_S * 2 ** (self.auth_failures - 1))
                self.error = (
                    "ONVIF login rejected - check the camera account in .env. "
                    f"Not retrying for {delay:.0f} s to avoid locking the account."
                )
                self.last_error_kind = "auth"
            else:
                self.failures += 1
                delay = min(CONNECT_BACKOFF_MAX_S, CONNECT_BACKOFF_MIN_S * 2 ** (self.failures - 1))
                self.error = f"ONVIF not reachable ({type(exc).__name__}: {str(exc)[:120]}). Retry possible in {delay:.0f} s."
                self.last_error_kind = "network"
            self.retry_at = self.clock() + delay
        print(f"[PTZ] {self.camera_id}: connect failed ({'auth' if auth else 'network'}), backing off {delay:.0f} s.")

    def _command_failed(self, exc: BaseException) -> None:
        if is_auth_error(exc):
            # Session no longer accepted: drop it and back off like a failed login.
            self._connect_failed(exc)
            return
        with self._cond:
            self.error = f"PTZ command failed: {type(exc).__name__}: {str(exc)[:120]}"
            self.last_error_kind = "command"
        print(f"[PTZ] {self.camera_id}: {self.error}")


class PtzNotReady(Exception):
    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


class PtzUnsupportedOp(Exception):
    pass


class PtzBusy(Exception):
    pass
