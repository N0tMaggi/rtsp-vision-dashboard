"""Per-camera depth worker: one background thread, low rate, never on the video path.

Lifecycle: the render path calls ``ensure(camera_state)`` for every rendered
frame (= the camera is being viewed). A worker thread starts only when the
``depth`` setting is on; it exits on its own when the setting goes off, the
camera closes, or no frame was rendered for ``IDLE_STOP_S`` (nobody watches).

Each step: newest frame -> resize to a static input shape -> inference on the
resolved device -> relative depth map. Then the metric scale is updated from
the persons currently tracked (``calib``) and a coarse floor scan is derived.
Readers (targets payload, depth view) only take a short lock to grab the
latest ``DepthFrame``.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import calib, geometry, model
from .render import ViewCache

IDLE_STOP_S = 3.0             # no rendered frame for this long -> nobody watches -> stop
FAIL_RETRY_S = 10.0
TRACK_TAU_S = 0.6             # smoothing of per-target depth
MATCH_S = 0.35                # a track observation belongs to a depth frame within this time
DEFAULTS = {
    "depth": False, "depth_view": False, "depth_view_opacity": 0.5, "depth_view_palette": "turbo",
    "depth_rate": 3.0, "depth_size": 364, "depth_fov": 90, "depth_scale": 0.0,
}


def _num(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return max(lo, min(hi, x))


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


@dataclass(frozen=True)
class DepthSettings:
    enabled: bool
    view: bool
    opacity: float
    palette: str
    rate: float
    size: int
    fov: float
    user_scale: float

    @classmethod
    def of(cls, camera_state) -> "DepthSettings":
        raw = getattr(camera_state, "esp_settings", None) or {}
        g = lambda k: raw.get(k, DEFAULTS[k])  # noqa: E731
        pal = str(g("depth_view_palette") or "turbo").lower()
        return cls(
            enabled=_bool(g("depth")),
            view=_bool(g("depth_view")),
            opacity=_num(g("depth_view_opacity"), 0.5, 0.0, 1.0),
            palette=pal if pal in ("turbo", "inferno") else "turbo",
            rate=_num(g("depth_rate"), 3.0, 0.5, 10.0),
            size=model.snap14(_num(g("depth_size"), 364, 196, 518)),
            fov=_num(g("depth_fov"), 90.0, 30.0, 170.0),
            user_scale=_num(g("depth_scale"), 0.0, 0.0, 1e5),
        )


@dataclass
class DepthFrame:
    seq: int
    disp: np.ndarray               # (H, W) relative inverse depth
    t: float                       # perf_counter capture time of the source frame
    frame_w: int
    frame_h: int
    latency_ms: float
    scan: List[List[float]] = field(default_factory=list)
    scan_world: bool = False


BackendFactory = Callable[[Any, str, Tuple[int, int, int, int]], Any]


class DepthWorker:
    def __init__(self, camera_state, backend_factory: Optional[BackendFactory] = None,
                 model_path: Optional[Callable[[], Any]] = None, clock: Callable[[], float] = time.perf_counter):
        self.cs = camera_state
        self.camera_id = getattr(camera_state, "camera_id", "")
        self._factory = backend_factory or model.build_backend
        self._model_path = model_path or _asset_path
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last_ping = clock()
        self.state = "idle"            # idle | downloading | loading | running | failed | stopped
        self.error: Optional[str] = None
        self.backend = None
        self.device: Optional[str] = None
        self.device_source = "auto"
        self.latency_ms: Optional[float] = None
        self.fps: Optional[float] = None
        self.frame: Optional[DepthFrame] = None
        self.calibrator = calib.ScaleCalibrator()
        self.view_cache = ViewCache()
        self._seq = 0
        self._last_src = None
        self._last_step_t: Optional[float] = None
        self._retry_at = 0.0
        self._track_depth: Dict[Any, Tuple[float, float]] = {}   # key -> (depth_m, t)

    # ------------------------------------------------------------------ lifecycle
    def ping(self) -> None:
        self.last_ping = self._clock()

    def start(self) -> None:
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"DepthWorker-{self.camera_id}")
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive() and not self._stop.is_set()

    def should_run(self) -> bool:
        if getattr(self.cs, "closed", False) or self._stop.is_set():
            return False
        if not DepthSettings.of(self.cs).enabled:
            return False
        return self._clock() - self.last_ping < IDLE_STOP_S

    def _run(self) -> None:
        try:
            while self.should_run():
                s = DepthSettings.of(self.cs)
                t0 = self._clock()
                try:
                    self.step(s)
                except Exception as exc:          # never let one bad frame end the worker
                    traceback.print_exc(limit=2)
                    self.error = f"{type(exc).__name__}: {exc}"[:300]
                period = 1.0 / s.rate
                wait = max(0.02, period - (self._clock() - t0))
                if self.state in ("downloading", "failed", "loading"):
                    wait = max(wait, 0.5)
                self._stop.wait(wait)
        finally:
            self.state = "stopped"
            self.backend = None            # free the compiled model / session
            _forget(self)

    # ------------------------------------------------------------------ one step
    def _frame(self):
        lock = getattr(self.cs, "frame_lock", None)
        if lock is None:
            return getattr(self.cs, "latest_frame", None), getattr(self.cs, "latest_frame_ts", 0.0)
        with lock:
            return self.cs.latest_frame, self.cs.latest_frame_ts

    def _ensure_backend(self, shape: Tuple[int, int, int, int]) -> bool:
        now = self._clock()
        res = model.resolve()
        dev = res["device"]
        if self.backend is not None and self.device == dev and tuple(self.backend.shape) == shape:
            return True
        if now < self._retry_at:
            return False
        path = self._model_path()
        if path is None:
            self.state = "downloading"
            return False
        self.state = "loading"
        try:
            self.backend = self._factory(path, dev, shape)
            self.device, self.device_source = dev, res["source"]
            self.error = None
            print(f"[DEPTH] {self.camera_id}: running on {getattr(self.backend, 'label', dev)} "
                  f"({shape[3]}x{shape[2]}).")
            return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
            self._device_failed(dev, res["source"], msg)
            return False

    def _device_failed(self, dev: str, source: str, msg: str) -> None:
        self.backend = None
        self.error = msg
        print(f"[DEPTH] {self.camera_id}: {dev} failed: {msg}")
        if source == "user":
            from ..compute import assignments
            assignments.mark_failed("depth", dev, msg)      # -> resolves to auto next time
        elif dev != "cpu":
            model.mark_auto_failed(dev, msg)
        else:
            self.state = "failed"
            self._retry_at = self._clock() + FAIL_RETRY_S

    def step(self, s: Optional[DepthSettings] = None) -> bool:
        """Process the newest frame once (synchronous; the thread loop calls it). True if a map was made."""
        s = s or DepthSettings.of(self.cs)
        frame, ts = self._frame()
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return False
        if frame is self._last_src and self.state == "running":
            return False
        fh, fw = frame.shape[:2]
        h, w = model.input_shape(s.size, fw, fh)
        shape = (1, 3, h, w)
        if not self._ensure_backend(shape):
            return False
        x = model.preprocess(frame, (h, w))
        t0 = time.perf_counter()
        try:
            out = self.backend(x)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
            self._device_failed(self.device or "cpu", self.device_source, msg)
            return False
        lat = (time.perf_counter() - t0) * 1000.0
        disp = model.to_map(out)
        self._last_src = frame
        now = self._clock()
        if self._last_step_t is not None and now > self._last_step_t:
            inst = 1.0 / (now - self._last_step_t)
            self.fps = inst if self.fps is None else self.fps + (inst - self.fps) * 0.3
        self._last_step_t = now
        self.latency_ms = lat if self.latency_ms is None else self.latency_ms + (lat - self.latency_ms) * 0.3
        self._seq += 1
        df = DepthFrame(self._seq, disp, float(ts or 0.0), fw, fh, lat)
        self._calibrate(df)
        self._scan(df, s)
        with self._lock:
            self.frame = df
        self.state = "running"
        return True

    # ------------------------------------------------------------------ calibration
    def _tracks(self) -> List[Tuple[Any, str, float, List[float], int, int, float, bool]]:
        """(key, label, conf, obs box px, frame_w, frame_h, obs_t, present) of the tracks (ESP lock)."""
        from ..esp import motion
        from ..esp.pipeline import _esp_lock
        out = []
        with _esp_lock(self.cs):
            st = motion.get_state(self.cs)
            for key, tr in st.tracks.items():
                out.append((key, tr.det.cls_name, float(tr.conf), list(tr.obs), st.frame_w, st.frame_h,
                            float(tr.obs_t), tr.missing_since is None))
        return out

    def _calibrate(self, df: DepthFrame) -> None:
        from ..esp import trackstats
        try:
            tracks = self._tracks()
        except Exception:
            return
        for key, label, conf, obs, fw, fh, obs_t, present in tracks:
            if not present or fw <= 0 or fh <= 0 or (df.t and abs(obs_t - df.t) > MATCH_S):
                continue
            wpx, hpx = obs[2] - obs[0], obs[3] - obs[1]
            box = [obs[0] / fw, obs[1] / fh, obs[2] / fw, obs[3] / fh]
            if not calib.person_ok(label, conf, box, wpx, hpx):
                continue
            d = geometry.region_median(df.disp, box)
            self.calibrator.add(d, trackstats.estimate_distance_m(hpx), self._clock(), track=key)

    def calibration(self) -> calib.Calibration:
        return self.calibrator.current(DepthSettings.of(self.cs).user_scale)

    # ------------------------------------------------------------------ geometry
    def lens_pose(self, fw: int, fh: int, s: Optional[DepthSettings] = None):
        """(lens, pose, world) for ground projection: PTZ view when anchored, else the FOV setting."""
        from .. import anchor
        from ..geo import Pose
        try:
            view = anchor.view_for(self.cs)
        except Exception:
            view = None
        if view is not None and view.active:
            return view.lens, view.pose, True
        s = s or DepthSettings.of(self.cs)
        return geometry.lens_for_fov(s.fov, fw, fh), Pose(), False

    def _scan(self, df: DepthFrame, s: DepthSettings) -> None:
        cal = self.calibrator.current(s.user_scale)
        if not cal.ready:
            return
        lens, pose, world = self.lens_pose(df.frame_w, df.frame_h, s)
        dm = calib.depth_map_m(cal.scale, df.disp)
        df.scan = geometry.scan_points(dm, lens, pose, world)
        df.scan_world = world

    def latest(self) -> Optional[DepthFrame]:
        with self._lock:
            return self.frame

    def target_depth(self, key: Any, box: List[float], cal: calib.Calibration, t: float) -> Optional[float]:
        """Smoothed metric depth of a target (normalized box) from the latest map, or None."""
        df = self.latest()
        if df is None or not cal.ready:
            return None
        z = calib.depth_m(cal.scale, geometry.region_median(df.disp, box))
        if z is None:
            return None
        prev = self._track_depth.get(key)
        if prev is not None and t > prev[1]:
            z = prev[0] + (z - prev[0]) * (1.0 - math.exp(-(t - prev[1]) / TRACK_TAU_S))
        elif prev is not None:
            z = prev[0]
        self._track_depth[key] = (z, t)
        if len(self._track_depth) > 256:
            for k in list(self._track_depth)[:128]:
                self._track_depth.pop(k, None)
        return z


def _asset_path():
    from .. import deps
    return deps.asset_path(model.ASSET)


# ----------------------------------------------------------------------------- registry
_workers: Dict[str, DepthWorker] = {}
_reg_lock = threading.Lock()
WORKER_FACTORY: Callable[[Any], DepthWorker] = DepthWorker     # injectable for tests


def _forget(w: DepthWorker) -> None:
    with _reg_lock:
        if _workers.get(w.camera_id) is w:
            _workers.pop(w.camera_id, None)


def ensure(camera_state) -> Optional[DepthWorker]:
    """Called per rendered frame: keep (or start) the camera's worker while depth is on."""
    cid = getattr(camera_state, "camera_id", "")
    raw = getattr(camera_state, "esp_settings", None) or {}
    if not _bool(raw.get("depth", False)):
        return None
    with _reg_lock:
        w = _workers.get(cid)
        if w is not None and w.alive():
            w.ping()
            return w
        w = WORKER_FACTORY(camera_state)
        w.ping()
        _workers[cid] = w
    w.start()
    return w


def get(camera_id: str) -> Optional[DepthWorker]:
    with _reg_lock:
        return _workers.get(camera_id)


def stop_all() -> None:
    with _reg_lock:
        ws = list(_workers.values())
        _workers.clear()
    for w in ws:
        w.stop()
