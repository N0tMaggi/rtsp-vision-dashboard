"""Per-camera PTZ pose estimate (pan / tilt in degrees, zoom) for world-anchored overlays.

Three sources are fused into one pose per camera:

* **Vision** (every frame while moving, every ``IDLE_EVERY``-th frame when idle):
  features of a *keyframe* (320 px wide grayscale, detection boxes masked out)
  are tracked into the current frame with pyramidal Lucas-Kanade, seeded by
  the predicted pose. Outliers (people, cars) are dropped with a RANSAC
  similarity, then pan / tilt (/ zoom) are solved by Gauss-Newton on the
  pinhole reprojection error of the keyframe rays (``geo``). Measuring
  against a keyframe instead of the previous frame means a still camera
  never random-walks; error only accumulates when the keyframe is replaced
  (overlap too small), i.e. during real moves.
* **ONVIF absolute position** (``GetStatus``), polled by the PTZ worker thread
  (fast while moving, slow when idle), mapped from the generic -1..1 space to
  degrees with the calibrated pan / tilt ranges. Readings are only trusted once
  the camera has settled; they slowly pull the pose (and the keyframe) toward
  ``onvif + offset``. The offset aligns the ONVIF frame with the world frame
  of the existing shapes once, so the first reading never makes overlays jump.
  Cameras that report nothing, a constant value or inverted axes are detected
  by comparing settled ONVIF deltas with vision deltas.
* **Commanded moves** (relative move / center, presets): used as a prior when
  vision lost track during the move (dark, blur).

Non-PTZ cameras (no connected PTZ controller with pan/tilt, never seen one)
and cameras with the ``world_anchor`` ESP setting off are not processed at
all; their overlays stay in image space exactly as before.
"""
from __future__ import annotations

import bisect
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import geo
from ..geo import IDENTITY, Lens, Pose

SMALL_W = 320                 # vision runs on frames this wide
MAX_FEATURES = 160
MIN_FEATURES = 24
MIN_INLIERS = 12
RANSAC_PX = 1.5               # small-frame pixels
TRIM_PX = 2.5
REKEY_INLIER_FRAC = 0.45      # rekey when fewer keyframe features survive
REKEY_ZOOM = 0.12             # or zoom changed by this fraction
IDLE_EVERY = 3                # vision cadence while idle
IDLE_SPEED = 0.35             # deg/s below which the camera counts as still
DEADBAND_PX = 0.12            # idle pose changes below this (small-frame px) are ignored
SETTLE_S = 1.0                # no command and no motion for this long = settled
HISTORY_S = 12.0              # pose history kept for look-ups by timestamp
ONVIF_TAU_S = 1.5             # time constant of the ONVIF drift correction
ONVIF_DEADBAND = 0.05         # deg
ONVIF_SNAP = 8.0              # deg: larger disagreement while settled -> trust ONVIF, jump
ONVIF_FRESH_S = 15.0
ONVIF_AVG_N = 8               # settled readings averaged
ONVIF_MAX_RATE = 2.0          # deg/s: drift correction never moves overlays faster than this
DRIFT_PER_REKEY = 0.02        # deg, vision-only drift estimate
DRIFT_PER_DEG = 0.004         # fraction of traveled angle
SPEED_BASE_S = 0.3            # baseline for the speed estimate
FAST_DEG_S = 15.0             # "fast camera motion" for alarm robustness


def _now() -> float:
    return time.perf_counter()


@dataclass
class VisionResult:
    ok: bool
    pose: Pose
    inliers: int = 0
    tracked: int = 0
    rms: float = 0.0
    rekey: bool = False


# ----------------------------------------------------------------------------- vision
def small_gray(frame: np.ndarray, width: int = SMALL_W) -> np.ndarray:
    h, w = frame.shape[:2]
    sw = min(width, w)
    sh = max(8, int(round(h * sw / w)))
    if frame.ndim == 3:
        # area-averaging a full 1080p color frame costs ~1.3 ms; a bilinear step to 2x the
        # target first (~0.2 ms in total) keeps enough anti-aliasing for feature tracking
        src = frame
        if w > sw * 2:
            src = cv2.resize(frame, (sw * 2, sh * 2), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)


def _project_px(rays: np.ndarray, p: np.ndarray, lens: Lens, size, zoom_fixed: Optional[float]) -> np.ndarray:
    pose = Pose(p[0], p[1], zoom_fixed if zoom_fixed is not None else math.exp(p[2]))
    uv, _ = geo.rays_to_image(rays, pose, lens)
    return uv * np.asarray(size, dtype=np.float64) - 0.5


def solve_pose(rays: np.ndarray, obs_px: np.ndarray, init: Pose, lens: Lens, size: Tuple[int, int],
               estimate_zoom: bool, iters: int = 6) -> Tuple[Pose, np.ndarray]:
    """Gauss-Newton for the pose that projects ``rays`` onto ``obs_px`` (small-frame pixels)."""
    zoom_fixed = None if estimate_zoom else init.zoom
    p = np.array([init.pan, init.tilt, math.log(max(1e-3, init.zoom))])
    n_par = 3 if estimate_zoom else 2
    eps = np.array([1e-3, 1e-3, 1e-4])
    obs = obs_px.reshape(-1)
    for _ in range(iters):
        base = _project_px(rays, p, lens, size, zoom_fixed).reshape(-1)
        r = base - obs
        valid = np.isfinite(r)
        if valid.sum() < 6:
            break
        jac = np.empty((len(r), n_par))
        for k in range(n_par):
            q = p.copy()
            q[k] += eps[k]
            jac[:, k] = (_project_px(rays, q, lens, size, zoom_fixed).reshape(-1) - base) / eps[k]
        jv, rv = jac[valid], r[valid]
        jtj = jv.T @ jv
        jtj += np.eye(n_par) * (1e-9 + 1e-6 * np.trace(jtj) / n_par)
        try:
            dp = -np.linalg.solve(jtj, jv.T @ rv)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dp)):
            break
        p[:n_par] += dp
        if abs(dp[0]) < 1e-5 and abs(dp[1]) < 1e-5 and (n_par == 2 or abs(dp[2]) < 1e-6):
            break
    pose = Pose(float(p[0]), max(-89.0, min(89.0, float(p[1]))),
                zoom_fixed if zoom_fixed is not None else float(min(100.0, max(0.05, math.exp(p[2])))))
    res = _project_px(rays, np.array([pose.pan, pose.tilt, math.log(pose.zoom)]), lens, size, zoom_fixed) - obs_px
    return pose, np.linalg.norm(res, axis=1)


class VisionTracker:
    """Keyframe-based rotation tracker (see module docstring)."""

    def __init__(self):
        self.key_gray: Optional[np.ndarray] = None
        self.key_px: Optional[np.ndarray] = None        # (N, 2) float32 small-frame pixels
        self.key_uv: Optional[np.ndarray] = None
        self.key_rays: Optional[np.ndarray] = None
        self.key_pose: Pose = IDENTITY
        self.size: Tuple[int, int] = (0, 0)

    @property
    def has_key(self) -> bool:
        return self.key_gray is not None

    def reset(self) -> None:
        self.key_gray = None

    def rekey(self, gray: np.ndarray, pose: Pose, lens: Lens, mask: Optional[np.ndarray] = None) -> bool:
        pts = cv2.goodFeaturesToTrack(gray, MAX_FEATURES, 0.01, 7, mask=mask, blockSize=5)
        if (pts is None or len(pts) < MIN_FEATURES) and mask is not None:
            pts = cv2.goodFeaturesToTrack(gray, MAX_FEATURES, 0.01, 7, blockSize=5)
        if pts is None or len(pts) < MIN_FEATURES:
            self.key_gray = None
            return False
        h, w = gray.shape[:2]
        self.size = (w, h)
        self.key_px = pts.reshape(-1, 2).astype(np.float32)
        self.key_uv = (self.key_px.astype(np.float64) + 0.5) / np.array([w, h], dtype=np.float64)
        self.key_pose = pose
        self.key_rays = geo.image_to_rays(self.key_uv, pose, lens)
        self.key_gray = gray
        return True

    def rebase(self, pose: Pose, lens: Lens) -> None:
        """Re-derive the keyframe rays for a corrected keyframe pose (ONVIF / prior / FOV change)."""
        if self.key_gray is None:
            return
        self.key_pose = pose
        self.key_rays = geo.image_to_rays(self.key_uv, pose, lens)

    def track(self, gray: np.ndarray, pred: Pose, lens: Lens, estimate_zoom: bool) -> VisionResult:
        if self.key_gray is None or gray.shape != self.key_gray.shape:
            return VisionResult(False, pred, rekey=True)
        w, h = self.size
        size = np.array([w, h], dtype=np.float64)
        uv, front = geo.rays_to_image(self.key_rays, pred, lens)
        pred_px = uv * size - 0.5
        m = 3.0
        sel = front & (pred_px[:, 0] > m) & (pred_px[:, 0] < w - 1 - m) & (pred_px[:, 1] > m) & (pred_px[:, 1] < h - 1 - m)
        n_sel = int(sel.sum())
        if n_sel < MIN_INLIERS:
            return VisionResult(False, pred, rekey=True)
        idx = np.nonzero(sel)[0]
        p0 = self.key_px[idx]
        p1 = pred_px[idx].astype(np.float32)
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.key_gray, gray, p0, p1.copy(), winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if p1 is None:
            return VisionResult(False, pred, rekey=True)
        good = (status.reshape(-1) == 1) & (err.reshape(-1) < 30.0)
        if good.sum() < MIN_INLIERS:
            return VisionResult(False, pred, tracked=int(good.sum()), rekey=True)
        gi = idx[good]
        obs = p1[good].astype(np.float64)
        predg = pred_px[gi]
        _m, inl = cv2.estimateAffinePartial2D(predg.astype(np.float32), obs.astype(np.float32),
                                              method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                              maxIters=200, confidence=0.99)
        if inl is None:
            return VisionResult(False, pred, tracked=len(gi), rekey=True)
        inl = inl.reshape(-1).astype(bool)
        if inl.sum() < MIN_INLIERS:
            return VisionResult(False, pred, tracked=len(gi), rekey=True)
        rays = self.key_rays[gi[inl]]
        o = obs[inl]
        pose, res = solve_pose(rays, o, pred, lens, (w, h), estimate_zoom)
        keep = res < TRIM_PX
        if keep.sum() >= MIN_INLIERS and keep.sum() < len(keep):
            pose, res = solve_pose(rays[keep], o[keep], pose, lens, (w, h), estimate_zoom, iters=3)
        n_in = int(keep.sum()) if keep.sum() >= MIN_INLIERS else int(inl.sum())
        rms = float(np.sqrt(np.mean(np.square(res)))) if len(res) else 0.0
        rekey = (n_in < REKEY_INLIER_FRAC * len(self.key_px)
                 or abs(math.log(pose.zoom / max(1e-6, self.key_pose.zoom))) > REKEY_ZOOM)
        return VisionResult(True, pose, inliers=n_in, tracked=len(gi), rms=rms, rekey=rekey)


# ----------------------------------------------------------------------------- fusion
class PoseFusion:
    def __init__(self, camera_id: str, clock: Callable[[], float] = _now):
        self.camera_id = camera_id
        self.clock = clock
        self.lock = threading.RLock()
        self.pose: Pose = IDENTITY
        self.ptz_known = False
        self.estimate_zoom = False
        self.lens_override: Optional[Lens] = None          # tests
        self._lens_used: Optional[Lens] = None
        self.vision = VisionTracker()
        self.vision_state = "off"                           # off | ok | lost | idle
        self._ht: List[float] = []
        self._hp: List[Pose] = []
        self.frame_i = 0
        self.last_frame_t: Optional[float] = None
        self.vel = (0.0, 0.0)                               # deg/s pan, tilt
        self._vis = (0.0, 0.0)                              # integrated vision-only motion (deg)
        self._vh: List[Tuple[float, float, float]] = []
        self.speed = 0.0
        self.last_motion_t = -1e9
        self.last_command_t = -1e9
        self.commanding = False
        self.settled = True
        self.lost_during_move = False
        self.move_start: Optional[Tuple[Pose, Optional[Tuple[float, float]]]] = None
        self.expected: Optional[Pose] = None
        self.preset_poses: Dict[str, Pose] = {}
        # ONVIF
        self.onvif_supported: Optional[bool] = None
        self.onvif_raw: Optional[Tuple[float, float, Optional[float]]] = None   # pan deg, tilt deg, zoom generic
        self.onvif_t = -1e9
        self.onvif_moving: Optional[bool] = None
        self.offset: Optional[Tuple[float, float]] = None
        self.sign = [1.0, 1.0]
        self.onvif_stale = False
        self._stale_votes = 0
        self.scale_hint: Optional[float] = None
        self.pending_check = None                           # (move start, settle time)
        self.settled_at = -1e9
        self.settled_readings: List[Tuple[float, float]] = []   # raw ONVIF degrees since settling
        self.align_open = False                             # offset still being averaged
        self.last_move_lost = False
        self.pose_uncertain = False                         # restored from disk, not measured yet
        # drift bookkeeping
        self.drift = 0.0
        self.drift_measured = False
        self.rekeys_since_fix = 0
        self.travel_since_fix = 0.0
        self.calibration: Dict[str, Any] = {"state": "idle"}
        self.on_settle: Optional[Callable[[], None]] = None
        self._mask_boxes: Callable[[], List[Tuple[float, float, float, float]]] = lambda: []

    # ------------------------------------------------------------------ settings
    def lens(self) -> Lens:
        if self.lens_override is not None:
            return self.lens_override
        from . import settings_for
        s = settings_for(self.camera_id)
        return Lens(float(s.get("fov_h", 100.0)), float(s.get("fov_v", 56.0)))

    def _ranges(self) -> Tuple[float, float, float]:
        from . import settings_for
        s = settings_for(self.camera_id)
        return float(s.get("pan_range", 360.0)), float(s.get("tilt_range", 114.0)), float(s.get("zoom_max", 1.0))

    # ------------------------------------------------------------------ queries
    def current(self) -> Pose:
        with self.lock:
            return self.pose

    def pose_at(self, t: Optional[float]) -> Pose:
        with self.lock:
            if t is None or not self._ht:
                return self.pose
            i = bisect.bisect_left(self._ht, t)
            if i >= len(self._ht):
                return self._hp[-1]
            if i == 0 or self._ht[i] == t:
                return self._hp[i]
            t0, t1 = self._ht[i - 1], self._ht[i]
            a, b = self._hp[i - 1], self._hp[i]
            k = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return Pose(a.pan + (b.pan - a.pan) * k, a.tilt + (b.tilt - a.tilt) * k, a.zoom + (b.zoom - a.zoom) * k)

    def speed_at(self, t: Optional[float]) -> float:
        """Angular speed (deg/s) around time ``t``."""
        with self.lock:
            if t is None or len(self._ht) < 2:
                return self.speed
            i = min(max(1, bisect.bisect_left(self._ht, t)), len(self._ht) - 1)
            j = max(0, i - 3)
            dt = self._ht[i] - self._ht[j]
            if dt <= 1e-6:
                return self.speed
            a, b = self._hp[j], self._hp[i]
            return math.hypot(geo.wrap180(b.pan - a.pan) * math.cos(math.radians(b.tilt)), b.tilt - a.tilt) / dt

    def _record(self, t: float) -> None:
        if self._ht and t <= self._ht[-1]:
            # same / older timestamp (e.g. a skipped frame re-used): update the last entry
            if t == self._ht[-1]:
                self._hp[-1] = self.pose
            return
        self._ht.append(t)
        self._hp.append(self.pose)
        if len(self._ht) > 64 and self._ht[0] < t - HISTORY_S:
            k = bisect.bisect_left(self._ht, t - HISTORY_S)
            del self._ht[:k]
            del self._hp[:k]

    # ------------------------------------------------------------------ pose updates
    def _shift(self, new: Pose) -> None:
        """Move the pose (and the vision keyframe by the same rotation) to ``new``."""
        old = self.pose
        self.pose = new
        if self.vision.has_key:
            kp = self.vision.key_pose
            self.vision.rebase(Pose(kp.pan + (new.pan - old.pan), kp.tilt + (new.tilt - old.tilt),
                                    kp.zoom * (new.zoom / max(1e-6, old.zoom))), self.lens())

    def set_pose(self, pose: Pose, t: Optional[float] = None) -> None:
        """Hard set (restore / tests). Resets vision."""
        with self.lock:
            self.pose = pose
            self.vision.reset()
            if t is not None:
                self._record(t)

    def _moving_hint(self, now: float) -> bool:
        return (self.commanding or now - self.last_command_t < SETTLE_S + 1.0 or self.speed > IDLE_SPEED
                or self.vision_state == "lost" or not self.settled)

    def process(self, frame: np.ndarray, t: float, mask_boxes=None) -> None:
        """Vision step for one camera frame captured at ``t`` (perf_counter)."""
        with self.lock:
            self.frame_i += 1
            now = t
            moving = self._moving_hint(now)
            if not moving and self.vision.has_key and self.frame_i % IDLE_EVERY:
                self._fuse_onvif(now, 0.0)
                self._record(t)
                return
            lens = self.lens()
            if lens != self._lens_used:
                # FOV calibration changed: keyframe rays must use the new intrinsics
                self._lens_used = lens
                self.vision.rebase(self.vision.key_pose, lens)
            gray = small_gray(frame)
            dt = 0.0 if self.last_frame_t is None else max(0.0, min(0.5, t - self.last_frame_t))
            self.last_frame_t = t
            if not self.vision.has_key:
                self.vision.rekey(gray, self.pose, lens, self._mask(gray, mask_boxes))
                self.vision_state = "ok" if self.vision.has_key else "lost"
                if not self.vision.has_key and not self.settled:
                    self.lost_during_move = True
                self._fuse_onvif(now, dt)
                self._settle_check(now)
                self._record(t)
                return
            pred = Pose(self.pose.pan + self.vel[0] * dt, self.pose.tilt + self.vel[1] * dt, self.pose.zoom)
            res = self.vision.track(gray, pred, lens, self.estimate_zoom)
            old = self.pose
            if res.ok:
                new = res.pose
                fx, fy = lens.focal(old.zoom)
                px_deg = max(1e-6, gray.shape[1] * fx * math.pi / 180.0)    # small px per degree at center
                moved_px = math.hypot(geo.wrap180(new.pan - old.pan) * math.cos(math.radians(old.tilt)),
                                      new.tilt - old.tilt) * px_deg
                quiet = not self.commanding and now - self.last_command_t > SETTLE_S
                if quiet and moved_px < DEADBAND_PX and abs(math.log(new.zoom / old.zoom)) < 1e-3:
                    new = old
                self.pose = new
                self.vision_state = "ok"
                if res.rekey:
                    if self.vision.rekey(gray, new, lens, self._mask(gray, mask_boxes)):
                        self.rekeys_since_fix += 1
            else:
                self.vision_state = "lost"
                if not self.settled or self.commanding:
                    self.lost_during_move = True
                self.vision.rekey(gray, self.pose, lens, self._mask(gray, mask_boxes))
            if dt > 0:
                dpan = geo.wrap180(self.pose.pan - old.pan) / dt
                dtil = (self.pose.tilt - old.tilt) / dt
                a = 1.0 - math.exp(-dt / 0.12)
                self.vel = (self.vel[0] + (dpan - self.vel[0]) * a, self.vel[1] + (dtil - self.vel[1]) * a)
                inst = math.hypot(dpan * math.cos(math.radians(self.pose.tilt)), dtil)
                if res.ok:
                    self.travel_since_fix += inst * dt
                # Speed from vision-measured motion only (drift corrections are not camera motion),
                # over a ~0.3 s baseline so per-frame tracking jitter does not look like motion.
                cos_t = math.cos(math.radians(self.pose.tilt))
                self._vis = (self._vis[0] + geo.wrap180(self.pose.pan - old.pan) * cos_t,
                             self._vis[1] + self.pose.tilt - old.tilt)
                self._vh.append((t, self._vis[0], self._vis[1]))
                while len(self._vh) > 2 and self._vh[1][0] <= t - SPEED_BASE_S:
                    self._vh.pop(0)
                t0, v0, w0 = self._vh[0]
                span = t - t0
                base = math.hypot(self._vis[0] - v0, self._vis[1] - w0) / span if span > 0.5 * SPEED_BASE_S else inst
                self.speed += (base - self.speed) * min(1.0, 2.0 * a)
            if not res.ok:
                self.vel = (0.0, 0.0)
            if self.speed > IDLE_SPEED:
                self.last_motion_t = now
            self._fuse_onvif(now, dt)
            self._settle_check(now)
            self._record(t)

    def _mask(self, gray: np.ndarray, mask_boxes) -> Optional[np.ndarray]:
        boxes = mask_boxes() if callable(mask_boxes) else (mask_boxes or [])
        if not boxes:
            return None
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[max(0, int(y1 * h) - 2):min(h, int(y2 * h) + 3), max(0, int(x1 * w) - 2):min(w, int(x2 * w) + 3)] = 0
        return mask

    # ------------------------------------------------------------------ settle / priors / ONVIF
    def _begin_move(self) -> None:
        self.settled = False
        self.move_start = (self.pose, self._onvif_mean() or self._onvif_world_free())
        self.lost_during_move = False
        self.settled_readings = []
        self.align_open = False

    def _settle_check(self, now: float) -> None:
        recent_onvif_moving = self.onvif_moving is True and now - self.onvif_t < 1.0
        settled = (not self.commanding and now - self.last_command_t > SETTLE_S
                   and now - self.last_motion_t > 0.5 and not recent_onvif_moving)
        if settled and not self.settled:
            self.settled = True
            self.settled_at = now
            self._on_settled(now)
        elif not settled and self.settled:
            self._begin_move()

    def _onvif_world_free(self) -> Optional[Tuple[float, float]]:
        """Last ONVIF reading (degrees, sign applied, no offset) if usable."""
        if self.onvif_raw is None or self.onvif_stale:
            return None
        return self.onvif_raw[0] * self.sign[0], self.onvif_raw[1] * self.sign[1]

    def _onvif_mean(self) -> Optional[Tuple[float, float]]:
        """Mean of the readings taken since the camera settled (noise averaging)."""
        if not self.settled_readings or self.onvif_stale:
            return None
        n = len(self.settled_readings)
        ref = self.settled_readings[-1][0]
        pan = sum(ref + geo.wrap180(r[0] - ref) for r in self.settled_readings) / n
        tilt = sum(r[1] for r in self.settled_readings) / n
        return pan * self.sign[0], tilt * self.sign[1]

    def _on_settled(self, now: float) -> None:
        start = self.move_start
        self.move_start = None
        self.last_move_lost = self.lost_during_move
        if self.expected is not None and self.lost_during_move:
            # vision lost the move: trust the commanded target
            self._shift(self.expected)
        self.expected = None
        self.lost_during_move = False
        # ONVIF plausibility is checked with the first reading taken after settling
        if start is not None and start[1] is not None:
            self.pending_check = (start, now)
        if self.on_settle is not None:
            try:
                self.on_settle()
            except Exception:
                pass

    def _reading_settled(self, now: float) -> bool:
        return (self.settled and self.onvif_moving is not True and not self.commanding
                and now > self.settled_at and now > self.last_command_t + SETTLE_S)

    def on_onvif_position(self, pos: Optional[dict]) -> None:
        """Called from the PTZ worker thread with ``backend.get_position()``."""
        with self.lock:
            now = self.clock()
            if pos is None:
                return
            try:
                x, y = float(pos["x"]), float(pos["y"])
            except (KeyError, TypeError, ValueError):
                return
            if not (math.isfinite(x) and math.isfinite(y)):
                return
            pan_range, tilt_range, _zmax = self._ranges()
            zoom = pos.get("zoom")
            z = float(zoom) if isinstance(zoom, (int, float)) and math.isfinite(zoom) else None
            prev = self.onvif_raw
            self.onvif_raw = (x * pan_range / 2.0, y * tilt_range / 2.0, z)
            self.onvif_t = now
            self.onvif_moving = pos.get("moving")
            self.onvif_supported = True
            if self.onvif_stale and prev is not None and (abs(prev[0] - self.onvif_raw[0]) > 0.3
                                                          or abs(prev[1] - self.onvif_raw[1]) > 0.3):
                self.onvif_stale = False      # it does report changes after all
                self._stale_votes = 0
            if not self.ptz_known or not self._reading_settled(now):
                return
            self.settled_readings.append(self.onvif_raw[:2])
            del self.settled_readings[:-ONVIF_AVG_N]
            if self.pending_check is not None:
                start, _ = self.pending_check
                self.pending_check = None
                self._check_onvif_follow(start)
            if self.offset is None or self.align_open:
                self._align_offset()

    def _check_onvif_follow(self, start) -> None:
        """Did the reported position follow the last move? Detects frozen values and inverted axes."""
        s_pose, s_onv = start
        cur = self._onvif_world_free()
        if cur is None or self.vision_state != "ok" or self.last_move_lost:
            return
        dv = (geo.wrap180(self.pose.pan - s_pose.pan), self.pose.tilt - s_pose.tilt)
        do = (geo.wrap180(cur[0] - s_onv[0]), cur[1] - s_onv[1])
        if math.hypot(*dv) <= 3.0:
            return
        if math.hypot(*do) < 0.3:
            self._stale_votes += 1
            if self._stale_votes >= 2:
                self.onvif_stale = True
            return
        self._stale_votes = 0
        flipped = False
        for k in (0, 1):
            if abs(dv[k]) > 3.0 and abs(do[k]) > 1.0 and dv[k] * do[k] < 0:
                self.sign[k] = -self.sign[k]
                flipped = True
        if flipped:
            self.offset = None          # re-align with the corrected sign
        elif abs(dv[0]) > 3.0 and abs(do[0]) > 1.0:
            self.scale_hint = round(dv[0] / do[0], 3)

    def _align_offset(self) -> None:
        """Tie the ONVIF frame to the world frame of the current pose (refined while the camera
        stays where it was aligned, so the offset averages the reading noise)."""
        w = self._onvif_mean() or self._onvif_world_free()
        if w is None:
            return
        first = self.offset is None
        self.offset = (self.pose.pan - w[0], self.pose.tilt - w[1])
        self.align_open = True
        if first:
            self.drift = 0.0
            self.drift_measured = True
            self.rekeys_since_fix = 0
            self.travel_since_fix = 0.0
            self.pose_uncertain = False
            self._notify_dirty()

    def onvif_target(self, averaged: bool = True) -> Optional[Pose]:
        w = (self._onvif_mean() if averaged else None) or self._onvif_world_free()
        if w is None or self.offset is None:
            return None
        pan = w[0] + self.offset[0]
        pan += 360.0 * round((self.pose.pan - pan) / 360.0)
        zoom = self.pose.zoom
        if self.estimate_zoom and self.onvif_raw and self.onvif_raw[2] is not None:
            zmax = self._ranges()[2]
            if zmax > 1.0:
                zoom = 1.0 + max(0.0, min(1.0, self.onvif_raw[2])) * (zmax - 1.0)
        return Pose(pan, max(-89.0, min(89.0, w[1] + self.offset[1])), zoom)

    def _fuse_onvif(self, now: float, dt: float) -> None:
        if self.onvif_raw is None or self.onvif_stale or now - self.onvif_t > ONVIF_FRESH_S:
            return
        if self.offset is None or self.align_open or self.pending_check is not None:
            return
        if self.vision_state == "lost" or not self.vision.has_key:
            # vision blind: ONVIF is all we have, also during moves (latest reading, not averaged)
            target = self.onvif_target(averaged=False)
            if target is not None:
                self._shift(target)
            return
        if not self.settled or not self.settled_readings:
            return
        target = self.onvif_target()
        err_pan = geo.wrap180(target.pan - self.pose.pan)
        err_tilt = target.tilt - self.pose.tilt
        err = math.hypot(err_pan, err_tilt)
        self.drift = err
        self.drift_measured = True
        if err > ONVIF_SNAP and (self.pose_uncertain or self.last_move_lost):
            self._shift(target)          # restored / blind pose: the camera's absolute position wins
            self.pose_uncertain = False
        elif err > ONVIF_DEADBAND and dt > 0:
            k = 1.0 - math.exp(-dt / ONVIF_TAU_S)
            step = min(1.0, ONVIF_MAX_RATE * dt / err) if err * k > ONVIF_MAX_RATE * dt else k
            self._shift(Pose(self.pose.pan + err_pan * step, self.pose.tilt + err_tilt * step, self.pose.zoom))
        if err < 0.5:
            self.rekeys_since_fix = 0
            self.travel_since_fix = 0.0

    # ------------------------------------------------------------------ commands (PTZ worker thread)
    def on_command(self, kind: str, *args) -> None:
        with self.lock:
            now = self.clock()
            if kind == "connected":
                caps = args[0] if args else {}
                if caps.get("pan_tilt", True):
                    if not self.ptz_known:
                        self.ptz_known = True
                        self._notify_dirty()
                self.estimate_zoom = bool(caps.get("zoom"))
                return
            if kind == "motion":
                x, y, z = args[0]
                self.commanding = bool(x or y or z)
                self.last_command_t = now
            elif kind == "stop":
                self.commanding = False
                self.last_command_t = now
            elif kind == "relative":
                dpan, dtilt = args[0]
                self.expected = Pose(self.pose.pan + dpan, max(-89.0, min(89.0, self.pose.tilt + dtilt)), self.pose.zoom)
                self.last_command_t = now
            elif kind == "goto":
                self.expected = self.preset_poses.get(str(args[0]))
                self.last_command_t = now
            elif kind == "home":
                self.expected = None
                self.last_command_t = now
            elif kind == "preset_saved":
                name, presets = args[0], args[1]
                for p in presets or []:
                    if p.get("name") == name:
                        self.preset_poses[str(p["token"])] = self.pose
                self._notify_dirty()
                return
            elif kind == "preset_removed":
                self.preset_poses.pop(str(args[0]), None)
                return
            if self.settled:
                self._begin_move()

    def wants_fast_poll(self) -> bool:
        with self.lock:
            return not self.settled or self.commanding or self.pending_check is not None

    # ------------------------------------------------------------------ user actions
    def reanchor(self) -> dict:
        """Accept the current alignment: new keyframe, ONVIF re-aligned, drift estimate cleared."""
        with self.lock:
            self.vision.reset()
            self.rekeys_since_fix = 0
            self.travel_since_fix = 0.0
            self.drift = 0.0
            self.expected = None
            if self._onvif_world_free() is not None:
                self._align_offset()
            self._notify_dirty()
            return self.status()

    def sync_onvif(self) -> bool:
        with self.lock:
            target = self.onvif_target()
            if target is None:
                return False
            self._shift(target)
            self.align_open = False
            self.drift = 0.0
            self.rekeys_since_fix = 0
            self.travel_since_fix = 0.0
            self._notify_dirty()
            return True

    def _notify_dirty(self) -> None:
        try:
            from .. import persistence
            persistence.mark_dirty()
        except Exception:
            pass

    # ------------------------------------------------------------------ status / persistence
    def source(self) -> str:
        onvif = self.onvif_raw is not None and not self.onvif_stale and self.offset is not None and \
            self.clock() - self.onvif_t < ONVIF_FRESH_S
        vis = self.vision_state == "ok"
        return "both" if onvif and vis else "onvif" if onvif else "vision" if vis else "none"

    def drift_estimate(self) -> Tuple[float, str]:
        if self.drift_measured and self.source() in ("both", "onvif"):
            return self.drift, "measured"
        return DRIFT_PER_REKEY * self.rekeys_since_fix + DRIFT_PER_DEG * self.travel_since_fix, "estimated"

    def status(self) -> dict:
        with self.lock:
            lens = self.lens()
            fh, fv = lens.effective_fov(self.pose.zoom)
            drift, kind = self.drift_estimate()
            onvif_pos = None
            if self.onvif_raw is not None:
                onvif_pos = {"pan": round(self.onvif_raw[0], 3), "tilt": round(self.onvif_raw[1], 3)}
            return {
                "ptz_known": self.ptz_known,
                "pose": self.pose.as_dict(),
                "fov_h": round(fh, 3),
                "fov_v": round(fv, 3),
                "source": self.source(),
                "vision": self.vision_state,
                "moving": not self.settled,
                "speed": round(self.speed, 2),
                "drift_deg": round(drift, 3),
                "drift_kind": kind,
                "onvif": {
                    "supported": self.onvif_supported,
                    "stale": self.onvif_stale,
                    "aligned": self.offset is not None,
                    "position": onvif_pos,
                    "age_s": round(self.clock() - self.onvif_t, 1) if self.onvif_raw is not None else None,
                    "inverted": [self.sign[0] < 0, self.sign[1] < 0],
                    "scale_hint": self.scale_hint,
                },
                "calibration": dict(self.calibration),
            }

    def dump(self) -> dict:
        with self.lock:
            out: Dict[str, Any] = {"pose": self.pose.as_dict(), "ptz": self.ptz_known,
                                   "sign": list(self.sign)}
            if self.offset is not None:
                out["offset"] = [round(self.offset[0], 4), round(self.offset[1], 4)]
            if self.preset_poses:
                out["presets"] = {k: p.as_dict() for k, p in self.preset_poses.items()}
            return out

    def restore(self, data: dict) -> None:
        with self.lock:
            p = geo.pose_from(data.get("pose"))
            if p is not None:
                self.pose = p
                self.vision.reset()
                self.pose_uncertain = True       # the camera may have moved while the server was off
            self.ptz_known = bool(data.get("ptz", self.ptz_known))
            off = data.get("offset")
            if isinstance(off, (list, tuple)) and len(off) == 2 and all(isinstance(v, (int, float)) for v in off):
                self.offset = (float(off[0]), float(off[1]))
            sign = data.get("sign")
            if isinstance(sign, (list, tuple)) and len(sign) == 2:
                self.sign = [-1.0 if s == -1 else 1.0 for s in sign]
            presets = data.get("presets")
            if isinstance(presets, dict):
                for k, v in presets.items():
                    pp = geo.pose_from(v)
                    if pp is not None:
                        self.preset_poses[str(k)] = pp


def clean_dump(raw: Any) -> dict:
    """Validate a persisted ``{camera: dump}`` mapping (unknown / bad entries dropped)."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for cam, data in raw.items():
        if isinstance(cam, str) and isinstance(data, dict) and geo.pose_from(data.get("pose")) is not None:
            out[cam] = data
    return out


# ----------------------------------------------------------------------------- registry
_lock = threading.Lock()
_fusions: Dict[str, PoseFusion] = {}


def get(camera_id: str) -> PoseFusion:
    with _lock:
        f = _fusions.get(camera_id)
        if f is None:
            f = _fusions[camera_id] = PoseFusion(camera_id)
        return f


def peek(camera_id: str) -> Optional[PoseFusion]:
    return _fusions.get(camera_id)


def reset() -> None:
    with _lock:
        _fusions.clear()


def anchor_enabled(camera_state) -> bool:
    settings = getattr(camera_state, "esp_settings", None) or {}
    v = settings.get("world_anchor", True)
    return v is True or (isinstance(v, (int, float)) and bool(v))


def active(camera_state) -> bool:
    f = _fusions.get(getattr(camera_state, "camera_id", ""))
    return f is not None and f.ptz_known and anchor_enabled(camera_state)


def _detection_boxes(camera_state):
    def boxes():
        st = getattr(camera_state, "esp_motion", None)
        if st is None or not getattr(st, "frame_w", 0) or not getattr(st, "frame_h", 0):
            return []
        fw, fh = float(st.frame_w), float(st.frame_h)
        try:
            tracks = list(st.tracks.values())
        except RuntimeError:
            return []
        return [(t.obs[0] / fw, t.obs[1] / fh, t.obs[2] / fw, t.obs[3] / fh) for t in tracks if t.missing_since is None]
    return boxes


def on_frame(camera_state, frame, ts: float) -> None:
    """Reader-thread hook: one vision step per camera frame (no-op unless anchoring is active)."""
    if frame is None or not active(camera_state):
        f = _fusions.get(getattr(camera_state, "camera_id", ""))
        if f is not None:
            f.vision_state = "off"
        return
    try:
        get(camera_state.camera_id).process(frame, ts, _detection_boxes(camera_state))
    except Exception as exc:  # pragma: no cover - never break the reader
        print(f"[PTZ pose] {camera_state.camera_id}: {exc.__class__.__name__}: {exc}")


def attach(controller) -> None:
    """Wire a PtzController's callbacks to the camera's fusion."""
    f = get(controller.camera_id)
    f.on_settle = lambda: f._notify_dirty()
    controller.position_listener = f.on_onvif_position
    controller.command_listener = f.on_command
    controller.fast_poll = f.wants_fast_poll


# ----------------------------------------------------------------------------- FOV calibration
def calibrate_fov(camera_id: str, controller, degrees: float = 20.0, timeout_s: float = 20.0,
                  sleep=time.sleep, update_settings=None) -> None:
    """Background job: pan by ``degrees`` (relative move) and back, compare with vision, fix the FOV.

    The true angle is the ONVIF position delta when the camera reports one, else
    the commanded angle (via the calibrated pan range). Results land in
    ``fusion.calibration``.
    """
    f = get(camera_id)

    def wait_settled(t_start: float) -> bool:
        deadline = time.monotonic() + timeout_s
        sleep(SETTLE_S * 0.6)
        while time.monotonic() < deadline:
            with f.lock:
                ok = f.settled and f.last_command_t >= t_start - 1e-6
            if ok:
                return True
            sleep(0.1)
        return False

    def fail(msg: str) -> None:
        with f.lock:
            f.calibration = {"state": "failed", "message": msg}

    with f.lock:
        if f.calibration.get("state") == "running":
            return
        f.calibration = {"state": "running", "message": f"Panning {degrees:.0f}° and back"}
        if f.vision_state not in ("ok", "idle"):
            f.calibration = {"state": "failed", "message": "The picture has too little texture to track."}
            return
        p0 = f.pose
        o0 = f._onvif_world_free()
        lens0 = f.lens()
    try:
        t_cmd = f.clock()
        controller.relative_degrees(degrees, 0.0)
        if not wait_settled(t_cmd):
            fail("The camera did not settle in time.")
            return
        with f.lock:
            p1 = f.pose
            o1 = f._onvif_world_free() if f.onvif_t > t_cmd else None
            lost = f.lost_during_move or f.vision_state != "ok"
        t_back = f.clock()
        controller.relative_degrees(-degrees, 0.0)
        if lost:
            fail("Tracking was lost during the move; aim at a more textured scene.")
            return
        measured = geo.wrap180(p1.pan - p0.pan)
        true = degrees
        if o0 is not None and o1 is not None and abs(geo.wrap180(o1[0] - o0[0])) > 1.0:
            true = abs(geo.wrap180(o1[0] - o0[0])) * math.copysign(1.0, degrees)
        if abs(measured) < 1.0 or measured * true <= 0:
            fail("No usable motion was measured.")
            return
        fx0, fy0 = lens0.focal(1.0)
        ratio = math.tan(math.radians(abs(measured))) / math.tan(math.radians(abs(true)))
        fx1, fy1 = fx0 * ratio, fy0 * ratio
        fov_h = math.degrees(2.0 * math.atan(0.5 / fx1))
        fov_v = math.degrees(2.0 * math.atan(0.5 / fy1))
        if update_settings is None:
            from . import update_settings as update_settings_fn
        else:
            update_settings_fn = update_settings
        new = update_settings_fn(camera_id, {"fov_h": round(fov_h, 2), "fov_v": round(fov_v, 2)})
        wait_settled(t_back)
        f.reanchor()
        with f.lock:
            f.calibration = {"state": "done", "message": f"Measured {abs(measured):.1f}° for {abs(true):.1f}°.",
                             "fov_h": new.get("fov_h", round(fov_h, 2)), "fov_v": new.get("fov_v", round(fov_v, 2))}
    except Exception as exc:
        fail(f"Calibration failed ({exc.__class__.__name__}).")
