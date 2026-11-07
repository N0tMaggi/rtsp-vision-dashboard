"""Render-time motion model for ESP boxes.

YOLO typically runs slower than the camera (e.g. 8-20 fps vs 25 fps), so drawing
the raw inference boxes makes them jump at inference rate. Instead every
detection feeds a ``Track``; each *rendered* frame then asks the tracks where
their box is at that frame's capture time:

* constant-velocity extrapolation from the last observations (velocity is an
  EMA of observed deltas), limited to a short horizon and to a fraction of the
  box size so it never overshoots wildly,
* exponential smoothing of the displayed box toward that target (time based, so
  it behaves the same at any render rate),
* fade-in for new tracks and a short fade-out grace period for lost ones.

Tracks are keyed by the tracker id when tracking is on; otherwise detections
are associated to existing tracks by IoU (same class), so smoothing and fades
still work, only identity across occlusions is weaker.

All functions mutate per-camera state and must be called with the camera's
ESP lock held. ``step`` returns immutable ``Detection`` snapshots that can be
drawn outside the lock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detections import Detection, HistoryPoint

FADE_IN_S = 0.18          # new boxes fade in over this time
FADE_OUT_S = 0.40         # lost boxes stay (fading) this long before they are dropped
IOU_MATCH = 0.25          # min IoU to associate untracked detections with a track
VEL_GAIN = 0.55           # EMA gain for the velocity estimate per observation
MAX_EXTRAP_FRACTION = 0.6  # extrapolated offset is capped at this fraction of the box size
MIN_HORIZON_S, MAX_HORIZON_S = 0.06, 0.40
MAX_TRACKS = 256
STALE_S = 3.0             # drop tracks not observed for this long (e.g. YOLO was paused)
# Ids of tracks without a tracker id start here, so they never collide with
# tracker ids (one id space for the targets API, lock and history).
UNTRACKED_ID_BASE = 1_000_000
PEEK_MAX_LEAD_S = 0.10    # peek() extrapolates the last rendered box at most this far


@dataclass
class Track:
    key: Any
    # Boxes are plain float lists [x1, y1, x2, y2]: for 4 numbers, Python floats
    # are several times faster than tiny numpy arrays and this runs per frame.
    det: Detection                 # last observed detection (metadata, keypoints)
    obs: List[float]               # last observed box
    obs_t: float                   # capture time of the frame the observation came from
    disp: List[float]              # currently displayed box
    disp_t: float                  # render time ``disp`` corresponds to
    born: float
    vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    interval: float = 0.1          # EMA of time between observations
    conf: float = 0.0
    missing_since: Optional[float] = None
    seen: int = 1
    tracked: bool = True           # key is a tracker id (else an internal UNTRACKED_ID_BASE+n id)


@dataclass
class MotionState:
    tracks: Dict[Any, Track] = field(default_factory=dict)
    next_uid: int = 0
    last_results: Any = None
    signature: Tuple = ()
    frame_w: int = 0               # size of the frame the track coordinates refer to
    frame_h: int = 0


def get_state(camera_state) -> MotionState:
    st = getattr(camera_state, "esp_motion", None)
    if not isinstance(st, MotionState):
        st = MotionState()
        camera_state.esp_motion = st
    return st


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _predict(tr: Track, t: float) -> List[float]:
    dt = min(max(0.0, t - tr.obs_t), MAX_HORIZON_S)
    return [o + v * dt for o, v in zip(tr.obs, tr.vel)]


def _box(det: Detection) -> List[float]:
    return [float(det.x1), float(det.y1), float(det.x2), float(det.y2)]


def ingest(st: MotionState, detections: Sequence[Detection], t_obs: float, now: float,
           history: Dict[Any, List[HistoryPoint]], max_hist: int) -> None:
    """Fold one inference result into the tracks (call once per new result)."""
    tracks = st.tracks
    matched: Dict[Any, Detection] = {}

    # 1) tracker ids
    untracked: List[Detection] = []
    for det in detections:
        if det.obj_id != -1 and det.obj_id not in matched:
            matched[det.obj_id] = det
        else:
            untracked.append(det)

    # 2) IoU association for detections without ids (greedy, same class)
    if untracked:
        free = [tr for k, tr in tracks.items() if k not in matched and not tr.tracked]
        pairs = []
        for di, det in enumerate(untracked):
            box = _box(det)
            for tr in free:
                if tr.det.cls_id != det.cls_id:
                    continue
                iou = _iou(box, _predict(tr, t_obs))
                if iou >= IOU_MATCH:
                    pairs.append((iou, di, tr.key))
        pairs.sort(key=lambda p: -p[0])
        used_det, used_key = set(), set()
        for _, di, key in pairs:
            if di in used_det or key in used_key:
                continue
            used_det.add(di)
            used_key.add(key)
            matched[key] = untracked[di]
        for di, det in enumerate(untracked):
            if di not in used_det:
                st.next_uid += 1
                matched[UNTRACKED_ID_BASE + st.next_uid] = det

    # 3) update / create tracks
    for key, det in matched.items():
        box = _box(det)
        tr = tracks.get(key)
        if tr is None:
            if len(tracks) >= MAX_TRACKS:
                continue
            tracks[key] = Track(key=key, det=det, obs=box, obs_t=t_obs, disp=list(box), disp_t=now,
                                born=now, conf=det.conf, tracked=det.obj_id != -1 and det.obj_id == key)
        else:
            dt = t_obs - tr.obs_t
            size = max(box[2] - box[0], box[3] - box[1], 1.0)
            jump = abs((box[0] + box[2]) - (tr.obs[0] + tr.obs[2])) * 0.5 + \
                abs((box[1] + box[3]) - (tr.obs[1] + tr.obs[3])) * 0.5
            if jump > 1.5 * size:
                # Implausible jump (id switch / re-acquired elsewhere): snap.
                tr.vel = [0.0, 0.0, 0.0, 0.0]
                tr.disp = list(box)
            elif dt > 1e-3:
                gain = VEL_GAIN if tr.seen > 1 else 0.35
                vmax = 3.0 * size
                tr.vel = [min(vmax, max(-vmax, v + ((b - o) / dt - v) * gain))
                          for v, b, o in zip(tr.vel, box, tr.obs)]
                tr.interval += (min(dt, 1.0) - tr.interval) * 0.3
            if tr.missing_since is not None:
                # Re-acquired: restart the fade-in from the current fade level.
                tr.born = now - FADE_IN_S * _fade_out(tr, now)
                tr.missing_since = None
            tr.det = det
            tr.obs = box
            tr.obs_t = max(t_obs, tr.obs_t)
            tr.conf += (det.conf - tr.conf) * 0.5
            tr.seen += 1

        hist = history.get(key)
        if hist is None:
            hist = history[key] = []
        hist.append(((det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2, t_obs))
        if len(hist) > max_hist:
            del hist[:-max_hist]

    for key, tr in tracks.items():
        if key not in matched and tr.missing_since is None:
            tr.missing_since = now


def mark_all_missing(st: MotionState, now: float) -> None:
    for tr in st.tracks.values():
        if tr.missing_since is None:
            tr.missing_since = now


def _alive(tr: Track, t: float) -> bool:
    return _fade_out(tr, t) > 0.0 and t - tr.obs_t <= max(STALE_S, 3.0 * tr.interval)


def peek(st: MotionState, t: float) -> List[Tuple[Any, Track, List[float]]]:
    """Read-only view of the visible tracks at time ``t``: (key, track, box).

    The box is the last rendered (smoothed/extrapolated) box, advanced by the
    track velocity for at most PEEK_MAX_LEAD_S; if nothing has been rendered
    recently, it is extrapolated from the last observation. Does not mutate
    any track, so it is safe to call at any rate (e.g. the targets API).
    """
    out = []
    for key, tr in st.tracks.items():
        if not _alive(tr, t):
            continue
        if t - tr.disp_t <= 0.25:
            base, dt = tr.disp, min(max(0.0, t - tr.disp_t), PEEK_MAX_LEAD_S)
        else:
            base, dt = tr.obs, min(max(0.0, t - tr.obs_t), MAX_HORIZON_S)
        out.append((key, tr, [b + v * dt for b, v in zip(base, tr.vel)]))
    return out


def _fade_out(tr: Track, t: float) -> float:
    if tr.missing_since is None:
        return 1.0
    return max(0.0, 1.0 - (t - tr.missing_since) / FADE_OUT_S)


def step(st: MotionState, t: float, smoothing: float, history: Dict[Any, List[HistoryPoint]],
         want_history: bool) -> List[Detection]:
    """Advance all tracks to render time ``t`` and return drawable detections."""
    out: List[Detection] = []
    dead = []
    tau = 0.015 + 0.17 * smoothing          # EMA time constant in seconds
    for key, tr in st.tracks.items():
        if not _alive(tr, t):
            dead.append(key)
            continue
        fade = _fade_out(tr, t)
        fade *= min(1.0, max(0.0, (t - tr.born) / FADE_IN_S)) if FADE_IN_S > 0 else 1.0

        obs = tr.obs
        if smoothing <= 0.0:
            tr.disp = list(obs)
        else:
            # Extrapolate over the observation's age (inference latency + time since
            # the last result), at most ~2 inference intervals, plus a lead of ~tau
            # that compensates the lag of the exponential follower below.
            horizon = min(MAX_HORIZON_S, max(MIN_HORIZON_S, 2.0 * tr.interval))
            dt_e = min(max(0.0, t - tr.obs_t), horizon) + 0.8 * tau
            dt_r = t - tr.disp_t
            if dt_r > 0.0:
                a = 1.0 - math.exp(-dt_r / tau)
                mx = MAX_EXTRAP_FRACTION * (obs[2] - obs[0])
                my = MAX_EXTRAP_FRACTION * (obs[3] - obs[1])
                disp = tr.disp
                for i in range(4):
                    m = mx if i % 2 == 0 else my
                    off = min(m, max(-m, tr.vel[i] * dt_e))
                    disp[i] += (obs[i] + off - disp[i]) * a
        if t > tr.disp_t:
            tr.disp_t = t

        out.append(_snapshot(tr, fade, history.get(key) if want_history else None))

    for key in dead:
        st.tracks.pop(key, None)
        history.pop(key, None)
    # history entries whose track vanished some other way
    if len(history) > len(st.tracks):
        for key in [k for k in history if k not in st.tracks]:
            history.pop(key, None)
    return out


def _snapshot(tr: Track, alpha: float, hist: Optional[List[HistoryPoint]]) -> Detection:
    d = tr.det
    disp, obs = tr.disp, tr.obs
    x1, y1, x2, y2 = int(round(disp[0])), int(round(disp[1])), int(round(disp[2])), int(round(disp[3]))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    keypoints = d.keypoints
    if keypoints is not None:
        ow, oh = max(obs[2] - obs[0], 1.0), max(obs[3] - obs[1], 1.0)
        sx, sy = (disp[2] - disp[0]) / ow, (disp[3] - disp[1]) / oh
        if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3 or abs(disp[0] - obs[0]) > 0.5 or abs(disp[1] - obs[1]) > 0.5:
            kp = (keypoints - (obs[0], obs[1])) * (sx, sy) + (disp[0], disp[1])
            keypoints = kp.astype(np.int32)
    vx = (tr.vel[0] + tr.vel[2]) * 0.5
    vy = (tr.vel[1] + tr.vel[3]) * 0.5
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    history: Tuple[HistoryPoint, ...] = ()
    if hist:
        history = tuple(hist)
    return Detection(
        index=d.index, obj_id=d.obj_id, cls_id=d.cls_id, cls_name=d.cls_name,
        x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy=cy, conf=tr.conf,
        keypoints=keypoints, kp_visible=d.kp_visible, history=history,
        alpha=alpha, vx=vx, vy=vy, track_key=tr.key, ref_height=int(obs[3] - obs[1]),
        mask=d.mask,   # box-relative: drawn onto the displayed box = same warp as the keypoints above
    )
