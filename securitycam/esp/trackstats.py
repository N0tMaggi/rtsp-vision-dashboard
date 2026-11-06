"""Per-track statistics: first seen, dwell (loitering), path length, speed, distance.

Stats live next to the motion model (``camera_state.esp_trackstats``, keyed by
the same stable track ids as ``/targets``) and are folded in once per new
observation of a track, so the per-frame cost is a dict walk over the tracks.

Units
-----
* ``path`` / ``speed_n``: normalized frame units (diagonal-free euclidean in 0..1 coords).
* ``distance``: the same rough estimate the ESP label shows (``1000 / box height px``).
* ``path_m`` / ``speed``: meters, assuming a ~1.7 m tall target, i.e. one pixel at the
  target spans ``REF_HEIGHT_M / box_height_px`` meters (pinhole model consistent with
  the distance estimate), plus the change of the estimated distance (depth).

Jitter
------
YOLO boxes jitter by a few pixels per inference. Path length therefore only
grows when the target left a dead-band around the last path anchor
(``PATH_DEADBAND`` x box height), and dwell is measured against a slowly
following anchor, so a person pacing in a small area still counts as loitering.

All functions must be called with the camera's ESP lock held.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DISTANCE_K = 1000.0        # distance_m ~= DISTANCE_K / box_height_px (same as the label)
REF_HEIGHT_M = 1.7         # assumed target height for pixel -> meter conversion
PATH_DEADBAND = 0.25       # path grows once the center moved > this x box height
DWELL_RADIUS = 1.0         # leaving the dwell anchor by > this x box height resets dwell
DWELL_ANCHOR_TAU = 8.0     # s; the dwell anchor follows the target this slowly
DEPTH_TAU = 1.5            # s; smoothing of the distance estimate (box height is noisy)
SPEED_WINDOW = 1.0         # s; speed = displacement over this baseline (robust to box jitter)
SPEED_TAU = 0.5            # s; extra smoothing of the reported speed


def estimate_distance_m(height_px: float) -> float:
    return DISTANCE_K / max(float(height_px), 1.0)


@dataclass
class TrackStats:
    first_seen: float          # wall clock (time.time()) of the first observation
    first_t: float             # perf_counter of the first observation
    last_t: float              # perf_counter of the last folded observation
    cx: float                  # last observed center (px)
    cy: float
    height: float              # last observed box height (px)
    depth: float               # smoothed distance estimate (m)
    path_ax: float             # path anchor (px) and its depth
    path_ay: float
    path_ad: float
    path_n: float = 0.0        # accumulated path, normalized units
    path_m: float = 0.0        # accumulated path, meters
    dwell_ax: float = 0.0      # slowly following dwell anchor (px)
    dwell_ay: float = 0.0
    dwell_since: float = 0.0   # perf_counter when the target settled at the anchor
    speed_n: float = 0.0       # smoothed speed, normalized units / s
    speed_m: float = 0.0       # smoothed speed, m / s
    samples: deque = field(default_factory=lambda: deque(maxlen=64))  # (t, cx, cy, depth)


def get_stats(camera_state) -> Dict[Any, TrackStats]:
    stats = getattr(camera_state, "esp_trackstats", None)
    if not isinstance(stats, dict):
        stats = camera_state.esp_trackstats = {}
    return stats


def _new(tr, cx: float, cy: float, h: float, t: float, wall_now: float, t_now: float) -> TrackStats:
    d = estimate_distance_m(h)
    st = TrackStats(
        first_seen=wall_now - max(0.0, t_now - t), first_t=t, last_t=t,
        cx=cx, cy=cy, height=h, depth=d,
        path_ax=cx, path_ay=cy, path_ad=d,
        dwell_ax=cx, dwell_ay=cy, dwell_since=t,
    )
    st.samples.append((t, cx, cy, d))
    return st


def update(camera_state, mst, t_now: float, wall_now: Optional[float] = None) -> Dict[Any, TrackStats]:
    """Fold the latest observation of every track in ``mst`` (a MotionState) into its stats.

    Cheap to call every rendered frame: tracks whose observation did not change
    since the last call are skipped. Stats of vanished tracks are dropped.
    """
    stats = get_stats(camera_state)
    fw, fh = mst.frame_w, mst.frame_h
    if fw <= 0 or fh <= 0:
        return stats
    wall_now = time.time() if wall_now is None else wall_now
    tracks = mst.tracks
    for key, tr in tracks.items():
        obs = tr.obs
        t = tr.obs_t
        cx, cy = (obs[0] + obs[2]) * 0.5, (obs[1] + obs[3]) * 0.5
        h = max(1.0, obs[3] - obs[1])
        s = stats.get(key)
        if s is None:
            stats[key] = _new(tr, cx, cy, h, t, wall_now, t_now)
            continue
        dt = t - s.last_t
        if dt <= 1e-4:
            continue
        # smoothed depth (box height is noisy)
        d_raw = estimate_distance_m(h)
        s.depth += (d_raw - s.depth) * (1.0 - math.exp(-dt / DEPTH_TAU))
        m_per_px = REF_HEIGHT_M / h

        # speed: displacement over a ~1 s baseline (per-step deltas are dominated by jitter)
        smp = s.samples
        smp.append((t, cx, cy, s.depth))
        while len(smp) > 2 and t - smp[1][0] >= SPEED_WINDOW:
            smp.popleft()
        t0, x0, y0, d0 = smp[0]
        span = t - t0
        if span >= 0.25:
            lat_px = math.hypot(cx - x0, cy - y0)
            v_n = math.hypot((cx - x0) / fw, (cy - y0) / fh) / span
            v_m = min(15.0, math.hypot(lat_px * m_per_px, s.depth - d0) / span)
            a = 1.0 - math.exp(-dt / SPEED_TAU)
            s.speed_n += (v_n - s.speed_n) * a
            s.speed_m += (v_m - s.speed_m) * a

        # path length with a dead-band around the last anchor
        far = math.hypot(cx - s.path_ax, cy - s.path_ay)
        if far > PATH_DEADBAND * h:
            s.path_n += math.hypot((cx - s.path_ax) / fw, (cy - s.path_ay) / fh)
            s.path_m += math.hypot(far * m_per_px, s.depth - s.path_ad)
            s.path_ax, s.path_ay, s.path_ad = cx, cy, s.depth

        # dwell: anchor follows slowly; leaving its radius restarts the dwell timer
        k = 1.0 - math.exp(-dt / DWELL_ANCHOR_TAU)
        s.dwell_ax += (cx - s.dwell_ax) * k
        s.dwell_ay += (cy - s.dwell_ay) * k
        if math.hypot(cx - s.dwell_ax, cy - s.dwell_ay) > DWELL_RADIUS * h:
            s.dwell_ax, s.dwell_ay, s.dwell_since = cx, cy, t

        s.cx, s.cy, s.height, s.last_t = cx, cy, h, t

    if len(stats) > len(tracks):
        for key in [k for k in stats if k not in tracks]:
            del stats[key]
    return stats


def dwell_seconds(s: Optional[TrackStats], t_now: float) -> float:
    return max(0.0, t_now - s.dwell_since) if s is not None else 0.0


def format_dwell(seconds: float) -> str:
    """``m:ss`` (``h:mm:ss`` from one hour on)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def target_fields(camera_state, key, tr, t_now: float) -> Dict[str, Any]:
    """Additive ``/targets`` fields for one track (distance always, stats once known)."""
    h = max(1.0, tr.obs[3] - tr.obs[1])
    out: Dict[str, Any] = {"distance": round(estimate_distance_m(h), 2)}
    s = get_stats(camera_state).get(key)
    if s is None:
        return out
    out.update({
        "first_seen": round(s.first_seen, 3),
        "age": round(max(0.0, t_now - s.first_t), 1),
        "dwell": round(dwell_seconds(s, t_now), 1),
        "path": round(s.path_n, 4),
        "path_m": round(s.path_m, 2),
        "speed": round(s.speed_m, 2),
        "speed_n": round(s.speed_n, 4),
    })
    return out
