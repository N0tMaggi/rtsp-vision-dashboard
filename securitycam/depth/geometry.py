"""Image point + metric depth -> position on the floor plan (pure numpy).

Same camera model as ``geo`` (pinhole, x right / y up / z forward, pan about
the vertical, tilt about the camera's horizontal axis). A target at normalized
image point (u, v) with z-depth ``z`` (meters along the optical axis) is the
camera-space point ``z * ((u - .5) / fx, (.5 - v) / fy, 1)``. Rotating it by
the camera pose gives a world point; its horizontal components (x, z) are the
floor-plan position.

* ``ground``: the camera's own horizontal frame (tilt removed, pan = 0), i.e.
  "x meters right, z meters ahead of the camera".
* ``ground_world``: rotated by the pan as well (PTZ with world anchoring), so a
  standing person keeps the same map position while the camera pans.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geo import Lens, Pose, rotation


def lens_for_fov(fov_h: float, frame_w: int, frame_h: int) -> Lens:
    """Lens with the given horizontal FOV and a square-pixel vertical FOV for the frame aspect."""
    fov_h = min(170.0, max(10.0, float(fov_h)))
    aspect = (frame_h / frame_w) if frame_w > 0 and frame_h > 0 else 9.0 / 16.0
    fov_v = math.degrees(2.0 * math.atan(math.tan(math.radians(fov_h) * 0.5) * aspect))
    return Lens(fov_h, fov_v)


def camera_point(u: float, v: float, z: float, lens: Lens, zoom: float = 1.0) -> np.ndarray:
    fx, fy = lens.focal(zoom)
    return np.array([(u - 0.5) / fx * z, (0.5 - v) / fy * z, z], np.float64)


def ground_point(u: float, v: float, z: float, lens: Lens, pose: Pose = Pose(),
                 world: bool = False) -> Tuple[float, float]:
    """Floor-plan (x, z) in meters: camera frame (pan ignored) or world frame (``world=True``)."""
    p = camera_point(u, v, z, lens, pose.zoom)
    rot = rotation(pose if world else Pose(0.0, pose.tilt, pose.zoom))
    w = rot @ p
    return float(w[0]), float(w[2])


def bearing_deg(x: float, z: float) -> float:
    """Azimuth of a floor-plan point (0 = straight ahead, + = right)."""
    return math.degrees(math.atan2(x, z))


def scan_points(depth_m: np.ndarray, lens: Lens, pose: Pose = Pose(), world: bool = False,
                cols: int = 48, rows: int = 27, cell: float = 0.3, max_range: float = 40.0,
                max_points: int = 260) -> List[List[float]]:
    """Coarse top-down occupancy from a metric depth map: [[x, z, weight 0..1], ...].

    A grid of ``cols`` x ``rows`` pixels is projected onto the floor plan and
    binned into ``cell`` meter squares; the strongest cells are returned.
    """
    if depth_m is None or depth_m.ndim != 2 or depth_m.size == 0:
        return []
    h, w = depth_m.shape
    ys = ((np.arange(rows) + 0.5) / rows * h).astype(np.int32).clip(0, h - 1)
    xs = ((np.arange(cols) + 0.5) / cols * w).astype(np.int32).clip(0, w - 1)
    z = depth_m[np.ix_(ys, xs)].astype(np.float64)
    uu, vv = np.meshgrid((xs + 0.5) / w, (ys + 0.5) / h)
    ok = np.isfinite(z) & (z > 0.2) & (z < max_range)
    if not ok.any():
        return []
    fx, fy = lens.focal(pose.zoom)
    pts = np.stack([(uu[ok] - 0.5) / fx * z[ok], (0.5 - vv[ok]) / fy * z[ok], z[ok]], axis=1)
    rot = rotation(pose if world else Pose(0.0, pose.tilt, pose.zoom))
    wp = pts @ rot.T
    gx = np.round(wp[:, 0] / cell).astype(np.int64)
    gz = np.round(wp[:, 2] / cell).astype(np.int64)
    keys, counts = np.unique(np.stack([gx, gz], 1), axis=0, return_counts=True)
    order = np.argsort(-counts)[:max_points]
    top = counts[order[0]] if len(order) else 1
    return [[round(float(keys[i, 0] * cell), 2), round(float(keys[i, 1] * cell), 2),
             round(float(min(1.0, counts[i] / max(1, top))), 2)] for i in order]


def region_median(depth: np.ndarray, box: Sequence[float], fx0: float = 0.35, fx1: float = 0.65,
                  fy0: float = 0.25, fy1: float = 0.7) -> Optional[float]:
    """Median of ``depth`` over the central part of a normalized box (torso for a person)."""
    if depth is None or depth.ndim != 2:
        return None
    h, w = depth.shape
    x1, y1, x2, y2 = (float(b) for b in box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    c0 = int(max(0, min(w - 1, math.floor((x1 + bw * fx0) * w))))
    c1 = int(max(c0 + 1, min(w, math.ceil((x1 + bw * fx1) * w))))
    r0 = int(max(0, min(h - 1, math.floor((y1 + bh * fy0) * h))))
    r1 = int(max(r0 + 1, min(h, math.ceil((y1 + bh * fy1) * h))))
    patch = depth[r0:r1, c0:c1]
    if patch.size == 0:
        return None
    v = float(np.median(patch))
    return v if math.isfinite(v) else None
