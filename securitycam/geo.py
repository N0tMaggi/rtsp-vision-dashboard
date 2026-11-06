"""Camera-world geometry for pan/tilt/zoom cameras (pure numpy, no state).

World frame: the PTZ head's pivot. A world direction is a unit vector
``(x, y, z)`` with x = right, y = up, z = forward at pan = tilt = 0, or the
angles ``(az, el)`` in degrees (azimuth to the right, elevation up).

Camera model: ideal pinhole rotating about its optical center. ``pan`` turns
about the world vertical axis, ``tilt`` about the camera's horizontal axis
(no roll, like a PTZ head). Intrinsics come from the horizontal / vertical
field of view at zoom 1 (``Lens``); ``zoom`` multiplies both focal lengths.
Normalized image coordinates ``(u, v)`` are 0..1 from the top-left corner.

Straight image lines are great circles in the world, so a polygon or segment
projected from one view into another is still exactly a polygon / segment
(only its vertices need projecting) as long as it stays in front of the
camera; the parts behind are clipped in camera space before projecting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

NEAR = 1e-3            # camera-space z below which a ray counts as "behind"


@dataclass(frozen=True)
class Lens:
    fov_h: float = 100.0      # degrees at zoom 1
    fov_v: float = 56.0

    def focal(self, zoom: float = 1.0) -> Tuple[float, float]:
        """Focal lengths in normalized image units (u per unit tan, v per unit tan)."""
        fh = min(179.0, max(1.0, float(self.fov_h)))
        fv = min(179.0, max(1.0, float(self.fov_v)))
        z = max(1e-3, float(zoom))
        return 0.5 * z / math.tan(math.radians(fh) * 0.5), 0.5 * z / math.tan(math.radians(fv) * 0.5)

    def effective_fov(self, zoom: float = 1.0) -> Tuple[float, float]:
        fx, fy = self.focal(zoom)
        return math.degrees(2.0 * math.atan(0.5 / fx)), math.degrees(2.0 * math.atan(0.5 / fy))


@dataclass(frozen=True)
class Pose:
    pan: float = 0.0          # degrees, + = right
    tilt: float = 0.0         # degrees, + = up
    zoom: float = 1.0         # focal multiplier (1 = widest)

    def as_dict(self) -> dict:
        return {"pan": round(self.pan, 4), "tilt": round(self.tilt, 4), "zoom": round(self.zoom, 4)}

    def close_to(self, other: "Pose", eps: float = 1e-6) -> bool:
        return (abs(self.pan - other.pan) <= eps and abs(self.tilt - other.tilt) <= eps
                and abs(self.zoom - other.zoom) <= eps * max(1.0, self.zoom))


IDENTITY = Pose()


def pose_from(value) -> Optional[Pose]:
    """Pose from a dict / sequence (persisted data); None if malformed."""
    try:
        if isinstance(value, Pose):
            return value
        if isinstance(value, dict):
            pan, tilt, zoom = value.get("pan", 0.0), value.get("tilt", 0.0), value.get("zoom", 1.0)
        else:
            pan, tilt, zoom = value
        vals = [float(pan), float(tilt), float(zoom)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals) or vals[2] <= 0 or abs(vals[0]) > 1e5 or abs(vals[1]) > 1e5:
        return None
    return Pose(vals[0], max(-90.0, min(90.0, vals[1])), max(0.05, min(100.0, vals[2])))


# ----------------------------------------------------------------------------- rotations
def rotation(pose: Pose) -> np.ndarray:
    """3x3 matrix taking camera-frame vectors to world vectors (R = Ry(pan) @ Rx(tilt))."""
    p, t = math.radians(pose.pan), math.radians(pose.tilt)
    cp, sp, ct, st = math.cos(p), math.sin(p), math.cos(t), math.sin(t)
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, ct, st), (0.0, -st, ct)))
    return ry @ rx


def _as_points(points) -> np.ndarray:
    a = np.asarray(points, dtype=np.float64)
    return a.reshape(-1, 2)


# ----------------------------------------------------------------------------- image <-> world
def image_to_rays(uv, pose: Pose, lens: Lens) -> np.ndarray:
    """(N, 2) normalized image points -> (N, 3) world unit vectors."""
    uv = _as_points(uv)
    fx, fy = lens.focal(pose.zoom)
    cam = np.empty((len(uv), 3))
    cam[:, 0] = (uv[:, 0] - 0.5) / fx
    cam[:, 1] = (0.5 - uv[:, 1]) / fy
    cam[:, 2] = 1.0
    cam /= np.linalg.norm(cam, axis=1, keepdims=True)
    return cam @ rotation(pose).T


def camera_coords(rays, pose: Pose) -> np.ndarray:
    """World vectors -> camera-frame vectors (x right, y up, z forward)."""
    return np.asarray(rays, dtype=np.float64).reshape(-1, 3) @ rotation(pose)


def rays_to_image(rays, pose: Pose, lens: Lens) -> Tuple[np.ndarray, np.ndarray]:
    """World vectors -> (normalized (N, 2) image points, (N,) bool "in front of the camera").

    Points behind the camera get NaN coordinates.
    """
    cam = camera_coords(rays, pose)
    fx, fy = lens.focal(pose.zoom)
    z = cam[:, 2]
    front = z > NEAR
    zs = np.where(front, z, np.nan)
    uv = np.empty((len(cam), 2))
    uv[:, 0] = 0.5 + fx * cam[:, 0] / zs
    uv[:, 1] = 0.5 - fy * cam[:, 1] / zs
    return uv, front


def rays_to_angles(rays) -> np.ndarray:
    r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    az = np.degrees(np.arctan2(r[:, 0], r[:, 2]))
    el = np.degrees(np.arcsin(np.clip(r[:, 1], -1.0, 1.0)))
    return np.stack([az, el], axis=1)


def angles_to_rays(azel) -> np.ndarray:
    a = np.radians(_as_points(azel))
    ce = np.cos(a[:, 1])
    return np.stack([ce * np.sin(a[:, 0]), np.sin(a[:, 1]), ce * np.cos(a[:, 0])], axis=1)


def image_to_world(uv, pose: Pose, lens: Lens) -> np.ndarray:
    """Normalized image points -> (N, 2) (azimuth, elevation) in degrees."""
    return rays_to_angles(image_to_rays(uv, pose, lens))


def world_to_image(azel, pose: Pose, lens: Lens) -> Tuple[np.ndarray, np.ndarray]:
    return rays_to_image(angles_to_rays(azel), pose, lens)


def transfer(uv, pose_from: Pose, pose_to: Pose, lens: Lens) -> Tuple[np.ndarray, np.ndarray]:
    """Map image points seen at ``pose_from`` into the image at ``pose_to``."""
    return rays_to_image(image_to_rays(uv, pose_from, lens), pose_to, lens)


def pose_axis(pose: Pose) -> np.ndarray:
    return rotation(pose)[:, 2]


def pose_of_axis(axis, zoom: float = 1.0, pan_hint: float = 0.0) -> Pose:
    """Pose whose optical axis points along ``axis``; pan is unwrapped next to ``pan_hint``."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) or 1.0)
    pan = math.degrees(math.atan2(a[0], a[2]))
    pan += 360.0 * round((pan_hint - pan) / 360.0)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, a[1]))))
    return Pose(pan, tilt, zoom)


def angle_between(pose_a: Pose, pose_b: Pose) -> float:
    """Angle in degrees between the optical axes of two poses."""
    d = float(np.dot(pose_axis(pose_a), pose_axis(pose_b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------- clipping
def _clip_near(cam: np.ndarray, closed: bool) -> List[np.ndarray]:
    """Clip a polygon (closed) or polyline (open) in camera space against z >= NEAR."""
    n = len(cam)
    if n == 0:
        return []
    out: List[np.ndarray] = []
    if closed:
        for i in range(n):
            cur, nxt = cam[i], cam[(i + 1) % n]
            cin, nin = cur[2] >= NEAR, nxt[2] >= NEAR
            if cin:
                out.append(cur)
            if cin != nin:
                t = (NEAR - cur[2]) / (nxt[2] - cur[2])
                out.append(cur + (nxt - cur) * t)
        return out
    # open segment(s): only 2-point segments are used here
    a, b = cam[0], cam[-1]
    ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
    if not ain and not bin_:
        return []
    if ain and bin_:
        return [a, b]
    t = (NEAR - a[2]) / (b[2] - a[2])
    m = a + (b - a) * t
    return [a, m] if ain else [m, b]


def _project_cam(pts: Sequence[np.ndarray], fx: float, fy: float) -> List[Tuple[float, float]]:
    return [(0.5 + fx * p[0] / p[2], 0.5 - fy * p[1] / p[2]) for p in pts]


def clip_polygon_rect(poly: Sequence[Tuple[float, float]], x0=0.0, y0=0.0, x1=1.0, y1=1.0) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clip of a polygon against an axis-aligned rectangle."""
    def clip(points, inside, inter):
        out = []
        n = len(points)
        for i in range(n):
            cur, prev = points[i], points[i - 1]
            ci, pi = inside(cur), inside(prev)
            if ci:
                if not pi:
                    out.append(inter(prev, cur))
                out.append(cur)
            elif pi:
                out.append(inter(prev, cur))
        return out

    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + (b[1] - a[1]) * t)

    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + (b[0] - a[0]) * t, y)

    pts = list(poly)
    for inside, inter in (
        (lambda p: p[0] >= x0, lambda a, b: ix(a, b, x0)),
        (lambda p: p[0] <= x1, lambda a, b: ix(a, b, x1)),
        (lambda p: p[1] >= y0, lambda a, b: iy(a, b, y0)),
        (lambda p: p[1] <= y1, lambda a, b: iy(a, b, y1)),
    ):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return pts


def clip_segment_rect(a, b, x0=0.0, y0=0.0, x1=1.0, y1=1.0):
    """Liang-Barsky; returns the clipped (a, b) or None."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return (ax + dx * t0, ay + dy * t0), (ax + dx * t1, ay + dy * t1)


def project_polygon(rays, pose: Pose, lens: Lens, clip: bool = True) -> List[Tuple[float, float]]:
    """World polygon (vertex rays) -> normalized image polygon at ``pose``.

    Edges are great circles, so they stay straight. Parts behind the camera are
    removed; with ``clip`` the result is also clipped to the image (0..1).
    An empty list means "not visible".
    """
    cam = camera_coords(rays, pose)
    pts = _clip_near(cam, closed=True)
    if len(pts) < 3:
        return []
    fx, fy = lens.focal(pose.zoom)
    poly = _project_cam(pts, fx, fy)
    if clip:
        poly = clip_polygon_rect(poly)
    return poly if len(poly) >= 3 else []


def project_segment(rays, pose: Pose, lens: Lens, clip: bool = True):
    """World segment (2 rays) -> clipped normalized image segment ((u, v), (u, v)) or None."""
    cam = camera_coords(rays, pose)
    pts = _clip_near(cam, closed=False)
    if len(pts) < 2:
        return None
    fx, fy = lens.focal(pose.zoom)
    a, b = _project_cam(pts, fx, fy)
    if not clip:
        return a, b
    return clip_segment_rect(a, b)


def box_corners(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def bbox(points: Iterable[Tuple[float, float]]) -> Optional[List[float]]:
    pts = list(points)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def point_in_polygon(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd rule; points on the boundary may go either way."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def homography(pose_from: Pose, pose_to: Pose, lens: Lens, size_from: Tuple[int, int],
               size_to: Tuple[int, int]) -> Optional[np.ndarray]:
    """3x3 pixel homography mapping the image at ``pose_from`` onto ``pose_to`` (pure rotation + zoom).

    ``size`` = (width, height) in pixels of the respective raster. Pixel centers:
    u = (x + 0.5) / w.
    """
    fx0, fy0 = lens.focal(pose_from.zoom)
    fx1, fy1 = lens.focal(pose_to.zoom)
    w0, h0 = size_from
    w1, h1 = size_to
    # pixel -> camera ray at pose_from
    k0_inv = np.array(((1.0 / (fx0 * w0), 0.0, (0.5 / w0 - 0.5) / fx0),
                       (0.0, -1.0 / (fy0 * h0), (0.5 - 0.5 / h0) / fy0),
                       (0.0, 0.0, 1.0)))
    # camera ray at pose_to -> pixel (homogeneous)
    k1 = np.array(((fx1 * w1, 0.0, 0.5 * w1 - 0.5),
                   (0.0, -fy1 * h1, 0.5 * h1 - 0.5),
                   (0.0, 0.0, 1.0)))
    r = rotation(pose_to).T @ rotation(pose_from)
    hmat = k1 @ r @ k0_inv
    if not np.all(np.isfinite(hmat)) or abs(hmat[2, 2]) < 1e-12:
        return None
    return hmat / hmat[2, 2]
