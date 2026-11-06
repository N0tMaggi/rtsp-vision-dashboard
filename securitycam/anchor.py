"""World anchoring of zones, tripwires and markers for PTZ cameras.

Every anchored shape keeps the image geometry it was drawn with (``box`` /
``a``, ``b``, normalized) plus the camera pose at that moment (``anchor``:
``{"pan", "tilt", "zoom"}``). Together with the lens (FOV calibration) that
defines fixed world directions (``geo``); each frame they are projected into
the current view (``View``). Shapes without an ``anchor`` count as drawn at
the identity pose.

When anchoring is off (no PTZ, ``world_anchor`` setting off) ``View.active``
is False and every helper returns the stored image geometry unchanged, so
fixed cameras behave exactly as before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geo
from .geo import IDENTITY, Lens, Pose


@dataclass(frozen=True)
class View:
    active: bool
    pose: Pose = IDENTITY
    lens: Lens = field(default_factory=Lens)
    speed: float = 0.0           # deg/s around the frame time
    t: Optional[float] = None
    camera_id: str = ""

    def identical(self, anchor: Pose) -> bool:
        return not self.active or anchor.close_to(self.pose, 1e-7)


INACTIVE = View(False)


def lens_for(camera_id: str) -> Lens:
    from . import ptz
    s = ptz.settings_for(camera_id)
    return Lens(float(s.get("fov_h", 100.0)), float(s.get("fov_v", 56.0)))


def view_for(camera_state, t: Optional[float] = None) -> View:
    """The view of ``camera_state`` at frame time ``t`` (perf_counter); current pose if None."""
    from .ptz import pose as ptz_pose
    if not ptz_pose.active(camera_state):
        return View(False, camera_id=getattr(camera_state, "camera_id", ""))
    f = ptz_pose.get(camera_state.camera_id)
    lens = f.lens_override or lens_for(camera_state.camera_id)
    return View(True, f.pose_at(t), lens, f.speed_at(t), t, camera_state.camera_id)


def pose_now(camera_state) -> Pose:
    """Pose new shapes are anchored at (the frozen last pose if anchoring is currently off)."""
    from .ptz import pose as ptz_pose
    f = ptz_pose.peek(getattr(camera_state, "camera_id", ""))
    return f.current() if f is not None else IDENTITY


def current_anchor(camera_state) -> Dict[str, float]:
    return pose_now(camera_state).as_dict()


def anchor_of(shape: Dict[str, Any]) -> Pose:
    return geo.pose_from(shape.get("anchor")) or IDENTITY


def clean_anchor(value: Any) -> Optional[Dict[str, float]]:
    p = geo.pose_from(value) if value is not None else None
    return p.as_dict() if p is not None else None


def pose_at(camera_state, t: Optional[float]) -> Pose:
    from .ptz import pose as ptz_pose
    f = ptz_pose.peek(getattr(camera_state, "camera_id", ""))
    return f.pose_at(t) if f is not None else IDENTITY


# ----------------------------------------------------------------------------- zones
def zone_world(zone: Dict[str, Any], lens: Lens) -> List[List[float]]:
    """World corners (az, el in degrees) of a zone, clockwise from its drawn top-left corner."""
    rays = geo.image_to_rays(geo.box_corners(zone["box"]), anchor_of(zone), lens)
    return [[round(float(a), 4), round(float(e), 4)] for a, e in geo.rays_to_angles(rays)]


def project_zone(view: View, zone: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[float]], bool]:
    """(polygon or None for "axis-aligned box", normalized bbox, in_view) of a zone in ``view``.

    The polygon is clipped to the frame; ``None`` means the stored box applies
    unchanged (anchoring off or the camera is exactly where the zone was drawn).
    """
    box = [float(v) for v in zone["box"]]
    anchor = anchor_of(zone)
    if view.identical(anchor):
        return None, box, True
    rays = geo.image_to_rays(geo.box_corners(box), anchor, view.lens)
    poly = geo.project_polygon(rays, view.pose, view.lens)
    if not poly:
        return [], None, False
    bb = geo.bbox(poly)
    if bb[2] - bb[0] < 1e-4 or bb[3] - bb[1] < 1e-4:
        return [], None, False
    return poly, bb, True


def project_zones(view: View, zones: Sequence[Dict[str, Any]], w: int, h: int) -> List[Dict[str, Any]]:
    """Copies of ``zones`` for drawing / alarm evaluation in ``view``.

    Adds ``_poly_px`` (pixel polygon, or None for a plain box) and ``_hidden``;
    ``box`` becomes the projected bounding box (normalized).
    """
    out = []
    for zone in zones:
        try:
            poly, bb, in_view = project_zone(view, zone)
        except (KeyError, TypeError, ValueError):
            continue
        z = dict(zone)
        if not in_view:
            z["_hidden"] = True
            z["_poly_px"] = None
        else:
            z["box"] = bb
            z["_poly_px"] = None if poly is None else [(u * w, v * h) for u, v in poly]
        out.append(z)
    return out


def zone_api_fields(view: View, zone: Dict[str, Any]) -> Dict[str, Any]:
    """Extra API fields: projected ``box`` / ``polygon`` for the current view, ``in_view``, world data."""
    poly, bb, in_view = project_zone(view, zone)
    drawn = [float(v) for v in zone["box"]]
    out: Dict[str, Any] = {
        "drawn_box": drawn,
        "anchor": anchor_of(zone).as_dict(),
        "anchored": view.active,
        "in_view": in_view,
    }
    if poly is None:
        out["polygon"] = [[round(u, 5), round(v, 5)] for u, v in geo.box_corners(drawn)]
        out["box"] = drawn
    elif in_view:
        out["polygon"] = [[round(u, 5), round(v, 5)] for u, v in poly]
        out["box"] = [round(v, 5) for v in bb]
    else:
        out["polygon"] = []
        out["box"] = drawn
    out["world"] = zone_world(zone, view.lens if view.active else _lens_or_default(view))
    return out


def _lens_or_default(view: View) -> Lens:
    if view.camera_id:
        return lens_for(view.camera_id)
    return view.lens


def count_inside(poly_or_box, points: Sequence[Tuple[float, float]]) -> int:
    """Number of points inside a polygon (list of (x, y)) or box [x1, y1, x2, y2]."""
    if not poly_or_box:
        return 0
    if len(poly_or_box) == 4 and not isinstance(poly_or_box[0], (tuple, list)):
        x1, y1, x2, y2 = poly_or_box
        return sum(1 for x, y in points if x1 <= x <= x2 and y1 <= y <= y2)
    return sum(1 for x, y in points if geo.point_in_polygon(x, y, poly_or_box))


# ----------------------------------------------------------------------------- tripwires
def project_wire(view: View, wire: Dict[str, Any]):
    """((ax, ay), (bx, by)) of a tripwire in ``view`` (normalized, clipped), or None if not visible."""
    a, b = tuple(wire["a"]), tuple(wire["b"])
    anchor = anchor_of(wire)
    if view.identical(anchor):
        return a, b
    rays = geo.image_to_rays(np.array([a, b], dtype=np.float64), anchor, view.lens)
    seg = geo.project_segment(rays, view.pose, view.lens)
    if seg is None or math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) < 1e-4:
        return None
    return seg


def wire_api_fields(view: View, wire: Dict[str, Any]) -> Dict[str, Any]:
    seg = project_wire(view, wire)
    lens = view.lens if view.active else _lens_or_default(view)
    rays = geo.image_to_rays(np.array([wire["a"], wire["b"]], dtype=np.float64), anchor_of(wire), lens)
    world = geo.rays_to_angles(rays)
    out = {
        "drawn_a": list(wire["a"]),
        "drawn_b": list(wire["b"]),
        "anchor": anchor_of(wire).as_dict(),
        "anchored": view.active,
        "in_view": seg is not None,
        "world": {"a": [round(float(world[0, 0]), 4), round(float(world[0, 1]), 4)],
                  "b": [round(float(world[1, 0]), 4), round(float(world[1, 1]), 4)]},
    }
    if seg is not None:
        out["a"] = [round(seg[0][0], 5), round(seg[0][1], 5)]
        out["b"] = [round(seg[1][0], 5), round(seg[1][1], 5)]
    return out


class PointMapper:
    """Maps observed image points (with their capture time) into another anchored view.

    Used by the alarm engine to evaluate tripwire crossings in a world-fixed
    plane: a point is un-projected with the pose of *its* frame and projected
    into the view the wire was drawn in, so camera motion cannot fake a crossing.
    """

    def __init__(self, camera_state, view: View):
        self.view = view
        self.camera_state = camera_state

    def rays(self, points_px: Sequence[Tuple[float, float, float]], w: int, h: int) -> Optional[np.ndarray]:
        if not self.view.active or not points_px:
            return None
        out = np.empty((len(points_px), 3))
        for i, (px, py, pt) in enumerate(points_px):
            pose = pose_at(self.camera_state, pt)
            out[i] = geo.image_to_rays(((px / w, py / h),), pose, self.view.lens)[0]
        return out

    def to_anchor(self, rays: np.ndarray, anchor: Pose, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        uv, front = geo.rays_to_image(rays, anchor, self.view.lens)
        return uv * np.array([w, h], dtype=np.float64), front


# ----------------------------------------------------------------------------- boxes (abandoned objects)
def transfer_box(box: Sequence[float], pose_from: Pose, pose_to: Pose, lens: Lens) -> Optional[List[float]]:
    """Bounding box of ``box`` (normalized) seen at ``pose_from``, as seen at ``pose_to`` (unclipped)."""
    if pose_from.close_to(pose_to, 1e-7):
        return [float(v) for v in box]
    uv, front = geo.transfer(geo.box_corners(box), pose_from, pose_to, lens)
    if not front.all():
        return None
    return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]


# ----------------------------------------------------------------------------- migration
def ensure_zone_anchors(camera_state) -> None:
    """Give zones without a valid ``anchor`` the current pose (call with the ESP lock held)."""
    zones = getattr(camera_state, "esp_zones", None)
    if not zones:
        return
    pose = None
    for i, z in enumerate(zones):
        if clean_anchor(z.get("anchor")) is None:
            if pose is None:
                pose = pose_now(camera_state).as_dict()
            zones[i] = dict(z, anchor=dict(pose))


def default_wire_anchor(camera_id: str) -> Dict[str, float]:
    from .ptz import pose as ptz_pose
    f = ptz_pose.peek(camera_id)
    return (f.current() if f is not None else IDENTITY).as_dict()
