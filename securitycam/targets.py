"""Live targets, target lock and user zones (backing the /targets, /lock, /zones API).

Everything reads/writes per-camera state under the camera's ESP lock, the
same lock the overlay renderer uses. Coordinates in the API are normalized
0..1 ([x1, y1, x2, y2]) relative to the frame.

Zones on PTZ cameras are world-anchored (``anchor.py``): a new / moved zone
stores the camera pose it was drawn at; responses describe it in the
*current* view (``box`` = projected bounding box, ``polygon``, ``in_view``)
plus ``drawn_box``, ``anchor`` and ``world`` (corner azimuth / elevation).
"""
from __future__ import annotations

import time
from typing import Any, Optional

from . import anchor
from .esp import aim as _aim
from .esp import motion, trackstats
from .esp.pipeline import _esp_lock
from .validation import ZONE_DEFAULT_COLOR, ZoneError, parse_zone_patch

MAX_ZONES = 16


def _frame_size(camera_state) -> tuple[int, int]:
    with camera_state.frame_lock:
        frame = camera_state.latest_frame
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return 0, 0
    return int(frame.shape[1]), int(frame.shape[0])


def _settings_filter(camera_state):
    s = camera_state.esp_settings
    person_only = bool(s.get("person_only", False))
    try:
        thr = float(s.get("confidence_threshold", 0.0))
    except (TypeError, ValueError):
        thr = 0.0
    return person_only, thr


def _visible_targets_locked(camera_state, t: float) -> list[dict[str, Any]]:
    """Visible targets (normalized). Caller holds the ESP lock."""
    if not camera_state.yolo_enabled:
        return []
    with camera_state.frame_lock:
        has_results = camera_state.latest_results is not None
    if not has_results:
        return []
    st = motion.get_state(camera_state)
    fw, fh = st.frame_w, st.frame_h
    if fw <= 0 or fh <= 0:
        return []
    person_only, thr = _settings_filter(camera_state)
    out = []
    for key, tr, box in motion.peek(st, t):
        det = tr.det
        if person_only and det.cls_name.lower() != "person":
            continue
        if det.conf < thr:
            continue
        x1 = min(1.0, max(0.0, box[0] / fw))
        y1 = min(1.0, max(0.0, box[1] / fh))
        x2 = min(1.0, max(0.0, box[2] / fw))
        y2 = min(1.0, max(0.0, box[3] / fh))
        if x2 <= x1 or y2 <= y1:
            continue
        aim, aim_src = _aim.compute(tr, box, fw, fh)
        out.append({
            "id": int(key),
            "label": det.cls_name,
            "conf": round(float(tr.conf), 3),
            "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            "aim": _aim.smoothed(camera_state, key, aim, t),
            "aim_src": aim_src,
            **trackstats.target_fields(camera_state, key, tr, t),
        })
    _aim.prune(camera_state, st.tracks)
    return out


def _validate_lock_locked(camera_state, targets: Optional[list]) -> Optional[int]:
    """Drop a lock whose track no longer exists (only while detection is running)."""
    locked = camera_state.esp_locked
    if locked is None or targets is None:
        return locked
    st = motion.get_state(camera_state)
    if locked not in st.tracks:
        camera_state.esp_locked = locked = None
    return locked


def targets_payload(camera_state) -> dict[str, Any]:
    t = time.perf_counter()
    w, h = _frame_size(camera_state)
    with _esp_lock(camera_state):
        targets = _visible_targets_locked(camera_state, t)
        running = camera_state.yolo_enabled and camera_state.latest_results is not None
        locked = _validate_lock_locked(camera_state, targets if running else None)
    out = {"ts": time.time(), "frame": {"w": w, "h": h}, "locked": locked, "targets": targets}
    view = anchor.view_for(camera_state)
    if view.active:
        fov_h, fov_v = view.lens.effective_fov(view.pose.zoom)
        out["pose"] = {**view.pose.as_dict(), "fov_h": round(fov_h, 3), "fov_v": round(fov_v, 3),
                       "moving": view.speed > 0.35}
    from .tags import decorate_payload   # tags, identity, solo / hidden (additive fields)

    out = decorate_payload(camera_state, out)
    # pose pack: iff / posture / action / gaze / fall / gestures (additive fields)
    from .esp.posepack import decorate_targets
    with _esp_lock(camera_state):
        return decorate_targets(camera_state, out)


def set_lock(camera_state, target_id: Optional[int]) -> bool:
    """Lock ``target_id`` (must be a currently visible target) or unlock with None."""
    with _esp_lock(camera_state):
        if target_id is None:
            camera_state.esp_locked = None
            return True
        visible = {tg["id"] for tg in _visible_targets_locked(camera_state, time.perf_counter())}
        if target_id not in visible:
            return False
        camera_state.esp_locked = int(target_id)
        return True


def get_locked(camera_state) -> Optional[int]:
    return camera_state.esp_locked


# ---------------------------------------------------------------------------- zones
def _count_inside(box, targets) -> int:
    x1, y1, x2, y2 = box
    n = 0
    for tg in targets:
        bx1, by1, bx2, by2 = tg["box"]
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            n += 1
    return n


def _zone_view_out(view, zone, targets) -> dict[str, Any]:
    out = dict(zone)
    out.update(anchor.zone_api_fields(view, zone))
    if not out["in_view"]:
        out["count"] = 0
    else:
        centers = [((tg["box"][0] + tg["box"][2]) * 0.5, (tg["box"][1] + tg["box"][3]) * 0.5) for tg in targets]
        poly = [tuple(p) for p in out["polygon"]]
        out["count"] = anchor.count_inside(poly, centers) if view.active and not zone_is_box(view, zone) \
            else _count_inside(out["box"], targets)
    return out


def zone_is_box(view, zone) -> bool:
    return view.identical(anchor.anchor_of(zone))


def zones_payload(camera_state) -> list[dict[str, Any]]:
    view = anchor.view_for(camera_state)
    with _esp_lock(camera_state):
        anchor.ensure_zone_anchors(camera_state)
        targets = _visible_targets_locked(camera_state, time.perf_counter())
        zones = [dict(z) for z in camera_state.esp_zones]
    return [_zone_view_out(view, z, targets) for z in zones]


def _zone_out(camera_state, zone) -> dict[str, Any]:
    targets = _visible_targets_locked(camera_state, time.perf_counter())
    return _zone_view_out(anchor.view_for(camera_state), zone, targets)


def create_zone(camera_state, data: Any) -> dict[str, Any]:
    fields = parse_zone_patch(data, require_box=True)   # raises ZoneError
    with _esp_lock(camera_state):
        if len(camera_state.esp_zones) >= MAX_ZONES:
            raise ZoneError("too_many_zones")
        camera_state.esp_zone_seq += 1
        n = camera_state.esp_zone_seq
        zone = {
            "id": f"z{n}",
            "label": fields.get("label") or f"Zone {n}",
            "box": fields["box"],
            "color": fields.get("color") or ZONE_DEFAULT_COLOR,
            "anchor": anchor.current_anchor(camera_state),   # drawn in the current view
        }
        camera_state.esp_zones.append(zone)
        return _zone_out(camera_state, zone)


def update_zone(camera_state, zone_id: str, data: Any) -> Optional[dict[str, Any]]:
    """Returns the updated zone, or None if ``zone_id`` is unknown (raises ZoneError on bad input)."""
    with _esp_lock(camera_state):
        zone = next((z for z in camera_state.esp_zones if z["id"] == zone_id), None)
        if zone is None:
            return None
        fields = parse_zone_patch(data, require_box=False)
        new = dict(zone)
        if "box" in fields:
            new["box"] = fields["box"]
            new["anchor"] = anchor.current_anchor(camera_state)
        if "label" in fields:
            new["label"] = fields["label"] or f"Zone {zone_id.lstrip('z')}"
        if "color" in fields:
            new["color"] = fields["color"]
        # replace instead of mutate: the renderer may hold a snapshot of the old dict
        idx = camera_state.esp_zones.index(zone)
        camera_state.esp_zones[idx] = new
        return _zone_out(camera_state, new)


def delete_zone(camera_state, zone_id: str) -> bool:
    with _esp_lock(camera_state):
        for i, z in enumerate(camera_state.esp_zones):
            if z["id"] == zone_id:
                del camera_state.esp_zones[i]
                return True
    return False
