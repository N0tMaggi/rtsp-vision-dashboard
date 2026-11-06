"""Live targets, target lock and user zones (backing the /targets, /lock, /zones API).

Everything reads/writes per-camera state under the camera's ESP lock, the
same lock the overlay renderer uses. Coordinates in the API are normalized
0..1 ([x1, y1, x2, y2]) relative to the frame.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .esp import motion
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
        out.append({
            "id": int(key),
            "label": det.cls_name,
            "conf": round(float(tr.conf), 3),
            "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
        })
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
    return {"ts": time.time(), "frame": {"w": w, "h": h}, "locked": locked, "targets": targets}


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


def zones_payload(camera_state) -> list[dict[str, Any]]:
    with _esp_lock(camera_state):
        targets = _visible_targets_locked(camera_state, time.perf_counter())
        zones = [dict(z) for z in camera_state.esp_zones]
    for z in zones:
        z["box"] = list(z["box"])
        z["count"] = _count_inside(z["box"], targets)
    return zones


def _zone_out(camera_state, zone) -> dict[str, Any]:
    out = dict(zone)
    out["box"] = list(zone["box"])
    targets = _visible_targets_locked(camera_state, time.perf_counter())
    out["count"] = _count_inside(zone["box"], targets)
    return out


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
