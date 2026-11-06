"""Render-path hooks for the ESP pipeline (cheap lookups only, no inference).

``prepare()`` is called once per rendered frame and returns a small context;
the other functions take that context. All coordinates coming from the
recognition worker are in source-frame pixels and get scaled by
``coord_scale`` to the rendered frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..esp import renderers as R
from ..esp.colors import ANTHRACITE, DetectionColors, mix
from .common import FaceHit, PlateRead, RecogSettings, TrackInfo, is_pet, runtime
from .worker import ensure_worker

FRESH_S = 1.5        # worker results older than this are not drawn / associated


@dataclass
class FrameContext:
    s: RecogSettings
    k: float
    track_info: Dict[int, TrackInfo] = field(default_factory=dict)
    faces: List[FaceHit] = field(default_factory=list)
    plates: List[Tuple[Tuple[int, int, int, int], PlateRead, Optional[int]]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.s.known_faces or self.s.plates or not self.s.pets_show


def prepare(camera_state, coord_scale: float = 1.0, now: Optional[float] = None) -> FrameContext:
    s = RecogSettings.from_mapping(camera_state.esp_settings)
    ctx = FrameContext(s=s, k=float(coord_scale or 1.0))
    if not s.any_worker_feature:
        return ctx
    rt = runtime(camera_state)
    ensure_worker(camera_state)
    now = time.perf_counter() if now is None else now
    with rt.lock:
        ctx.track_info = dict(rt.track_info)
        if s.known_faces and now - rt.faces_ts <= FRESH_S:
            ctx.faces = list(rt.faces)
        if s.plates and now - rt.plates_ts <= FRESH_S:
            ctx.plates = list(rt.plates_now)
    return ctx


def filter_detections(ctx: FrameContext, detections):
    if ctx.s.pets_show:
        return detections
    return [d for d in detections if not is_pet(d.cls_name)]


def adjust_palettes(ctx: FrameContext, detections, palettes, rgb_mode: bool = False):
    if rgb_mode:
        return palettes
    out = list(palettes)
    for i, d in enumerate(detections):
        if is_pet(d.cls_name):
            c = ctx.s.color_pet
            out[i] = DetectionColors(box=c, text=out[i].text, fill=c, skeleton=out[i].skeleton)
    return out


def _scaled(box, k: float):
    return tuple(int(v * k) for v in box)


def _iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def name_for(ctx: FrameContext, det) -> Optional[str]:
    if det.cls_name.lower() != "person":
        return None
    if det.obj_id != -1:
        ti = ctx.track_info.get(det.obj_id)
        if ti is not None and ti.name:
            return ti.name
    # untracked (or not yet attributed): face centre inside the upper half of the box
    best = None
    for hit in ctx.faces:
        if not hit.name:
            continue
        fx, fy, fw, fh = _scaled(hit.box, ctx.k)
        cx, cy = fx + fw / 2, fy + fh / 2
        if det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y1 + 0.5 * (det.y2 - det.y1):
            if best is None or hit.score > best.score:
                best = hit
    return best.name if best else None


def plate_for(ctx: FrameContext, det) -> Optional[str]:
    if det.obj_id != -1:
        ti = ctx.track_info.get(det.obj_id)
        if ti is not None and ti.plate_best:
            return ti.plate_best
    box = (det.x1, det.y1, det.x2, det.y2)
    best, best_iou = None, 0.3
    for vbox, read, _ in ctx.plates:
        iou = _iou(box, _scaled(vbox, ctx.k))
        if iou > best_iou:
            best, best_iou = read.text, iou
    return best


def decorate_label(ctx: FrameContext, det, runs: list, text_color) -> list:
    """Append name / plate / pet marker runs to a detection label."""
    if not ctx.active and not is_pet(det.cls_name):
        return runs
    extra = []
    if ctx.s.known_faces:
        name = name_for(ctx, det)
        if name:
            extra.append((name.upper(), True, ctx.s.color_known))
    if ctx.s.plates and det.cls_name.lower() in ("car", "truck", "bus", "motorcycle"):
        plate = plate_for(ctx, det)
        if plate:
            extra.append((plate, True, text_color))
    if is_pet(det.cls_name):
        extra.append(("PET", True, ctx.s.color_pet))
    if not extra:
        return runs
    out = list(runs)
    sep = mix(text_color, ANTHRACITE, 0.55)
    for run in extra:
        if out:
            out.append(("  ", False, sep))
        out.append(run)
    return out


def draw_faces(ctx: FrameContext, frame, faces: Sequence[Tuple[int, int, int, int]], settings, st,
               placed: list) -> bool:
    """Draw known/unknown face tags. Returns False if the default face boxes should be drawn."""
    if not ctx.s.known_faces or not ctx.faces:
        return False
    h, w = frame.shape[:2]
    hits = [(_scaled(f.box, ctx.k), f) for f in ctx.faces]
    # Prefer the fresh render-path boxes (no worker latency) and attach names to them.
    base = list(faces) if faces else [b for b, _ in hits]
    for (fx, fy, fw, fh) in base:
        cx, cy = fx + fw / 2, fy + fh / 2
        hit = None
        best_d = None
        for (hx, hy, hw, hh), f in hits:
            d = ((hx + hw / 2 - cx) ** 2 + (hy + hh / 2 - cy) ** 2) ** 0.5
            if d <= 0.6 * max(fw, hw) and (best_d is None or d < best_d):
                hit, best_d = f, d
        if hit is not None and hit.name:
            color, text = ctx.s.color_known, hit.name.upper()
        elif hit is not None and hit.recognizable:
            color, text = ctx.s.color_unknown, "UNKNOWN"
        else:
            color, text = settings.color_face, "FACE"
        corners = R._box_geometry(fx, fy, fx + fw, fy + fh, "corner")[0]
        R.cv2.polylines(frame, corners, False, ANTHRACITE, st.thin + 2, R.AA)
        R.cv2.polylines(frame, corners, False, color, st.thin, R.AA)
        tag = R.make_tag([(text, True, settings.color_text)], st, color, 1.0, settings.label_background)
        if tag is not None:
            R.place_tag(tag, (fx, fy, fx + fw, fy + fh), w, h, st, placed)
            R.draw_tag(frame, tag)
    return True
