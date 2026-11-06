"""Per-feature drawing primitives. All functions draw in place on ``frame``.

Look: anti-aliased strokes scaled to the frame resolution (``Style``), compact
anthracite label pills with a monospace font, a soft translucent layer for glow
and secondary geometry, and opaque crisp strokes on top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace as dc_replace
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .colors import ANTHRACITE, BGR, WHITE, mix
from .detections import Detection, HistoryPoint
from .sensors import FaceBox, HandPoints, clip_rect
from .text import line_height, rounded_mask, separator, text_mask

# COCO keypoint limb connections
SKELETON_LIMBS = (
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
)
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12

# MediaPipe 21-point hand connections
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
)

AA = cv2.LINE_AA
VELOCITY_ARROW_SECONDS = 0.5
PREDICTION_SECONDS = 1.0
FACE_ZOOM_PAD = 0.2
SOFT_ALPHA = 0.30         # opacity of the soft layer (glow, faint frame, 3d back face)
LABEL_BG_ALPHA = 0.78
FILL_ALPHA = 0.12


# --------------------------------------------------------------------------- style
@dataclass(frozen=True)
class Style:
    """Resolution-scaled stroke and type metrics (720p is the 1.0 reference)."""
    ui: float
    t: int            # main stroke
    thin: int         # secondary strokes
    glow: int         # glow stroke width
    font_px: int
    pad_x: int
    pad_y: int
    radius: int
    gap: int
    accent_w: int
    margin: int       # screen-edge margin for insets


@lru_cache(maxsize=64)
def style_for(w: int, h: int, line_thickness: int) -> Style:
    ui = max(0.6, min(w / 1280.0, h / 720.0))
    t = max(1, int(round(line_thickness * ui)))
    font_px = max(10, int(round(14 * ui)))
    return Style(
        ui=ui,
        t=t,
        thin=max(1, int(round(t * 0.5))),
        glow=max(3, int(round(t * 3.5))),
        font_px=font_px,
        pad_x=max(3, int(round(font_px * 0.5))),
        pad_y=max(2, int(round(font_px * 0.22))),
        radius=max(2, int(round(font_px * 0.32))),
        gap=max(2, int(round(4 * ui))),
        accent_w=max(2, int(round(2.5 * ui))),
        margin=max(6, int(round(12 * ui))),
    )


# --------------------------------------------------------------------------- helpers
def _clamp_point(x: float, y: float, w: int, h: int, margin: int) -> Tuple[int, int]:
    """Keep far-off projections representable (cv2 rejects huge ints)."""
    return (int(max(-margin, min(w + margin, x))), int(max(-margin, min(h + margin, y))))


def blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: BGR, alpha: float) -> None:
    """Alpha-blend a solid rect into the frame, in place on the (clipped) ROI."""
    if alpha <= 0.0:
        return
    h, w = frame.shape[:2]
    rect = clip_rect(x1, y1, x2 - x1, y2 - y1, w, h)
    if rect is None:
        return
    rx, ry, rw, rh = rect
    roi = frame[ry:ry + rh, rx:rx + rw]
    if alpha >= 1.0:
        roi[:] = color
        return
    cv2.addWeighted(roi, 1.0 - alpha, _solid(color, rh, rw), alpha, 0, dst=roi)


_SOLIDS: dict = {}


def _solid(color: BGR, h: int, w: int) -> np.ndarray:
    """Read-only solid-color image of at least h x w (cached; filling a 3-channel
    color per call is slower than the blend itself)."""
    key = tuple(int(c) for c in color)
    img = _SOLIDS.get(key)
    if img is None or img.shape[0] < h or img.shape[1] < w:
        hh = max(h, img.shape[0] if img is not None else 0)
        ww = max(w, img.shape[1] if img is not None else 0)
        img = np.empty((hh, ww, 3), np.uint8)
        img[:] = key
        img.setflags(write=False)
        if len(_SOLIDS) > 64:
            _SOLIDS.clear()
        _SOLIDS[key] = img
    return img[:h, :w]


def _blend_layer(frame: np.ndarray, rects, draw, alpha: float) -> None:
    """Draw ``draw(overlay, ox, oy)`` onto copies of the given ROIs and blend them in.

    ``draw`` receives the ROI copy and the offset to subtract from frame coords.
    Only the listed rects are copied, so thin strips around a large box cost
    little compared with blending the whole box.
    """
    h, w = frame.shape[:2]
    for (x0, y0, x1, y1) in rects:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = frame[y0:y1, x0:x1]
        overlay = roi.copy()
        draw(overlay, x0, y0)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _edge_rects(x1: int, y1: int, x2: int, y2: int, pad: int):
    """ROIs covering a band of +-pad around a rectangle outline (strips for big boxes)."""
    X1, Y1, X2, Y2 = x1 - pad, y1 - pad, x2 + pad + 1, y2 + pad + 1
    band = 2 * pad + 1
    # Strips cost 4x the Python/draw overhead; only worth it for large boxes.
    if (X2 - X1) * (Y2 - Y1) <= 12 * band * ((X2 - X1) + (Y2 - Y1)):
        return [(X1, Y1, X2, Y2)]
    return [
        (X1, Y1, X2, y1 + pad + 1),          # top
        (X1, y2 - pad, X2, Y2),              # bottom
        (X1, y1 + pad + 1, x1 + pad + 1, y2 - pad),  # left
        (x2 - pad, y1 + pad + 1, X2, y2 - pad),      # right
    ]


def _shift(polys, ox: int, oy: int):
    off = np.array((ox, oy), np.int32)
    return [p - off for p in polys]


def apply_fade(frame: np.ndarray, backup: np.ndarray, rect, alpha: float) -> None:
    """Blend the drawn ROI back toward ``backup`` (the ROI before drawing)."""
    x0, y0, x1, y1 = rect
    roi = frame[y0:y1, x0:x1]
    cv2.addWeighted(roi, alpha, backup, 1.0 - alpha, 0, dst=roi)


# --------------------------------------------------------------------------- full-frame effects
def apply_thermal(frame: np.ndarray, gray: np.ndarray) -> None:
    cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO, dst=frame)


def draw_center_dot(frame: np.ndarray, color: BGR, st: Style) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    arm, gap = int(14 * st.ui), int(5 * st.ui)
    segs = [
        np.array(((cx - arm, cy), (cx - gap, cy)), np.int32), np.array(((cx + gap, cy), (cx + arm, cy)), np.int32),
        np.array(((cx, cy - arm), (cx, cy - gap)), np.int32), np.array(((cx, cy + gap), (cx, cy + arm)), np.int32),
    ]
    cv2.polylines(frame, segs, False, ANTHRACITE, st.thin + 2, AA)
    cv2.polylines(frame, segs, False, color, st.thin, AA)
    cv2.circle(frame, (cx, cy), max(1, st.thin), color, -1, AA)


# --------------------------------------------------------------------------- boxes
def _box_geometry(x1: int, y1: int, x2: int, y2: int, style: str):
    """Return (crisp_polys, crisp_closed, soft_polys, back_polys, extra_pad).

    crisp: opaque main stroke (+ glow); soft: thin lines in the translucent layer;
    back: thin opaque lines in a dimmed color drawn behind the main stroke (3d).
    """
    if style == "corner":
        l = max(2, int(min(x2 - x1, y2 - y1) * 0.22))
        corners = [
            np.array(((x1, y1 + l), (x1, y1), (x1 + l, y1)), np.int32),
            np.array(((x2 - l, y1), (x2, y1), (x2, y1 + l)), np.int32),
            np.array(((x2, y2 - l), (x2, y2), (x2 - l, y2)), np.int32),
            np.array(((x1 + l, y2), (x1, y2), (x1, y2 - l)), np.int32),
        ]
        frame_rect = [np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), np.int32)]
        return corners, False, frame_rect, [], 0
    rect = np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), np.int32)
    if style == "3d":
        s = _depth(x2 - x1, y2 - y1)
        back = np.array(((x1 + s, y1 - s), (x2 + s, y1 - s), (x2 + s, y2 - s), (x1 + s, y2 - s), (x1 + s, y1 - s)), np.int32)
        links = [np.array(((px, py), (px + s, py - s)), np.int32) for (px, py) in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
        return [rect], True, [], [back] + links, s
    return [rect], True, [], [], 0


def _depth(w: int, h: int) -> int:
    return max(4, int(min(w, h) * 0.12))


def draw_box(frame: np.ndarray, det: Detection, style: str, color: BGR, st: Style, glow: bool) -> None:
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
    crisp, closed, soft, back, extra = _box_geometry(x1, y1, x2, y2, style)

    if back:
        cv2.polylines(frame, back, False, mix(color, ANTHRACITE, 0.35), st.thin, AA)

    if soft or glow:
        def draw(ov, ox, oy):
            if glow:
                cv2.polylines(ov, _shift(crisp, ox, oy), closed, color, st.glow, AA)
            if soft:
                cv2.polylines(ov, _shift(soft, ox, oy), False, color, st.thin, AA)

        pad = (st.glow // 2 + 1) if glow else st.thin + 1
        _blend_layer(frame, _edge_rects(x1, y1, x2, y2, pad), draw, SOFT_ALPHA)

    # thin dark keyline under the stroke keeps it readable on bright backgrounds
    cv2.polylines(frame, crisp, closed, ANTHRACITE, st.t + 2, AA)
    cv2.polylines(frame, crisp, closed, color, st.t, AA)


def box_extent(det: Detection, style: str, st: Style) -> Tuple[int, int, int, int]:
    """Frame-space rect that ``draw_box``/skeleton/head can touch (for fades)."""
    pad = st.glow // 2 + st.t + 2
    extra = _depth(det.width, det.height) if style == "3d" else 0
    return det.x1 - pad, det.y1 - pad - extra, det.x2 + pad + extra + 1, det.y2 + pad + 1


def draw_fill(frame: np.ndarray, det: Detection, color: BGR, alpha: float = FILL_ALPHA) -> None:
    blend_rect(frame, det.x1, det.y1, det.x2, det.y2, color, alpha * det.alpha)


def draw_chams(frame: np.ndarray, det: Detection, color: BGR, alpha: float) -> None:
    blend_rect(frame, det.x1, det.y1, det.x2, det.y2, color, alpha * det.alpha)


# --------------------------------------------------------------------------- labels
Run = Tuple[str, bool, BGR]   # text, bold, color


@dataclass
class Tag:
    inv: np.ndarray       # uint8 (H, W, 3): 255 * (1 - coverage)
    color: np.ndarray     # uint8 (H, W, 3): premultiplied color
    cover: np.ndarray     # float32 (H, W, 1) coverage, used only while fading
    alpha: float
    key: tuple = ()       # _tag_image cache key (for the float path while fading)
    x: int = 0
    y: int = 0

    @property
    def w(self) -> int:
        return self.inv.shape[1]

    @property
    def h(self) -> int:
        return self.inv.shape[0]


def build_label(det: Detection, settings, text_color: BGR) -> List[Run]:
    """Label runs: NAME (bold), then dimmed meta (conf, distance) and the track id."""
    dim = mix(text_color, ANTHRACITE, 0.28)
    meta: List[Run] = []
    if settings.conf:
        meta.append((f"{int(det.conf * 100)}%", False, dim))
    if settings.distance:
        # from the observed (not the smoothed, per-frame changing) box height
        meta.append((f"{1000.0 / max(det.ref_height or det.height, 1):.1f}m", False, dim))
    if settings.tracking and det.obj_id != -1:
        meta.append((f"#{det.obj_id}", False, text_color))
    runs: List[Run] = []
    if settings.names:
        runs.append((det.cls_name.upper(), True, text_color))
    sep = separator()
    for i, run in enumerate(meta):
        if runs:
            runs.append((f" {sep} " if i else "  ", False, mix(text_color, ANTHRACITE, 0.55)))
        runs.append(run)
    return runs


@lru_cache(maxsize=1024)
def _tag_image(runs: Tuple[Run, ...], st: Style, accent: Optional[BGR], background: bool):
    """Pre-composited label (pill, accent bar, text). Cached: labels only change per inference."""
    masks = [(text_mask(text, st.font_px, bold), color) for text, bold, color in runs]
    th = line_height(st.font_px)
    left = st.pad_x + (st.accent_w + st.pad_x // 2 if accent is not None and background else 0)
    w = left + sum(m.shape[1] for m, _ in masks) + st.pad_x
    h = th + 2 * st.pad_y

    # text coverage and per-pixel text color
    text = np.zeros((h, w), np.float32)
    tcol = np.zeros((h, w, 3), np.float32)
    x, y0 = left, st.pad_y
    for mask, rgb in masks:
        mh, mw = mask.shape
        text[y0:y0 + mh, x:x + mw] = mask
        tcol[:, x:x + mw] = rgb
        x += mw
    text *= 1.0 / 255.0

    # base layer: rounded pill (+ accent bar), or a soft drop shadow without pill
    if background:
        base = rounded_mask(w, h, st.radius) * LABEL_BG_ALPHA
        bcol = np.empty((h, w, 3), np.float32)
        bcol[:] = ANTHRACITE
        if accent is not None:
            aw = st.accent_w
            base[:, :aw] = rounded_mask(w, h, st.radius)[:, :aw]
            bcol[:, :aw] = accent
    else:
        base = np.zeros((h, w), np.float32)
        base[1:, 1:] = text[:-1, :-1] * 0.75
        bcol = np.empty((h, w, 3), np.float32)
        bcol[:] = ANTHRACITE

    t3 = text[:, :, None]
    cover = base * (1.0 - text) + text
    color = bcol * (base * (1.0 - text))[:, :, None] + tcol * t3
    cover3 = cover[:, :, None]
    inv = np.repeat(((1.0 - cover3) * 255.0 + 0.5).astype(np.uint8), 3, axis=2)
    col_u8 = (color + 0.5).clip(0, 255).astype(np.uint8)
    for a in (inv, col_u8, cover3, color):
        a.setflags(write=False)
    return inv, col_u8, cover3, color


def make_tag(runs: Sequence[Run], st: Style, accent: Optional[BGR], alpha: float,
             background: bool = True) -> Optional[Tag]:
    if not runs or alpha <= 0.0:
        return None
    key = (tuple(runs), st, accent, background)
    inv, col_u8, cover, _ = _tag_image(*key)
    return Tag(inv=inv, color=col_u8, cover=cover, alpha=alpha, key=key)


def place_tag(tag: Tag, box: Tuple[int, int, int, int], w: int, h: int, st: Style,
              placed: List[Tuple[int, int, int, int]]) -> None:
    """Put the tag above the box (inside/below if no room), clamp on screen, nudge off others."""
    x1, y1, x2, y2 = box
    tw, th = tag.w, tag.h
    x = min(max(0, x1), max(0, w - tw))
    y = y1 - st.gap - th
    if y < 0:
        y = y1 + st.gap if (y2 - y1) > 2 * th else y2 + st.gap
    y = min(max(0, y), max(0, h - th))
    for _ in range(6):
        hit = None
        for (px0, py0, px1, py1) in placed:
            if x < px1 and x + tw > px0 and y < py1 and y + th > py0:
                hit = (py0, py1)
                break
        if hit is None:
            break
        up = hit[0] - th - 2
        y = up if up >= 0 else hit[1] + 2
        if y + th > h:
            y = h - th
            break
    tag.x, tag.y = x, y
    placed.append((x, y, x + tw, y + th))


def draw_tag(frame: np.ndarray, tag: Tag) -> None:
    h, w = frame.shape[:2]
    x0, y0 = max(0, tag.x), max(0, tag.y)
    x1, y1 = min(w, tag.x + tag.w), min(h, tag.y + tag.h)
    if x1 <= x0 or y1 <= y0:
        return
    sy, sx = slice(y0 - tag.y, y1 - tag.y), slice(x0 - tag.x, x1 - tag.x)
    roi = frame[y0:y1, x0:x1]
    if tag.alpha >= 0.995:
        # roi = roi * (1 - cover) + premultiplied color, in two saturating uint8 ops
        cv2.multiply(roi, tag.inv[sy, sx], dst=roi, scale=1.0 / 255.0)
        cv2.add(roi, tag.color[sy, sx], dst=roi)
        return
    a = tag.alpha
    _, _, cover, color = _tag_image(*tag.key)
    out = roi * (1.0 - cover[sy, sx] * a) + color[sy, sx] * a
    out += 0.5
    roi[:] = out.astype(np.uint8)


# --------------------------------------------------------------------------- per-detection extras
def draw_highlight_marker(frame: np.ndarray, det: Detection, color: BGR, st: Style) -> None:
    r = max(4, int(7 * st.ui))
    pts = np.array(((det.cx, det.cy - r), (det.cx + r, det.cy), (det.cx, det.cy + r), (det.cx - r, det.cy)), np.int32)
    cv2.polylines(frame, [pts], True, ANTHRACITE, st.thin + 2, AA)
    cv2.polylines(frame, [pts], True, color, st.thin, AA)


def draw_snapline(frame: np.ndarray, det: Detection, color: BGR, st: Style) -> None:
    h, w = frame.shape[:2]
    origin = (w // 2, h - 1)
    cv2.line(frame, origin, (det.cx, det.y2), color, st.thin, AA)
    cv2.circle(frame, (det.cx, det.y2), max(2, st.thin + 1), color, -1, AA)


def draw_tracer(frame: np.ndarray, history: Sequence[HistoryPoint], head: Tuple[int, int],
                color: BGR, st: Style) -> None:
    if len(history) < 2:
        return
    pts = np.empty((len(history) + 1, 2), np.int32)
    for i, p in enumerate(history):
        pts[i] = (p[0], p[1])
    pts[-1] = head
    n = len(pts)
    # older half thinner and darker: a cheap "fading" trail without blending
    half = n // 2
    if half >= 2:
        cv2.polylines(frame, [pts[:half + 1]], False, mix(color, ANTHRACITE, 0.5), st.thin, AA)
    cv2.polylines(frame, [pts[max(0, half):]], False, color, st.thin + (1 if st.t > 2 else 0), AA)


def draw_motion(frame: np.ndarray, det: Detection, settings, st: Style) -> None:
    """Velocity arrow and/or predicted position from the smoothed track velocity."""
    vx, vy = det.vx, det.vy
    if abs(vx) < 1.0 and abs(vy) < 1.0:
        return
    h, w = frame.shape[:2]
    margin = max(w, h)
    origin = (det.cx, det.cy)
    if settings.velocity:
        end = _clamp_point(det.cx + vx * VELOCITY_ARROW_SECONDS, det.cy + vy * VELOCITY_ARROW_SECONDS, w, h, margin)
        if end != origin:
            cv2.arrowedLine(frame, origin, end, settings.color_velocity, st.thin + 1, AA, tipLength=0.25)
    if settings.prediction:
        pred = _clamp_point(det.cx + vx * PREDICTION_SECONDS, det.cy + vy * PREDICTION_SECONDS, w, h, margin)
        cv2.line(frame, origin, pred, mix(settings.color_prediction, ANTHRACITE, 0.45), st.thin, AA)
        cv2.circle(frame, pred, max(3, int(6 * st.ui)), settings.color_prediction, st.thin, AA)
        cv2.circle(frame, pred, max(1, st.thin), settings.color_prediction, -1, AA)


_LIMB_PAIRS = np.array(SKELETON_LIMBS + ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)), np.intp)


def _kp_points(det: Detection):
    kps = det.keypoints
    vis = det.kp_visible
    if kps is None or vis is None:
        return None, None
    return kps, vis


def draw_skeleton(frame: np.ndarray, det: Detection, color: BGR, st: Style) -> None:
    kps, vis = _kp_points(det)
    if kps is None or len(kps) <= R_HIP:
        return
    pairs = _LIMB_PAIRS
    ok = vis[pairs[:, 0]] & vis[pairs[:, 1]]
    segs = list(kps[pairs[ok]].astype(np.int32, copy=False))
    sh, hip = ok[-2], ok[-1]
    if sh:
        mid_sh = (kps[L_SHOULDER] + kps[R_SHOULDER]) // 2
        if hip:
            segs.append(np.array((mid_sh, (kps[L_HIP] + kps[R_HIP]) // 2), np.int32))
        if vis[NOSE]:
            segs.append(np.array((kps[NOSE], mid_sh), np.int32))
    if segs:
        cv2.polylines(frame, segs, False, color, st.thin + (1 if st.t >= 3 else 0), AA)


def draw_head(frame: np.ndarray, det: Detection, color: BGR, st: Style) -> None:
    kps, vis = _kp_points(det)
    if kps is None or len(kps) <= NOSE or not vis[NOSE]:
        return
    radius = max(det.width // 8, int(5 * st.ui))
    c = (int(kps[NOSE][0]), int(kps[NOSE][1]))
    # polyline ring: ~3x cheaper than an anti-aliased cv2.circle of the same look
    ring = cv2.ellipse2Poly(c, (radius, radius), 0, 0, 360, 15)
    cv2.polylines(frame, [ring], True, color, st.thin + 1, AA)


# --------------------------------------------------------------------------- faces / hands
FACE_PAD = 0.25          # grow face boxes by 25% per side: Haar boxes are tight (no hair/chin)


def anonymize_faces(frame: np.ndarray, faces: Sequence[FaceBox], mode: str, strength: int) -> None:
    """
    Hide faces in place.

    mode: "blur" (heavy, size-relative blur), "pixelate" (coarse mosaic) or
    "blackbox" (solid black). strength 1..100 scales blur/mosaic relative to the
    face size, so it looks the same for near and far faces; blackbox ignores it.
    """
    p = min(1.0, max(0.0, (int(strength) - 1) / 99.0))
    h, w = frame.shape[:2]
    for (fx, fy, fw, fh) in faces:
        pad_x, pad_y = int(fw * FACE_PAD), int(fh * FACE_PAD)
        rect = clip_rect(fx - pad_x, fy - pad_y, fw + 2 * pad_x, fh + 2 * pad_y, w, h)
        if rect is None:
            continue
        x, y, rw, rh = rect
        roi = frame[y:y + rh, x:x + rw]
        if mode == "blackbox":
            roi[:] = 0
            continue
        if mode == "pixelate":
            # 8 blocks across at 1%, 3 blocks across at 100%
            blocks = max(2, round(8 - 5 * p))
            sw = max(1, blocks)
            sh = max(1, round(blocks * rh / max(1, rw)))
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            roi[:] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            continue
        # blur: shrink hard, blur, scale back up -> strong, cheap, size-independent.
        # 1% keeps ~1/18 of the resolution (features already gone), 100% ~1/80.
        factor = 0.055 - (0.055 - 0.0125) * p
        sw, sh = max(2, round(rw * factor)), max(2, round(rh * factor))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_CUBIC)
        # Soft elliptical edge so it reads as a face mask, not a square; core fully covered.
        m_in, m_out = _face_masks(rw, rh)
        roi[:] = cv2.blendLinear(blurred, roi, m_in, m_out)


_mask_cache: dict = {}


def _face_masks(rw: int, rh: int):
    """Cached feathered ellipse weights (inside, outside) for a face ROI size."""
    key = (rw // 4, rh // 4)          # 4px buckets keep the cache small while faces move
    hit = _mask_cache.get(key)
    if hit is not None and hit[0].shape == (rh, rw):
        return hit
    mask = np.zeros((rh, rw), np.float32)
    cv2.ellipse(mask, (rw // 2, rh // 2), (max(1, rw // 2), max(1, rh // 2)), 0, 0, 360, 1.0, -1)
    feather = max(3, (min(rw, rh) // 6) | 1)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    cv2.ellipse(mask, (rw // 2, rh // 2), (max(1, int(rw * 0.42)), max(1, int(rh * 0.42))), 0, 0, 360, 1.0, -1)
    pair = (mask, 1.0 - mask)
    if len(_mask_cache) > 64:
        _mask_cache.clear()
    _mask_cache[key] = pair
    return pair


def blur_faces(frame: np.ndarray, faces: Sequence[FaceBox], kernel: int) -> None:
    """Backward-compatible wrapper (old odd-kernel API)."""
    anonymize_faces(frame, faces, "blur", max(1, min(100, int(kernel) * 3)))


def draw_face_boxes(frame: np.ndarray, faces: Sequence[FaceBox], color: BGR, text_color: BGR,
                    st: Style, background: bool, placed: List[Tuple[int, int, int, int]]) -> None:
    h, w = frame.shape[:2]
    for (fx, fy, fw, fh) in faces:
        corners = _box_geometry(fx, fy, fx + fw, fy + fh, "corner")[0]
        cv2.polylines(frame, corners, False, ANTHRACITE, st.thin + 2, AA)
        cv2.polylines(frame, corners, False, color, st.thin, AA)
        tag = make_tag([("FACE", True, text_color)], st, color, 1.0, background)
        place_tag(tag, (fx, fy, fx + fw, fy + fh), w, h, st, placed)
        draw_tag(frame, tag)


def make_face_inset(frame: np.ndarray, faces: Sequence[FaceBox], zoom: float, margin: int = 10) -> Optional[np.ndarray]:
    """Crop + scale the largest face, keeping aspect ratio and fitting a quarter of the frame."""
    if not faces:
        return None
    h, w = frame.shape[:2]
    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    pad = int(FACE_ZOOM_PAD * max(fw, fh))
    rect = clip_rect(fx - pad, fy - pad, fw + 2 * pad, fh + 2 * pad, w, h)
    if rect is None:
        return None
    x, y, cw, ch = rect
    max_w = min(w // 4, w - 2 * margin)
    max_h = min(h // 4, h - 2 * margin)
    scale = min(max(1.0, zoom), max_w / cw, max_h / ch)
    out_w, out_h = int(cw * scale), int(ch * scale)
    if out_w < 1 or out_h < 1:
        return None
    # resize allocates a new array, so the inset is independent of later drawing
    return cv2.resize(frame[y:y + ch, x:x + cw], (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def paste_inset(frame: np.ndarray, inset: np.ndarray, color: BGR = WHITE, st: Optional[Style] = None,
                label: str = "ZOOM", text_color: BGR = WHITE, background: bool = True) -> None:
    ih, iw = inset.shape[:2]
    m = st.margin if st is not None else 10
    fh, fw = frame.shape[:2]
    ih, iw = min(ih, fh - m), min(iw, fw - m)
    if ih <= 0 or iw <= 0:
        return
    frame[m:m + ih, m:m + iw] = inset[:ih, :iw]
    t = st.thin if st is not None else 1
    cv2.rectangle(frame, (m - t, m - t), (m + iw + t - 1, m + ih + t - 1), color, t, AA)
    if st is not None and label:
        tag = make_tag([(label, True, text_color)], st, color, 1.0, background)
        tag.x, tag.y = m, min(fh - tag.h, m + ih + st.gap)
        draw_tag(frame, tag)


def draw_hands(frame: np.ndarray, hands: Sequence[HandPoints], color: BGR, st: Style) -> None:
    for pts in hands:
        n = len(pts)
        segs = [np.array((pts[a], pts[b]), np.int32) for a, b in HAND_CONNECTIONS if a < n and b < n]
        if segs:
            cv2.polylines(frame, segs, False, color, st.thin, AA)
        r = max(1, st.thin)
        for p in pts:
            cv2.circle(frame, p, r, color, -1, AA)


# --------------------------------------------------------------------------- target lock
LOCK_PULSE_HZ = 1.1


def draw_lock(frame: np.ndarray, det: Detection, style: str, color: BGR, st: Style, t: float,
              center_line: bool) -> None:
    """Locked target: heavier box, pulsing outer brackets, optional line from frame center."""
    h, w = frame.shape[:2]
    pulse = 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * LOCK_PULSE_HZ)   # smooth, time based
    if center_line:
        cv2.line(frame, (w // 2, h // 2), (det.cx, det.cy), mix(color, ANTHRACITE, 0.35), st.thin, AA)
    heavy = dc_replace(st, t=st.t + max(1, int(round(st.ui))), glow=int(st.glow * 1.3))
    draw_box(frame, det, style, color, heavy, True)
    off = st.t + int(round((4 + 6 * pulse) * st.ui))
    x1, y1, x2, y2 = det.x1 - off, det.y1 - off, det.x2 + off, det.y2 + off
    brackets = _box_geometry(x1, y1, x2, y2, "corner")[0]
    bracket_color = mix(color, (255, 255, 255), 0.45 * pulse)
    tb = max(1, st.t)
    cv2.polylines(frame, brackets, False, ANTHRACITE, tb + 2, AA)
    cv2.polylines(frame, brackets, False, bracket_color, tb, AA)
    # small center reticle
    r = max(3, int(5 * st.ui))
    cx, cy = det.cx, det.cy
    segs = [np.array(((cx - 2 * r, cy), (cx - r, cy)), np.int32), np.array(((cx + r, cy), (cx + 2 * r, cy)), np.int32),
            np.array(((cx, cy - 2 * r), (cx, cy - r)), np.int32), np.array(((cx, cy + r), (cx, cy + 2 * r)), np.int32)]
    cv2.polylines(frame, segs, False, ANTHRACITE, st.thin + 2, AA)
    cv2.polylines(frame, segs, False, color, st.thin, AA)


# --------------------------------------------------------------------------- zones
ZONE_FILL_ALPHA = 0.05
ZONE_FILL_ALPHA_ACTIVE = 0.09


def _dashed_rect(x1: int, y1: int, x2: int, y2: int, dash: int, gap: int):
    segs = []
    period = dash + gap
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        pos = 0
        while pos < length:
            end = min(length, pos + dash)
            segs.append(np.array(((ax + dx * pos, ay + dy * pos), (ax + dx * end, ay + dy * end)), np.int32))
            pos += period
    return segs


def draw_zone(frame: np.ndarray, box: Tuple[int, int, int, int], color: BGR, label: str, count: int,
              text_color: BGR, st: Style, background: bool,
              placed: List[Tuple[int, int, int, int]]) -> Optional[Tag]:
    """Zone: low-alpha fill, dashed stroke (solid and brighter while occupied).

    Returns the positioned corner pill (already reserved in ``placed``); the caller
    draws it after the detection boxes so it stays readable.
    """
    x1, y1, x2, y2 = box
    active = count > 0
    blend_rect(frame, x1, y1, x2, y2, color, ZONE_FILL_ALPHA_ACTIVE if active else ZONE_FILL_ALPHA)
    if active:
        rect = np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), np.int32)
        cv2.polylines(frame, [rect], True, color, st.thin + 1, AA)
    else:
        dash, gap = max(4, int(10 * st.ui)), max(3, int(6 * st.ui))
        segs = _dashed_rect(x1, y1, x2, y2, dash, gap)
        if segs:
            cv2.polylines(frame, segs, False, mix(color, ANTHRACITE, 0.3), st.thin, AA)
    runs = [(label, True, text_color), (f" {separator()} ", False, mix(text_color, ANTHRACITE, 0.55)),
            (str(count), True, color if active else mix(text_color, ANTHRACITE, 0.28))]
    tag = make_tag(runs, st, color, 1.0, background)
    if tag is None:
        return None
    h, w = frame.shape[:2]
    tag.x = min(max(0, x1 + st.gap), max(0, w - tag.w))
    tag.y = min(max(0, y1 + st.gap), max(0, h - tag.h))
    placed.append((tag.x, tag.y, tag.x + tag.w, tag.y + tag.h))
    return tag
