"""Drawing of world-anchored zones whose projection is no longer an axis-aligned box.

Same look as ``renderers.draw_zone`` (low-alpha fill, dashed outline that turns
solid while occupied, corner pill) but for an arbitrary convex polygon in
pixel coordinates (already clipped to the frame by ``anchor.project_zones``).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import renderers as R
from .colors import ANTHRACITE, BGR, mix
from .text import separator


def _dashed_polygon(pts: Sequence[Tuple[float, float]], dash: int, gap: int) -> List[np.ndarray]:
    segs = []
    period = dash + gap
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        length = math.hypot(bx - ax, by - ay)
        if length <= 0.5:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        pos = 0.0
        while pos < length:
            end = min(length, pos + dash)
            segs.append(np.array(((ax + dx * pos, ay + dy * pos), (ax + dx * end, ay + dy * end)), np.int32))
            pos += period
    return segs


def blend_polygon(frame: np.ndarray, poly: np.ndarray, color: BGR, alpha: float) -> None:
    """Alpha-blend a filled polygon (int32 (N, 2)) in place, touching only its bounding box."""
    if alpha <= 0.0 or len(poly) < 3:
        return
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(poly[:, 0].min())), max(0, int(poly[:, 1].min()))
    x1, y1 = min(w, int(poly[:, 0].max()) + 1), min(h, int(poly[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - (x0, y0)], 255)
    sel = mask.astype(bool)
    if not sel.any():
        return
    px = roi[sel].astype(np.float32)
    roi[sel] = (px * (1.0 - alpha) + np.asarray(color, np.float32) * alpha + 0.5).astype(np.uint8)


def expand_polygon(poly: Sequence[Tuple[float, float]], off: float) -> np.ndarray:
    """Push each vertex ``off`` px away from the centroid (good enough for convex outlines)."""
    p = np.asarray(poly, np.float64)
    c = p.mean(axis=0)
    d = p - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return (p + d / n * off * math.sqrt(2.0)).astype(np.int32)


def draw_zone_poly(frame: np.ndarray, poly_px: Sequence[Tuple[float, float]], color: BGR, label: str, count: int,
                   text_color: BGR, st: R.Style, background: bool,
                   placed: List[Tuple[int, int, int, int]]) -> Optional[R.Tag]:
    pts = np.asarray(poly_px, np.float64)
    ipts = np.round(pts).astype(np.int32)
    active = count > 0
    blend_polygon(frame, ipts, color, R.ZONE_FILL_ALPHA_ACTIVE if active else R.ZONE_FILL_ALPHA)
    if active:
        cv2.polylines(frame, [ipts], True, color, st.thin + 1, R.AA)
    else:
        dash, gap = max(4, int(10 * st.ui)), max(3, int(6 * st.ui))
        segs = _dashed_polygon(pts, dash, gap)
        if segs:
            cv2.polylines(frame, segs, False, mix(color, ANTHRACITE, 0.3), st.thin, R.AA)
    runs = [(label, True, text_color), (f" {separator()} ", False, mix(text_color, ANTHRACITE, 0.55)),
            (str(count), True, color if active else mix(text_color, ANTHRACITE, 0.28))]
    tag = R.make_tag(runs, st, color, 1.0, background)
    if tag is None:
        return None
    h, w = frame.shape[:2]
    # pill at the top-left-most vertex (the drawn top-left corner while it is visible)
    k = int(np.argmin(pts[:, 0] + pts[:, 1]))
    x1, y1 = int(pts[k, 0]), int(pts[k, 1])
    tag.x = min(max(0, x1 + st.gap), max(0, w - tag.w))
    tag.y = min(max(0, y1 + st.gap), max(0, h - tag.h))
    placed.append((tag.x, tag.y, tag.x + tag.w, tag.y + tag.h))
    return tag
