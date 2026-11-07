"""Silhouette chams: body-shaped fills / outlines from instance-segmentation masks.

``chams_mode`` (with ``chams`` on):

* ``box``                 -- classic translucent box (``renderers.draw_chams``), the default
* ``silhouette``          -- translucent body fill
* ``outline``             -- crisp body outline with a soft glow (``outline_glow``)
* ``silhouette_outline``  -- both
* ``xray``                -- wallhack look: the frame around people is darkened, people stay
                             bright with a faint tint and a bright glowing outline

Each detection carries a small box-relative alpha crop (``masks.py``). Per
rendered frame everything is mapped onto the box the motion model displays
*now*, so the shape glides with the person between inferences:

* fill: the crop is resized with linear interpolation and sharpened with a LUT
  ramp ~1.5 output px wide -> smooth anti-aliased edges, never blocky,
* outline: the crop's contour (traced at 2x, smoothed) is scaled onto the box
  and stroked with ``cv2.LINE_AA`` at sub-pixel precision,
* glow: blurred at crop resolution and upsampled (a soft signal needs no
  full-resolution blur); fill and glow share one blend per person.

Detections without a mask (detect / pose models) fall back to the box chams.
Lost tracks keep their last silhouette while ``motion`` fades them out
(``FADE_OUT_S``): a short "ghost" for free.

Keypoint features (skeleton, head, pose aim points) need a Pose model; one model
can't return keypoints and masks at the same time.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import renderers as R
from .colors import BGR, WHITE, mix
from .detections import Detection

MODES = ("box", "silhouette", "outline", "silhouette_outline", "xray")
SEG_SUFFIX = "-seg"
XRAY_DIM = 0.32            # brightness of the frame outside people in xray mode
XRAY_TINT = 0.18           # minimum body tint in xray mode
GLOW_GAIN = 0.8            # glow peak opacity
EDGE_PX = 1.5              # width of the anti-aliased fill edge in output pixels
_OPAQUE = 0.985
_SHIFT = 3                 # polyline sub-pixel bits

Rect = Tuple[int, int, int, int]


def is_seg_model(name: Optional[str]) -> bool:
    stem = str(name or "").rsplit(".", 1)[0]
    return stem.endswith(SEG_SUFFIX)


def uses_masks(settings) -> bool:
    return bool(settings.chams) and getattr(settings, "chams_mode", "box") != "box"


def chams_color(settings, pal_box: BGR) -> BGR:
    """chams_color, unless the palette recolored this detection (class colors, RGB, IFF, lock)."""
    return pal_box if tuple(pal_box) != tuple(settings.color_boxes) else settings.chams_color


# --------------------------------------------------------------------------- LUTs / blending
@lru_cache(maxsize=256)
def _ramp_lut(k_q: int, gain_q: int) -> np.ndarray:
    """uint8 LUT: steep anti-aliased edge for a linearly upsampled mask, scaled to gain."""
    v = np.arange(256, dtype=np.float32)
    ramp = np.clip((v - 127.5) * (k_q / 4.0) + 127.5, 0, 255)
    return np.clip(ramp * (gain_q / 100.0) + 0.5, 0, 255).astype(np.uint8)


@lru_cache(maxsize=256)
def _odds_lut(gain_q: int) -> np.ndarray:
    """float32 LUT x/(1-x) for x = a*g. cv2.blendLinear normalizes by (w1 + w2), so with
    w1 = 1 this single weight map gives roi*(1-x) + src*x: one LUT pass per blend."""
    x = (np.arange(256, dtype=np.float64) / 255.0) * (gain_q / 100.0)
    return np.minimum(x / np.maximum(1.0 - x, 1e-6), 1e6).astype(np.float32)


_ONES: dict = {}


def _ones(h: int, w: int) -> np.ndarray:
    img = _ONES.get("w")
    if img is None or img.shape[0] < h or img.shape[1] < w:
        img = np.ones((max(h, img.shape[0] if img is not None else 0),
                       max(w, img.shape[1] if img is not None else 0)), np.float32)
        img.setflags(write=False)
        _ONES["w"] = img
    return img[:h, :w]


@lru_cache(maxsize=64)
def _scale_lut(k_q: int) -> np.ndarray:
    return np.clip(np.arange(256, dtype=np.float32) * (k_q / 100.0), 0, 255).astype(np.uint8)


def _blend(roi: np.ndarray, src: np.ndarray, a: np.ndarray, gain: float) -> None:
    """roi = roi*(1 - a*gain) + src*a*gain for a uint8 alpha ``a`` (in place)."""
    g = int(round(min(1.0, max(0.0, gain)) * 100))
    if g <= 0:
        return
    cv2.blendLinear(roi, src, _ones(a.shape[0], a.shape[1]), cv2.LUT(a, _odds_lut(g)), dst=roi)


def _blend_color(roi: np.ndarray, color: BGR, a: np.ndarray, gain: float) -> None:
    _blend(roi, R._solid(color, roi.shape[0], roi.shape[1]), a, gain)


def _clip(rect: Rect, w: int, h: int) -> Optional[Rect]:
    x0, y0, x1, y1 = max(0, rect[0]), max(0, rect[1]), min(w, rect[2]), min(h, rect[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _sub(img: np.ndarray, rect: Rect, clip: Rect) -> np.ndarray:
    """Part of ``img`` (covering ``rect``) that lies inside ``clip``."""
    return img[clip[1] - rect[1]:clip[3] - rect[1], clip[0] - rect[0]:clip[2] - rect[0]]


# --------------------------------------------------------------------------- mask geometry
def box_alpha(det: Detection, gain: float = 1.0) -> Optional[np.ndarray]:
    """Mask crop resized onto the displayed box (uint8 0..255*gain, shape = box h x w), or None."""
    crop = det.mask
    if crop is None:
        return None
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    if bw < 2 or bh < 2:
        return None
    ch, cw = crop.shape[:2]
    up = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_LINEAR)
    k = max(1.0, min(bw / cw, bh / ch) / EDGE_PX)
    g = int(round(min(1.0, max(0.0, gain)) * 100))
    if k > 1.0 or g < 100:
        cv2.LUT(up, _ramp_lut(int(round(min(k, 32.0) * 4)), g), dst=up)
    return up


_contour_cache: dict = {}


def _crop_contours(crop: np.ndarray) -> List[np.ndarray]:
    """Smoothed, simplified float contours of a crop in crop px units (cached per crop object)."""
    key = id(crop)
    hit = _contour_cache.get(key)
    if hit is not None and hit[0] is crop:
        return hit[1]
    up = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)
    found, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    min_area = max(4.0, 0.002 * up.shape[0] * up.shape[1])
    for c in found:
        if len(c) < 6 or cv2.contourArea(c) < min_area:
            continue
        p = c.reshape(-1, 2).astype(np.float32)
        # circular moving average (5 taps) removes the pixel staircase ...
        p = (np.roll(p, 2, 0) + np.roll(p, 1, 0) + p + np.roll(p, -1, 0) + np.roll(p, -2, 0)) * 0.2
        # ... and a light simplification keeps the AA stroke cheap
        p = cv2.approxPolyDP(p.reshape(-1, 1, 2), 0.35, True).reshape(-1, 2)
        if len(p) >= 3:
            out.append((p + 0.5) * 0.5)       # 2x px centers -> crop px units
    if len(_contour_cache) > 128:
        _contour_cache.clear()
    _contour_cache[key] = (crop, out)
    return out


Cut = Tuple[bool, bool, bool, bool]      # box side (left, top, right, bottom) lies on the frame edge


EDGE_CUT = 0.012           # a box side this close to the frame edge (fraction) counts as cut off


def _cut_sides(det: Detection, w: int, h: int) -> Cut:
    mx, my = max(2.0, EDGE_CUT * w), max(2.0, EDGE_CUT * h)
    return det.x1 <= mx, det.y1 <= my, det.x2 >= w - mx, det.y2 >= h - my


def _open_runs(p: np.ndarray, cw: int, ch: int, cut: Cut) -> List[Tuple[np.ndarray, bool]]:
    """Split a closed contour where it runs along a box side cut by the frame edge.

    A person leaving the picture is cut off by the frame, not outlined along it.
    """
    on = np.zeros(len(p), bool)
    if cut[0]:
        on |= p[:, 0] <= 1.0
    if cut[1]:
        on |= p[:, 1] <= 1.0
    if cut[2]:
        on |= p[:, 0] >= cw - 1.0
    if cut[3]:
        on |= p[:, 1] >= ch - 1.0
    if not on.any():
        return [(p, True)]
    if on.all():
        return []
    start = int(np.argmax(on))               # rotate so the contour starts on a cut point
    p, on = np.roll(p, -start, 0), np.roll(on, -start)
    n = len(p)
    runs = []
    i = 0
    while i < n:
        if on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not on[j + 1]:
            j += 1
        # keep the cut points on both ends: the segments leading to the edge stay, only the
        # stretch running along the edge (cut point -> cut point) is dropped
        idx = [i - 1] + list(range(i, j + 1)) + [(j + 1) % n]
        runs.append((p[idx].astype(np.float32), False))
        i = j + 1
    return runs


def _polys_for(det: Detection, ox: int, oy: int, cut: Cut = (False,) * 4) -> List[Tuple[np.ndarray, bool]]:
    """Contours scaled onto the displayed box, relative to (ox, oy), fixed-point for polylines:
    [(points, closed)]."""
    crop = det.mask
    ch, cw = crop.shape[:2]
    sx, sy = (det.x2 - det.x1) / cw, (det.y2 - det.y1) / ch
    one = float(1 << _SHIFT)
    out = []
    for contour in _crop_contours(crop):
        for p, closed in (_open_runs(contour, cw, ch, cut) if any(cut) else [(contour, True)]):
            q = np.empty_like(p)
            q[:, 0] = (p[:, 0] * sx + (det.x1 - ox)) * one
            q[:, 1] = (p[:, 1] * sy + (det.y1 - oy)) * one
            out.append((q.astype(np.int32).reshape(-1, 1, 2), closed))
    return out


def _glow(det: Detection, glow_px: int, gain: float, cut: Cut = (False,) * 4) -> Tuple[np.ndarray, Rect]:
    """Glow alpha (0..255*gain) around the contour and the frame rect it covers.

    Blurred at crop resolution and upsampled: a soft signal needs no full-res blur.
    """
    crop = det.mask
    ch, cw = crop.shape[:2]
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    fx, fy = bw / cw, bh / ch
    plx, ply = int(math.ceil(1.6 * glow_px / fx)) + 1, int(math.ceil(1.6 * glow_px / fy)) + 1
    lo = cv2.copyMakeBorder(crop, ply, ply, plx, plx, cv2.BORDER_CONSTANT, value=0)
    # sides on the frame edge: extend the mask outward so no glow runs along the cut
    if cut[0]:
        lo[ply:ply + ch, :plx] = crop[:, :1]
    if cut[1]:
        lo[:ply, plx:plx + cw] = crop[:1, :]
    if cut[2]:
        lo[ply:ply + ch, plx + cw:] = crop[:, -1:]
    if cut[3]:
        lo[ply + ch:, plx:plx + cw] = crop[-1:, :]
    band = cv2.morphologyEx(lo, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=max(0.6, 0.5 * glow_px / fx), sigmaY=max(0.6, 0.5 * glow_px / fy))
    peak = int(band.max())
    if peak > 0:      # blurring a thin band flattens it: renormalize to full strength * gain
        cv2.LUT(band, _scale_lut(min(25500, int(25500 * gain / peak))), dst=band)
    px, py = int(round(plx * fx)), int(round(ply * fy))
    rect = (det.x1 - px, det.y1 - py, det.x2 + px, det.y2 + py)
    out = cv2.resize(band, (rect[2] - rect[0], rect[3] - rect[1]), interpolation=cv2.INTER_LINEAR)
    return out, rect


def _stroke(frame: np.ndarray, det: Detection, color: BGR, t: int) -> None:
    h, w = frame.shape[:2]
    pad = t + 2
    rect = _clip((det.x1 - pad, det.y1 - pad, det.x2 + pad, det.y2 + pad), w, h)
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    polys = _polys_for(det, x0, y0, _cut_sides(det, w, h))
    if not polys:
        return
    roi = frame[y0:y1, x0:x1]
    target = roi if det.alpha >= _OPAQUE else roi.copy()   # fading: stroke a copy, blend it back
    closed = [q for q, c in polys if c]
    opened = [q for q, c in polys if not c]
    if closed:
        cv2.polylines(target, closed, True, color, t, cv2.LINE_AA, _SHIFT)
    if opened:
        cv2.polylines(target, opened, False, color, t, cv2.LINE_AA, _SHIFT)
    if target is not roi:
        cv2.addWeighted(target, det.alpha, roi, 1.0 - det.alpha, 0, dst=roi)


# --------------------------------------------------------------------------- drawing
def draw(frame: np.ndarray, detections: Sequence[Detection], palettes: Sequence, settings, st: R.Style) -> int:
    """Draw silhouette chams for ``detections`` in place; returns how many used a mask.

    Detections without a mask get the classic box chams instead.
    """
    mode = settings.chams_mode if settings.chams_mode in MODES else "box"
    h, w = frame.shape[:2]
    opacity = float(settings.chams_opacity)
    if mode == "box":
        for det, pal in zip(detections, palettes):
            R.draw_chams(frame, det, chams_color(settings, pal.box), opacity)
        return 0

    xray = mode == "xray"
    fill = mode in ("silhouette", "silhouette_outline", "xray")
    line = mode in ("outline", "silhouette_outline", "xray")
    t = max(1, st.t + (1 if xray else 0))
    glow_px = int(round(max(0, int(settings.outline_glow)) * st.ui)) if line else 0

    items: List[tuple] = []
    for det, pal in zip(detections, palettes):
        color = chams_color(settings, pal.box)
        if det.mask is None:
            R.draw_chams(frame, det, color, opacity)
        elif det.x2 - det.x1 >= 2 and det.y2 - det.y1 >= 2 and \
                _clip((det.x1, det.y1, det.x2, det.y2), w, h) is not None:
            items.append((det, color))
    if not items:
        return 0

    if xray:
        # Darken everything, then bring people back from the undimmed copy (their faint
        # tint rides along with the glow blend below). The dim strength follows the
        # people's fade, so it eases in / out with the tracks.
        fade = max(det.alpha for det, _ in items)
        dim = 1.0 - (1.0 - XRAY_DIM) * fade
        restore = []
        for det, color in items:
            box = (det.x1, det.y1, det.x2, det.y2)
            clip = _clip(box, w, h)
            src = frame[clip[1]:clip[3], clip[0]:clip[2]].copy()
            restore.append((det, _sub(box_alpha(det), box, clip), clip, src))
        cv2.LUT(frame, _scale_lut(int(round(dim * 100))), dst=frame)
        for det, a, (x0, y0, x1, y1), src in restore:
            _blend(frame[y0:y1, x0:x1], src, a, det.alpha)

    for det, color in items:
        # fill and glow share the color: merge them into one alpha and blend once
        box = (det.x1, det.y1, det.x2, det.y2)
        alpha = rect = None
        if glow_px > 0:
            alpha, rect = _glow(det, glow_px, GLOW_GAIN, _cut_sides(det, w, h))
        if fill:
            fa = box_alpha(det, max(XRAY_TINT, opacity * 0.5) if xray else opacity)
            if alpha is None:
                alpha, rect = fa, box
            else:
                sub = _sub(alpha, rect, box)
                cv2.max(sub, fa, dst=sub)
        if alpha is not None:
            clip = _clip(rect, w, h)
            if clip is not None:
                _blend_color(frame[clip[1]:clip[3], clip[0]:clip[2]], color, _sub(alpha, rect, clip), det.alpha)
        if line:
            _stroke(frame, det, mix(color, WHITE, 0.45) if xray else color, t)
    return len(items)
