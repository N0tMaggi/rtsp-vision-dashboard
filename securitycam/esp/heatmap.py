"""Motion heatmap: where targets were over the last 5 min / 1 h / 24 h.

Per camera a low-res grid per window accumulates *time* spent at each foot
point (bottom center of the observed box) and decays exponentially with the
window as time constant, so old activity fades out smoothly. Accumulation is
a handful of float64 multiply-adds per rendered frame.

Rendering blurs and normalizes the grid, maps it through a theme colormap
(anthracite -> signal yellow -> white, alpha rising with heat) and caches the
premultiplied overlay per (window, size, opacity) for ``RENDER_EVERY_S``; the
per-frame cost is one uint8 multiply + add over the bounding box of the heat.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

WINDOWS: Dict[str, float] = {"5m": 300.0, "1h": 3600.0, "24h": 86400.0}
DEFAULT_WINDOW = "1h"
GRID_W = 96                    # grid width; height follows the frame aspect
MAX_STEP_S = 0.5               # cap per-update time weight (pauses / stalls)
RENDER_EVERY_S = 0.5           # overlay cache lifetime
MIN_ALPHA = 6                  # alpha (0..255) below which a pixel counts as empty

# colormap stops: (position, BGR) - anthracite #222427, yellow #ffd21f, white #f4f4f2
_STOPS = ((0.0, (39, 36, 34)), (0.55, (31, 210, 255)), (1.0, (242, 244, 244)))


def _build_lut() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 256)
    col = np.zeros((256, 3), np.float32)
    for c in range(3):
        col[:, c] = np.interp(x, [p for p, _ in _STOPS], [bgr[c] for _, bgr in _STOPS])
    # alpha: transparent at 0, quickly visible, ~full at the hot end
    alpha = np.clip(np.power(x, 0.6) * 1.15, 0.0, 1.0).astype(np.float32)
    alpha[0] = 0.0
    return col, alpha


LUT_COLOR, LUT_ALPHA = _build_lut()


@dataclass
class HeatState:
    grid_h: int = 0
    grids: Dict[str, np.ndarray] = field(default_factory=dict)
    last_t: Optional[float] = None
    total_s: Dict[str, float] = field(default_factory=dict)   # decayed observed time
    cache: Dict[tuple, tuple] = field(default_factory=dict)   # overlay cache


def get_state(camera_state) -> HeatState:
    hs = getattr(camera_state, "esp_heatmap", None)
    if not isinstance(hs, HeatState):
        hs = camera_state.esp_heatmap = HeatState()
    return hs


def _ensure(hs: HeatState, fw: int, fh: int) -> None:
    gh = max(8, int(round(GRID_W * fh / max(fw, 1))))
    if hs.grid_h != gh or not hs.grids:
        hs.grid_h = gh
        hs.grids = {k: np.zeros((gh, GRID_W), np.float64) for k in WINDOWS}
        hs.total_s = {k: 0.0 for k in WINDOWS}
        hs.cache.clear()


# 3x3 splat kernel (sum 1) so single points don't alias to one cell
_SPLAT = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16.0


def accumulate(camera_state, mst, t_now: float) -> None:
    """Add the time since the last call at every live track's foot point (ESP lock held)."""
    fw, fh = mst.frame_w, mst.frame_h
    if fw <= 0 or fh <= 0:
        return
    hs = get_state(camera_state)
    _ensure(hs, fw, fh)
    last = hs.last_t
    hs.last_t = t_now
    if last is None:
        return
    dt = t_now - last
    if dt <= 0.0:
        return
    dt = min(dt, MAX_STEP_S)
    gw, gh = GRID_W, hs.grid_h
    pts = []
    for tr in mst.tracks.values():
        if tr.missing_since is not None:
            continue
        o = tr.obs
        gx = int((o[0] + o[2]) * 0.5 / fw * gw)
        gy = int(o[3] / fh * gh - 0.5)
        if 0 <= gx < gw and 0 <= gy < gh:
            pts.append((gx, gy))
    for key, window in WINDOWS.items():
        g = hs.grids[key]
        decay = math.exp(-dt / window)
        g *= decay
        hs.total_s[key] = hs.total_s[key] * decay + dt
        for gx, gy in pts:
            x0, x1 = max(0, gx - 1), min(gw, gx + 2)
            y0, y1 = max(0, gy - 1), min(gh, gy + 2)
            g[y0:y1, x0:x1] += dt * _SPLAT[y0 - gy + 1:y1 - gy + 1, x0 - gx + 1:x1 - gx + 1]


def reset(camera_state) -> None:
    hs = get_state(camera_state)
    for g in hs.grids.values():
        g[:] = 0.0
    for k in hs.total_s:
        hs.total_s[k] = 0.0
    hs.cache.clear()


def normalized(camera_state, window: str) -> Optional[np.ndarray]:
    """Blurred heat in 0..1 (float32, grid resolution), or None if nothing accumulated."""
    hs = get_state(camera_state)
    g = hs.grids.get(window if window in WINDOWS else DEFAULT_WINDOW)
    if g is None:
        return None
    heat = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    peak = float(heat.max())
    if peak <= 1e-9:
        return None
    # soft compression so a single long loiter doesn't hide everything else
    return np.sqrt(heat / peak)


def colorize(heat: np.ndarray, size: Tuple[int, int], opacity: float) -> Tuple[np.ndarray, np.ndarray]:
    """heat (0..1, any size) -> (premultiplied BGR uint8, alpha float32 0..1) at ``size`` (w, h)."""
    w, h = size
    up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    idx = np.clip(up * 255.0, 0, 255).astype(np.uint8)
    alpha = LUT_ALPHA[idx] * float(opacity)
    col = LUT_COLOR[idx]
    return col, alpha


def _overlay(camera_state, window: str, w: int, h: int, opacity: float, t_now: float):
    hs = get_state(camera_state)
    key = (window, w, h, round(opacity, 3))
    hit = hs.cache.get(key)
    if hit is not None and t_now - hit[0] < RENDER_EVERY_S:
        return hit[1]
    heat = normalized(camera_state, window)
    result = None
    if heat is not None:
        # colorize at 1/4 size (smooth field), crop to the heat, then upscale the uint8 layers
        sw, sh = max(1, w // 4), max(1, h // 4)
        col, alpha = colorize(heat, (sw, sh), opacity)
        ys, xs = np.nonzero(alpha * 255.0 >= MIN_ALPHA)
        if len(xs):
            sx0, sx1 = max(0, int(xs.min()) - 1), min(sw, int(xs.max()) + 2)
            sy0, sy1 = max(0, int(ys.min()) - 1), min(sh, int(ys.max()) + 2)
            a = alpha[sy0:sy1, sx0:sx1, None]
            pre_s = (col[sy0:sy1, sx0:sx1] * a + 0.5).astype(np.uint8)
            inv_s = np.repeat(((1.0 - a) * 255.0 + 0.5).astype(np.uint8), 3, axis=2)
            kx, ky = w / sw, h / sh
            x0, x1 = int(sx0 * kx), min(w, int(math.ceil(sx1 * kx)))
            y0, y1 = int(sy0 * ky), min(h, int(math.ceil(sy1 * ky)))
            if x1 > x0 and y1 > y0:
                pre = cv2.resize(pre_s, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                inv = cv2.resize(inv_s, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                result = (x0, y0, x1, y1, pre, inv)
    hs.cache = {key: (t_now, result)}     # only the current configuration stays cached
    return result


def draw(frame: np.ndarray, camera_state, window: str, opacity: float, t_now: float) -> None:
    """Blend the heat overlay onto ``frame`` in place (call outside the ESP lock is fine:
    the cache read races only with itself in the single render thread per camera)."""
    if opacity <= 0.0:
        return
    h, w = frame.shape[:2]
    ov = _overlay(camera_state, window, w, h, opacity, t_now)
    if ov is None:
        return
    x0, y0, x1, y1, pre, inv = ov
    roi = frame[y0:y1, x0:x1]
    cv2.multiply(roi, inv, dst=roi, scale=1.0 / 255.0)
    cv2.add(roi, pre, dst=roi)


def png_bytes(camera_state, window: str, width: int = 0) -> Optional[bytes]:
    """RGBA PNG of the heatmap (transparent where cold); None if empty."""
    heat = normalized(camera_state, window)
    if heat is None:
        return None
    gh, gw = heat.shape
    width = int(width) if width else gw * 8
    width = max(gw, min(1920, width))
    height = max(1, int(round(width * gh / gw)))
    col, alpha = colorize(heat, (width, height), 1.0)
    bgra = np.dstack([np.clip(col + 0.5, 0, 255).astype(np.uint8),
                      (alpha * 255.0 + 0.5).astype(np.uint8)])
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes() if ok else None


def info(camera_state) -> dict:
    hs = get_state(camera_state)
    return {
        "windows": list(WINDOWS),
        "grid": [GRID_W, hs.grid_h],
        "observed_s": {k: round(v, 1) for k, v in hs.total_s.items()},
        "updated": time.time() if hs.last_t is not None else None,
    }
