"""World-space motion heatmap for PTZ cameras (used by ``heatmap.py`` while anchoring is active).

Heat accumulates in an azimuth / elevation grid (``RES_DEG`` per cell, full
360 x 180 degrees, azimuth wraps), so it stays on the floor where people
walked whatever the camera looks at. For drawing, the visible part is
resampled for the current view at heat-grid resolution (a few thousand rays,
refreshed every ``heatmap.RENDER_EVERY_S`` or faster while the camera moves);
between refreshes the cached quarter-resolution overlay is warped with the
exact rotation homography, so it glides with the picture every frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .. import geo

RES_DEG = 1.0
AZ_CELLS = int(round(360.0 / RES_DEG))
EL_CELLS = int(round(180.0 / RES_DEG))
PAD = 3                                    # wrapped columns on each side for sampling across +-180
MOVING_REFRESH_S = 0.15

_SPLAT = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16.0


@dataclass
class WorldHeat:
    grids: Dict[str, np.ndarray] = field(default_factory=dict)
    total_s: Dict[str, float] = field(default_factory=dict)
    last_t: Optional[float] = None
    blurred: Dict[str, Tuple[float, np.ndarray, float]] = field(default_factory=dict)   # window -> (t, padded, peak)
    cache: Dict[tuple, tuple] = field(default_factory=dict)

    def ensure(self, windows) -> None:
        if not self.grids:
            self.grids = {k: np.zeros((EL_CELLS, AZ_CELLS), np.float64) for k in windows}
            self.total_s = {k: 0.0 for k in windows}

    def reset(self) -> None:
        for g in self.grids.values():
            g[:] = 0.0
        for k in self.total_s:
            self.total_s[k] = 0.0
        self.blurred.clear()
        self.cache.clear()


def cell_of(az: float, el: float) -> Tuple[int, int]:
    gx = int(math.floor((az + 180.0) / RES_DEG)) % AZ_CELLS
    gy = min(EL_CELLS - 1, max(0, int(math.floor((90.0 - el) / RES_DEG))))
    return gx, gy


def accumulate(wh: WorldHeat, windows: Dict[str, float], foot_rays: np.ndarray, dt: float) -> None:
    pts = []
    if len(foot_rays):
        for az, el in geo.rays_to_angles(foot_rays):
            pts.append(cell_of(float(az), float(el)))
    for key, window in windows.items():
        g = wh.grids[key]
        decay = math.exp(-dt / window)
        g *= decay
        wh.total_s[key] = wh.total_s[key] * decay + dt
        for gx, gy in pts:
            for dy in (-1, 0, 1):
                y = gy + dy
                if not 0 <= y < EL_CELLS:
                    continue
                for dx in (-1, 0, 1):
                    g[y, (gx + dx) % AZ_CELLS] += dt * _SPLAT[dy + 1, dx + 1]


def _blurred(wh: WorldHeat, window: str, t_now: float, max_age: float) -> Tuple[Optional[np.ndarray], float]:
    hit = wh.blurred.get(window)
    if hit is not None and t_now - hit[0] < max_age:
        return hit[1], hit[2]
    g = wh.grids.get(window)
    if g is None:
        return None, 0.0
    padded = np.concatenate([g[:, -PAD:], g, g[:, :PAD]], axis=1).astype(np.float32)
    padded = cv2.GaussianBlur(padded, (0, 0), 1.2)
    peak = float(padded[:, PAD:-PAD].max())
    wh.blurred[window] = (t_now, padded, peak)
    return padded, peak


def view_heat(wh: WorldHeat, window: str, pose: geo.Pose, lens: geo.Lens, gw: int, gh: int,
              t_now: float, max_age: float = 0.5) -> Optional[np.ndarray]:
    """Heat 0..1 (sqrt-compressed, normalized by the *global* peak) sampled for a view, (gh, gw) float32."""
    padded, peak = _blurred(wh, window, t_now, max_age)
    if padded is None or peak <= 1e-9:
        return None
    xs = (np.arange(gw) + 0.5) / gw
    ys = (np.arange(gh) + 0.5) / gh
    u, v = np.meshgrid(xs, ys)
    rays = geo.image_to_rays(np.stack([u.ravel(), v.ravel()], axis=1), pose, lens)
    ang = geo.rays_to_angles(rays)
    mx = ((ang[:, 0] + 180.0) / RES_DEG - 0.5 + PAD).astype(np.float32).reshape(gh, gw)
    my = ((90.0 - ang[:, 1]) / RES_DEG - 0.5).astype(np.float32).reshape(gh, gw)
    heat = cv2.remap(padded, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    heat = np.clip(heat / peak, 0.0, 1.0)
    if float(heat.max()) <= 1e-6:
        return None
    return np.sqrt(heat)
