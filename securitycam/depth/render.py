"""Depth view: colorized depth map blended into the video ("scanner" look).

The colorized map is built once per depth frame and per output size, so the
per-rendered-frame cost is one ``addWeighted`` (plus a scanline sweep).
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

PALETTES = {"turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO}


def colorize(disp: np.ndarray, palette: str = "turbo", lo_hi: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """(H, W) relative inverse depth -> (H, W, 3) BGR uint8 (near = hot)."""
    d = np.asarray(disp, np.float32)
    if lo_hi is None:
        lo, hi = np.percentile(d, (2.0, 98.0)) if d.size else (0.0, 1.0)
    else:
        lo, hi = lo_hi
    span = max(1e-6, float(hi) - float(lo))
    g = np.clip((d - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, PALETTES.get(palette, cv2.COLORMAP_TURBO))


class ViewCache:
    """Colorized + resized depth image for the current depth frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._img: Optional[np.ndarray] = None
        self._range: Optional[Tuple[float, float]] = None

    def image(self, seq: int, disp: np.ndarray, palette: str, w: int, h: int) -> np.ndarray:
        key = (seq, palette, w, h)
        with self._lock:
            if key == self._key and self._img is not None:
                return self._img
            lo, hi = np.percentile(disp, (2.0, 98.0)) if disp.size else (0.0, 1.0)
            if self._range is not None:          # smooth the color range: no flicker between frames
                lo = self._range[0] + (lo - self._range[0]) * 0.35
                hi = self._range[1] + (hi - self._range[1]) * 0.35
            self._range = (float(lo), float(hi))
            col = colorize(disp, palette, self._range)
            self._img = cv2.resize(col, (w, h), interpolation=cv2.INTER_LINEAR)
            self._key = key
            return self._img


def blend(frame: np.ndarray, colored: np.ndarray, opacity: float) -> None:
    """Blend ``colored`` into ``frame`` in place."""
    a = float(max(0.0, min(1.0, opacity)))
    if a <= 0.0 or colored.shape != frame.shape:
        return
    cv2.addWeighted(colored, a, frame, 1.0 - a, 0.0, dst=frame)
