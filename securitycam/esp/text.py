"""Cached, anti-aliased text masks for ESP labels.

Text is rasterized once per (string, pixel size, weight) into an 8-bit coverage
mask and cached; drawing a label is then a cheap alpha blend of a small ROI.
That is both much faster than ``cv2.putText(..., LINE_AA)`` per frame and lets
us use a real monospace TrueType font (Pillow) when one is available. Without
Pillow or a usable font the masks fall back to OpenCV's Hershey font.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

from .. import config

try:  # Pillow ships with ultralytics, but keep it optional.
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - depends on environment
    Image = ImageDraw = ImageFont = None

_WIN_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_REGULAR_CANDIDATES = (
    os.path.join(_WIN_FONTS, "consola.ttf"),
    os.path.join(_WIN_FONTS, "CascadiaMono.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "DejaVuSansMono.ttf",
)
_BOLD_CANDIDATES = (
    os.path.join(_WIN_FONTS, "consolab.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "DejaVuSansMono-Bold.ttf",
)

HERSHEY = cv2.FONT_HERSHEY_SIMPLEX


@lru_cache(maxsize=64)
def _font(px: int, bold: bool):
    if ImageFont is None:
        return None
    candidates = ([config.ESP_FONT] if config.ESP_FONT else [])
    candidates += list(_BOLD_CANDIDATES if bold else ()) + list(_REGULAR_CANDIDATES)
    for path in candidates:
        try:
            return ImageFont.truetype(path, px)
        except (OSError, ValueError):
            continue
    return None


def has_truetype() -> bool:
    return _font(14, False) is not None


def separator() -> str:
    # Hershey has no middle dot.
    return "\u00b7" if has_truetype() else "|"


@lru_cache(maxsize=2048)
def text_mask(text: str, px: int, bold: bool = False) -> np.ndarray:
    """Coverage mask (uint8, H x W) for ``text`` at roughly ``px`` pixel cap height.

    All masks of one ``px``/weight share the same height, so runs line up.
    """
    font = _font(px, bold)
    if font is not None:
        ascent, descent = font.getmetrics()
        width = max(1, int(np.ceil(font.getlength(text))))
        img = Image.new("L", (width, ascent + descent), 0)
        ImageDraw.Draw(img).text((0, 0), text, fill=255, font=font)
        return np.asarray(img, dtype=np.uint8)

    scale = px / 22.0
    thickness = 2 if bold and px >= 16 else 1
    (tw, th), base = cv2.getTextSize(text, HERSHEY, scale, thickness)
    pad = thickness
    mask = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv2.putText(mask, text, (pad, th + pad), HERSHEY, scale, 255, thickness, cv2.LINE_AA)
    return mask


def line_height(px: int) -> int:
    return text_mask("Ag", px).shape[0]


@lru_cache(maxsize=512)
def rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    """Float32 coverage (0..1) of an anti-aliased rounded rectangle."""
    r = max(0, min(r, w // 2, h // 2))
    ss = 4  # supersample so the corners are smooth
    big = np.zeros((h * ss, w * ss), np.uint8)
    R = r * ss
    if R > 0:
        cv2.rectangle(big, (R, 0), (w * ss - 1 - R, h * ss - 1), 255, -1)
        cv2.rectangle(big, (0, R), (w * ss - 1, h * ss - 1 - R), 255, -1)
        for cx, cy in ((R, R), (w * ss - 1 - R, R), (R, h * ss - 1 - R), (w * ss - 1 - R, h * ss - 1 - R)):
            cv2.circle(big, (cx, cy), R, 255, -1, cv2.LINE_AA)
    else:
        big[:] = 255
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    out = small.astype(np.float32) / 255.0
    out.setflags(write=False)
    return out
