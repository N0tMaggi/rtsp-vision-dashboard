"""Color helpers and per-detection palette resolution for the ESP overlay.

All colors are OpenCV BGR tuples.
"""
from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

BGR = Tuple[int, int, int]

# Dashboard theme: anthracite #222427 / signal yellow #ffd21f / white #f4f4f2
ANTHRACITE: BGR = (39, 36, 34)
SIGNAL_YELLOW: BGR = (31, 210, 255)
WHITE: BGR = (242, 244, 244)
DEFAULT_BGR: BGR = SIGNAL_YELLOW


def get_rainbow_color(offset: float = 0.0, speed: float = 1.0, now: Optional[float] = None) -> BGR:
    """Time-based rainbow color (RGB mode). Pass ``now`` to keep one frame consistent."""
    t = (time.time() if now is None else now) * speed + offset
    r, g, b = colorsys.hsv_to_rgb(t % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


@lru_cache(maxsize=256)
def parse_hex(hex_str: str) -> Optional[BGR]:
    """Parse ``#rrggbb`` (leading ``#`` optional) into BGR, or ``None`` if invalid."""
    if not isinstance(hex_str, str):
        return None
    value = hex_str.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    return (b, g, r)


def hex_to_bgr(hex_str: str) -> BGR:
    """Backward-compatible helper: invalid input falls back to the theme yellow."""
    parsed = parse_hex(hex_str) if isinstance(hex_str, str) else None
    return parsed if parsed is not None else DEFAULT_BGR


@lru_cache(maxsize=1024)
def class_id_to_bgr(cls_id: int) -> BGR:
    """Deterministic class color derived from the class id."""
    hue = (cls_id * 0.11) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


@dataclass(frozen=True)
class DetectionColors:
    box: BGR
    text: BGR
    fill: BGR
    skeleton: BGR


def mix(a: BGR, b: BGR, t: float) -> BGR:
    """Linear blend ``a`` -> ``b`` (t=0 gives a)."""
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def resolve_detection_colors(settings, order: int, obj_id: int, cls_id: int,
                             highlighted: bool, now: float) -> DetectionColors:
    """Pick colors for one detection: rgb mode > class colors > per-feature colors.

    The detection closest to the frame center is outlined in the secondary color
    (and its fill follows when the fill follows the box color).
    """
    if settings.rgb_mode:
        box = get_rainbow_color(offset=order * 0.1 + obj_id * 0.2, speed=0.5, now=now)
        text = WHITE
    elif settings.class_colors:
        box = class_id_to_bgr(cls_id)
        text = WHITE
    else:
        box = settings.color_boxes
        text = settings.color_text

    if highlighted:
        box = settings.color_secondary

    fill = box if (settings.class_colors or settings.rgb_mode) else settings.color_secondary
    skeleton = (get_rainbow_color(order * 0.5, 2.0, now=now)
                if settings.rgb_mode else settings.color_skeleton)
    return DetectionColors(box=box, text=text, fill=fill, skeleton=skeleton)


def center_dot_color(settings, now: float) -> BGR:
    return get_rainbow_color(0.5, now=now) if settings.rgb_mode else settings.color_primary
