"""Game-style ESP overlay for camera frames.

Layout:
    settings.py    -- ``EspSettings``: typed snapshot of ``camera_state.esp_settings``
    colors.py      -- hex parsing, rainbow / class palettes, per-detection colors
    detections.py  -- results -> ``Detection`` extraction
    motion.py      -- render-time tracks: extrapolation, smoothing, fades, history
    text.py        -- cached anti-aliased label text (TrueType monospace / Hershey fallback)
    sensors.py     -- throttled face (Haar) and hand (MediaPipe) detection
    renderers.py   -- resolution-scaled per-feature drawing functions
    pipeline.py    -- ``draw_custom_esp`` orchestration
"""
from .colors import class_id_to_bgr, get_rainbow_color, hex_to_bgr
from .pipeline import draw_custom_esp
from .renderers import SKELETON_LIMBS
from .settings import EspSettings

__all__ = [
    "draw_custom_esp",
    "EspSettings",
    "SKELETON_LIMBS",
    "class_id_to_bgr",
    "get_rainbow_color",
    "hex_to_bgr",
]
