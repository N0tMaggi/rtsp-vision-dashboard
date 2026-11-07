"""Typed, immutable per-frame snapshot of a camera's ESP settings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .. import config
from .colors import BGR, DEFAULT_BGR, WHITE, parse_hex

BOX_STYLES = ("full", "corner", "3d")
CHAMS_MODES = ("box", "silhouette", "outline", "silhouette_outline", "xray")   # silhouette chams

# Legacy setting names still accepted when reading a settings mapping.
SETTING_ALIASES = {"gay_mode": "rgb_mode"}


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not math.isfinite(result):
        result = float(default)
    return max(lo, min(hi, result))


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    return int(round(_float(value, default, lo, hi)))


FACE_BLUR_MODES = ("blur", "pixelate", "blackbox", "image")
FACE_IMAGE_FITS = ("cover", "contain", "stretch")


def _blur_mode(value: Any) -> str:
    m = str(value or "blur").strip().lower()
    return m if m in FACE_BLUR_MODES else "blur"


def _odd_int(value: Any, default: int, lo: int, hi: int) -> int:
    k = _int(value, default, lo, hi)
    if k % 2 == 0:
        k = k + 1 if k + 1 <= hi else k - 1
    return k


def _color(value: Any, fallback: BGR) -> BGR:
    parsed = parse_hex(value) if isinstance(value, str) else None
    return parsed if parsed is not None else fallback


@dataclass(frozen=True)
class EspSettings:
    # geometry / style
    boxes: bool
    box_style: str
    line_thickness: int
    fill_box: bool
    chams: bool
    chams_opacity: float
    # filtering / coloring
    class_colors: bool
    person_only: bool
    highlight_center: bool
    rgb_mode: bool
    thermal: bool
    # labels
    names: bool
    conf: bool
    tracking: bool
    distance: bool
    # pose
    skeleton: bool
    head: bool
    # motion
    snaplines: bool
    tracers: bool
    prediction: bool
    velocity: bool
    center_dot: bool
    # faces / hands
    face_boxes: bool
    face_zoom: bool
    face_blur: bool
    face_blur_strength: int
    face_blur_mode: str
    face_blur_image: str
    face_blur_image_fit: str
    face_blur_image_underlay: bool
    face_zoom_scale: float
    hand_skeleton: bool
    # motion / look / output
    smoothing: float
    confidence_threshold: float
    zones: bool
    label_background: bool
    glow: bool
    jpeg_quality: int
    # colors (BGR)
    color_primary: BGR
    color_secondary: BGR
    color_boxes: BGR
    color_text: BGR
    color_skeleton: BGR
    color_head: BGR
    color_tracers: BGR
    color_velocity: BGR
    color_prediction: BGR
    color_face: BGR
    color_hand: BGR
    chams_color: BGR
    # --- silhouette chams (silhouette.py); defaults keep duck-typed constructions working ---
    chams_mode: str = "box"
    outline_glow: int = 8

    # ---- derived flags -------------------------------------------------
    @property
    def needs_faces(self) -> bool:
        return self.face_boxes or self.face_zoom or self.face_blur

    @property
    def needs_keypoints(self) -> bool:
        return self.skeleton or self.head

    @property
    def needs_history(self) -> bool:
        return self.tracers or self.velocity or self.prediction

    @property
    def needs_labels(self) -> bool:
        return self.names or self.conf or self.tracking or self.distance

    # ---- construction --------------------------------------------------
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EspSettings":
        """Build a snapshot from a settings dict (one shallow copy, then no more lookups)."""
        d = dict(config.DEFAULT_ESP_SETTINGS)
        for legacy, canonical in SETTING_ALIASES.items():
            if legacy in raw and canonical not in raw:
                d[canonical] = raw[legacy]
        d.update(raw)
        D = config.DEFAULT_ESP_SETTINGS

        def b(key: str) -> bool:
            return _bool(d.get(key), bool(D.get(key, False)))

        primary = _color(d.get("color_primary"), DEFAULT_BGR)
        secondary = _color(d.get("color_secondary"), WHITE)
        box_style = str(d.get("box_style", "corner")).strip().lower()

        return cls(
            boxes=b("boxes"),
            box_style=box_style if box_style in BOX_STYLES else "full",
            line_thickness=_int(d.get("line_thickness"), 2, 1, 6),
            fill_box=b("fill_box"),
            chams=b("chams"),
            chams_opacity=_float(d.get("chams_opacity"), 0.35, 0.0, 1.0),
            class_colors=b("class_colors"),
            person_only=b("person_only"),
            highlight_center=b("highlight_center"),
            rgb_mode=b("rgb_mode"),
            thermal=b("thermal"),
            names=b("names"),
            conf=b("conf"),
            tracking=b("tracking"),
            distance=b("distance"),
            skeleton=b("skeleton"),
            head=b("head"),
            snaplines=b("snaplines"),
            tracers=b("tracers"),
            prediction=b("prediction"),
            velocity=b("velocity"),
            center_dot=b("center_dot"),
            face_boxes=b("face_boxes"),
            face_zoom=b("face_zoom"),
            face_blur=b("face_blur"),
            face_blur_strength=_int(d.get("face_blur_strength"), 70, 1, 100),
            face_blur_mode=_blur_mode(d.get("face_blur_mode")),
            face_blur_image=str(d.get("face_blur_image") or ""),
            face_blur_image_fit=d.get("face_blur_image_fit") if d.get("face_blur_image_fit") in FACE_IMAGE_FITS else "cover",
            face_blur_image_underlay=b("face_blur_image_underlay"),
            face_zoom_scale=_float(d.get("face_zoom_scale"), 2.0, 1.0, 4.0),
            hand_skeleton=b("hand_skeleton"),
            smoothing=_float(d.get("smoothing"), 0.6, 0.0, 1.0),
            confidence_threshold=_float(d.get("confidence_threshold"), 0.35, 0.0, 1.0),
            zones=b("zones"),
            label_background=b("label_background"),
            glow=b("glow"),
            jpeg_quality=_int(d.get("jpeg_quality"), config.JPEG_QUALITY, 30, 95),
            color_primary=primary,
            color_secondary=secondary,
            color_boxes=_color(d.get("color_boxes"), primary),
            color_text=_color(d.get("color_text"), primary),
            color_skeleton=_color(d.get("color_skeleton"), primary),
            color_head=_color(d.get("color_head"), (87, 107, 255)),
            color_tracers=_color(d.get("color_tracers"), secondary),
            color_velocity=_color(d.get("color_velocity"), secondary),
            color_prediction=_color(d.get("color_prediction"), secondary),
            color_face=_color(d.get("color_face"), secondary),
            color_hand=_color(d.get("color_hand"), primary),
            chams_color=_color(d.get("chams_color"), primary),
            # silhouette chams
            chams_mode=d.get("chams_mode") if d.get("chams_mode") in CHAMS_MODES else "box",
            outline_glow=_int(d.get("outline_glow"), 8, 0, 24),
        )
