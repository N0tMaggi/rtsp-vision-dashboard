import re
from typing import Any


HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

BOOL_KEYS = {
    "boxes",
    "names",
    "conf",
    "skeleton",
    "tracking",
    "thermal",
    "face_boxes",
    "face_blur",
}

FLOAT_KEYS = {
    "confidence_threshold": (0.05, 1.0),
    "line_thickness": (1.0, 6.0),
    "face_blur_strength": (3.0, 35.0),
}

ENUM_KEYS = {
    "box_style": {"full", "corner", "3d"},
}

COLOR_KEYS = {
    "color_primary",
    "color_secondary",
    "color_text",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def sanitize_esp_patch(raw_data: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}

    for key in BOOL_KEYS:
        if key in raw_data:
            sanitized[key] = _as_bool(raw_data[key])

    for key, (minimum, maximum) in FLOAT_KEYS.items():
        if key not in raw_data:
            continue
        try:
            value = float(raw_data[key])
        except (TypeError, ValueError):
            continue
        value = max(minimum, min(maximum, value))
        if key in {"line_thickness", "face_blur_strength"}:
            value = float(round(value))
        sanitized[key] = value

    for key, valid_values in ENUM_KEYS.items():
        if key not in raw_data:
            continue
        value = str(raw_data[key]).strip().lower()
        if value in valid_values:
            sanitized[key] = value

    for key in COLOR_KEYS:
        if key not in raw_data:
            continue
        value = str(raw_data[key]).strip()
        if HEX_COLOR_RE.match(value):
            sanitized[key] = value.lower()

    return sanitized
