"""Validation of client-supplied ESP settings patches.

Key sets are derived from ``config.DEFAULT_ESP_SETTINGS`` and checked at import
time, so adding a default without a validation rule fails fast instead of
silently accepting (or dropping) the new key.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

from .config import DEFAULT_ESP_SETTINGS

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

# Legacy key -> canonical key. The canonical key wins if both are sent.
ESP_KEY_ALIASES = {"gay_mode": "rgb_mode"}

# Every bool default is a toggle.
BOOL_KEYS = frozenset(k for k, v in DEFAULT_ESP_SETTINGS.items() if isinstance(v, bool))

# Every string default that is a hex color is a color.
COLOR_KEYS = frozenset(
    k for k, v in DEFAULT_ESP_SETTINGS.items() if isinstance(v, str) and HEX_COLOR_RE.match(v)
)

ENUM_KEYS: dict[str, frozenset[str]] = {
    "box_style": frozenset({"full", "corner", "3d"}),
    "face_blur_mode": frozenset({"blur", "pixelate", "blackbox"}),
    "heatmap_window": frozenset({"5m", "1h", "24h"}),
    "night_vision": frozenset({"off", "on", "auto"}),
}

# key -> (min, max, kind); kind is "float", "int" or "odd_int".
NUMERIC_KEYS: dict[str, tuple[float, float, str]] = {
    "confidence_threshold": (0.05, 1.0, "float"),
    "line_thickness": (1, 6, "int"),
    "face_zoom_scale": (1.0, 4.0, "float"),
    "face_blur_strength": (1, 100, "int"),
    "chams_opacity": (0.0, 1.0, "float"),
    "smoothing": (0.0, 1.0, "float"),
    "jpeg_quality": (30, 95, "int"),
    "heatmap_opacity": (0.0, 1.0, "float"),
    "loiter_seconds": (5, 3600, "int"),
}


def _check_coverage() -> None:
    covered = BOOL_KEYS | COLOR_KEYS | set(ENUM_KEYS) | set(NUMERIC_KEYS)
    missing = set(DEFAULT_ESP_SETTINGS) - covered
    unknown = (set(ENUM_KEYS) | set(NUMERIC_KEYS)) - set(DEFAULT_ESP_SETTINGS)
    if missing or unknown:
        raise RuntimeError(
            f"ESP validator out of sync with DEFAULT_ESP_SETTINGS: "
            f"missing rules for {sorted(missing)}, rules for unknown keys {sorted(unknown)}"
        )


_check_coverage()

_INVALID = object()


def _as_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    return _INVALID


def _as_number(value: Any, minimum: float, maximum: float, kind: str) -> Any:
    if isinstance(value, bool):
        return _INVALID
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _INVALID
    if not math.isfinite(number):
        return _INVALID
    number = max(minimum, min(maximum, number))
    if kind == "float":
        return number
    integer = int(round(number))
    if kind == "odd_int" and integer % 2 == 0:
        integer = integer + 1 if integer + 1 <= maximum else integer - 1
    return integer


def _as_enum(value: Any, valid: frozenset[str]) -> Any:
    if not isinstance(value, str):
        return _INVALID
    v = value.strip().lower()
    return v if v in valid else _INVALID


def _as_color(value: Any) -> Any:
    if not isinstance(value, str):
        return _INVALID
    v = value.strip()
    return v.lower() if HEX_COLOR_RE.match(v) else _INVALID


def _validator_for(key: str) -> Callable[[Any], Any] | None:
    if key in BOOL_KEYS:
        return _as_bool
    if key in COLOR_KEYS:
        return _as_color
    if key in ENUM_KEYS:
        return lambda v, valid=ENUM_KEYS[key]: _as_enum(v, valid)
    if key in NUMERIC_KEYS:
        lo, hi, kind = NUMERIC_KEYS[key]
        return lambda v: _as_number(v, lo, hi, kind)
    return None


_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    key: fn for key in DEFAULT_ESP_SETTINGS if (fn := _validator_for(key)) is not None
}


def sanitize_esp_patch(raw_data: Any) -> dict[str, Any]:
    """Return only known keys with valid, clamped values; invalid values are dropped."""
    if not isinstance(raw_data, dict):
        return {}

    data = dict(raw_data)
    for legacy, canonical in ESP_KEY_ALIASES.items():
        if legacy in data:
            legacy_value = data.pop(legacy)
            data.setdefault(canonical, legacy_value)

    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        validate = _VALIDATORS.get(key)
        if validate is None:
            continue
        result = validate(value)
        if result is not _INVALID:
            sanitized[key] = result
    return sanitized


# ---------------------------------------------------------------------------- zones / lock
ZONE_LABEL_MAX = 32
ZONE_MIN_SIZE = 0.01          # fraction of frame width / height
ZONE_DEFAULT_COLOR = "#ffd21f"


class ZoneError(ValueError):
    """Invalid zone input; ``code`` is the API error string."""

    def __init__(self, code: str = "invalid_zone"):
        super().__init__(code)
        self.code = code


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_zone_box(value: Any) -> list[float]:
    """4 finite numbers -> clamped to 0..1, sorted, at least ZONE_MIN_SIZE wide/high."""
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_finite_number(v) for v in value):
        raise ZoneError()
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in value)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < ZONE_MIN_SIZE or y2 - y1 < ZONE_MIN_SIZE:
        raise ZoneError()
    return [round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)]


def parse_zone_patch(data: Any, require_box: bool) -> dict[str, Any]:
    """Validate a zone create (``require_box``) or update body.

    Returns only the provided fields (``box`` / ``label`` / ``color``). An empty
    or whitespace label means "use the default name" (key present, value None).
    Labels longer than ZONE_LABEL_MAX are rejected.
    """
    if not isinstance(data, dict):
        raise ZoneError()
    out: dict[str, Any] = {}
    if "box" in data or require_box:
        out["box"] = parse_zone_box(data.get("box"))
    if "label" in data and data["label"] is not None:
        label = data["label"]
        if not isinstance(label, str):
            raise ZoneError()
        label = " ".join(label.split())
        if len(label) > ZONE_LABEL_MAX:
            raise ZoneError()
        out["label"] = label or None
    if "color" in data and data["color"] is not None:
        color = _as_color(data["color"])
        if color is _INVALID:
            raise ZoneError()
        out["color"] = color
    if not require_box and not out:
        raise ZoneError()
    return out


def parse_lock_id(data: Any) -> Any:
    """``{"id": int}`` -> int, ``{"id": null}`` -> None; anything else raises ValueError."""
    if not isinstance(data, dict) or "id" not in data:
        raise ValueError("invalid_lock")
    value = data["id"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError("invalid_lock")
    return value
