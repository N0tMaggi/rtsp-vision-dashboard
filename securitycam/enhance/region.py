"""Request validation and crop math for the Enhance view (pure functions)."""
from __future__ import annotations

import math
import re
from typing import Any

MIN_CROP_PX = 8
SCALES = (2, 4)
JOB_ID = re.compile(r"^[0-9a-f]{20}$")
DEVICES = ("auto", "cuda", "openvino", "cpu")


class EnhanceInputError(ValueError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def parse_region(raw: Any) -> tuple[float, float, float, float]:
    """``[x1, y1, x2, y2]`` normalized to the frame (0..1). Missing -> full frame.

    Values are clamped to the frame; corners may come in any order. Non-finite
    numbers, wrong lengths or a degenerate box are rejected.
    """
    if raw is None:
        return (0.0, 0.0, 1.0, 1.0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise EnhanceInputError("invalid_region")
    vals = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise EnhanceInputError("invalid_region")
        vals.append(min(1.0, max(0.0, float(v))))
    x1, x2 = sorted((vals[0], vals[2]))
    y1, y2 = sorted((vals[1], vals[3]))
    if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
        raise EnhanceInputError("invalid_region")
    return (x1, y1, x2, y2)


def crop_box(region: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel box ``(x1, y1, x2, y2)`` (x2/y2 exclusive) for a normalized region.

    Rounds outward so the crop covers everything that was visible, and grows
    tiny boxes to at least ``MIN_CROP_PX`` (kept inside the frame).
    """
    if width <= 0 or height <= 0:
        raise EnhanceInputError("no_frame", 409)
    x1 = int(math.floor(region[0] * width + 1e-6))
    y1 = int(math.floor(region[1] * height + 1e-6))
    x2 = int(math.ceil(region[2] * width - 1e-6))
    y2 = int(math.ceil(region[3] * height - 1e-6))
    x1, x2 = _grow(x1, x2, width)
    y1, y2 = _grow(y1, y2, height)
    return (x1, y1, x2, y2)


def _grow(a: int, b: int, size: int) -> tuple[int, int]:
    a = max(0, min(a, size))
    b = max(0, min(b, size))
    need = min(MIN_CROP_PX, size)
    if b - a < need:
        c = (a + b) / 2.0
        a = int(round(c - need / 2.0))
        a = max(0, min(a, size - need))
        b = a + need
    return a, b


def parse_options(data: Any, model_ids: tuple[str, ...], default_model: str) -> dict[str, Any]:
    """Validate the JSON body of ``POST /api/cameras/<id>/enhance``."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise EnhanceInputError("invalid_body")
    region = parse_region(data.get("region"))
    model = data.get("model", default_model)
    if not isinstance(model, str) or model not in model_ids:
        raise EnhanceInputError("invalid_model")
    scale = data.get("scale", 4)
    if isinstance(scale, bool) or scale not in SCALES:
        raise EnhanceInputError("invalid_scale")
    denoise = data.get("denoise", 0.5)
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(denoise):
        raise EnhanceInputError("invalid_denoise")
    device = data.get("device", "auto")
    if not isinstance(device, str) or device not in DEVICES:
        raise EnhanceInputError("invalid_device")
    overlay = data.get("include_overlay", False)
    if not isinstance(overlay, bool):
        raise EnhanceInputError("invalid_overlay")
    return {
        "region": region,
        "model": model,
        "scale": int(scale),
        "denoise": round(min(1.0, max(0.0, float(denoise))), 2),
        "device": device,
        "include_overlay": overlay,
    }


def valid_job_id(job_id: Any) -> bool:
    return isinstance(job_id, str) and bool(JOB_ID.match(job_id))
