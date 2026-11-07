"""Metric scale for relative depth: z_m = scale / disparity.

Depth Anything outputs relative inverse depth. One number per camera turns it
into meters (the shift term of the affine-invariant output is ignored, which
is fine at room / yard distances):

* **auto**: every standing, fully visible person gives a sample
  ``scale = distance_from_box_height * disparity_at_person`` (the same rough
  box-height estimate as the ESP distance label, ``trackstats``). The robust
  median of the last samples is the scale; the spread (MAD) gives a confidence.
* **user**: "this person is X m away" sets ``scale = X * disparity`` once and
  overrides auto (stored as the ``depth_scale`` setting, 0 = auto).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

MIN_SAMPLES = 6            # auto scale is used from this many samples on
MAX_SAMPLES = 240
SAMPLE_GAP_S = 1.0         # one sample per track and second
MAX_SPREAD = 0.45          # relative MAD at which the confidence reaches 0
MIN_CONFIDENCE = 0.35      # below: not "calibrated" (distance falls back to the box estimate)
MIN_BOX_PX = 40.0          # persons smaller than this are too noisy
MIN_ASPECT = 1.6           # height / width: standing persons only (sitting / cut-off boxes lie)
EDGE = 0.015               # boxes touching the frame edge are cut off


@dataclass
class Calibration:
    mode: str              # "user" | "auto" | "none"
    scale: Optional[float]
    samples: int
    confidence: float

    @property
    def ready(self) -> bool:
        return self.scale is not None and (self.mode == "user" or self.confidence >= MIN_CONFIDENCE)

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "scale": round(self.scale, 4) if self.scale else None,
                "samples": self.samples, "confidence": round(self.confidence, 2), "ready": self.ready}


def person_ok(label: str, conf: float, box_n: Sequence[float], width_px: float, height_px: float) -> bool:
    """Whether a detection is a good scale reference (standing person, fully in frame)."""
    if str(label).lower() != "person" or conf < 0.5 or height_px < MIN_BOX_PX or width_px <= 0:
        return False
    x1, y1, x2, y2 = box_n
    if y1 <= EDGE or y2 >= 1.0 - EDGE or x1 <= EDGE or x2 >= 1.0 - EDGE:
        return False
    return height_px / width_px >= MIN_ASPECT


class ScaleCalibrator:
    def __init__(self):
        self._samples: deque = deque(maxlen=MAX_SAMPLES)
        self._last_by_track: Dict[Any, float] = {}

    def reset(self) -> None:
        self._samples.clear()
        self._last_by_track.clear()

    def add(self, disparity: Optional[float], ref_distance_m: float, t: float, track: Any = None) -> bool:
        if disparity is None or not math.isfinite(disparity) or disparity <= 1e-6:
            return False
        if not math.isfinite(ref_distance_m) or not 0.3 <= ref_distance_m <= 80.0:
            return False
        if track is not None:
            last = self._last_by_track.get(track)
            if last is not None and t - last < SAMPLE_GAP_S:
                return False
            self._last_by_track[track] = t
            if len(self._last_by_track) > 512:
                for k in list(self._last_by_track)[:256]:
                    self._last_by_track.pop(k, None)
        self._samples.append(float(disparity) * float(ref_distance_m))
        return True

    def auto(self) -> Calibration:
        n = len(self._samples)
        if n < MIN_SAMPLES:
            return Calibration("none", None, n, 0.0)
        arr = np.asarray(self._samples, np.float64)
        med = float(np.median(arr))
        if med <= 0:
            return Calibration("none", None, n, 0.0)
        mad = float(np.median(np.abs(arr - med))) / med
        conf = max(0.0, min(1.0, 1.0 - mad / MAX_SPREAD)) * min(1.0, n / (MIN_SAMPLES * 3))
        return Calibration("auto", med, n, conf)

    def current(self, user_scale: float = 0.0) -> Calibration:
        if user_scale and user_scale > 0:
            return Calibration("user", float(user_scale), len(self._samples), 1.0)
        return self.auto()


def depth_m(scale: Optional[float], disparity: Optional[float]) -> Optional[float]:
    if scale is None or disparity is None or not math.isfinite(disparity) or disparity <= 1e-6:
        return None
    z = scale / disparity
    return z if 0.1 <= z <= 500.0 else None


def depth_map_m(scale: float, disp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(disp > 1e-6, scale / np.maximum(disp, 1e-6), np.nan).astype(np.float32)


def user_scale_for(distance_m: float, disparity: Optional[float]) -> Optional[float]:
    if disparity is None or disparity <= 1e-6 or not 0.3 <= float(distance_m) <= 200.0:
        return None
    return float(distance_m) * float(disparity)
