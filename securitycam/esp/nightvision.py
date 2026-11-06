"""Night-vision look (green phosphor) and IR/night frame auto detection.

Detection (mode ``auto``): every ``CHECK_EVERY_S`` a 64 px wide thumbnail is
analyzed. IR frames from Tapo cameras are near-grayscale, sometimes with a
uniform purple cast (R ~ B > G). Both have one thing in common: the per-pixel
chroma vector (B-G, R-G) hardly varies across the image, while a daylight
scene has varied colors. The score is the mean distance of the chroma vectors
from their image mean (cast removed); a hysteresis with consecutive-sample
counts switches the mode so it never flickers.

Look: luminance -> vignette (+ optional scanlines) -> film grain -> green
phosphor LUT. LUT, vignette mask and grain tiles are cached per frame size;
all per-frame work is uint8 OpenCV ops (~2-4 ms at 1080p).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

MODES = ("off", "on", "auto")
CHECK_EVERY_S = 0.5
THUMB_W = 64
ENTER_SCORE = 6.0      # chroma spread below this -> looks like IR
EXIT_SCORE = 10.0      # above this -> looks like daylight
ENTER_COUNT = 3        # consecutive samples needed to switch on
EXIT_COUNT = 4         # ... and off
GRAIN_TILES = 4
GRAIN_AMPLITUDE = 10   # +/- gray levels


@dataclass
class NightState:
    active: bool = False
    score: Optional[float] = None
    streak: int = 0
    last_check: float = -1e9
    frames: int = 0


def get_state(camera_state) -> NightState:
    ns = getattr(camera_state, "esp_nightvision", None)
    if not isinstance(ns, NightState):
        ns = camera_state.esp_nightvision = NightState()
    return ns


def chroma_score(frame: np.ndarray) -> float:
    """Spread of per-pixel chroma around the image mean (0 = perfectly uniform tint)."""
    h, w = frame.shape[:2]
    tw = min(THUMB_W, w)
    th = max(1, int(round(h * tw / max(w, 1))))
    small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    cb, cr = b - g, r - g
    # ignore near-black / blown-out pixels: they carry no reliable chroma
    lum = (b + g + r) * (1.0 / 3.0)
    mask = (lum > 12) & (lum < 245)
    if mask.sum() < 16:
        return 0.0
    cb, cr = cb[mask], cr[mask]
    return float(np.mean(np.hypot(cb - cb.mean(), cr - cr.mean())))


def update_auto(ns: NightState, frame: np.ndarray, t: float) -> bool:
    """Feed one frame (throttled internally); returns whether night mode is active."""
    if t - ns.last_check < CHECK_EVERY_S:
        return ns.active
    ns.last_check = t
    score = chroma_score(frame)
    ns.score = score
    if not ns.active:
        ns.streak = ns.streak + 1 if score < ENTER_SCORE else 0
        if ns.streak >= ENTER_COUNT:
            ns.active, ns.streak = True, 0
    else:
        ns.streak = ns.streak + 1 if score > EXIT_SCORE else 0
        if ns.streak >= EXIT_COUNT:
            ns.active, ns.streak = False, 0
    return ns.active


def is_active(camera_state, mode: str, frame: np.ndarray, t: float) -> bool:
    if mode == "on":
        return True
    if mode != "auto":
        return False
    return update_auto(get_state(camera_state), frame, t)


@lru_cache(maxsize=1)
def phosphor_lut() -> np.ndarray:
    """256x1x3 LUT gray -> green phosphor (BGR): lifted blacks, gentle S-curve, hot whites."""
    x = np.linspace(0.0, 1.0, 256)
    y = np.clip(1.08 * x ** 0.8, 0.0, 1.0)
    y = y * y * (3.0 - 2.0 * y) * 0.55 + y * 0.45        # mild contrast
    g = 18 + y * (255 - 18)
    r = 6 + np.clip(y - 0.55, 0, 1) / 0.45 * 150 + y * 25    # whites drift to mint
    b = 8 + np.clip(y - 0.6, 0, 1) / 0.4 * 110 + y * 20
    lut = np.stack([b, g, r], axis=1).clip(0, 255).astype(np.uint8)
    return lut.reshape(256, 1, 3)


@lru_cache(maxsize=4)
def _vignette(w: int, h: int, scanlines: bool) -> np.ndarray:
    """uint8 gain mask (255 = unchanged)."""
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = xs * xs * 0.85 + ys * ys
    gain = np.clip(1.0 - 0.42 * np.power(r2 / 1.85, 1.6), 0.35, 1.0)
    if scanlines:
        period = max(2, h // 360)          # ~360 lines regardless of resolution
        rows = (np.arange(h) // max(1, period // 2)) % 2 == 1
        gain[rows, :] *= 0.82
    out = (gain * 255.0 + 0.5).astype(np.uint8)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=2)
def _grain(w: int, h: int):
    """GRAIN_TILES pairs of (add, sub) uint8 noise maps."""
    rng = np.random.default_rng(1234)
    tiles = []
    for _ in range(GRAIN_TILES):
        n = rng.normal(0.0, GRAIN_AMPLITUDE * 0.5, (h, w)).clip(-GRAIN_AMPLITUDE, GRAIN_AMPLITUDE)
        add = np.maximum(n, 0).astype(np.uint8)
        sub = np.maximum(-n, 0).astype(np.uint8)
        add.setflags(write=False)
        sub.setflags(write=False)
        tiles.append((add, sub))
    return tuple(tiles)


def apply(frame: np.ndarray, frame_idx: int, scanlines: bool, gray: Optional[np.ndarray] = None) -> None:
    """Render the night-vision look onto ``frame`` in place."""
    h, w = frame.shape[:2]
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray.copy()
    cv2.multiply(gray, _vignette(w, h, bool(scanlines)), dst=gray, scale=1.0 / 255.0)
    add, sub = _grain(w, h)[frame_idx % GRAIN_TILES]
    cv2.add(gray, add, dst=gray)
    cv2.subtract(gray, sub, dst=gray)
    cv2.LUT(cv2.merge((gray, gray, gray)), phosphor_lut(), dst=frame)


def info(camera_state) -> dict:
    ns = get_state(camera_state)
    return {"active": ns.active, "score": None if ns.score is None else round(ns.score, 2)}
