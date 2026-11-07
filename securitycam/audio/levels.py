"""Audio level math on 16 kHz mono int16 PCM: dBFS, loudness meter, noise floor,
loud-event detection and a small mel spectrum for the on-stage spectrogram.

Everything here is pure numpy and works on one ~100 ms chunk at a time, so it
costs next to nothing per camera (one 2048-point FFT per chunk).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600            # 100 ms -> meter / spectrogram at 10 Hz
MIN_DB = -96.0                  # 16-bit noise floor; silence reads as this
FULL_SCALE = 32768.0


def to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    """s16le bytes (or an int16 array) -> float32 in [-1, 1)."""
    arr = np.frombuffer(pcm, dtype="<i2") if isinstance(pcm, (bytes, bytearray, memoryview)) else np.asarray(pcm)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / FULL_SCALE
    return arr


def db(value: float) -> float:
    """Linear amplitude (1.0 = full scale) -> dBFS, clamped to ``MIN_DB``."""
    if value <= 0 or not math.isfinite(value):
        return MIN_DB
    return max(MIN_DB, 20.0 * math.log10(value))


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (a full-scale square wave is 0 dBFS, a full-scale sine -3 dBFS)."""
    if x.size == 0:
        return MIN_DB
    return db(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))


def peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return MIN_DB
    return db(float(np.max(np.abs(x))))


class Meter:
    """Smoothed loudness for the level bar: fast attack, slow release (in dB)."""

    def __init__(self, attack: float = 0.6, release: float = 0.15):
        self.attack = attack
        self.release = release
        self.value = MIN_DB

    def update(self, level_db: float) -> float:
        a = self.attack if level_db > self.value else self.release
        self.value += (level_db - self.value) * a
        return self.value


class NoiseFloor:
    """Adaptive background level: drops quickly to quieter levels, rises slowly.

    ``down`` / ``up`` are the per-chunk follow rates (10 chunks per second): with
    the defaults the floor reaches a quieter room within ~0.5 s and needs ~10 s
    to accept a permanently louder one, so short loud sounds barely move it.
    """

    def __init__(self, down: float = 0.25, up: float = 0.01):
        self.down = down
        self.up = up
        self.value: Optional[float] = None

    def update(self, level_db: float, frozen: bool = False) -> float:
        if self.value is None:
            self.value = level_db
        elif level_db < self.value:
            self.value += (level_db - self.value) * self.down
        elif not frozen:
            self.value += (level_db - self.value) * self.up
        return self.value


@dataclass
class LoudEvent:
    level_db: float           # loudest chunk of the episode so far (dBFS)
    above_db: float           # how far above the noise floor
    duration_ms: float


class LoudDetector:
    """A "loud" episode: level above ``floor + threshold_db`` for at least ``min_ms``.

    Fires once per episode; the episode ends when the level falls back below
    ``floor + threshold_db - release_db``. Levels quieter than ``min_level_db``
    never count (a whisper over a dead-silent floor is not "loud").
    """

    def __init__(self, threshold_db: float = 20.0, min_ms: float = 300.0,
                 release_db: float = 3.0, min_level_db: float = -55.0):
        self.threshold_db = threshold_db
        self.min_ms = min_ms
        self.release_db = release_db
        self.min_level_db = min_level_db
        self.above_ms = 0.0
        self.fired = False
        self.peak = MIN_DB

    @property
    def active(self) -> bool:
        return self.above_ms > 0

    def update(self, level_db: float, floor_db: float, dt_ms: float) -> Optional[LoudEvent]:
        above = level_db - floor_db
        if above >= self.threshold_db and level_db >= self.min_level_db:
            self.above_ms += dt_ms
            self.peak = max(self.peak, level_db)
            if not self.fired and self.above_ms >= self.min_ms:
                self.fired = True
                return LoudEvent(self.peak, self.peak - floor_db, self.above_ms)
            return None
        if above < self.threshold_db - self.release_db or level_db < self.min_level_db:
            self.above_ms = 0.0
            self.fired = False
            self.peak = MIN_DB
        return None


# ----------------------------------------------------------------------------- mel spectrum
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_bands: int, n_fft: int, sr: int = SAMPLE_RATE,
                   fmin: float = 60.0, fmax: Optional[float] = None) -> np.ndarray:
    """(n_bands, n_fft // 2 + 1) triangular mel filters with peak 1 (band power = weighted sum)."""
    fmax = fmax or sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bands + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_bands, bins.size), np.float32)
    for i in range(n_bands):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-9)
        down = (hi - bins) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
        if fb[i].sum() <= 0:      # band narrower than one FFT bin: take the nearest bin
            fb[i, int(np.argmin(np.abs(bins - mid)))] = 1.0
    return fb


class MelSpectrum:
    """One spectrogram column (``n_bands`` values 0..255) per chunk."""

    def __init__(self, n_bands: int = 32, n_fft: int = 2048, lo_db: float = -95.0, hi_db: float = -5.0):
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.lo_db = lo_db
        self.hi_db = hi_db
        self.fb = mel_filterbank(n_bands, n_fft)
        self._windows: dict[int, np.ndarray] = {}

    def _window(self, n: int) -> np.ndarray:
        w = self._windows.get(n)
        if w is None:
            w = self._windows[n] = np.hanning(n).astype(np.float32)
        return w

    def band_db(self, x: np.ndarray) -> np.ndarray:
        """Mel band power in dB (relative to a full-scale sine ~ 0 dB)."""
        x = x[-self.n_fft:]
        w = self._window(x.size)
        spec = np.fft.rfft(x * w, n=self.n_fft)
        # Parseval: one-sided sum |X|^2 = N/2 * sum(x^2 w^2); scaled so a full-scale sine
        # (mean power 0.5) reads ~0 dB in the band that contains it
        power = (np.abs(spec) ** 2) * (4.0 / (self.n_fft * max(1e-9, float(np.sum(w * w)))))
        bands = self.fb @ power.astype(np.float32)
        return 10.0 * np.log10(np.maximum(bands, 1e-12))

    def column(self, x: np.ndarray) -> np.ndarray:
        d = self.band_db(x)
        scaled = (d - self.lo_db) / (self.hi_db - self.lo_db) * 255.0
        return np.clip(scaled, 0, 255).astype(np.uint8)
