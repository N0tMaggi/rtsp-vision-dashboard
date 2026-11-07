"""YAMNet sound classification through OpenVINO (Intel NPU by default, CPU fallback).

Model: YAMNet (AudioSet, 521 classes, Apache-2.0, Google) as the TFLite model
of the TensorFlow Lite Task Library. OpenVINO reads TFLite directly; the input
is a static 15600-sample waveform (0.975 s at 16 kHz, float32 -1..1), so the
NPU compiles it as is (log-mel front end included). The AudioSet class names
are embedded in the file (zip metadata ``yamnet_label_list.txt``).

One background worker classifies the newest window of every camera that
submitted one (stale windows are dropped, so a slow device never builds a
backlog). The "Sound" compute workload picks the device: ``auto`` = Intel NPU
when present, else CPU (both OpenVINO); a failing NPU falls back to the CPU.
"""
from __future__ import annotations

import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

WINDOW_SAMPLES = 15600
ASSET = "sound_yamnet"
WORKLOAD = "sound"
MODEL_URL = ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/"
             "audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite")
MODEL_FILE = "yamnet_classification.tflite"
MODEL_LICENSE = "Apache-2.0 (YAMNet, Google / TensorFlow Model Garden)"

_labels_cache: Dict[str, List[str]] = {}
_auto_failed: set = set()
_lock = threading.Lock()


def labels_from_model(path: Path) -> List[str]:
    key = str(path)
    if key in _labels_cache:
        return _labels_cache[key]
    names: List[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".txt"):
                    names = [ln.strip() for ln in z.read(n).decode("utf-8", "replace").splitlines() if ln.strip()]
                    break
    except (OSError, zipfile.BadZipFile):
        names = []
    _labels_cache[key] = names
    return names


def model_path() -> Optional[Path]:
    """Local model path; starts the download on first use and returns None meanwhile."""
    from .. import deps

    return deps.asset_path(ASSET)


def model_status() -> str:
    from .. import config, deps

    if (config.MODELS_DIR / MODEL_FILE).is_file():
        return "ready"
    return deps.asset_status.get(ASSET, "missing")


# ----------------------------------------------------------------------------- device choice
def resolve_device() -> Tuple[Optional[str], str]:
    """(device id or None, reason). auto = NPU if present (and not failed) else OpenVINO CPU."""
    from ..compute import assignments, devices

    if not devices.openvino_installed():
        return None, "Sound classification needs OpenVINO (python -m pip install openvino)."
    devices.start()
    if not devices.ready():
        return None, "Detecting hardware…"
    eff = assignments.effective(WORKLOAD)
    dev = eff["device"]
    if dev != "auto":
        return dev, ""
    with _lock:
        failed = set(_auto_failed)
    for cand in ("npu", "cpu"):
        if cand not in failed and devices.ov_device(cand):
            return cand, ""
    return None, "No OpenVINO device can run the sound model."


def auto_label() -> Dict[str, str]:
    """Shown as "Auto runs on ..." in the compute panel."""
    try:
        from ..compute import devices

        if devices.ov_device("npu"):
            return {"device": "npu", "label": "Intel NPU (OpenVINO)"}
    except Exception:
        pass
    return {"device": "cpu", "label": "CPU (OpenVINO)"}


def _on_fail(workload: str, msg: str) -> None:
    from ..compute import assignments

    eff = assignments.effective(WORKLOAD)
    if eff["requested"] == "auto":
        dev, _ = _current
        if dev:
            with _lock:
                _auto_failed.add(dev)
            print(f"[Sound] Classifier on {dev} failed, falling back: {msg}")
    else:
        assignments.mark_failed(WORKLOAD, eff["requested"], msg)


_current: Tuple[Optional[str], str] = (None, "")


def get_net(path: Path):
    """Ready OVNet for the current device or None while compiling / unavailable."""
    global _current
    from ..compute.ov_runtime import POOL

    dev, reason = resolve_device()
    _current = (dev, reason)
    if dev is None:
        return None
    return POOL.get(WORKLOAD, path, dev, (WINDOW_SAMPLES,), on_fail=_on_fail)


# ----------------------------------------------------------------------------- worker
Callback = Callable[[np.ndarray, float, str], None]


class Classifier:
    """Single worker; ``submit(key, window, cb)`` keeps only the newest window per key."""

    def __init__(self, infer: Optional[Callable[[np.ndarray], Tuple[np.ndarray, str]]] = None):
        self._cond = threading.Condition()
        self._pending: Dict[str, Tuple[np.ndarray, Callback]] = {}
        self._thread: Optional[threading.Thread] = None
        self._infer = infer or self._default_infer
        self.status = "idle"          # idle | downloading | preparing | ready | unavailable
        self.reason = ""
        self.device: Optional[str] = None
        self.last_ms: Optional[float] = None
        self.labels: List[str] = []

    def submit(self, key: str, window: np.ndarray, cb: Callback) -> None:
        with self._cond:
            self._pending[key] = (window, cb)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="SoundClassifier")
                self._thread.start()
            self._cond.notify()

    def drop(self, key: str) -> None:
        with self._cond:
            self._pending.pop(key, None)

    def _run(self) -> None:
        idle_since = time.monotonic()
        while True:
            with self._cond:
                while not self._pending:
                    if time.monotonic() - idle_since > 30.0:
                        self._thread = None
                        return
                    self._cond.wait(1.0)
                key = next(iter(self._pending))
                window, cb = self._pending.pop(key)
            idle_since = time.monotonic()
            try:
                t0 = time.perf_counter()
                result = self._infer(window)
                if result is None:
                    continue
                scores, device = result
                ms = (time.perf_counter() - t0) * 1000.0
                self.last_ms = ms
                cb(scores, ms, device)
            except Exception as exc:
                self.status, self.reason = "unavailable", f"{type(exc).__name__}: {exc}"[:200]
                time.sleep(0.5)

    def _default_infer(self, window: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        path = model_path()
        if path is None:
            st = model_status()
            self.status = "downloading" if st in ("downloading", "missing") else "unavailable"
            self.reason = "Downloading the sound model…" if self.status == "downloading" else \
                "Sound model download failed - retrying."
            return None
        if not self.labels:
            self.labels = labels_from_model(path)
        net = get_net(path)
        dev, reason = _current
        if net is None:
            self.status = "unavailable" if dev is None else "preparing"
            self.reason = reason or "Compiling the sound model…"
            self.device = dev
            return None
        x = np.ascontiguousarray(window[-WINDOW_SAMPLES:], np.float32)
        out = net(x)
        self.status, self.reason, self.device = "ready", "", dev
        return np.asarray(out, np.float32).reshape(-1), dev or ""

    def info(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "device": self.device,
                "ms": round(self.last_ms, 2) if self.last_ms is not None else None}


CLASSIFIER = Classifier()


# ----------------------------------------------------------------------------- benchmark ("Test speed")
def bench(job, set_row, time_fn) -> None:
    """compute.benchmark runner for the Sound workload: raw inference per 0.975 s window."""
    from ..compute import devices, ov_runtime

    path = model_path()
    if path is None:
        raise RuntimeError("The sound model is still downloading - try again in a moment.")
    t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / 16000.0
    x = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    for row in list(job["rows"]):
        dev = row["device"]
        set_row(dev, status="running")
        try:
            real = dev
            if dev == "auto":
                real = auto_label()["device"]
                set_row(dev, label=f"Auto ({devices.LABELS.get(real, real)})")
            compiled, _ = ov_runtime.compile_raw(path, real)
            req = compiled.create_infer_request()
            set_row(dev, ms=round(time_fn(lambda: req.infer({0: x})), 2), status="done")
        except Exception as exc:
            set_row(dev, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}",
                    status="failed")
