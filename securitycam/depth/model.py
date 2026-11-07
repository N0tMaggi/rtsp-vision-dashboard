"""Depth Anything V2 Small (ONNX): device resolution, backends and preprocessing.

Model: ``onnx-community/depth-anything-v2-small`` (Apache-2.0), fp32 ONNX,
dynamic input ``pixel_values [N, 3, H, W]`` -> ``predicted_depth [N, H, W]``
(relative inverse depth: larger = nearer). H and W must be multiples of 14.

Backends
--------
* OpenVINO (Intel NPU / Intel GPU / CPU): compiled for one *static* shape
  (NPUs need static shapes), shared compile cache in ``models/ov_cache``.
* NVIDIA GPU: onnxruntime with the CUDA execution provider. torch is imported
  first so onnxruntime finds the CUDA / cuDNN DLLs torch ships; every call
  runs under ``gpu.guard`` (never overlaps TensorRT work of the camera workers).
* CPU without OpenVINO: onnxruntime CPU.

"auto" for this workload (a new one, so it may differ from SecurityCam's usual
"auto = old behaviour"): Intel NPU if present, else NVIDIA GPU, else Intel
GPU, else CPU. Devices that failed in this process are skipped.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

ASSET = "depth_anything_v2_small"
MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/"
             "4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx")
MODEL_FILE = "depth_anything_v2_small.onnx"
MODEL_MIN_BYTES = 90_000_000          # the fp32 export is 99 MB; anything smaller is truncated
LICENSE = "Apache-2.0 (Depth Anything V2 Small)"

PATCH = 14
MIN_SIDE = PATCH * 8                  # 112 px
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

AUTO_ORDER = ("npu", "cuda", "igpu", "cpu")
DEVICES = ("cuda", "npu", "igpu", "cpu")

_failed_auto: Dict[str, str] = {}     # device -> error (auto skips it for the rest of the process)
_fail_lock = threading.Lock()


def snap14(v: float) -> int:
    return max(MIN_SIDE, int(round(float(v) / PATCH)) * PATCH)


def input_shape(size: int, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """(H, W) model input for a frame: the long side is ``size`` (multiple of 14), aspect kept."""
    size = snap14(size)
    if frame_w <= 0 or frame_h <= 0:
        return size, size
    if frame_w >= frame_h:
        return snap14(size * frame_h / frame_w), size
    return size, snap14(size * frame_w / frame_h)


def preprocess(frame_bgr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """BGR frame -> (1, 3, H, W) float32, ImageNet-normalized RGB."""
    h, w = hw
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = small[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


# ----------------------------------------------------------------------------- device choice
def _available(dev: str) -> bool:
    from ..compute import devices
    if dev == "cpu":
        return True
    return bool(devices.availability(dev)[0])


def mark_auto_failed(dev: str, error: str) -> None:
    with _fail_lock:
        _failed_auto[dev] = str(error)[:300]


def auto_failures() -> Dict[str, str]:
    with _fail_lock:
        return dict(_failed_auto)


def clear_auto_failures() -> None:
    with _fail_lock:
        _failed_auto.clear()


def auto_device(available: Optional[Callable[[str], bool]] = None) -> str:
    """Device "auto" stands for: NPU if present, else NVIDIA GPU, else Intel GPU, else CPU."""
    ok = available or _available
    failed = auto_failures()
    for dev in AUTO_ORDER:
        if dev in failed and dev != "cpu":
            continue
        try:
            if ok(dev):
                return dev
        except Exception:
            continue
    return "cpu"


def resolve() -> Dict[str, Any]:
    """{requested, device, source, fallback_reason, error} for the depth workload right now."""
    from ..compute import assignments
    eff = assignments.effective("depth")
    req = eff["requested"]
    dev = eff["device"]
    out = {"requested": req, "device": dev, "source": "user", "fallback_reason": eff.get("fallback_reason"),
           "error": eff.get("error")}
    if dev == "auto":
        out["device"] = auto_device()
        out["source"] = "auto"
    return out


# ----------------------------------------------------------------------------- backends
class Backend:
    device: str = "cpu"
    label: str = ""
    shape: Tuple[int, int, int, int] = (1, 3, 0, 0)

    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class OVBackend(Backend):
    def __init__(self, path, dev: str, shape: Sequence[int]):
        from ..compute import devices, ov_runtime
        ovname = devices.ov_device(dev) or {"cpu": "CPU", "npu": "NPU", "igpu": "GPU"}.get(dev)
        if not ovname:
            raise RuntimeError(f"{devices.LABELS.get(dev, dev)} is not available")
        compiled, self.compile_s = ov_runtime.compile_raw(path, dev, tuple(int(s) for s in shape)) \
            if devices.ov_device(dev) else _compile_direct(path, ovname, shape)
        self._req = compiled.create_infer_request()
        self.device = dev
        self.shape = tuple(int(s) for s in shape)
        self.label = f"OpenVINO {ovname}"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._req.infer({0: np.ascontiguousarray(x, np.float32)})
        return np.array(self._req.get_output_tensor(0).data, np.float32)


def _compile_direct(path, ovname: str, shape):
    """Compile without discovery (benchmarks / tests before the device scan finished)."""
    from ..compute import devices
    c = devices.core()
    model = c.read_model(str(path))
    model.reshape({model.inputs[0].get_any_name(): list(shape)})
    t0 = time.perf_counter()
    return c.compile_model(model, ovname, {"PERFORMANCE_HINT": "LATENCY"}), time.perf_counter() - t0


_ort_lock = threading.Lock()


def _ort():
    with _ort_lock:
        try:
            import torch  # noqa: F401  - loads the CUDA / cuDNN DLLs onnxruntime-gpu links against
        except Exception:
            pass
        import onnxruntime as ort
        return ort


class OrtBackend(Backend):
    def __init__(self, path, dev: str, shape: Sequence[int]):
        ort = _ort()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if dev == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("onnxruntime has no CUDA support here (install onnxruntime-gpu)")
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        t0 = time.perf_counter()
        self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        if dev == "cuda" and "CUDAExecutionProvider" not in self._sess.get_providers():
            raise RuntimeError("onnxruntime could not start CUDA (CUDA / cuDNN libraries missing)")
        self.compile_s = time.perf_counter() - t0
        self._name = self._sess.get_inputs()[0].name
        self.device = dev
        self.shape = tuple(int(s) for s in shape)
        self.label = "onnxruntime CUDA" if dev == "cuda" else "onnxruntime CPU"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.device == "cuda":
            from .. import gpu
            with gpu.guard("cuda"):
                return np.asarray(self._sess.run(None, {self._name: x})[0], np.float32)
        return np.asarray(self._sess.run(None, {self._name: x})[0], np.float32)


def build_backend(path, dev: str, shape: Sequence[int]) -> Backend:
    """Load + compile the model for ``dev`` and one static input shape, then check one run."""
    from ..compute import devices
    if dev == "cuda":
        be: Backend = OrtBackend(path, dev, shape)
    elif dev == "cpu" and not devices.openvino_installed():
        be = OrtBackend(path, dev, shape)
    else:
        be = OVBackend(path, dev, shape)
    out = be(np.zeros(tuple(int(s) for s in shape), np.float32))
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"{devices.LABELS.get(dev, dev)} returns invalid (non-finite) depth values")
    return be


def to_map(out: np.ndarray) -> np.ndarray:
    """Model output -> (H, W) float32 relative inverse depth, >= 0."""
    a = np.asarray(out, np.float32)
    while a.ndim > 2:
        a = a[0]
    return np.maximum(a, 0.0)
