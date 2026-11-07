"""Which device runs which workload: assignments, failures and effective resolution.

Everything defaults to ``"auto"`` = the behaviour SecurityCam always had (YOLO
on CUDA when available else CPU, TensorRT engine when built, plate models via
onnxruntime/OpenCV, faces via OpenCV). A non-auto choice that is unavailable or
failed resolves back to auto; the reason is kept for the UI.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from . import devices

WORKLOADS: Dict[str, Dict[str, Any]] = {
    "yolo": {
        "label": "Object detection",
        "description": "YOLO detection, tracking and pose for every camera.",
        "devices": ("cuda", "npu", "igpu", "cpu"),
        "per_camera": True,
    },
    "faces": {
        "label": "Faces",
        "description": "YuNet face detection and SFace recognition (known faces, face boxes, blur).",
        "devices": ("npu", "igpu", "cpu"),
        "per_camera": False,
    },
    "plates": {
        "label": "License plates",
        "description": "Plate detector and plate OCR on vehicles.",
        "devices": ("igpu", "cpu"),
        "per_camera": False,
        "unsupported": {
            # verified with OpenVINO 2026.4: the end-to-end NMS of the plate detector returns
            # garbage boxes / inf on the NPU (and can hang it); the OCR alone would work
            "npu": "The plate detector's built-in NMS gives wrong results on the Intel NPU. Use Intel GPU or CPU.",
        },
    },
    # --- depth (securitycam/depth): new workload, so its Auto may differ from "old behaviour":
    # Auto = Intel NPU if present, else NVIDIA GPU, else Intel GPU, else CPU (depth.model.auto_device)
    "depth": {
        "label": "Depth",
        "description": "Depth Anything V2 for the room map and depth view (only while Depth is on).",
        "devices": ("cuda", "npu", "igpu", "cpu"),
        "per_camera": False,
    },
    # --- Sound ESP (securitycam/audio): its Auto = Intel NPU when present, else CPU (OpenVINO) ---
    "sound": {
        "label": "Sound",
        "description": "YAMNet sound classification (Sound ESP), one inference per 0.5 s per camera.",
        "devices": ("npu", "igpu", "cpu"),
        "per_camera": False,
    },
}
CHOICES = ("auto",) + devices.DEVICE_IDS
NO_CUDA_REASON = "These models have no CUDA backend here. Auto already runs them on the CPU, which is fast enough."


class AssignmentError(ValueError):
    def __init__(self, code: str, detail: Optional[str] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


_lock = threading.RLock()
_global: Dict[str, str] = {w: "auto" for w in WORKLOADS}
_cameras: Dict[str, str] = {}               # camera id -> device (YOLO override; "auto" allowed)
_failures: Dict[Tuple[str, str], Dict[str, Any]] = {}   # (workload, device) -> {error, model, at}
_benchmarks: Dict[str, Dict[str, Any]] = {}  # workload -> {model or "_": result}
_version = 0                                 # bumped on every change (cache keys for workers)
_listeners: list = []


def version() -> int:
    return _version


def on_change(fn) -> None:
    _listeners.append(fn)


def _changed() -> None:
    global _version
    _version += 1
    for fn in list(_listeners):
        try:
            fn()
        except Exception as exc:
            print(f"[COMPUTE] change listener failed: {exc}")


def supports(workload: str, dev: str) -> tuple[bool, Optional[str]]:
    """(supported, reason) for a workload/device pair, including hardware availability."""
    if dev == "auto":
        return True, None
    if workload not in WORKLOADS or dev not in CHOICES:
        return False, "Unknown device."
    if dev not in WORKLOADS[workload]["devices"]:
        return False, WORKLOADS[workload].get("unsupported", {}).get(dev, NO_CUDA_REASON)
    return devices.availability(dev)


# ----------------------------------------------------------------------------- read
def assignment(workload: str) -> str:
    with _lock:
        return _global.get(workload, "auto")


def camera_override(camera_id: str) -> Optional[str]:
    with _lock:
        return _cameras.get(camera_id)


def is_default() -> bool:
    with _lock:
        return all(v == "auto" for v in _global.values()) and not any(v != "auto" for v in _cameras.values())


def failure(workload: str, dev: str) -> Optional[Dict[str, Any]]:
    with _lock:
        f = _failures.get((workload, dev))
        return dict(f) if f else None


def requested(workload: str, camera_id: Optional[str] = None) -> Tuple[str, str]:
    """(device, source) the user asked for: source is "camera" for a per-camera override."""
    with _lock:
        if workload == "yolo" and camera_id is not None and camera_id in _cameras:
            return _cameras[camera_id], "camera"
        return _global.get(workload, "auto"), "global"


def effective(workload: str, camera_id: Optional[str] = None) -> Dict[str, Any]:
    """What actually runs: {requested, source, device, fallback_reason, error}."""
    req, source = requested(workload, camera_id)
    out = {"requested": req, "source": source, "device": req, "fallback_reason": None, "error": None}
    if req == "auto":
        return out
    ok, reason = supports(workload, req)
    if not ok:
        out.update(device="auto", fallback_reason=reason)
        return out
    f = failure(workload, req)
    if f is not None:
        out.update(device="auto", fallback_reason="failed", error=f.get("error"))
    return out


# ----------------------------------------------------------------------------- write
def mark_failed(workload: str, dev: str, error: str, model: Optional[str] = None) -> bool:
    """Remember that ``dev`` failed for ``workload`` (falls back to auto). True if newly failed."""
    with _lock:
        new = (workload, dev) not in _failures
        _failures[(workload, dev)] = {"error": str(error)[:400] or "error", "model": model, "at": time.time()}
    if new:
        print(f"[COMPUTE] {WORKLOADS.get(workload, {}).get('label', workload)} on "
              f"{devices.LABELS.get(dev, dev)} failed, using Auto: {str(error).splitlines()[0][:200] if error else ''}")
        _changed()
    return new


def clear_failures(workload: Optional[str] = None) -> None:
    with _lock:
        for k in [k for k in _failures if workload is None or k[0] == workload]:
            del _failures[k]


def validate_patch(body: Any, known_cameras: set) -> Dict[str, Any]:
    """Normalize a PATCH body; raises AssignmentError. Unavailable devices are rejected."""
    if not isinstance(body, dict):
        raise AssignmentError("invalid_body")
    out: Dict[str, Any] = {"workloads": {}, "cameras": {}}
    wl = body.get("workloads", {})
    cams = body.get("cameras", {})
    if not isinstance(wl, dict) or not isinstance(cams, dict):
        raise AssignmentError("invalid_body")
    if not wl and not cams:
        raise AssignmentError("empty_patch")
    for name, dev in wl.items():
        if name not in WORKLOADS:
            raise AssignmentError("invalid_workload", str(name)[:40])
        if not isinstance(dev, str) or dev not in CHOICES:
            raise AssignmentError("invalid_device", str(dev)[:40])
        ok, reason = supports(name, dev)
        if not ok:
            raise AssignmentError("device_unavailable", reason)
        out["workloads"][name] = dev
    for cam_id, dev in cams.items():
        if cam_id not in known_cameras:
            raise AssignmentError("unknown_camera", str(cam_id)[:64])
        if dev is None:
            out["cameras"][cam_id] = None
            continue
        if not isinstance(dev, str) or dev not in CHOICES:
            raise AssignmentError("invalid_device", str(dev)[:40])
        ok, reason = supports("yolo", dev)
        if not ok:
            raise AssignmentError("device_unavailable", reason)
        out["cameras"][cam_id] = dev
    return out


def apply_patch(clean: Dict[str, Any]) -> None:
    with _lock:
        for name, dev in clean.get("workloads", {}).items():
            _global[name] = dev
            # picking a device again is an explicit retry
            for k in [k for k in _failures if k[0] == name]:
                del _failures[k]
        for cam_id, dev in clean.get("cameras", {}).items():
            if dev is None:
                _cameras.pop(cam_id, None)
            else:
                _cameras[cam_id] = dev
            if dev:
                _failures.pop(("yolo", dev), None)
    _changed()


def reset() -> None:
    with _lock:
        for w in _global:
            _global[w] = "auto"
        _cameras.clear()
        _failures.clear()
    _changed()


# ----------------------------------------------------------------------------- benchmarks
def store_benchmark(workload: str, key: str, result: Dict[str, Any]) -> None:
    with _lock:
        _benchmarks.setdefault(workload, {})[key] = result
        # keep it small: last 6 models per workload
        items = sorted(_benchmarks[workload].items(), key=lambda kv: -float(kv[1].get("finished") or 0))
        _benchmarks[workload] = dict(items[:6])


def benchmark(workload: str, key: str) -> Optional[Dict[str, Any]]:
    with _lock:
        b = (_benchmarks.get(workload) or {}).get(key)
        return dict(b) if b else None


def benchmarks() -> Dict[str, Any]:
    with _lock:
        return {w: dict(v) for w, v in _benchmarks.items()}


# ----------------------------------------------------------------------------- persistence
def dump() -> Dict[str, Any]:
    with _lock:
        return {
            "workloads": dict(_global),
            "cameras": dict(_cameras),
            "benchmarks": {w: dict(v) for w, v in _benchmarks.items()},
        }


def _clean_bench(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list):
        return None
    rows = []
    for r in raw["rows"][:12]:
        if not isinstance(r, dict) or r.get("device") not in CHOICES + ("cpu_torch", "opencv"):
            continue
        ms = r.get("ms")
        rows.append({
            "device": r["device"],
            "label": str(r.get("label") or "")[:60],
            "ms": float(ms) if isinstance(ms, (int, float)) and not isinstance(ms, bool) else None,
            "error": str(r["error"])[:200] if r.get("error") else None,
            "auto": bool(r.get("auto")),
        })
    fin = raw.get("finished")
    return {"workload": str(raw.get("workload") or "")[:16], "model": str(raw.get("model") or "")[:64] or None,
            "rows": rows, "finished": float(fin) if isinstance(fin, (int, float)) else 0.0}


def validate(raw: Any) -> Tuple[Dict[str, Any], list]:
    if not isinstance(raw, dict):
        raise ValueError("compute section is not an object")
    warnings = []
    clean = {"workloads": {}, "cameras": {}, "benchmarks": {}}
    for name, dev in (raw.get("workloads") or {}).items() if isinstance(raw.get("workloads"), dict) else []:
        if name in WORKLOADS and dev in CHOICES:
            clean["workloads"][name] = dev
        else:
            warnings.append(f"ignored assignment {name!r} -> {dev!r}")
    cams = raw.get("cameras")
    if isinstance(cams, dict):
        for cam_id, dev in cams.items():
            if isinstance(cam_id, str) and dev in CHOICES:
                clean["cameras"][cam_id[:64]] = dev
            else:
                warnings.append(f"ignored camera override {cam_id!r} -> {dev!r}")
    bench = raw.get("benchmarks")
    if isinstance(bench, dict):
        for w, per in bench.items():
            if w in WORKLOADS and isinstance(per, dict):
                for k, v in list(per.items())[:6]:
                    b = _clean_bench(v)
                    if b is not None:
                        clean["benchmarks"].setdefault(w, {})[str(k)[:64]] = b
    return clean, warnings


def load(clean: Dict[str, Any]) -> None:
    with _lock:
        for w in _global:
            _global[w] = clean.get("workloads", {}).get(w, "auto")
        _cameras.clear()
        _cameras.update(clean.get("cameras", {}))
        _benchmarks.clear()
        _benchmarks.update({w: dict(v) for w, v in clean.get("benchmarks", {}).items()})
        _failures.clear()
    _changed()
