"""Model catalog and TensorRT engine export ("Optimize for GPU").

Exports run in one background thread at a time; the server never waits on
them. A finished ``models/<stem>.engine`` is picked up by ``load_model()``
the next time the worker (re)loads that model; cameras using the model are
told to reload as soon as the export finishes.
"""
from __future__ import annotations

import importlib.util
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config, gpu, state

TRT_INSTALL_CMD = "python -m pip install tensorrt onnx onnxslim"

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_engine_failed: Dict[str, str] = {}       # model name -> error of the last engine load


def model_info(name: str) -> Dict[str, Any]:
    stem = Path(name).stem
    if stem.startswith("rtdetr"):
        family = "RT-DETR"
    elif stem.startswith("yolo26"):
        family = "YOLO26"
    elif stem.startswith("yolo11"):
        family = "YOLO11"
    elif stem.startswith("yolov8"):
        family = "YOLOv8"
    else:
        family = "Other"
    pose = stem.endswith("-pose")
    size = stem.replace("-pose", "")[-1:] if family != "RT-DETR" else stem.split("-")[-1]
    return {
        "name": name,
        "family": family,
        "task": "pose" if pose else "detect",
        "size": size.upper(),
        "label": f"{family} {size.upper()}{' Pose' if pose else ''}",
        "tracking": True,           # Ultralytics tracks any detect/pose model, RT-DETR included
        "keypoints": pose,
    }


def engine_path(name: str) -> Path:
    return config.MODELS_DIR / f"{Path(name).stem}.engine"


def tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def gpu_available() -> bool:
    return bool(state.CUDA_AVAILABLE) and str(state.DEVICE).startswith("cuda")


def capabilities() -> Dict[str, Any]:
    trt = tensorrt_available()
    gpu = gpu_available()
    reason = None
    if not gpu:
        reason = "no_cuda"
    elif not trt:
        reason = "tensorrt_missing"
    return {
        "available": trt and gpu,
        "tensorrt_installed": trt,
        "cuda": gpu,
        "reason": reason,
        "install_cmd": None if trt else TRT_INSTALL_CMD,
    }


def catalog(active: Optional[set] = None) -> list:
    out = []
    for name in config.AVAILABLE_MODELS:
        info = model_info(name)
        eng = engine_path(name)
        with _jobs_lock:
            job = dict(_jobs.get(name) or {})
        info.update({
            "weights_downloaded": config.model_path(name).is_file(),
            "engine": eng.is_file(),
            "engine_size_mb": round(eng.stat().st_size / 1e6, 1) if eng.is_file() else None,
            "engine_error": _engine_failed.get(name),
            "job": _public_job(job) if job else None,
            "in_use": bool(active and name in active),
        })
        out.append(info)
    return out


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    if job.get("status") == "running":
        # Ultralytics' export has no progress callback: stage progress + a time-based estimate.
        elapsed = time.time() - job.get("started", time.time())
        out["progress"] = min(0.95, max(job.get("progress", 0.0), 0.15 + 0.8 * (1 - 2.718 ** (-elapsed / 240))))
        out["elapsed_s"] = round(elapsed, 1)
    return out


def job_status(name: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(name)
        return _public_job(dict(job)) if job else None


def any_running() -> bool:
    with _jobs_lock:
        return any(j.get("status") == "running" for j in _jobs.values())


class EngineError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def start_export(name: str, exporter=None) -> Dict[str, Any]:
    """Start a background TensorRT export for ``name``. ``exporter`` is injectable for tests."""
    if name not in config.AVAILABLE_MODELS:
        raise EngineError("invalid_model")
    if exporter is None:
        caps = capabilities()
        if not caps["available"]:
            raise EngineError(caps["reason"] or "unavailable")
    with _jobs_lock:
        if any(j.get("status") == "running" for j in _jobs.values()):
            raise EngineError("busy")
        job = _jobs[name] = {"model": name, "status": "running", "stage": "queued", "progress": 0.02,
                             "started": time.time(), "finished": None, "error": None}
    threading.Thread(target=_run_export, args=(name, exporter or _export_tensorrt), daemon=True,
                     name=f"TensorRTExport-{name}").start()
    return _public_job(dict(job))


def _set(name: str, **kw) -> None:
    with _jobs_lock:
        if name in _jobs:
            _jobs[name].update(kw)


def _export_tensorrt(name: str) -> Path:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    src = config.model_path(name)
    if not src.is_file():
        _set(name, stage="downloading", progress=0.05)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(src)
    _set(name, stage="loading", progress=0.1)
    model = YOLO(str(src))
    _set(name, stage="exporting", progress=0.15)
    # The TensorRT build captures CUDA streams heavily: camera inference pauses meanwhile
    # (securitycam/gpu.py) instead of crashing with cudaErrorStreamCaptureUnsupported.
    with gpu.LOCK:
        out = model.export(format="engine", half=True, device=0, imgsz=640, verbose=False)
    return Path(out)


def _run_export(name: str, exporter) -> None:
    try:
        out = Path(exporter(name))
        target = engine_path(name)
        if out.resolve() != target.resolve() and out.is_file():
            out.replace(target)
        if not target.is_file():
            raise RuntimeError("export finished without an .engine file")
        _engine_failed.pop(name, None)
        _set(name, status="done", stage="done", progress=1.0, finished=time.time())
        print(f"[TensorRT] {target.name} ready.")
        reload_cameras(name)
    except Exception as exc:
        traceback.print_exc()
        _set(name, status="failed", stage="failed", error=str(exc)[:400], finished=time.time())


def delete_engine(name: str) -> bool:
    if name not in config.AVAILABLE_MODELS:
        raise EngineError("invalid_model")
    with _jobs_lock:
        if (_jobs.get(name) or {}).get("status") == "running":
            raise EngineError("busy")
        _jobs.pop(name, None)
    p = engine_path(name)
    existed = p.is_file()
    p.unlink(missing_ok=True)
    _engine_failed.pop(name, None)
    if existed:
        reload_cameras(name)
    return existed


def reload_cameras(name: str) -> None:
    for cam in state.list_camera_states():
        if cam.yolo_model_name == name:
            cam.yolo_model = None


def load_model(name: str, default_source: str):
    """YOLO model for ``name``: the TensorRT engine when present and usable, else the weights."""
    eng = engine_path(name)
    if eng.is_file() and gpu_available() and tensorrt_available() and name not in _engine_failed:
        try:
            model = state.YOLO(str(eng), task=model_info(name)["task"])
            print(f"[TensorRT] Using engine {eng.name} for {name}.")
            return model
        except Exception as exc:
            _engine_failed[name] = str(exc)[:300]
            print(f"[TensorRT] Engine {eng.name} could not be loaded ({exc}); using {default_source}.")
    return state.YOLO(default_source)


def uses_engine(model) -> bool:
    return str(getattr(model, "ckpt_path", None) or getattr(model, "model_name", "") or "").endswith(".engine") or \
        str(getattr(model, "model", "")).endswith(".engine")
