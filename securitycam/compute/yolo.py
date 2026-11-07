"""YOLO on a chosen device: background preparation, hand-over to the camera worker, fallback.

Flow for a camera whose YOLO device is not auto:

1. ``deps.get_yolo_model`` asks ``load()``; while nothing is prepared it gets
   ``None`` and loads the usual (auto) model, so detection never pauses.
2. A single background thread prepares the model: OpenVINO export (once per
   model), load, compile on the device and one warm-up inference.
3. When it is ready the camera's model is dropped; the next ``load()`` hands
   the prepared model to the worker and ``device()`` returns the matching
   Ultralytics device string for ``predict`` / ``track``.
4. Any preparation or inference error marks the device failed for YOLO, the
   camera reloads the auto model, and the error is shown in the Compute tab.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import weakref
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .. import config, state
from . import assignments, devices, ov_runtime

_lock = threading.RLock()
_active: Dict[str, Dict[str, Any]] = {}          # camera id -> {ref, device, ul_device, backend, name}
_prepared: Dict[Tuple[str, str, str], Any] = {}  # (camera id, model, device) -> ready YOLO model
_jobs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}   # (camera id, model, device) -> status
_queue: "queue.Queue[Tuple[str, str, str]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_busy = False


def _cam(camera_id: str):
    return state.get_camera_state(camera_id)


def _uses_openvino(dev: str) -> bool:
    return dev in ("npu", "igpu") or (dev == "cpu" and devices.openvino_installed())


def device(camera_state, model) -> str:
    """Device argument for ``model.predict`` / ``model.track``; ``state.DEVICE`` unless a prepared model runs."""
    ent = _active.get(camera_state.camera_id)
    if ent is not None and model is not None and ent["ref"]() is model:
        return ent["ul_device"]
    return state.DEVICE


def active(camera_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        ent = _active.get(camera_id)
        return {k: v for k, v in ent.items() if k != "ref"} if ent else None


def load(camera_state, name: str):
    """Prepared model for this camera's non-auto device, or None (caller loads the auto model)."""
    cid = camera_state.camera_id
    eff = assignments.effective("yolo", cid)
    dev = eff["device"]
    with _lock:
        _active.pop(cid, None)
        if dev in ("auto", "cuda"):
            # "NVIDIA GPU" is the regular CUDA / TensorRT path
            return None
        key = (cid, name, dev)
        model = _prepared.pop(key, None)
        if model is not None:
            _active[cid] = {"ref": weakref.ref(model), "device": dev, "name": name,
                            "ul_device": model._securitycam_device, "backend": model._securitycam_backend}
            return model
    _schedule(cid, name, dev)
    return None


def _schedule(cid: str, name: str, dev: str) -> None:
    global _worker
    key = (cid, name, dev)
    with _lock:
        job = _jobs.get(key)
        if job is not None and job["status"] in ("queued", "running"):
            return
        if job is not None and job["status"] == "ready" and key in _prepared:
            return
        _jobs[key] = {"camera": cid, "model": name, "device": dev, "status": "queued", "stage": "queued",
                      "error": None, "started": time.time(), "finished": None}
        _queue.put(key)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, daemon=True, name="ComputeYoloPrepare")
            _worker.start()


def _set(key, **kw) -> None:
    with _lock:
        if key in _jobs:
            _jobs[key].update(kw)


def _wanted(key) -> bool:
    cid, name, dev = key
    cs = _cam(cid)
    if cs is None or cs.closed or cs.yolo_model_name != name:
        return False
    return assignments.effective("yolo", cid)["device"] == dev


def _expected_tracker(cs, model) -> Optional[str]:
    try:
        from ..recognition import reid
        from ..recognition.common import RecogSettings

        s = RecogSettings.from_mapping(cs.esp_settings)
        return reid.tracker_for(s.tracking, s.reid, model) if s.tracking else None
    except Exception:
        return None


def prepare_model(name: str, dev: str, on_stage=None, camera_state=None):
    """Build a ready-to-run YOLO model for ``dev`` (blocking). Returns the model."""
    if state.YOLO is None:
        raise RuntimeError("Ultralytics is not installed")
    from ..recognition.engines import model_info

    task = model_info(name)["task"]
    if _uses_openvino(dev):
        ov_runtime.enable_compile_cache()
        src = ov_runtime.export_yolo(name, on_stage=on_stage)
        if on_stage:
            on_stage("compiling")
        model = state.YOLO(str(src), task=task)
        ul_device = ov_runtime.ultralytics_device(dev)
        backend = "openvino"
    else:  # "cpu" without OpenVINO: plain PyTorch on the CPU
        if on_stage:
            on_stage("loading")
        src = config.model_path(name)
        model = state.YOLO(str(src) if src.is_file() else name)
        ul_device = "cpu"
        backend = "pytorch"
    kw: Dict[str, Any] = {}
    if camera_state is not None:
        tracker = _expected_tracker(camera_state, model)
        if tracker:
            kw["tracker"] = tracker   # same predictor setup as the worker's track() -> no second compile
    dummy = np.full((config.YOLO_IMG_SIZE * 9 // 16, config.YOLO_IMG_SIZE, 3), 114, np.uint8)
    model.predict(dummy, device=ul_device, verbose=False, **kw)
    model._securitycam_device = ul_device
    model._securitycam_backend = backend
    return model


def _run() -> None:
    global _busy
    while True:
        key = _queue.get()
        _busy = True
        try:
            _prepare_one(key)
        finally:
            _busy = False


def idle(timeout: float = 5.0) -> bool:
    """True once no preparation is queued or running (for tests / shutdown)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        if _queue.empty() and not _busy:
            return True
        time.sleep(0.02)
    return False


def _prepare_one(key) -> None:
    cid, name, dev = key
    if not _wanted(key):
        _set(key, status="cancelled", stage="cancelled", finished=time.time())
        return
    _set(key, status="running", stage="preparing")
    try:
        cs = _cam(cid)
        model = prepare_model(name, dev, on_stage=lambda st: _set(key, stage=st), camera_state=cs)
    except Exception as exc:
        traceback.print_exc(limit=3)
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
        _set(key, status="failed", stage="failed", error=msg, finished=time.time())
        assignments.mark_failed("yolo", dev, msg, model=name)
        return
    with _lock:
        if not _wanted(key):
            _set(key, status="cancelled", stage="cancelled", finished=time.time())
            return
        _prepared[key] = model
        _set(key, status="ready", stage="ready", finished=time.time())
    print(f"[COMPUTE] {name} ready on {devices.LABELS.get(dev, dev)} for {cid}.")
    if cs is not None:
        cs.yolo_model = None          # worker reloads -> load() hands over the prepared model


def on_error(camera_state, exc: BaseException) -> bool:
    """Inference error in the worker: True if it came from a non-auto model (now falling back)."""
    cid = camera_state.camera_id
    model = camera_state.yolo_model
    with _lock:
        ent = _active.get(cid)
        if ent is None or model is None or ent["ref"]() is not model:
            return False
        _active.pop(cid, None)
    assignments.mark_failed("yolo", ent["device"], f"{type(exc).__name__}: {exc}", model=ent["name"])
    camera_state.yolo_model = None
    return True


def reconcile() -> None:
    """After an assignment change: reload cameras whose device changed, prepare new ones."""
    for cs in state.list_camera_states():
        cid = cs.camera_id
        dev = assignments.effective("yolo", cid)["device"]
        with _lock:
            ent = _active.get(cid)
            current = ent["device"] if ent else "auto"
            for k in [k for k in _prepared if k[0] == cid and (k[2] != dev or k[1] != cs.yolo_model_name)]:
                del _prepared[k]
        target = "auto" if dev in ("auto", "cuda") else dev
        if target == current:
            continue
        if target == "auto":
            with _lock:
                _active.pop(cid, None)
            cs.yolo_model = None
        elif cs.yolo_model is not None or cs.yolo_enabled:
            _schedule(cid, cs.yolo_model_name, target)


def job(camera_id: str, name: str, dev: str) -> Optional[Dict[str, Any]]:
    with _lock:
        j = _jobs.get((camera_id, name, dev))
        if not j:
            return None
        out = dict(j)
    if out["status"] in ("queued", "running"):
        out["elapsed_s"] = round(time.time() - out["started"], 1)
    return out


def camera_state_info(camera_id: str) -> Dict[str, Any]:
    """{state, backend, job} of YOLO on one camera for the API."""
    cs = _cam(camera_id)
    eff = assignments.effective("yolo", camera_id)
    dev = eff["device"]
    info: Dict[str, Any] = {"state": "auto", "backend": None, "job": None}
    if dev in ("auto", "cuda"):
        info["state"] = "active" if dev == "cuda" else "auto"
        return info
    ent = active(camera_id)
    if ent and ent["device"] == dev:
        info.update(state="active", backend=ent["backend"])
        return info
    j = job(camera_id, cs.yolo_model_name if cs else "", dev)
    info["job"] = j
    if j and j["status"] in ("queued", "running"):
        info["state"] = "preparing"
    elif j and j["status"] == "ready":
        info["state"] = "switching"
    else:
        info["state"] = "pending" if (cs and cs.yolo_enabled) else "idle"
    return info
