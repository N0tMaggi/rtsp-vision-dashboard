"""Compute devices: choose which hardware runs which AI workload (opt-in).

Default for every workload is ``"auto"`` = SecurityCam's normal behaviour; with
nothing configured none of the hooks below changes anything. Integration
points (each one call from the shared modules, all of them fail-safe):

    streaming.yolo_worker   -> ``yolo_device()`` for predict/track, ``on_yolo_error()``
    deps.get_yolo_model     -> ``load_yolo()`` (prepared OpenVINO / CPU model or None)
    recognition.faces       -> ``faces_detect()`` / ``face_feature()``
    recognition.plates      -> ``plate_nets()`` / ``small_key()``
    webapp.create_app       -> ``api.blueprint`` (also registers the persistence section)
"""
from __future__ import annotations

import threading
from typing import Optional

from .. import state

_warned: set = set()
_reg_lock = threading.Lock()
_registered = False


def _warn_once(key: str, exc: BaseException) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[COMPUTE] {key} failed, using Auto: {exc}")


def yolo_device(camera_state, model) -> str:
    try:
        from . import yolo

        return yolo.device(camera_state, model)
    except Exception as exc:
        _warn_once("yolo_device", exc)
        return state.DEVICE


def load_yolo(camera_state, name: str):
    try:
        from . import assignments, yolo

        if assignments.is_default() and not yolo._active:
            return None
        return yolo.load(camera_state, name)
    except Exception as exc:
        _warn_once("load_yolo", exc)
        return None


def on_yolo_error(camera_state, exc: BaseException) -> bool:
    try:
        from . import yolo

        return yolo.on_error(camera_state, exc)
    except Exception as e:
        _warn_once("on_yolo_error", e)
        return False


def small_key(workload: str) -> str:
    try:
        from . import assignments

        if assignments.assignment(workload) == "auto":
            return "auto"
        from . import small

        return small.key(workload)
    except Exception as exc:
        _warn_once(f"small_key:{workload}", exc)
        return "auto"


def faces_detect(img, score_thr: float, nms_thr: float, top_k: int):
    try:
        from . import assignments

        if assignments.assignment("faces") == "auto":
            return None
        from . import small

        return small.yunet_detect(img, score_thr, nms_thr, top_k)
    except Exception as exc:
        _warn_once("faces_detect", exc)
        return None


def face_feature(aligned):
    try:
        from . import assignments

        if assignments.assignment("faces") == "auto":
            return None
        from . import small

        return small.sface_feature(aligned)
    except Exception as exc:
        _warn_once("face_feature", exc)
        return None


def plate_nets(det_path, ocr_path, det_size: int, ocr_shape) -> Optional[tuple]:
    try:
        from . import assignments

        if assignments.assignment("plates") == "auto":
            return None
        from . import small

        return small.plate_nets(det_path, ocr_path, det_size, ocr_shape)
    except Exception as exc:
        _warn_once("plate_nets", exc)
        return None


def register() -> None:
    """Persistence section + change hooks + background discovery (idempotent)."""
    global _registered
    with _reg_lock:
        if _registered:
            return
        _registered = True
    from .. import persistence
    from . import assignments, devices, yolo

    assignments.on_change(yolo.reconcile)
    devices.on_ready(yolo.reconcile)
    persistence.register_section("compute", assignments.dump, assignments.load, assignments.validate)
    devices.start()
