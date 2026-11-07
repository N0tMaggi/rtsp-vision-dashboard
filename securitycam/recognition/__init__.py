"""Recognition package: re-identification, known faces, license plates, pets, model catalog, TensorRT.

Integration points (each a single call from the shared modules):
    streaming.yolo_worker  -> ``track_kwargs()`` for model.track(), ``after_inference()`` on every result
    esp.pipeline           -> ``overlay.prepare()`` + label/palette/face hooks
    esp.sensors            -> ``detect_face_rects()`` (YuNet instead of Haar)
    deps.get_yolo_model    -> ``engines.load_model()`` (TensorRT engine when built)
    webapp.create_app      -> ``api.blueprint``

Every feature degrades gracefully: missing models are downloaded in the
background and the feature simply stays inactive until they are ready.
Nothing is sent anywhere; the face gallery lives in data/faces/.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .common import RecogSettings, is_pet, is_vehicle, runtime

__all__ = ["is_pet", "is_vehicle", "pets_ignore_alarms", "track_kwargs", "after_inference",
           "detect_face_rects", "track_identity"]

_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(msg)


def pets_ignore_alarms(camera_state) -> bool:
    """Whether alarms should skip pets for this camera (setting ``pets_ignore_alarms``)."""
    return RecogSettings.from_mapping(getattr(camera_state, "esp_settings", {}) or {}).pets_ignore_alarms


def track_kwargs(camera_state, model) -> Dict[str, Any]:
    """Extra kwargs for ``model.track()``: the tracker config (BoT-SORT + ReID when enabled).

    Resets the tracker when the config changes; a new tracker (config or model
    change) restarts its ids, so the re-entry memory and per-track cache are cleared.
    """
    from . import reid

    try:
        s = RecogSettings.from_mapping(camera_state.esp_settings)
        rt = runtime(camera_state)
        tracker = reid.tracker_for(s.tracking, s.reid, model)
        new_model = rt.model_ref is not model
        if new_model or rt.tracker_cfg != tracker:
            if not new_model:
                predictor = getattr(model, "predictor", None)
                if predictor is not None and hasattr(predictor, "trackers"):
                    predictor.trackers = []      # rebuilt from the new tracker yaml on the next call
            rt.model_ref = model
            rt.tracker_cfg = tracker
            # a new tracker restarts its ids: forget id-keyed state
            if rt.reentry is not None:
                rt.reentry.reset()
            with rt.lock:
                rt.track_info.clear()
                rt.track_epoch += 1          # track-scoped tags / solo must not jump to a reused id
        return {"tracker": tracker}
    except Exception as exc:
        _warn_once("track_kwargs", f"[RECOG] tracker selection failed, using defaults: {exc}")
        return {}


def after_inference(camera_state, frame, results, frame_ts: Optional[float] = None):
    """Post-process fresh results in the YOLO worker: stable ids, then hand off to the recognition worker."""
    from . import reid, worker

    try:
        s = RecogSettings.from_mapping(camera_state.esp_settings)
        if s.tracking and s.reid and results:
            rt = runtime(camera_state)
            if rt.reentry is None:
                rt.reentry = reid.ReentryMatcher()
            reid.remap_results(rt.reentry, frame, results)
        if s.any_worker_feature and frame is not None:
            worker.submit(camera_state, frame, results, frame_ts or 0.0)
    except Exception as exc:
        _warn_once("after_inference", f"[RECOG] post-processing failed (continuing without it): {exc}")
    return results


def detect_face_rects(camera_state, frame, scale: float) -> Optional[List[Tuple[int, int, int, int]]]:
    """YuNet face rects for the render path, or None to fall back to the Haar cascade."""
    from . import faces

    try:
        s = RecogSettings.from_mapping(camera_state.esp_settings)
        if s.face_detector != "yunet":
            return None
        rows = faces.detect(frame, scale)
        if rows is None:
            return None
        h, w = frame.shape[:2]
        return faces.to_rects(rows, w, h)
    except Exception as exc:
        _warn_once("yunet", f"[RECOG] YuNet failed, using Haar: {exc}")
        return None


def track_identity(camera_id: str, track_id: int) -> Dict[str, Any]:
    """Name / plate known for a track id (for other packages, e.g. alarms or events)."""
    from .common import peek_runtime

    rt = peek_runtime(camera_id)
    if rt is None:
        return {}
    ti = rt.track_info.get(int(track_id))
    if ti is None:
        return {}
    return {"name": ti.name, "person_id": ti.person_id, "plate": ti.plate_best}
