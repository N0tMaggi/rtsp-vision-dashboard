"""HTTP API: ``/api/cameras/<id>/depth`` (status) and ``.../depth/calibrate`` (user scale).

POST ``/calibrate`` body (JSON):
  ``{"target_id": 12, "distance_m": 4.5}``  -> "this target is 4.5 m away"
  ``{"reset": true}``                        -> back to automatic calibration
The result is stored in the ``depth_scale`` setting (0 = auto), so it is saved
with the camera's other settings.
"""
from __future__ import annotations

import math

from flask import Blueprint, jsonify, request

from .. import state

blueprint = Blueprint("depth", __name__)


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


@blueprint.get("/api/cameras/<camera_id>/depth")
def depth_status(camera_id: str):
    from . import status
    cs = state.get_camera_state(camera_id)
    if cs is None:
        return _err("unknown_camera", 404)
    resp = jsonify({"ok": True, "depth": status(cs, include_scan=False)})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.post("/api/cameras/<camera_id>/depth/calibrate")
def depth_calibrate(camera_id: str):
    from . import calib, geometry, status, worker
    cs = state.get_camera_state(camera_id)
    if cs is None:
        return _err("unknown_camera", 404)
    if not request.is_json:
        return _err("unsupported_media_type", 415)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("invalid_body")
    if body.get("reset") is True:
        cs.esp_settings["depth_scale"] = 0.0
        w = worker.get(camera_id)
        if w is not None:
            w.calibrator.reset()
        _saved()
        return jsonify({"ok": True, "depth": status(cs, include_scan=False)})
    tid = body.get("target_id")
    dist = body.get("distance_m")
    if isinstance(tid, bool) or not isinstance(tid, int):
        return _err("invalid_target")
    if isinstance(dist, bool) or not isinstance(dist, (int, float)) or not math.isfinite(dist) \
            or not 0.3 <= float(dist) <= 200.0:
        return _err("invalid_distance")
    w = worker.get(camera_id)
    df = w.latest() if w is not None else None
    if df is None:
        return _err("depth_not_running", 409)
    from .. import targets
    tg = next((t for t in targets.targets_payload(cs).get("targets", []) if t.get("id") == tid), None)
    if tg is None:
        return _err("unknown_target", 404)
    scale = calib.user_scale_for(float(dist), geometry.region_median(df.disp, tg["box"]))
    if scale is None:
        return _err("no_depth_at_target", 409)
    cs.esp_settings["depth_scale"] = round(scale, 5)
    _saved()
    return jsonify({"ok": True, "depth": status(cs, include_scan=False)})


def _saved() -> None:
    try:
        from .. import live, persistence
        persistence.mark_dirty()
        live.notify_change()
    except Exception:
        pass
