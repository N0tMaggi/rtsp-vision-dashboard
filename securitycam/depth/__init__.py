"""Monocular depth (Depth Anything V2 Small) + floor-plan positions for the room map.

Integration points (each one fail-safe call from a shared module):

    streaming.render_jpeg   -> ``on_render()`` (keeps / starts the camera's worker while
                                ``depth`` is on and the camera is viewed) and ``wants_view()``
    esp.pipeline            -> ``draw_view()`` (depth view blended under every ESP stroke)
    streaming.render_jpeg   -> ``draw_view()`` too when detection is off
    targets.targets_payload -> ``decorate_targets()`` (``depth_m``, ``ground``, ``depth`` status)
    webapp.create_app       -> ``api.blueprint`` (``/api/cameras/<id>/depth``)
    compute                 -> workload "depth" (assignments / benchmark / advice / api)

Everything is off by default (``depth`` setting); with it off none of the hooks
does more than one dict lookup.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

NOTE = "Distances are estimated from a single camera."
_warned: set = set()


def _warn_once(key: str, exc: BaseException) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[DEPTH] {key} failed: {exc}")


def _on(camera_state) -> bool:
    raw = getattr(camera_state, "esp_settings", None) or {}
    v = raw.get("depth", False)
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")


def on_render(camera_state) -> None:
    if not _on(camera_state):
        return
    try:
        from . import worker
        worker.ensure(camera_state)
    except Exception as exc:
        _warn_once("on_render", exc)


def wants_view(camera_state) -> bool:
    """True when the depth view will draw into this camera's frames."""
    if not _on(camera_state):
        return False
    try:
        raw = camera_state.esp_settings
        if raw.get("depth_view", False) not in (True, 1, "true", "on", "1"):
            return False
        from . import worker
        w = worker.get(getattr(camera_state, "camera_id", ""))
        return w is not None and w.latest() is not None
    except Exception as exc:
        _warn_once("wants_view", exc)
        return False


def draw_view(camera_state, frame) -> None:
    """Blend the colorized depth map into ``frame`` (in place) when the depth view is on."""
    try:
        if not wants_view(camera_state):
            return
        from . import render, worker
        s = worker.DepthSettings.of(camera_state)
        w = worker.get(camera_state.camera_id)
        df = w.latest() if w is not None else None
        if df is None or frame is None or getattr(frame, "ndim", 0) != 3:
            return
        h, wd = frame.shape[:2]
        img = w.view_cache.image(df.seq, df.disp, s.palette, wd, h)
        render.blend(frame, img, s.opacity)
    except Exception as exc:
        _warn_once("draw_view", exc)


# ----------------------------------------------------------------------------- status / payload
def _asset_state() -> str:
    from .. import config, deps
    from .model import ASSET, MODEL_FILE
    if (config.MODELS_DIR / MODEL_FILE).is_file():
        return "ready"
    return deps.asset_status.get(ASSET, "missing")


def status(camera_state, include_scan: bool = True) -> Dict[str, Any]:
    from . import model, worker
    from ..compute import devices
    s = worker.DepthSettings.of(camera_state)
    w = worker.get(getattr(camera_state, "camera_id", ""))
    out: Dict[str, Any] = {
        "enabled": s.enabled,
        "state": "off" if not s.enabled else "idle",
        "model": {"state": _asset_state(), "name": "Depth Anything V2 Small", "license": model.LICENSE},
        "device": None, "device_label": None, "source": None, "backend": None,
        "input": None, "latency_ms": None, "fps": None, "rate": s.rate,
        "calibration": {"mode": "user" if s.user_scale > 0 else "none", "scale": s.user_scale or None,
                        "samples": 0, "confidence": 0.0, "ready": s.user_scale > 0},
        "fov_h": None, "error": None, "note": NOTE,
    }
    if w is None:
        return out
    cal = w.calibration()
    be = w.backend
    out.update({
        "state": w.state if s.enabled else "off",
        "device": w.device,
        "device_label": devices.LABELS.get(w.device or "", w.device),
        "source": w.device_source,
        "backend": getattr(be, "label", None),
        "input": f"{be.shape[3]}x{be.shape[2]}" if be is not None else None,
        "latency_ms": round(w.latency_ms, 1) if w.latency_ms is not None else None,
        "fps": round(w.fps, 2) if w.fps is not None else None,
        "calibration": cal.as_dict(),
        "error": w.error,
    })
    df = w.latest()
    if df is not None:
        lens, pose, world = w.lens_pose(df.frame_w, df.frame_h, s)
        out["fov_h"] = round(lens.effective_fov(pose.zoom)[0], 2)
        if include_scan and df.scan:
            out["scan"] = {"cell": 0.3, "world": df.scan_world, "pts": df.scan}
    return out


def decorate_targets(camera_state, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``depth_m`` / ``ground`` (/ ``ground_world``) per target and ``depth`` status (depth on only)."""
    if not _on(camera_state):
        return payload
    try:
        from . import geometry, worker
        payload["depth"] = st = status(camera_state)
        w = worker.get(getattr(camera_state, "camera_id", ""))
        df = w.latest() if w is not None else None
        fw = df.frame_w if df is not None else int((payload.get("frame") or {}).get("w") or 0)
        fh = df.frame_h if df is not None else int((payload.get("frame") or {}).get("h") or 0)
        if w is None:
            lens_pose = None
        else:
            lens_pose = w.lens_pose(fw, fh)
        if lens_pose is None:
            from ..geo import Pose
            lens_pose = (geometry.lens_for_fov(worker.DepthSettings.of(camera_state).fov, fw, fh), Pose(), False)
        lens, pose, world = lens_pose
        if st.get("fov_h") is None:
            st["fov_h"] = round(lens.effective_fov(pose.zoom)[0], 2)
        cal = w.calibration() if w is not None else None
        t = time.perf_counter()
        for tg in payload.get("targets") or ():
            box = tg.get("box")
            if not box:
                continue
            z = w.target_depth(tg.get("id"), box, cal, t) if (w is not None and cal is not None) else None
            src = "depth"
            if z is not None:
                tg["depth_m"] = round(z, 2)
                if "distance" in tg:
                    tg["distance_box"] = tg["distance"]
                tg["distance"] = round(z, 2)
                tg["distance_src"] = "depth"
            else:
                z = tg.get("distance")
                src = "box"
            if z is None or not math.isfinite(float(z)):
                continue
            u = (box[0] + box[2]) * 0.5
            v = (box[1] + box[3]) * 0.5
            gx, gz = geometry.ground_point(u, v, float(z), lens, pose, world=False)
            tg["ground"] = [round(gx, 2), round(gz, 2)]
            tg["ground_src"] = src
            if world:
                wx, wz = geometry.ground_point(u, v, float(z), lens, pose, world=True)
                tg["ground_world"] = [round(wx, 2), round(wz, 2)]
    except Exception as exc:
        _warn_once("decorate_targets", exc)
    return payload


def stop_all() -> None:
    from . import worker
    worker.stop_all()
