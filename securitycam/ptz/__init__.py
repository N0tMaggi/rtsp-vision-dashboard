"""Optional real PTZ via ONVIF (``pip install onvif-zeep``).

Configuration (.env):

* ``SECURITYCAM_PTZ`` - master switch (default true; still needs onvif-zeep).
* ``SECURITYCAM_ONVIF_PORT`` - ONVIF port for all cameras (default 2020, Tapo).
* ``SECURITYCAM_ONVIF_PORTS`` - per-camera override, ``camera-1:2020,garage:8000``.
* ``SECURITYCAM_ONVIF_TIMEOUT`` - seconds per ONVIF call (default 5).
* ``SECURITYCAM_PTZ_FOV_H`` / ``SECURITYCAM_PTZ_FOV_V`` - field of view in
  degrees used for right-click centering (default 100 x 56).
* ``SECURITYCAM_PTZ_PAN_RANGE`` / ``SECURITYCAM_PTZ_TILT_RANGE`` - mechanical
  range in degrees that ONVIF's generic -1..1 translation maps to (360 / 114).
* ``SECURITYCAM_PTZ_ZOOM_MAX`` - optical zoom factor at ONVIF zoom 1.0 (default 1:
  no optical zoom). Only used to read the absolute zoom position.

World-anchored overlays (zones, tripwires, heatmap ... stay put in the room
while the camera moves) use the pose estimate in ``pose.py``.

The ONVIF login reuses the camera account of the RTSP stream. FOV and speed
can be calibrated per camera in the UI; those values are persisted.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from .. import config, persistence
from . import autotrack, pose
from .controller import PtzBusy, PtzController, PtzNotReady, PtzUnsupportedOp  # noqa: F401
from .onvif_backend import INSTALL_HINT, OnvifBackend, onvif_available

ENABLED = config._env_bool("SECURITYCAM_PTZ", True)
ONVIF_PORT = config._env_int("SECURITYCAM_ONVIF_PORT", 2020, minimum=1, maximum=65535)
ONVIF_TIMEOUT = config._env_float("SECURITYCAM_ONVIF_TIMEOUT", 5.0, minimum=1.0, maximum=30.0)

SETTING_LIMITS: dict[str, tuple[float, float]] = {
    "fov_h": (10.0, 180.0),
    "fov_v": (5.0, 120.0),
    "pan_range": (10.0, 720.0),
    "tilt_range": (5.0, 360.0),
    "speed": (0.05, 1.0),
    "zoom_max": (1.0, 40.0),
}
DEFAULT_SETTINGS = {
    "fov_h": config._env_float("SECURITYCAM_PTZ_FOV_H", 100.0, *SETTING_LIMITS["fov_h"]),
    "fov_v": config._env_float("SECURITYCAM_PTZ_FOV_V", 56.0, *SETTING_LIMITS["fov_v"]),
    "pan_range": config._env_float("SECURITYCAM_PTZ_PAN_RANGE", 360.0, *SETTING_LIMITS["pan_range"]),
    "tilt_range": config._env_float("SECURITYCAM_PTZ_TILT_RANGE", 114.0, *SETTING_LIMITS["tilt_range"]),
    "speed": 0.5,
    "zoom_max": config._env_float("SECURITYCAM_PTZ_ZOOM_MAX", 1.0, *SETTING_LIMITS["zoom_max"]),
}


def _parse_ports(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in raw.split(","):
        if ":" not in chunk:
            continue
        cam, port = chunk.rsplit(":", 1)
        try:
            p = int(port.strip())
        except ValueError:
            continue
        if 1 <= p <= 65535 and cam.strip():
            out[cam.strip()] = p
    return out


ONVIF_PORTS = _parse_ports(os.getenv("SECURITYCAM_ONVIF_PORTS", ""))

_lock = threading.Lock()
_controllers: dict[str, PtzController] = {}
_settings: dict[str, dict[str, float]] = {}
# Tests replace this: (camera_config) -> backend object.
backend_factory = None
# Tests may force availability.
available_override: bool | None = None


def available() -> bool:
    if available_override is not None:
        return available_override
    return ENABLED and onvif_available()


def port_for(camera_id: str) -> int:
    return ONVIF_PORTS.get(camera_id, ONVIF_PORT)


def settings_for(camera_id: str) -> dict[str, float]:
    with _lock:
        s = _settings.get(camera_id)
        if s is None:
            s = _settings[camera_id] = dict(DEFAULT_SETTINGS)
        return s


def clean_settings(data: Any) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("invalid_settings")
    out: dict[str, float] = {}
    for key, (lo, hi) in SETTING_LIMITS.items():
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError("invalid_settings")
        out[key] = round(max(lo, min(hi, float(value))), 3)
    return out


def update_settings(camera_id: str, data: Any) -> dict[str, float]:
    patch = clean_settings(data)
    if not patch:
        raise ValueError("invalid_settings")
    s = settings_for(camera_id)
    with _lock:
        s.update(patch)
    persistence.mark_dirty()
    return dict(s)


def _rtsp_guard(camera_id: str):
    """Don't log in over ONVIF while the RTSP login is backing off (same account)."""
    def guard() -> str | None:
        try:
            from .. import streaming
            import time as _t

            failures, retry_after = streaming._connect_failures.get(camera_id, (0, 0.0))
            if failures and retry_after > _t.monotonic():
                return "The RTSP login of this camera is failing right now; PTZ waits so the account is not locked."
        except Exception:
            return None
        return None
    return guard


def _default_factory(camera):
    def make():
        return OnvifBackend(camera.ip, port_for(camera.id), camera.username, camera.password, ONVIF_TIMEOUT)
    return make


def get_controller(camera_id: str) -> PtzController | None:
    camera = config.get_camera(camera_id)
    if camera is None:
        return None
    with _lock:
        ctl = _controllers.get(camera_id)
    if ctl is not None:
        return ctl
    factory = backend_factory(camera) if backend_factory is not None else _default_factory(camera)
    ctl = PtzController(camera_id, factory, settings_for(camera_id), connect_guard=_rtsp_guard(camera_id))
    pose.attach(ctl)
    with _lock:
        return _controllers.setdefault(camera_id, ctl)


def peek_controller(camera_id: str) -> PtzController | None:
    with _lock:
        return _controllers.get(camera_id)


def pose_status(camera_id: str) -> dict[str, Any]:
    """World-anchoring state for the API (cheap, never logs in)."""
    from .. import state as _state

    cs = _state.get_camera_state(camera_id)
    f = pose.peek(camera_id)
    out: dict[str, Any] = {
        "world_anchor": pose.anchor_enabled(cs) if cs is not None else True,
        "active": bool(cs is not None and pose.active(cs)),
    }
    if f is None:
        out.update({"ptz_known": False, "pose": {"pan": 0.0, "tilt": 0.0, "zoom": 1.0}, "source": "none",
                    "vision": "off", "moving": False, "drift_deg": 0.0, "drift_kind": "estimated"})
        return out
    out.update(f.status())
    ctl = peek_controller(camera_id)
    if ctl is not None:
        out["onvif"]["supported"] = ctl.position_supported
    return out


def configured(camera_id: str) -> bool:
    camera = config.get_camera(camera_id)
    return bool(camera and camera.ip and camera.username and camera.password)


def status(camera_id: str) -> dict[str, Any]:
    base = {
        "available": available(),
        "enabled": ENABLED,
        "install_hint": INSTALL_HINT,
        "configured": configured(camera_id),
        "port": port_for(camera_id),
        "settings": dict(settings_for(camera_id)),
        "autotrack": autotrack.payload(camera_id),
    }
    if not base["available"] or not base["configured"]:
        base.update({"state": "unavailable", "error": "", "error_kind": "", "retry_in": 0, "caps": {}, "presets": [],
                     "busy": False, "moving": False})
        return base
    ctl = get_controller(camera_id)
    base.update(ctl.status() if ctl else {"state": "unavailable"})
    base["pose"] = pose_status(camera_id)
    return base


def reset() -> None:
    """Drop all controllers (tests)."""
    with _lock:
        for ctl in _controllers.values():
            ctl.close()
        _controllers.clear()
        _settings.clear()
    pose.reset()
    autotrack.reset()


# ---------------------------------------------------------------------------- persistence section
def _dump() -> dict[str, Any]:
    with _lock:
        out: dict[str, Any] = {"settings": {cam: dict(s) for cam, s in _settings.items() if s != DEFAULT_SETTINGS}}
    poses = {cam: f.dump() for cam, f in list(pose._fusions.items()) if f.ptz_known}
    if poses:
        out["pose"] = poses
    tracks = autotrack.dump()
    if tracks:
        out["autotrack"] = tracks
    return out


def _validate(raw: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        raise ValueError("ptz must be an object")
    out: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    for cam, s in (raw.get("settings") or {}).items():
        try:
            out[str(cam)] = clean_settings(s)
        except ValueError:
            warnings.append(f"PTZ settings for '{cam}' dropped.")
    clean: dict[str, Any] = {"settings": out}
    poses = pose.clean_dump(raw.get("pose"))
    if poses:
        clean["pose"] = poses
    tracks, track_warnings = autotrack.clean_dump(raw.get("autotrack"))
    warnings.extend(track_warnings)
    if tracks:
        clean["autotrack"] = tracks
    return clean, warnings


def _load(clean: dict[str, Any]) -> None:
    for cam, s in clean.get("settings", {}).items():
        merged = dict(DEFAULT_SETTINGS)
        merged.update(s)
        with _lock:
            current = _settings.get(cam)
            if current is None:
                _settings[cam] = merged
            else:
                current.update(merged)   # controllers hold a reference to this dict
    for cam, data in clean.get("pose", {}).items():
        pose.get(cam).restore(data)
    autotrack.load(clean.get("autotrack", {}))


def register() -> None:
    persistence.register_section("ptz", _dump, _load, _validate)
