"""Thin wrapper around the optional ``onvif-zeep`` package (``import onvif``).

All methods are blocking and are only ever called from the per-camera PTZ
worker thread (see ``controller.py``), never from Flask request threads.
"""
from __future__ import annotations

from typing import Any

INSTALL_HINT = "pip install onvif-zeep"

FOV_SPACE_SUFFIX = "TranslationSpaceFov"


def onvif_available() -> bool:
    try:
        import onvif  # noqa: F401
    except Exception:
        return False
    return True


class PtzUnsupported(Exception):
    """The device answers ONVIF but has no usable PTZ profile."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _space_list(spaces: Any, name: str) -> list:
    value = _get(spaces, name)
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


class OnvifBackend:
    def __init__(self, host: str, port: int, user: str, password: str, timeout: float = 5.0):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.timeout = float(timeout)
        self.ptz = None
        self.token = None
        self.fov_space: str | None = None

    def _camera(self):
        from onvif import ONVIFCamera

        transport = None
        try:
            from zeep.transports import Transport

            transport = Transport(timeout=self.timeout, operation_timeout=self.timeout)
        except Exception:
            transport = None
        if transport is not None:
            try:
                return ONVIFCamera(self.host, self.port, self.user, self.password, transport=transport)
            except TypeError:
                pass  # old onvif-zeep without the transport argument
        return ONVIFCamera(self.host, self.port, self.user, self.password)

    def connect(self) -> dict[str, Any]:
        cam = self._camera()
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()
        profiles = list(media.GetProfiles() or [])
        profile = next((p for p in profiles if _get(p, "PTZConfiguration") is not None), None)
        if profile is None:
            raise PtzUnsupported("no PTZ profile")
        self.ptz = ptz
        self.token = _get(profile, "token")

        caps = {"pan_tilt": True, "zoom": False, "relative": True, "relative_fov": False, "presets": False, "home": False}
        try:
            cfg_token = _get(_get(profile, "PTZConfiguration"), "token")
            opts = ptz.GetConfigurationOptions({"ConfigurationToken": cfg_token})
            spaces = _get(opts, "Spaces")
            if spaces is not None:
                caps["pan_tilt"] = bool(_space_list(spaces, "ContinuousPanTiltVelocitySpace"))
                caps["zoom"] = bool(_space_list(spaces, "ContinuousZoomVelocitySpace"))
                rel = _space_list(spaces, "RelativePanTiltTranslationSpace")
                caps["relative"] = bool(rel)
                for space in rel:
                    uri = str(_get(space, "URI", "") or "")
                    if uri.endswith(FOV_SPACE_SUFFIX):
                        caps["relative_fov"] = True
                        self.fov_space = uri
        except Exception as exc:
            if _is_auth_text(str(exc)):
                raise
            # Some firmwares (Tapo included) answer this sparsely; keep the optimistic defaults.
        try:
            ptz.GetPresets({"ProfileToken": self.token})
            caps["presets"] = True
        except Exception as exc:
            if _is_auth_text(str(exc)):
                raise
        caps["home"] = hasattr(ptz, "GotoHomePosition")
        return caps

    # ------------------------------------------------------------------ commands
    def continuous_move(self, x: float, y: float, zoom: float) -> None:
        velocity: dict[str, Any] = {}
        if x or y:
            velocity["PanTilt"] = {"x": float(x), "y": float(y)}
        if zoom:
            velocity["Zoom"] = {"x": float(zoom)}
        if not velocity:
            self.stop()
            return
        self.ptz.ContinuousMove({"ProfileToken": self.token, "Velocity": velocity})

    def stop(self) -> None:
        self.ptz.Stop({"ProfileToken": self.token, "PanTilt": True, "Zoom": True})

    def relative_move(self, x: float, y: float, use_fov_space: bool = False) -> None:
        pan_tilt: dict[str, Any] = {"x": float(x), "y": float(y)}
        if use_fov_space and self.fov_space:
            pan_tilt["space"] = self.fov_space
        self.ptz.RelativeMove({"ProfileToken": self.token, "Translation": {"PanTilt": pan_tilt}})

    def get_presets(self) -> list[dict[str, str]]:
        out = []
        for p in self.ptz.GetPresets({"ProfileToken": self.token}) or []:
            token = _get(p, "token")
            if token is None:
                continue
            out.append({"token": str(token), "name": str(_get(p, "Name") or token)})
        return out

    def goto_preset(self, token: str) -> None:
        self.ptz.GotoPreset({"ProfileToken": self.token, "PresetToken": token})

    def set_preset(self, name: str) -> str | None:
        result = self.ptz.SetPreset({"ProfileToken": self.token, "PresetName": name})
        return None if result is None else str(_get(result, "PresetToken", result))

    def remove_preset(self, token: str) -> None:
        self.ptz.RemovePreset({"ProfileToken": self.token, "PresetToken": token})

    def goto_home(self) -> None:
        self.ptz.GotoHomePosition({"ProfileToken": self.token})


AUTH_MARKERS = (
    "notauthorized", "not authorized", "unauthorized", "401", "authority failure",
    "sender not authorized", "authentication", "invalid username", "password",
    "wsse:failedauthentication", "failedauthentication",
)


def _is_auth_text(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in AUTH_MARKERS)


def is_auth_error(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    code = getattr(exc, "code", None)
    parts = [str(exc), str(code or ""), str(getattr(exc, "message", "") or "")]
    subcodes = getattr(exc, "subcodes", None)
    if subcodes:
        parts.extend(str(s) for s in subcodes)
    return _is_auth_text(" ".join(parts))
