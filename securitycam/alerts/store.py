"""Zone rules and tripwires, persisted in ``data/alerts.json``.

Per camera::

    {"zone_rules": {"z1": {...rule...}}, "tripwires": [{...}], "tripwire_seq": 3,
     "abandoned_notify": true}

Readers (the per-frame engine) get the current per-camera dict via
``camera_config``. Those dicts are never mutated: every change builds a new
one and swaps it in, so no locking is needed on the render path.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
from typing import Any, Optional

from . import conf

LABEL_MAX = 32
MAX_TRIPWIRES = 16
MAX_CLASSES = 24
WIRE_MIN_LEN = 0.02
TRIGGERS = ("both", "in", "out")
DEFAULT_WIRE_COLOR = "#ffd21f"
_HHMM = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_CLASS = re.compile(r"^[a-z][a-z0-9 _-]{0,31}$")


class AlertInputError(ValueError):
    def __init__(self, code: str = "invalid_input"):
        super().__init__(code)
        self.code = code


# ----------------------------------------------------------------------------- validation
def _bool(data: dict, key: str, default: bool) -> bool:
    if key not in data or data[key] is None:
        return default
    v = data[key]
    if isinstance(v, bool):
        return v
    raise AlertInputError(f"invalid_{key}")


def _classes(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_CLASSES:
        raise AlertInputError("invalid_classes")
    out: list[str] = []
    for c in value:
        if not isinstance(c, str):
            raise AlertInputError("invalid_classes")
        c = " ".join(c.strip().lower().split())
        if not c:
            continue
        if not _CLASS.match(c):
            raise AlertInputError("invalid_classes")
        if c not in out:
            out.append(c)
    return out


def _label(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise AlertInputError("invalid_label")
    v = " ".join(value.split())
    if len(v) > LABEL_MAX:
        raise AlertInputError("invalid_label")
    return v or fallback


def _schedule(value: Any) -> dict:
    if value is None:
        return {"mode": "always"}
    if not isinstance(value, dict):
        raise AlertInputError("invalid_schedule")
    mode = value.get("mode", "always")
    if mode == "always":
        return {"mode": "always"}
    if mode != "window":
        raise AlertInputError("invalid_schedule")
    start, end = value.get("start"), value.get("end")
    if not isinstance(start, str) or not isinstance(end, str) or not _HHMM.match(start) or not _HHMM.match(end):
        raise AlertInputError("invalid_schedule")
    return {"mode": "window", "start": start, "end": end}


def _num(value: Any, lo: float, hi: float, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AlertInputError(code)
    return float(max(lo, min(hi, value)))


def parse_rule(data: Any, current: Optional[dict] = None) -> dict:
    """Validate a zone rule (full or partial update of ``current``)."""
    if not isinstance(data, dict):
        raise AlertInputError("invalid_rule")
    base = dict(current) if current else {
        "enabled": True, "classes": [], "min_dwell_s": 0.0, "schedule": {"mode": "always"}, "notify": True,
    }
    if "enabled" in data:
        base["enabled"] = _bool(data, "enabled", True)
    if "notify" in data:
        base["notify"] = _bool(data, "notify", True)
    if "classes" in data:
        base["classes"] = _classes(data["classes"])
    if "min_dwell_s" in data:
        base["min_dwell_s"] = round(_num(data["min_dwell_s"], 0.0, 3600.0, "invalid_min_dwell_s"), 2)
    if "schedule" in data:
        base["schedule"] = _schedule(data["schedule"])
    return base


def _point(value: Any) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise AlertInputError("invalid_line")
    return [round(min(1.0, max(0.0, float(v))), 5) for v in value]


def parse_tripwire(data: Any, current: Optional[dict] = None) -> dict:
    if not isinstance(data, dict):
        raise AlertInputError("invalid_tripwire")
    out = dict(current) if current else {}
    if current is None or "a" in data or "b" in data:
        a = _point(data.get("a", out.get("a")))
        b = _point(data.get("b", out.get("b")))
        if math.hypot(b[0] - a[0], b[1] - a[1]) < WIRE_MIN_LEN:
            raise AlertInputError("line_too_short")
        out["a"], out["b"] = a, b
    if "label" in data:
        out["label"] = _label(data["label"], out.get("label", ""))
    if "trigger" in data:
        if data["trigger"] not in TRIGGERS:
            raise AlertInputError("invalid_trigger")
        out["trigger"] = data["trigger"]
    if "alarm" in data:
        out["alarm"] = _bool(data, "alarm", False)
    if "notify" in data:
        out["notify"] = _bool(data, "notify", True)
    if "classes" in data:
        out["classes"] = _classes(data["classes"])
    if "color" in data and data["color"] is not None:
        if not isinstance(data["color"], str) or not _HEX.match(data["color"].strip()):
            raise AlertInputError("invalid_color")
        out["color"] = data["color"].strip().lower()
    if data.get("flip") is True:
        out["a"], out["b"] = out["b"], out["a"]
    out.setdefault("trigger", "both")
    out.setdefault("alarm", False)
    out.setdefault("notify", True)
    out.setdefault("classes", [])
    out.setdefault("color", DEFAULT_WIRE_COLOR)
    return out


# ----------------------------------------------------------------------------- store
def _empty_camera() -> dict:
    return {"zone_rules": {}, "tripwires": [], "tripwire_seq": 0, "abandoned_notify": True}


_EMPTY = _empty_camera()


class AlertStore:
    def __init__(self, path=None):
        self.path = path or conf.ALERTS_FILE
        self._lock = threading.Lock()
        self._cams: dict[str, dict] = {}
        self._loaded = False
        self.version = 0

    # -- persistence
    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print(f"[Alerts] Could not read {self.path.name}: {exc.__class__.__name__} - starting empty.")
            return
        for cid, cam in (raw.get("cameras") or {}).items():
            if not isinstance(cid, str) or not isinstance(cam, dict):
                continue
            c = _empty_camera()
            for zid, rule in (cam.get("zone_rules") or {}).items():
                try:
                    c["zone_rules"][str(zid)] = parse_rule(rule)
                except AlertInputError:
                    continue
            for w in cam.get("tripwires") or []:
                try:
                    wire = parse_tripwire(w)
                except AlertInputError:
                    continue
                if isinstance(w.get("id"), str):
                    wire["id"] = w["id"]
                    wire.setdefault("label", w["id"])
                    c["tripwires"].append(wire)
            seq = cam.get("tripwire_seq")
            c["tripwire_seq"] = seq if isinstance(seq, int) and seq >= 0 else len(c["tripwires"])
            c["abandoned_notify"] = bool(cam.get("abandoned_notify", True))
            self._cams[cid] = c

    def _save_locked(self) -> None:
        self.version += 1
        payload = {"version": 1, "cameras": self._cams}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            print(f"[Alerts] Could not save {self.path.name}: {exc.__class__.__name__}")

    def _cam_locked(self, camera_id: str) -> dict:
        self._load_locked()
        cam = self._cams.get(camera_id)
        return cam if cam is not None else _empty_camera()

    def _commit_locked(self, camera_id: str, cam: dict) -> None:
        self._cams[camera_id] = cam
        self._save_locked()

    # -- reads
    def camera_config(self, camera_id: str) -> dict:
        """Current (shared, read-only) config for ``camera_id``."""
        if not self._loaded:
            with self._lock:
                self._load_locked()
        return self._cams.get(camera_id) or _EMPTY

    def payload(self, camera_id: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._cam_locked(camera_id))

    # -- zone rules
    def set_rule(self, camera_id: str, zone_id: str, data: Any) -> dict:
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            rule = parse_rule(data, cam["zone_rules"].get(zone_id))
            cam["zone_rules"][zone_id] = rule
            self._commit_locked(camera_id, cam)
            return copy.deepcopy(rule)

    def delete_rule(self, camera_id: str, zone_id: str) -> bool:
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            if cam["zone_rules"].pop(zone_id, None) is None:
                return False
            self._commit_locked(camera_id, cam)
            return True

    # -- tripwires
    def add_tripwire(self, camera_id: str, data: Any) -> dict:
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            if len(cam["tripwires"]) >= MAX_TRIPWIRES:
                raise AlertInputError("too_many_tripwires")
            wire = parse_tripwire(data)
            cam["tripwire_seq"] += 1
            n = cam["tripwire_seq"]
            wire["id"] = f"t{n}"
            if not wire.get("label"):
                wire["label"] = f"Tripwire {n}"
            cam["tripwires"].append(wire)
            self._commit_locked(camera_id, cam)
            return copy.deepcopy(wire)

    def update_tripwire(self, camera_id: str, wire_id: str, data: Any) -> Optional[dict]:
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            for i, w in enumerate(cam["tripwires"]):
                if w["id"] == wire_id:
                    new = parse_tripwire(data, w)
                    if not new.get("label"):
                        new["label"] = f"Tripwire {wire_id.lstrip('t')}"
                    cam["tripwires"][i] = new
                    self._commit_locked(camera_id, cam)
                    return copy.deepcopy(new)
            return None

    def delete_tripwire(self, camera_id: str, wire_id: str) -> bool:
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            before = len(cam["tripwires"])
            cam["tripwires"] = [w for w in cam["tripwires"] if w["id"] != wire_id]
            if len(cam["tripwires"]) == before:
                return False
            self._commit_locked(camera_id, cam)
            return True

    def set_options(self, camera_id: str, data: Any) -> dict:
        if not isinstance(data, dict) or "abandoned_notify" not in data:
            raise AlertInputError("invalid_options")
        with self._lock:
            cam = copy.deepcopy(self._cam_locked(camera_id))
            cam["abandoned_notify"] = _bool(data, "abandoned_notify", True)
            self._commit_locked(camera_id, cam)
            return {"abandoned_notify": cam["abandoned_notify"]}


store = AlertStore()
