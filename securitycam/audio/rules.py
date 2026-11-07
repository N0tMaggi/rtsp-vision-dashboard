"""Sound alarm rules: per curated class (enabled, confidence threshold, cooldown,
notify) plus the "loud" rule (dB above the noise floor for a minimum time).

Everything is opt-in: all rules start disabled. Rules are global (all cameras)
and persisted as the ``sound`` section of ``data/state.json``. Events go
through the regular alarm/event store with kinds ``sound:<class>`` /
``sound:loud``.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .classes import BY_ID, IDS

DEFAULT_THRESHOLD = 0.5
DEFAULT_COOLDOWN_S = 30.0
COOLDOWN_RANGE = (0.0, 3600.0)
THRESHOLD_RANGE = (0.05, 0.99)
LOUD_DB_RANGE = (6.0, 50.0)
LOUD_MS_RANGE = (100.0, 5000.0)
DISPLAY_RANGE = (0.05, 0.9)


def default_config() -> Dict[str, Any]:
    return {
        "classes": {cid: {"enabled": False, "threshold": DEFAULT_THRESHOLD,
                          "cooldown_s": DEFAULT_COOLDOWN_S, "notify": True} for cid in IDS},
        "loud": {"enabled": False, "db_above_floor": 20.0, "min_ms": 300.0,
                 "cooldown_s": DEFAULT_COOLDOWN_S, "notify": True},
        "display_threshold": 0.2,       # chips on the HUD from this confidence on
    }


class RuleError(ValueError):
    def __init__(self, code: str, detail: Optional[str] = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


def _num(v: Any, lo: float, hi: float, field: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise RuleError("invalid_value", field)
    return max(lo, min(hi, float(v)))


def _bool(v: Any, field: str) -> bool:
    if not isinstance(v, bool):
        raise RuleError("invalid_value", field)
    return v


def validate_patch(body: Any) -> Dict[str, Any]:
    """Normalize a PATCH body (partial). Raises RuleError on anything unknown or invalid."""
    if not isinstance(body, dict) or not body:
        raise RuleError("invalid_body")
    out: Dict[str, Any] = {}
    for key, val in body.items():
        if key == "classes":
            if not isinstance(val, dict):
                raise RuleError("invalid_body", "classes")
            out["classes"] = {}
            for cid, rule in val.items():
                if cid not in BY_ID:
                    raise RuleError("unknown_class", str(cid)[:40])
                if not isinstance(rule, dict):
                    raise RuleError("invalid_body", cid)
                clean: Dict[str, Any] = {}
                for f, v in rule.items():
                    if f in ("enabled", "notify"):
                        clean[f] = _bool(v, f"{cid}.{f}")
                    elif f == "threshold":
                        clean[f] = round(_num(v, *THRESHOLD_RANGE, f"{cid}.{f}"), 3)
                    elif f == "cooldown_s":
                        clean[f] = round(_num(v, *COOLDOWN_RANGE, f"{cid}.{f}"), 1)
                    else:
                        raise RuleError("unknown_field", f"{cid}.{f}"[:60])
                out["classes"][cid] = clean
        elif key == "loud":
            if not isinstance(val, dict):
                raise RuleError("invalid_body", "loud")
            clean = {}
            for f, v in val.items():
                if f in ("enabled", "notify"):
                    clean[f] = _bool(v, f"loud.{f}")
                elif f == "db_above_floor":
                    clean[f] = round(_num(v, *LOUD_DB_RANGE, f"loud.{f}"), 1)
                elif f == "min_ms":
                    clean[f] = round(_num(v, *LOUD_MS_RANGE, f"loud.{f}"))
                elif f == "cooldown_s":
                    clean[f] = round(_num(v, *COOLDOWN_RANGE, f"loud.{f}"), 1)
                else:
                    raise RuleError("unknown_field", f"loud.{f}"[:60])
            out["loud"] = clean
        elif key == "display_threshold":
            out[key] = round(_num(val, *DISPLAY_RANGE, key), 3)
        else:
            raise RuleError("unknown_field", str(key)[:40])
    return out


class Rules:
    def __init__(self):
        self._lock = threading.Lock()
        self._cfg = default_config()
        self._last: Dict[Tuple[str, str], float] = {}     # (camera, rule id) -> monotonic time
        self.listeners: List[Callable[[], None]] = []

    # ------------------------------------------------------------------ config
    def config(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._cfg)

    def apply(self, clean: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for cid, rule in clean.get("classes", {}).items():
                self._cfg["classes"][cid].update(rule)
            if "loud" in clean:
                self._cfg["loud"].update(clean["loud"])
            if "display_threshold" in clean:
                self._cfg["display_threshold"] = clean["display_threshold"]
            cfg = copy.deepcopy(self._cfg)
        for fn in list(self.listeners):
            try:
                fn()
            except Exception:
                pass
        return cfg

    def reset(self) -> None:
        with self._lock:
            self._cfg = default_config()
            self._last.clear()

    def any_enabled(self) -> bool:
        with self._lock:
            return self._cfg["loud"]["enabled"] or any(r["enabled"] for r in self._cfg["classes"].values())

    # ------------------------------------------------------------------ persistence
    def dump(self) -> Dict[str, Any]:
        return self.config()

    @staticmethod
    def validate(raw: Any) -> Tuple[Dict[str, Any], List[str]]:
        if not isinstance(raw, dict):
            raise ValueError("sound section is not an object")
        warnings: List[str] = []
        clean: Dict[str, Any] = {"classes": {}}
        for cid, rule in (raw.get("classes") or {}).items() if isinstance(raw.get("classes"), dict) else []:
            try:
                clean["classes"].update(validate_patch({"classes": {cid: rule}})["classes"])
            except RuleError as exc:
                warnings.append(f"ignored sound rule {cid!r}: {exc.code}")
        for key in ("loud", "display_threshold"):
            if key in raw:
                try:
                    clean.update(validate_patch({key: raw[key]}))
                except RuleError as exc:
                    warnings.append(f"ignored sound {key}: {exc.code}")
        return clean, warnings

    def load(self, clean: Dict[str, Any]) -> None:
        with self._lock:
            self._cfg = default_config()
        self.apply(clean)

    # ------------------------------------------------------------------ evaluation
    def _cooldown_ok(self, camera_id: str, rid: str, cooldown: float, now: float) -> bool:
        key = (camera_id, rid)
        last = self._last.get(key)
        if last is not None and now - last < cooldown:
            return False
        self._last[key] = now
        return True

    def check_classes(self, camera_id: str, curated: Dict[str, float],
                      now: Optional[float] = None) -> List[Tuple[str, float, bool]]:
        """Rules that fire for one classification: [(class id, score, notify)]."""
        now = time.monotonic() if now is None else now
        out = []
        with self._lock:
            for cid, rule in self._cfg["classes"].items():
                score = curated.get(cid, 0.0)
                if not rule["enabled"] or score < rule["threshold"]:
                    continue
                if self._cooldown_ok(camera_id, cid, rule["cooldown_s"], now):
                    out.append((cid, score, bool(rule["notify"])))
        return out

    def check_loud(self, camera_id: str, now: Optional[float] = None) -> Optional[bool]:
        """None if the loud rule is off or cooling down, else its notify flag."""
        now = time.monotonic() if now is None else now
        with self._lock:
            rule = self._cfg["loud"]
            if not rule["enabled"]:
                return None
            if not self._cooldown_ok(camera_id, "loud", rule["cooldown_s"], now):
                return None
            return bool(rule["notify"])


RULES = Rules()


def emit(camera_state, rid: str, label: str, message: str, notify: bool, extra: Optional[dict] = None) -> bool:
    """Create an event through the regular alarm store (never blocks)."""
    from ..alerts.events import events

    trig = {
        "camera": camera_state.camera_id,
        "camera_label": getattr(camera_state, "label", camera_state.camera_id),
        "ts": time.time(),
        "kind": f"sound:{rid}",
        "source_id": rid,
        "source_label": label,
        "cls": "sound",
        "track_id": None,
        "message": message,
        "notify": bool(notify),
    }
    if extra:
        trig.update(extra)
    return events.trigger(trig)
