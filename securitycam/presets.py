"""ESP presets (named settings snapshots) and time-based schedules.

A preset is a *patch* of ESP settings (plus optionally the detection on/off
flag). "Save current" stores a full snapshot; the built-ins ("Day", "Night",
"Away", "Minimal") only touch the keys that matter for them, so custom
colors survive. A preset is either global (``camera_id`` None, usable on every
camera) or bound to one camera.

A schedule applies a preset at a local wall-clock time on selected weekdays
(0 = Monday). The scheduler thread calls :meth:`Scheduler.tick` with the
current local time; ``tick`` fires every schedule whose due time lies in
``(previous tick, now]``, so it never double-fires and never misses a minute
because of sleep jitter. Clock jumps backwards just reset the window.
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import time
from typing import Any, Callable, Iterable

from . import config, persistence, state
from .validation import sanitize_esp_patch

NAME_MAX = 32
MAX_PRESETS = 64
MAX_SCHEDULES = 32
TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
PRESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
SCHEDULE_ID_RE = re.compile(r"^s\d+$")
ALL_DAYS = [0, 1, 2, 3, 4, 5, 6]

BUILTIN_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "day",
        "name": "Day",
        "yolo_enabled": None,
        "settings": {
            "boxes": True, "box_style": "corner", "names": True, "conf": True, "skeleton": True,
            "thermal": False, "glow": True, "label_background": True, "confidence_threshold": 0.35,
            "person_only": False, "tracers": False, "prediction": False, "velocity": False,
        },
    },
    {
        "id": "night",
        "name": "Night",
        "yolo_enabled": True,
        "settings": {
            "boxes": True, "box_style": "full", "names": True, "conf": True, "glow": True,
            "label_background": True, "confidence_threshold": 0.25, "person_only": True,
            "tracking": True, "tracers": True,
        },
    },
    {
        "id": "away",
        "name": "Away",
        "yolo_enabled": True,
        "settings": {
            "boxes": True, "person_only": True, "tracking": True, "tracers": True, "prediction": True,
            "snaplines": True, "zones": True, "confidence_threshold": 0.3, "names": True,
        },
    },
    {
        "id": "minimal",
        "name": "Minimal",
        "yolo_enabled": None,
        "settings": {
            "boxes": True, "box_style": "corner", "line_thickness": 1, "names": False, "conf": False,
            "skeleton": False, "head": False, "glow": False, "label_background": False, "fill_box": False,
            "chams": False, "snaplines": False, "center_dot": False, "distance": False, "tracers": False,
            "prediction": False, "velocity": False,
        },
    },
)

BUILTIN_SCHEDULES: tuple[dict[str, Any], ...] = (
    {"preset_id": "night", "time": "22:00", "days": ALL_DAYS, "camera_id": None, "enabled": False},
    {"preset_id": "day", "time": "07:00", "days": ALL_DAYS, "camera_id": None, "enabled": False},
)


class PresetError(ValueError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def _clean_name(value: Any) -> str:
    if not isinstance(value, str):
        raise PresetError("invalid_name")
    name = " ".join(value.split())
    if not name or len(name) > NAME_MAX:
        raise PresetError("invalid_name")
    return name


def _clean_camera(value: Any, known: Iterable[str] | None) -> str | None:
    if value is None or value == "" or value == "all":
        return None
    if not isinstance(value, str):
        raise PresetError("invalid_camera")
    if known is not None and value not in known:
        raise PresetError("unknown_camera", 404)
    return value


def _clean_days(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PresetError("invalid_days")
    days = set()
    for d in value:
        if isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 6:
            raise PresetError("invalid_days")
        days.add(d)
    return sorted(days)


def _clean_time(value: Any) -> str:
    if not isinstance(value, str) or not TIME_RE.match(value.strip()):
        raise PresetError("invalid_time")
    return value.strip()


def _clean_yolo(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise PresetError("invalid_yolo")


class PresetStore:
    def __init__(self, cameras: Callable[[], dict[str, Any]] = lambda: state.camera_states,
                 on_change: Callable[[], None] = persistence.mark_dirty):
        self._cameras = cameras
        self._on_change = on_change
        self.lock = threading.RLock()
        self.presets: dict[str, dict[str, Any]] = {}
        self.schedules: dict[str, dict[str, Any]] = {}
        self.preset_seq = 0
        self.schedule_seq = 0
        # camera_id -> {"preset_id", "at", "source"} (runtime only)
        self.last_applied: dict[str, dict[str, Any]] = {}
        self.seed_builtins()

    # ------------------------------------------------------------------ helpers
    def _known(self) -> set[str]:
        return set(self._cameras())

    def _changed(self) -> None:
        try:
            self._on_change()
        except Exception:
            pass

    def seed_builtins(self) -> None:
        with self.lock:
            self.presets.clear()
            self.schedules.clear()
            self.preset_seq = 0
            self.schedule_seq = 0
            now = time.time()
            for b in BUILTIN_PRESETS:
                self.presets[b["id"]] = {
                    "id": b["id"], "name": b["name"], "camera_id": None, "builtin": True,
                    "settings": sanitize_esp_patch(b["settings"]), "yolo_enabled": b["yolo_enabled"],
                    "created_at": now, "updated_at": now,
                }
            for s in BUILTIN_SCHEDULES:
                self.schedule_seq += 1
                sid = f"s{self.schedule_seq}"
                self.schedules[sid] = {"id": sid, **{k: (list(v) if isinstance(v, list) else v) for k, v in s.items()}}

    @staticmethod
    def capture(camera_state) -> dict[str, Any]:
        return dict(camera_state.esp_settings)

    def _visible(self, preset: dict[str, Any], camera_id: str | None) -> bool:
        return camera_id is None or preset["camera_id"] in (None, camera_id)

    # ------------------------------------------------------------------ presets
    def list_presets(self, camera_id: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            items = [dict(p, settings=dict(p["settings"])) for p in self.presets.values() if self._visible(p, camera_id)]
        return items

    def get(self, preset_id: str) -> dict[str, Any]:
        with self.lock:
            p = self.presets.get(preset_id)
            if p is None:
                raise PresetError("unknown_preset", 404)
            return dict(p, settings=dict(p["settings"]))

    def create(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise PresetError("invalid_preset")
        known = self._known()
        name = _clean_name(data.get("name"))
        camera_id = _clean_camera(data.get("camera_id"), known)
        source = data.get("from_camera") or camera_id
        if "settings" in data:
            settings = sanitize_esp_patch(data.get("settings"))
            if not settings:
                raise PresetError("invalid_settings")
        else:
            cs = self._cameras().get(source) if isinstance(source, str) else None
            if cs is None:
                raise PresetError("unknown_camera", 404)
            settings = self.capture(cs)
        include_yolo = data.get("include_yolo", False)
        yolo = None
        if "yolo_enabled" in data:
            yolo = _clean_yolo(data.get("yolo_enabled"))
        elif include_yolo is True and isinstance(source, str) and source in self._cameras():
            yolo = bool(self._cameras()[source].yolo_enabled)
        with self.lock:
            if len(self.presets) >= MAX_PRESETS:
                raise PresetError("too_many_presets")
            self.preset_seq += 1
            while f"p{self.preset_seq}" in self.presets:
                self.preset_seq += 1
            pid = f"p{self.preset_seq}"
            now = time.time()
            self.presets[pid] = {
                "id": pid, "name": name, "camera_id": camera_id, "builtin": False,
                "settings": settings, "yolo_enabled": yolo, "created_at": now, "updated_at": now,
            }
        self._changed()
        return self.get(pid)

    def update(self, preset_id: str, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not data:
            raise PresetError("invalid_preset")
        known = self._known()
        changes: dict[str, Any] = {}
        if "name" in data:
            changes["name"] = _clean_name(data["name"])
        if "camera_id" in data:
            changes["camera_id"] = _clean_camera(data["camera_id"], known)
        if "yolo_enabled" in data:
            changes["yolo_enabled"] = _clean_yolo(data["yolo_enabled"])
        if "settings" in data:
            settings = sanitize_esp_patch(data["settings"])
            if not settings:
                raise PresetError("invalid_settings")
            changes["settings"] = settings
        if data.get("capture_from") is not None:
            cs = self._cameras().get(data["capture_from"])
            if cs is None:
                raise PresetError("unknown_camera", 404)
            changes["settings"] = self.capture(cs)
        if not changes:
            raise PresetError("invalid_preset")
        with self.lock:
            p = self.presets.get(preset_id)
            if p is None:
                raise PresetError("unknown_preset", 404)
            p.update(changes)
            p["updated_at"] = time.time()
        self._changed()
        return self.get(preset_id)

    def delete(self, preset_id: str) -> list[str]:
        """Delete a preset and the schedules that use it; returns removed schedule ids."""
        with self.lock:
            if self.presets.pop(preset_id, None) is None:
                raise PresetError("unknown_preset", 404)
            removed = [sid for sid, s in self.schedules.items() if s["preset_id"] == preset_id]
            for sid in removed:
                del self.schedules[sid]
            for cam, info in list(self.last_applied.items()):
                if info.get("preset_id") == preset_id:
                    del self.last_applied[cam]
        self._changed()
        return removed

    def targets_for(self, preset: dict[str, Any], camera_id: str | None) -> list[str]:
        known = list(self._cameras())
        if preset["camera_id"] is not None:
            if camera_id not in (None, preset["camera_id"]):
                raise PresetError("preset_not_for_camera")
            return [preset["camera_id"]] if preset["camera_id"] in known else []
        if camera_id is None:
            return known
        if camera_id not in known:
            raise PresetError("unknown_camera", 404)
        return [camera_id]

    def apply(self, preset_id: str, camera_id: str | None = None, source: str = "manual") -> list[str]:
        preset = self.get(preset_id)
        cams = self.targets_for(preset, camera_id)
        states = self._cameras()
        now = time.time()
        for cid in cams:
            cs = states[cid]
            cs.esp_settings.update(preset["settings"])
            if preset.get("yolo_enabled") is not None:
                cs.yolo_enabled = bool(preset["yolo_enabled"])
            with self.lock:
                self.last_applied[cid] = {"preset_id": preset_id, "at": now, "source": source}
        if cams:
            self._changed()
        return cams

    # ------------------------------------------------------------------ schedules
    def list_schedules(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(s, days=list(s["days"])) for s in self.schedules.values()]

    def _schedule_fields(self, data: dict[str, Any], partial: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "preset_id" in data or not partial:
            pid = data.get("preset_id")
            with self.lock:
                if not isinstance(pid, str) or pid not in self.presets:
                    raise PresetError("unknown_preset", 404)
            out["preset_id"] = pid
        if "time" in data or not partial:
            out["time"] = _clean_time(data.get("time"))
        if "days" in data or not partial:
            out["days"] = _clean_days(data.get("days", ALL_DAYS if not partial else None))
        if "camera_id" in data:
            out["camera_id"] = _clean_camera(data.get("camera_id"), self._known())
        elif not partial:
            out["camera_id"] = None
        if "enabled" in data:
            if not isinstance(data["enabled"], bool):
                raise PresetError("invalid_enabled")
            out["enabled"] = data["enabled"]
        elif not partial:
            out["enabled"] = True
        return out

    def _check_schedule_scope(self, sched: dict[str, Any]) -> None:
        preset = self.presets.get(sched["preset_id"])
        if preset and preset["camera_id"] is not None and sched["camera_id"] not in (None, preset["camera_id"]):
            raise PresetError("preset_not_for_camera")

    def create_schedule(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise PresetError("invalid_schedule")
        if "days" not in data:
            data = dict(data, days=ALL_DAYS)
        fields = self._schedule_fields(data, partial=False)
        with self.lock:
            if len(self.schedules) >= MAX_SCHEDULES:
                raise PresetError("too_many_schedules")
            self._check_schedule_scope(fields)
            self.schedule_seq += 1
            while f"s{self.schedule_seq}" in self.schedules:
                self.schedule_seq += 1
            sid = f"s{self.schedule_seq}"
            self.schedules[sid] = {"id": sid, **fields}
            out = dict(self.schedules[sid], days=list(fields["days"]))
        self._changed()
        return out

    def update_schedule(self, schedule_id: str, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not data:
            raise PresetError("invalid_schedule")
        fields = self._schedule_fields(data, partial=True)
        if not fields:
            raise PresetError("invalid_schedule")
        with self.lock:
            sched = self.schedules.get(schedule_id)
            if sched is None:
                raise PresetError("unknown_schedule", 404)
            candidate = dict(sched, **fields)
            self._check_schedule_scope(candidate)
            self.schedules[schedule_id] = candidate
            out = dict(candidate, days=list(candidate["days"]))
        self._changed()
        return out

    def delete_schedule(self, schedule_id: str) -> None:
        with self.lock:
            if self.schedules.pop(schedule_id, None) is None:
                raise PresetError("unknown_schedule", 404)
        self._changed()

    # ------------------------------------------------------------------ persistence section
    def dump(self) -> dict[str, Any]:
        with self.lock:
            return {
                "presets": [
                    {k: p[k] for k in ("id", "name", "camera_id", "builtin", "settings", "yolo_enabled", "created_at", "updated_at")}
                    for p in self.presets.values()
                ],
                "schedules": [dict(s, days=list(s["days"])) for s in self.schedules.values()],
            }

    def validate(self, raw: Any) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(raw, dict):
            raise ValueError("presets must be an object")
        warnings: list[str] = []
        presets: list[dict[str, Any]] = []
        ids: set[str] = set()
        for item in raw.get("presets") or []:
            if len(presets) >= MAX_PRESETS:
                warnings.append("Too many presets, rest dropped.")
                break
            try:
                if not isinstance(item, dict):
                    raise PresetError("invalid_preset")
                pid = item.get("id")
                if not isinstance(pid, str) or not PRESET_ID_RE.match(pid) or pid in ids:
                    raise PresetError("invalid_id")
                name = _clean_name(item.get("name"))
                camera_id = _clean_camera(item.get("camera_id"), None)
                settings = sanitize_esp_patch(item.get("settings"))
                if not settings:
                    raise PresetError("invalid_settings")
                yolo = _clean_yolo(item.get("yolo_enabled"))
            except PresetError as exc:
                warnings.append(f"Preset dropped ({exc.code}).")
                continue
            ts = item.get("created_at")
            ts = float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else time.time()
            up = item.get("updated_at")
            up = float(up) if isinstance(up, (int, float)) and not isinstance(up, bool) else ts
            presets.append({
                "id": pid, "name": name, "camera_id": camera_id, "builtin": bool(item.get("builtin", False)),
                "settings": settings, "yolo_enabled": yolo, "created_at": ts, "updated_at": up,
            })
            ids.add(pid)
        schedules: list[dict[str, Any]] = []
        sids: set[str] = set()
        for item in raw.get("schedules") or []:
            if len(schedules) >= MAX_SCHEDULES:
                warnings.append("Too many schedules, rest dropped.")
                break
            try:
                if not isinstance(item, dict):
                    raise PresetError("invalid_schedule")
                sid = item.get("id")
                if not isinstance(sid, str) or not SCHEDULE_ID_RE.match(sid) or sid in sids:
                    raise PresetError("invalid_id")
                if item.get("preset_id") not in ids:
                    raise PresetError("unknown_preset")
                enabled = item.get("enabled", True)
                if not isinstance(enabled, bool):
                    raise PresetError("invalid_enabled")
                schedules.append({
                    "id": sid,
                    "preset_id": item["preset_id"],
                    "time": _clean_time(item.get("time")),
                    "days": _clean_days(item.get("days")),
                    "camera_id": _clean_camera(item.get("camera_id"), None),
                    "enabled": enabled,
                })
                sids.add(sid)
            except PresetError as exc:
                warnings.append(f"Schedule dropped ({exc.code}).")
        return {"presets": presets, "schedules": schedules}, warnings

    def load(self, clean: dict[str, Any]) -> None:
        with self.lock:
            self.presets = {p["id"]: dict(p, settings=dict(p["settings"])) for p in clean.get("presets", [])}
            self.schedules = {s["id"]: dict(s, days=list(s["days"])) for s in clean.get("schedules", [])}
            self.preset_seq = max([int(i[1:]) for i in self.presets if re.match(r"^p\d+$", i)] or [0])
            self.schedule_seq = max([int(i[1:]) for i in self.schedules] or [0])
            self.last_applied = {}


def next_run(schedule: dict[str, Any], now: dt.datetime) -> dt.datetime | None:
    if not schedule.get("enabled"):
        return None
    hh, mm = (int(x) for x in schedule["time"].split(":"))
    for offset in range(0, 8):
        day = (now + dt.timedelta(days=offset)).date()
        if day.weekday() not in schedule["days"]:
            continue
        due = dt.datetime.combine(day, dt.time(hh, mm))
        if due > now:
            return due
    return None


class Scheduler:
    """Applies due schedules. ``now_fn`` returns naive local time (injectable for tests)."""

    def __init__(self, store: PresetStore, now_fn: Callable[[], dt.datetime] = dt.datetime.now, interval_s: float = 10.0):
        self.store = store
        self.now_fn = now_fn
        self.interval_s = interval_s
        self._last: dt.datetime | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log: list[dict[str, Any]] = []          # recent firings (newest last)

    def tick(self, now: dt.datetime | None = None) -> list[dict[str, Any]]:
        now = now or self.now_fn()
        last = self._last
        self._last = now
        if last is None or now <= last or now - last > dt.timedelta(days=1):
            # first tick, clock went backwards, or a huge gap (sleep/hibernate): just re-anchor.
            if last is not None and now - last > dt.timedelta(days=1):
                last = now - dt.timedelta(minutes=1)
            else:
                return []
        due_items: list[tuple[dt.datetime, dict[str, Any]]] = []
        for sched in self.store.list_schedules():
            if not sched.get("enabled"):
                continue
            hh, mm = (int(x) for x in sched["time"].split(":"))
            day = last.date()
            while day <= now.date():
                due = dt.datetime.combine(day, dt.time(hh, mm))
                if day.weekday() in sched["days"] and last < due <= now:
                    due_items.append((due, sched))
                day += dt.timedelta(days=1)
        fired = []
        for due, sched in sorted(due_items, key=lambda x: x[0]):
            try:
                cams = self.store.apply(sched["preset_id"], sched.get("camera_id"), source="schedule")
            except PresetError as exc:
                print(f"[Presets] Schedule {sched['id']} failed: {exc.code}")
                continue
            entry = {"schedule_id": sched["id"], "preset_id": sched["preset_id"], "due": due.isoformat(timespec="minutes"), "cameras": cams}
            print(f"[Presets] Schedule {sched['id']} applied preset '{sched['preset_id']}' to {', '.join(cams) or 'no camera'}.")
            fired.append(entry)
        if fired:
            self.log = (self.log + fired)[-20:]
        return fired

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # never let the scheduler die
                print(f"[Presets] Scheduler error: {exc}")
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="PresetScheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


STORE = PresetStore()
SCHEDULER = Scheduler(STORE)


def register() -> None:
    persistence.register_section("presets", STORE.dump, STORE.load, STORE.validate)


def payload(camera_id: str | None = None) -> dict[str, Any]:
    now = dt.datetime.now()
    schedules = []
    for s in STORE.list_schedules():
        nxt = next_run(s, now)
        schedules.append(dict(s, next_run=nxt.isoformat(timespec="minutes") if nxt else None))
    with STORE.lock:
        applied = {k: dict(v) for k, v in STORE.last_applied.items()}
    return {
        "presets": STORE.list_presets(camera_id),
        "schedules": schedules,
        "last_applied": applied,
        "now": now.isoformat(timespec="seconds"),
    }
