"""Persist user configuration to ``data/state.json`` and restore it on startup.

The file is a versioned document made of named *sections*. Every feature that
owns user configuration registers one section::

    persistence.register_section("presets", dump_fn, load_fn, validate_fn)

* ``dump_fn() -> JSON-able`` returns the current configuration.
* ``validate_fn(raw) -> (clean, warnings)`` normalizes untrusted data
  (file contents or an import) without applying it; raises ``ValueError``
  when the section is unusable. Optional (identity when omitted).
* ``load_fn(clean)`` applies normalized data to the running app.

Writes are atomic (temp file + ``os.replace``), debounced, and skipped when
nothing changed. A corrupted file is kept as ``state.json.bak`` and the app
starts with defaults. Sections nobody registered (yet) are carried over
unchanged, so a feature that is temporarily missing never loses its data.

The built-in ``cameras`` section stores per-camera ESP settings, detection
on/off, YOLO model and zones (user configuration from ``CameraRuntime``;
runtime stats and the target lock are not persisted).
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import config

SCHEMA_VERSION = 1
APP_TAG = "securitycam"

DATA_DIR = Path(os.getenv("SECURITYCAM_DATA_DIR", "").strip() or (config.PROJECT_ROOT / "data"))
STATE_FILE = DATA_DIR / "state.json"
SAVE_DEBOUNCE_S = config._env_float("SECURITYCAM_SAVE_DEBOUNCE", 1.0, minimum=0.0, maximum=60.0)

Validator = Callable[[Any], "tuple[Any, list[str]]"]


class DocumentError(ValueError):
    """An import / state document that cannot be used at all."""


@dataclass
class Section:
    name: str
    dump: Callable[[], Any]
    load: Callable[[Any], None]
    validate: Validator | None = None

    def normalize(self, raw: Any) -> tuple[Any, list[str]]:
        if self.validate is None:
            return raw, []
        return self.validate(raw)


def _log(msg: str) -> None:
    print(f"[Persist] {msg}")


class Store:
    def __init__(self, path: Path, debounce_s: float = SAVE_DEBOUNCE_S, clock: Callable[[], float] = time.monotonic):
        self.path = Path(path)
        self.debounce_s = float(debounce_s)
        self.clock = clock
        self._sections: dict[str, Section] = {}
        self._raw: dict[str, Any] = {}           # file contents per section (kept for unregistered sections)
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._cond = threading.Condition(threading.Lock())
        self._dirty_at: float | None = None
        self._saver: threading.Thread | None = None
        self._last_written: str | None = None
        self.loaded = False
        self.last_error = ""
        self.last_saved_at = 0.0

    # ------------------------------------------------------------------ sections
    def register_section(self, name: str, dump_fn, load_fn, validate_fn=None) -> None:
        section = Section(name, dump_fn, load_fn, validate_fn)
        with self._lock:
            self._sections[name] = section
            if self.loaded and name in self._raw:
                self._load_section(section, self._raw[name])

    def sections(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sections)

    def _load_section(self, section: Section, raw: Any) -> None:
        try:
            clean, warnings = section.normalize(raw)
            for w in warnings:
                _log(f"{section.name}: {w}")
            section.load(clean)
        except Exception as exc:  # one broken section must not take the others down
            _log(f"Could not restore section '{section.name}': {exc}")

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Read the state file and apply all registered sections. Returns True if a file was applied."""
        with self._lock:
            self.loaded = True
            self._raw = {}
            if not self.path.exists():
                _log(f"No saved state at {self.path} - starting with defaults.")
                self._last_written = None
                return False
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
                sections = self._sections_of(doc)
            except (OSError, ValueError) as exc:
                self._quarantine(exc)
                return False
            self._raw = sections
            for name, section in self._sections.items():
                if name in sections:
                    self._load_section(section, sections[name])
            # Baseline: an unchanged state is never rewritten.
            self._last_written = self._serialize(self._dump_sections())
            _log(f"Restored {', '.join(sorted(sections)) or 'nothing'} from {self.path}.")
            return True

    @staticmethod
    def _sections_of(doc: Any) -> dict[str, Any]:
        if not isinstance(doc, dict):
            raise DocumentError("not a JSON object")
        version = doc.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise DocumentError("missing or invalid schema version")
        if version > SCHEMA_VERSION:
            _log(f"State file has newer schema version {version} (supported {SCHEMA_VERSION}); reading what we understand.")
        sections = doc.get("sections")
        if not isinstance(sections, dict):
            raise DocumentError("missing 'sections' object")
        return dict(sections)

    def _quarantine(self, exc: Exception) -> None:
        backup = self.path.with_name(self.path.name + ".bak")
        try:
            os.replace(self.path, backup)
            where = f"moved to {backup.name}"
        except OSError as move_exc:
            where = f"could not be moved aside ({move_exc})"
        self.last_error = f"State file was unreadable ({exc}); {where}. Started with defaults."
        _log(self.last_error)
        self._last_written = None

    # ------------------------------------------------------------------ dump / save
    def _dump_sections(self) -> dict[str, Any]:
        with self._lock:
            out = {name: copy.deepcopy(data) for name, data in self._raw.items() if name not in self._sections}
            for name, section in self._sections.items():
                try:
                    out[name] = section.dump()
                except Exception as exc:
                    _log(f"Could not dump section '{name}': {exc}")
                    if name in self._raw:
                        out[name] = copy.deepcopy(self._raw[name])
            return out

    @staticmethod
    def _serialize(sections: dict[str, Any]) -> str:
        return json.dumps(sections, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def document(self) -> dict[str, Any]:
        return {
            "app": APP_TAG,
            "version": SCHEMA_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sections": self._dump_sections(),
        }

    def save_now(self, force: bool = False) -> bool:
        """Write the state if it changed. Returns True when the file was written."""
        with self._io_lock:
            doc = self.document()
            key = self._serialize(doc["sections"])
            if not force and key == self._last_written:
                return False
            self._atomic_write(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True))
            self._last_written = key
            self.last_saved_at = time.time()
            return True

    def _atomic_write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    # Windows: a reader / virus scanner may hold the target briefly.
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    # ------------------------------------------------------------------ debounce
    def mark_dirty(self) -> None:
        if not self.loaded:
            return
        with self._cond:
            self._dirty_at = self.clock()
            if self._saver is None or not self._saver.is_alive():
                self._saver = threading.Thread(target=self._saver_loop, name="StateSaver", daemon=True)
                self._saver.start()
            self._cond.notify_all()

    def _saver_loop(self) -> None:
        while True:
            with self._cond:
                while self._dirty_at is None:
                    self._cond.wait()
                wait = self._dirty_at + self.debounce_s - self.clock()
                if wait > 0:
                    self._cond.wait(wait)
                    continue
                self._dirty_at = None
            self._safe_save()

    def _safe_save(self) -> None:
        try:
            self.save_now()
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"Saving state failed: {exc}"
            _log(self.last_error)

    def pending(self) -> bool:
        with self._cond:
            return self._dirty_at is not None

    def flush(self) -> None:
        with self._cond:
            dirty = self._dirty_at is not None
            self._dirty_at = None
        if dirty:
            self._safe_save()

    # ------------------------------------------------------------------ import / export
    def validate_document(self, doc: Any) -> tuple[dict[str, Any], list[str]]:
        """Normalize an import document. Returns ({section: clean}, warnings)."""
        try:
            sections = self._sections_of(doc)
        except DocumentError:
            raise
        if isinstance(doc, dict) and doc.get("app") not in (None, APP_TAG):
            raise DocumentError("not a SecurityCam settings file")
        clean: dict[str, Any] = {}
        warnings: list[str] = []
        with self._lock:
            for name, raw in sections.items():
                section = self._sections.get(name)
                if section is None:
                    warnings.append(f"Unknown section '{name}' ignored.")
                    continue
                try:
                    clean[name], w = section.normalize(raw)
                except (ValueError, TypeError) as exc:
                    warnings.append(f"Section '{name}' skipped: {exc}")
                    continue
                warnings.extend(f"{name}: {x}" for x in w)
        if not clean:
            raise DocumentError("no usable sections")
        return clean, warnings

    def diff(self, clean: dict[str, Any]) -> list[dict[str, Any]]:
        current = self._dump_sections()
        changes: list[dict[str, Any]] = []
        for name, new in clean.items():
            _diff_values(name, current.get(name), new, changes)
        return changes

    def apply_document(self, clean: dict[str, Any]) -> None:
        with self._lock:
            for name, data in clean.items():
                section = self._sections.get(name)
                if section is not None:
                    section.load(data)
        self.mark_dirty()
        self.flush()


_MISSING = object()
_MAX_DIFF = 500


def _id_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, dict) and "id" in x for x in value)


def _diff_values(path: str, old: Any, new: Any, out: list[dict[str, Any]]) -> None:
    if len(out) >= _MAX_DIFF:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new), key=str):
            _diff_values(f"{path}.{key}", old.get(key, _MISSING), new.get(key, _MISSING), out)
        return
    if _id_list(old) and _id_list(new):
        # Lists of records (presets, schedules, zones) are replaced as a whole on
        # import: diff per record id and report removed records too.
        old_by_id = {str(x["id"]): x for x in old}
        new_by_id = {str(x["id"]): x for x in new}
        for key in sorted(set(old_by_id) | set(new_by_id)):
            if key not in new_by_id:
                if len(out) < _MAX_DIFF:
                    out.append({"path": f"{path}[{key}]", "before": old_by_id[key], "after": None, "kind": "removed"})
            elif key not in old_by_id:
                if len(out) < _MAX_DIFF:
                    out.append({"path": f"{path}[{key}]", "before": None, "after": new_by_id[key], "kind": "added"})
            else:
                _diff_values(f"{path}[{key}]", old_by_id[key], new_by_id[key], out)
        return
    if old == new:
        return
    if old is _MISSING or new is _MISSING:
        # A key exists only on one side: replacing it is reported, removing it is not
        # (apply never deletes keys that the import does not mention).
        if new is _MISSING:
            return
        out.append({"path": path, "before": None, "after": new, "kind": "added"})
        return
    out.append({"path": path, "before": old, "after": new, "kind": "changed"})


# ---------------------------------------------------------------------------- cameras section
_ZONE_ID_RE = re.compile(r"^z(\d+)$")
_EXTRA_ZONE_KEYS_BLOCKED = {"count"}


def _camera_states():
    from . import state
    return state.camera_states


def dump_cameras() -> dict[str, Any]:
    from .esp.pipeline import _esp_lock

    out: dict[str, Any] = dict(_orphan_cameras)
    for cam_id, cs in _camera_states().items():
        with _esp_lock(cs):
            zones = [
                {k: (list(v) if k == "box" else copy.deepcopy(v)) for k, v in z.items() if k not in _EXTRA_ZONE_KEYS_BLOCKED}
                for z in cs.esp_zones
            ]
            seq = int(cs.esp_zone_seq)
        out[cam_id] = {
            "esp_settings": dict(cs.esp_settings),
            "yolo_enabled": bool(cs.yolo_enabled),
            "yolo_model": cs.yolo_model_name,
            "zones": zones,
            "zone_seq": seq,
        }
    return out


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_zones(raw: Any, warnings: list[str], cam_id: str) -> list[dict[str, Any]]:
    from .targets import MAX_ZONES
    from .validation import ZoneError, parse_zone_patch

    if not isinstance(raw, list):
        if raw is not None:
            warnings.append(f"{cam_id}: zones ignored (not a list).")
        return []
    zones: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if len(zones) >= MAX_ZONES:
            warnings.append(f"{cam_id}: more than {MAX_ZONES} zones, rest dropped.")
            break
        if not isinstance(item, dict):
            continue
        zid = item.get("id")
        if not isinstance(zid, str) or not _ZONE_ID_RE.match(zid) or zid in seen:
            warnings.append(f"{cam_id}: zone with invalid id dropped.")
            continue
        try:
            fields = parse_zone_patch(
                {"box": item.get("box"), "label": item.get("label"), "color": item.get("color")}, require_box=True
            )
        except ZoneError:
            warnings.append(f"{cam_id}: zone {zid} invalid, dropped.")
            continue
        zone = {
            "id": zid,
            "label": fields.get("label") or f"Zone {zid[1:]}",
            "box": fields["box"],
            "color": fields.get("color") or "#ffd21f",
        }
        # Keep extra fields other features may have added (JSON-safe values only).
        for k, v in item.items():
            if k not in zone and k not in _EXTRA_ZONE_KEYS_BLOCKED and isinstance(k, str) and _json_safe(v):
                zone[k] = v
        zones.append(zone)
        seen.add(zid)
    return zones


def validate_cameras(raw: Any) -> tuple[dict[str, Any], list[str]]:
    from .validation import sanitize_esp_patch

    if not isinstance(raw, dict):
        raise ValueError("cameras must be an object")
    known = _camera_states()
    clean: dict[str, Any] = {}
    warnings: list[str] = []
    for cam_id, entry in raw.items():
        if not isinstance(cam_id, str) or not isinstance(entry, dict):
            continue
        if cam_id not in known:
            warnings.append(f"Camera '{cam_id}' is not configured; kept but not applied.")
            clean[cam_id] = {"orphan": True, "data": entry if _json_safe(entry) else {}}
            continue
        item: dict[str, Any] = {}
        if "esp_settings" in entry:
            item["esp_settings"] = sanitize_esp_patch(entry.get("esp_settings"))
        if isinstance(entry.get("yolo_enabled"), bool):
            item["yolo_enabled"] = entry["yolo_enabled"]
        model = entry.get("yolo_model")
        if model is not None:
            if model in config.AVAILABLE_MODELS:
                item["yolo_model"] = model
            else:
                warnings.append(f"{cam_id}: unknown model '{model}' ignored.")
        if "zones" in entry:
            item["zones"] = _normalize_zones(entry.get("zones"), warnings, cam_id)
            seq = entry.get("zone_seq")
            top = max((int(_ZONE_ID_RE.match(z["id"]).group(1)) for z in item["zones"]), default=0)
            item["zone_seq"] = max(top, seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0 else 0)
        clean[cam_id] = item
    return clean, warnings


_orphan_cameras: dict[str, Any] = {}


def load_cameras(clean: dict[str, Any]) -> None:
    from .esp.pipeline import _esp_lock

    states = _camera_states()
    for cam_id, item in clean.items():
        if item.get("orphan"):
            _orphan_cameras[cam_id] = item.get("data", {})
            continue
        cs = states.get(cam_id)
        if cs is None:
            continue
        if "esp_settings" in item:
            merged = dict(config.DEFAULT_ESP_SETTINGS)
            merged.update(item["esp_settings"])
            cs.esp_settings.update(merged)
        if "yolo_enabled" in item:
            cs.yolo_enabled = item["yolo_enabled"]
        if "yolo_model" in item and item["yolo_model"] != cs.yolo_model_name:
            cs.yolo_model_name = item["yolo_model"]
            cs.yolo_model = None
        if "zones" in item:
            with _esp_lock(cs):
                cs.esp_zones[:] = [dict(z) for z in item["zones"]]
                cs.esp_zone_seq = max(int(item.get("zone_seq", 0)), int(cs.esp_zone_seq))
                # zones saved before world anchoring: anchor them at the pose known now
                from .anchor import ensure_zone_anchors
                ensure_zone_anchors(cs)


# ---------------------------------------------------------------------------- module API
STORE = Store(STATE_FILE)


def register_section(name: str, dump_fn, load_fn, validate_fn=None) -> None:
    """Register a configuration section (see module docstring)."""
    STORE.register_section(name, dump_fn, load_fn, validate_fn)


def mark_dirty() -> None:
    """Request a (debounced) save. Cheap; call after any configuration change."""
    STORE.mark_dirty()


_init_lock = threading.Lock()
_initialized = False


def init() -> None:
    """Register the built-in section, restore the state file and flush on exit (idempotent)."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        _initialized = True
        STORE.register_section("cameras", dump_cameras, load_cameras, validate_cameras)
        STORE.load()
        atexit.register(STORE.flush)
