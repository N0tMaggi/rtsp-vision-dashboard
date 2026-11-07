"""Target tags and per-camera view filters (solo / hidden), backing the right-click menu.

Tags belong to an *identity*, not just to a raw tracker id. For every target
the most durable identity available is used (best first)::

    p:<person id>          recognized known person (data/faces) - survives restarts, all cameras
    f:<face uid>           automatic unknown face id "Face #n" (unknown_faces.py) - survives restarts
    v:<plate key>          vehicle with a confirmed plate (OCR-confusion folded) - survives restarts
    t:<cam>:<epoch>:<id>   the track itself (ReID re-entry id) - this session only

A tag added while only the track is known is *followed*: as soon as the track
gets a better identity (the face is recognized, the plate is confirmed) its
track tags move there, so they come back next time that person / car shows up.
Identity tags are persisted in the ``tags`` section of data/state.json; track
tags live in memory and expire with the track.

Solo / hidden are per camera, visual only: the ESP overlay and the browser
overlays hide the other targets, while alarms, zone counts and recordings keep
seeing everything. They match the track id (same tracker epoch) or the
target's durable identity, so a solo'd person stays solo'd after re-entry.

The render path only reads immutable snapshots (``prepare()``), never locks.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LABEL_MAX = 24
MAX_TAGS_PER_ID = 8
MAX_IDENTITIES = 2000
TRACK_TTL_S = 900.0
MAX_HIDDEN = 32
FACE_RECENT_S = 3.0          # a face attributed this recently counts as "person has a visible face"
PLATE_KEY_READS = 3          # plate reads before a vehicle's plate becomes its identity
PALETTE = ("#ffd21f", "#ff5a5a", "#7ee0a1", "#5ad1ff", "#c792ea", "#ff9f43", "#f4f4f2")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
KEY_RE = re.compile(r"^(p:[a-z0-9][a-z0-9-]{0,47}|f:\d{1,7}|v:[A-Z0-9]{2,16})$")


class TagError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ----------------------------------------------------------------------------- validation
def clean_label(raw: Any) -> str:
    if not isinstance(raw, str):
        raise TagError("invalid_label")
    text = "".join(c for c in unicodedata.normalize("NFC", raw) if unicodedata.category(c)[0] != "C")
    text = " ".join(text.split())
    if not text or len(text) > LABEL_MAX:
        raise TagError("invalid_label")
    return text


def clean_color(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not HEX_RE.match(raw.strip()):
        raise TagError("invalid_color")
    return raw.strip().lower()


def default_color(label: str) -> str:
    h = 0
    for ch in label.casefold():
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


def hex_to_bgr(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


# ----------------------------------------------------------------------------- identities
def track_key(camera_id: str, epoch: int, tid: int) -> str:
    return f"t:{camera_id}:{int(epoch)}:{int(tid)}"


def identity_keys(camera_id: str, epoch: int, tid: int, ti: Any) -> List[str]:
    """Identity keys of one target, most durable first (track key always last)."""
    keys: List[str] = []
    if ti is not None:
        if getattr(ti, "person_id", None):
            keys.append(f"p:{ti.person_id}")
        if getattr(ti, "face_uid", None) is not None:
            keys.append(f"f:{int(ti.face_uid)}")
        plate = getattr(ti, "plate_best", None)
        if plate and getattr(ti, "plate_reads", 0) >= PLATE_KEY_READS:
            from .recognition.platelog import fold

            pk = fold(plate)
            if 2 <= len(pk) <= 16:
                keys.append(f"v:{pk}")
    keys.append(track_key(camera_id, epoch, tid))
    return keys


def _runtime_view(camera_state) -> Tuple[int, Dict[int, Any]]:
    """(tracker epoch, copy of track info) of a camera; cheap, no inference."""
    from .recognition.common import peek_runtime

    rt = peek_runtime(getattr(camera_state, "camera_id", "default"))
    if rt is None:
        return 0, {}
    with rt.lock:
        return rt.track_epoch, dict(rt.track_info)


# ----------------------------------------------------------------------------- store
class TagStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._tags: Dict[str, List[Dict[str, str]]] = {}
        self._touch: Dict[str, float] = {}               # track keys -> last use (monotonic)
        self._snap: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        # track key -> the durable key its tags went to. While the track lives its tags stay
        # visible even if recognition flips; a corrected plate reading carries them along.
        self._bound: Dict[str, str] = {}
        self._bound_snap: Dict[str, str] = {}
        self.version = 0

    # ---- internals
    def _publish_locked(self) -> None:
        self._snap = {k: tuple((t["label"], t["color"]) for t in v) for k, v in self._tags.items() if v}
        self._bound_snap = dict(self._bound)
        self.version += 1

    def _bind_locked(self, tkey: str, key: str) -> None:
        if tkey.startswith("t:") and not key.startswith("t:"):
            self._bound[tkey] = key
            self._touch[tkey] = time.monotonic()

    def with_bound(self, keys: Sequence[str]) -> List[str]:
        bound = self._bound_snap.get(keys[-1]) if keys else None
        return list(keys) + [bound] if bound and bound not in keys else list(keys)

    def _changed_locked(self, persist: bool = True) -> None:
        self._publish_locked()
        if persist:
            _mark_dirty()

    def _color_for_locked(self, label: str) -> str:
        fold = label.casefold()
        for tags in self._tags.values():
            for t in tags:
                if t["label"].casefold() == fold:
                    return t["color"]
        return default_color(label)

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - TRACK_TTL_S
        for k in [k for k, t in self._touch.items() if t < cutoff]:
            self._touch.pop(k, None)
            self._tags.pop(k, None)
            self._bound.pop(k, None)

    # ---- reads
    def snapshot(self) -> Dict[str, Tuple[Tuple[str, str], ...]]:
        return self._snap

    def get(self, keys: Iterable[str]) -> List[Dict[str, str]]:
        snap = self._snap
        out: List[Dict[str, str]] = []
        seen = set()
        for k in keys:
            for label, color in snap.get(k, ()):
                f = label.casefold()
                if f not in seen:
                    seen.add(f)
                    out.append({"label": label, "color": color})
        return out

    def catalog(self) -> List[Dict[str, Any]]:
        counts: Dict[str, Dict[str, Any]] = {}
        for tags in self._snap.values():
            for label, color in tags:
                c = counts.setdefault(label.casefold(), {"label": label, "color": color, "count": 0})
                c["count"] += 1
        return sorted(counts.values(), key=lambda c: (-c["count"], c["label"].casefold()))

    # ---- writes
    def add(self, keys: Sequence[str], label: Any, color: Any = None) -> List[Dict[str, str]]:
        label = clean_label(label)
        color = clean_color(color)
        key = keys[0]
        with self._lock:
            self._prune_locked()
            if key not in self._tags and len(self._tags) >= MAX_IDENTITIES:
                raise TagError("too_many_tags")
            color = color or self._color_for_locked(label)
            tags = self._tags.setdefault(key, [])
            fold = label.casefold()
            existing = next((t for t in tags if t["label"].casefold() == fold), None)
            if existing is not None:
                existing["label"], existing["color"] = label, color
            else:
                if len(tags) >= MAX_TAGS_PER_ID:
                    raise TagError("too_many_tags")
                tags.append({"label": label, "color": color})
            if key.startswith("t:"):
                self._touch[key] = time.monotonic()
            else:
                self._bind_locked(keys[-1], key)
            self._changed_locked(persist=not key.startswith("t:"))
        return self.get(self.with_bound(keys))

    def remove(self, keys: Sequence[str], label: Any) -> bool:
        label = clean_label(label)
        fold = label.casefold()
        removed = False
        with self._lock:
            persist = False
            for k in self.with_bound(keys):
                tags = self._tags.get(k)
                if not tags:
                    continue
                keep = [t for t in tags if t["label"].casefold() != fold]
                if len(keep) != len(tags):
                    removed = True
                    persist = persist or not k.startswith("t:")
                    if keep:
                        self._tags[k] = keep
                    else:
                        del self._tags[k]
            if removed:
                self._changed_locked(persist=persist)
        return removed

    def follow(self, keys: Sequence[str]) -> List[Dict[str, str]]:
        """Tags of a target; moves track tags onto a better identity once one is known."""
        snap = self._snap
        tkey, best = keys[-1], keys[0]
        bound = self._bound_snap.get(tkey)
        if bound is not None:
            self._touch[tkey] = time.monotonic()
            if bound != best and bound.startswith("v:") and best.startswith("v:") and snap.get(bound):
                self.move(bound, best)           # the plate reading of this very car was corrected
        if len(keys) > 1 and tkey.startswith("t:") and snap.get(tkey):
            with self._lock:
                track_tags = self._tags.pop(tkey, [])
                dst = self._tags.setdefault(best, [])
                have = {t["label"].casefold() for t in dst}
                for t in track_tags:
                    if t["label"].casefold() not in have and len(dst) < MAX_TAGS_PER_ID:
                        dst.append(dict(t))
                        have.add(t["label"].casefold())
                self._bind_locked(tkey, best)
                self._changed_locked()
        elif snap.get(tkey):
            self._touch[tkey] = time.monotonic()
        return self.get(self.with_bound(keys))

    def move(self, src: str, dst: str) -> None:
        """Identity changed (unknown face named / merged, plate corrected): its tags go along."""
        with self._lock:
            for k, v in list(self._bound.items()):
                if v == src:
                    self._bound[k] = dst
            tags = self._tags.pop(src, None)
            if not tags:
                self._publish_locked()
                return
            out = self._tags.setdefault(dst, [])
            have = {t["label"].casefold() for t in out}
            out.extend(t for t in tags if t["label"].casefold() not in have)
            del out[MAX_TAGS_PER_ID:]
            self._changed_locked()

    def drop(self, key: str) -> None:
        with self._lock:
            for k in [k for k, v in self._bound.items() if v == key]:
                del self._bound[k]
            if self._tags.pop(key, None) is not None:
                self._changed_locked()

    # ---- persistence
    def dump(self) -> Dict[str, Any]:
        with self._lock:
            return {"version": 1, "tags": {k: [dict(t) for t in v] for k, v in sorted(self._tags.items())
                                           if v and not k.startswith("t:")}}

    @staticmethod
    def validate(raw: Any) -> Tuple[Dict[str, Any], List[str]]:
        if not isinstance(raw, dict) or not isinstance(raw.get("tags", {}), dict):
            raise ValueError("tags section must be an object")
        out: Dict[str, List[Dict[str, str]]] = {}
        warnings: List[str] = []
        for key, items in raw.get("tags", {}).items():
            if not isinstance(key, str) or not KEY_RE.match(key) or not isinstance(items, list):
                warnings.append(f"dropped invalid tag owner {str(key)[:40]!r}")
                continue
            clean: List[Dict[str, str]] = []
            for it in items[:MAX_TAGS_PER_ID]:
                try:
                    label = clean_label(it.get("label") if isinstance(it, dict) else None)
                    color = clean_color(it.get("color")) or default_color(label)
                except TagError:
                    warnings.append(f"dropped an invalid tag of {key}")
                    continue
                if all(c["label"].casefold() != label.casefold() for c in clean):
                    clean.append({"label": label, "color": color})
            if clean:
                out[key] = clean
            if len(out) >= MAX_IDENTITIES:
                warnings.append("too many tagged identities; the rest was dropped")
                break
        return {"version": 1, "tags": out}, warnings

    def load(self, clean: Dict[str, Any]) -> None:
        with self._lock:
            track = {k: v for k, v in self._tags.items() if k.startswith("t:")}
            self._tags = {k: [dict(t) for t in v] for k, v in clean.get("tags", {}).items()}
            self._tags.update(track)
            self._publish_locked()


STORE = TagStore()


def _mark_dirty() -> None:
    try:
        from . import persistence

        persistence.mark_dirty()
    except Exception:
        pass


def _register() -> None:
    from . import persistence

    persistence.register_section("tags", STORE.dump, STORE.load, STORE.validate)


_register()


# ----------------------------------------------------------------------------- per-camera view (solo / hidden)
def _view_lock(camera_state):
    from .esp.pipeline import _esp_lock

    return _esp_lock(camera_state)


def view_state(camera_state) -> Dict[str, Any]:
    solo = getattr(camera_state, "view_solo", None)
    hidden = getattr(camera_state, "view_hidden", None) or []
    return {"solo": dict(solo) if solo else None, "hidden": [dict(h) for h in hidden]}


def _entry(camera_state, target: Dict[str, Any]) -> Dict[str, Any]:
    epoch, infos = _runtime_view(camera_state)
    tid = int(target["id"])
    keys = identity_keys(camera_state.camera_id, epoch, tid, infos.get(tid))
    ident = infos.get(tid)
    name = None
    if ident is not None and getattr(ident, "name", None):
        name = ident.name
    elif ident is not None and getattr(ident, "face_uid", None) is not None:
        name = f"Face #{ident.face_uid}"
    return {"id": tid, "epoch": epoch, "key": keys[0] if len(keys) > 1 else None,
            "label": str(target.get("label") or "target"), "name": name}


def set_solo(camera_state, target: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entry = _entry(camera_state, target) if target is not None else None
    with _view_lock(camera_state):
        camera_state.view_solo = entry
        if entry is not None:     # solo wins over hidden for the same target
            camera_state.view_hidden = [h for h in (getattr(camera_state, "view_hidden", None) or [])
                                        if not _same(h, entry)]
    return view_state(camera_state)


def hide(camera_state, target: Dict[str, Any]) -> Dict[str, Any]:
    entry = _entry(camera_state, target)
    with _view_lock(camera_state):
        hidden = [h for h in (getattr(camera_state, "view_hidden", None) or []) if not _same(h, entry)]
        hidden.append(entry)
        camera_state.view_hidden = hidden[-MAX_HIDDEN:]
        solo = getattr(camera_state, "view_solo", None)
        if solo and _same(solo, entry):
            camera_state.view_solo = None
    return view_state(camera_state)


def unhide(camera_state, target_id: Optional[int]) -> Dict[str, Any]:
    with _view_lock(camera_state):
        if target_id is None:
            camera_state.view_hidden = []
        else:
            camera_state.view_hidden = [h for h in (getattr(camera_state, "view_hidden", None) or [])
                                        if h["id"] != int(target_id)]
    return view_state(camera_state)


def clear_view(camera_state) -> Dict[str, Any]:
    with _view_lock(camera_state):
        camera_state.view_solo = None
        camera_state.view_hidden = []
    return view_state(camera_state)


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a["id"] == b["id"] and a["epoch"] == b["epoch"]) or (a.get("key") is not None and a.get("key") == b.get("key"))


def _matches(entry: Dict[str, Any], tid: int, epoch: int, keys: Sequence[str]) -> bool:
    if entry["id"] == tid and entry["epoch"] == epoch:
        return True
    key = entry.get("key")
    return key is not None and key in keys


def is_visible(solo, hidden, tid: int, epoch: int, keys: Sequence[str]) -> bool:
    if solo:
        return _matches(solo, tid, epoch, keys)
    return not any(_matches(h, tid, epoch, keys) for h in hidden)


# ----------------------------------------------------------------------------- /targets payload
def decorate_payload(camera_state, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``tags``, ``identity`` and ``hidden`` to each target and the camera's ``view`` (additive)."""
    epoch, infos = _runtime_view(camera_state)
    view = view_state(camera_state)
    solo, hidden = view["solo"], view["hidden"]
    now = time.monotonic()
    cam = getattr(camera_state, "camera_id", "default")
    for t in payload.get("targets", []):
        tid = int(t["id"])
        ti = infos.get(tid)
        keys = identity_keys(cam, epoch, tid, ti)
        tags = STORE.follow(keys)
        if tags:
            t["tags"] = tags
        if ti is not None:
            ident = {}
            if ti.name:
                ident["name"], ident["person_id"] = ti.name, ti.person_id
            if ti.face_uid is not None:
                ident["face_id"] = int(ti.face_uid)
            if ti.plate_best:
                ident["plate"] = ti.plate_best
            if ti.face_seen and now - ti.face_seen <= FACE_RECENT_S:
                ident["face"] = True
            if ident:
                t["identity"] = ident
        if (solo or hidden) and not is_visible(solo, hidden, tid, epoch, keys):
            t["hidden"] = True
    payload["view"] = view
    return payload


def target_keys(camera_state, target_id: int) -> List[str]:
    epoch, infos = _runtime_view(camera_state)
    return identity_keys(getattr(camera_state, "camera_id", "default"), epoch, int(target_id), infos.get(int(target_id)))


# ----------------------------------------------------------------------------- ESP render path
class ViewContext:
    """Per-frame snapshot for the ESP renderer: tag runs + solo / hidden filter (lookups only)."""

    __slots__ = ("cam", "epoch", "infos", "snap", "solo", "hidden", "bound")

    def __init__(self, cam, epoch, infos, snap, solo, hidden, bound=None):
        self.cam, self.epoch, self.infos, self.snap, self.solo, self.hidden = cam, epoch, infos, snap, solo, hidden
        self.bound = bound or {}

    def _keys(self, det) -> List[str]:
        tid = det.track_key if det.track_key is not None else det.obj_id
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            return []
        return identity_keys(self.cam, self.epoch, tid, self.infos.get(tid))

    def visible(self, det) -> bool:
        if not self.solo and not self.hidden:
            return True
        keys = self._keys(det)
        if not keys:
            return not self.solo
        tid = int(det.track_key if det.track_key is not None else det.obj_id)
        return is_visible(self.solo, self.hidden, tid, self.epoch, keys)

    def label_runs(self, det, runs: list, text_color) -> list:
        if not self.snap:
            return runs
        seen, extra = set(), []
        keys = self._keys(det)
        b = self.bound.get(keys[-1]) if keys else None
        for k in (keys + [b] if b else keys):
            for label, color in self.snap.get(k, ()):
                f = label.casefold()
                if f not in seen:
                    seen.add(f)
                    extra.append(("#" + label.upper(), True, hex_to_bgr(color)))
        if not extra:
            return runs
        from .esp.colors import ANTHRACITE, mix

        out = list(runs)
        sep = mix(text_color, ANTHRACITE, 0.55)
        for run in extra:
            if out:
                out.append(("  ", False, sep))
            out.append(run)
        return out

    def filter_faces(self, faces, shown, hidden_dets):
        """Drop face rects that belong to hidden targets (centre in a hidden box, in no shown box)."""
        if not hidden_dets or not faces:
            return faces

        def inside(cx, cy, d):
            return d.x1 <= cx <= d.x2 and d.y1 <= cy <= d.y2

        out = []
        for f in faces:
            fx, fy, fw, fh = f
            cx, cy = fx + fw / 2, fy + fh / 2
            if any(inside(cx, cy, d) for d in hidden_dets) and not any(inside(cx, cy, d) for d in shown):
                continue
            out.append(f)
        return out


def prepare(camera_state, rc=None) -> Optional[ViewContext]:
    """None when nothing is tagged and no view filter is set (the common, zero-cost case)."""
    snap = STORE.snapshot()
    solo = getattr(camera_state, "view_solo", None)
    hidden = getattr(camera_state, "view_hidden", None) or ()
    if not snap and not solo and not hidden:
        return None
    from .recognition.common import peek_runtime

    cam = getattr(camera_state, "camera_id", "default")
    rt = peek_runtime(cam)
    epoch = rt.track_epoch if rt is not None else 0
    infos = rc.track_info if rc is not None and rc.track_info else ({} if rt is None else dict(rt.track_info))
    return ViewContext(cam, epoch, infos, snap, solo, list(hidden), STORE._bound_snap)
