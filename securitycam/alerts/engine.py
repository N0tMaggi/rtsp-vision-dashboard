"""Per-frame alarm evaluation: zone rules, tripwire crossings, abandoned objects.

``update`` is called by the ESP pipeline once per rendered frame with the
camera's ESP lock held. It only does arithmetic over the (few) current
detections, zones and tripwires; everything expensive (thumbnail, clip,
notifications) is handed to ``events.trigger`` which just enqueues.

Coordinates: detections and the motion model's ``track_history`` are in pixels
of the rendered frame (``w`` x ``h``); zones, tripwires and abandoned boxes are
normalized 0..1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import clips, conf
from .events import events as _events
from .store import store as _store

ALARM_COLOR = "#ff5a4e"
BUFFER_IDLE_S = 3.0          # stop buffering frames this long after the last evaluated frame
RESTART_GAP_S = 5.0          # a pause this long (detection off / no viewer) restarts the baseline
ABANDON_BASELINE_S = 3.0     # objects already present right after a (re)start are not "new"
ABANDON_GRACE_S = 4.0        # an unseen candidate is forgotten after this long
ABANDON_IOU = 0.3
WIRE_MARGIN = 0.006          # min distance from the line (fraction of frame diagonal) to commit a side


@dataclass
class ZoneState:
    enter: dict = field(default_factory=dict)      # track key -> perf time it entered
    cond: bool = False
    active_until: float = 0.0


@dataclass
class WireState:
    n_in: int = 0
    n_out: int = 0
    sides: dict = field(default_factory=dict)      # track key -> (side, (x, y))
    flash_until: float = 0.0


@dataclass
class Candidate:
    uid: int
    cls: str
    box: tuple
    anchor: tuple
    first_t: float
    first_wall: float
    last_t: float
    baseline: bool
    flagged: bool = False


@dataclass
class FrameInfo:
    """What the overlay needs to draw for one frame (normalized coordinates)."""
    t: float
    alarm_zones: frozenset = frozenset()
    wires: tuple = ()          # (wire dict, n_in, n_out, alarm_active)
    abandoned: tuple = ()      # (box, label)


EMPTY = FrameInfo(t=0.0)


class CameraEngine:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.zones: dict[str, ZoneState] = {}
        self.wires: dict[str, WireState] = {}
        self.hist_t: dict[Any, float] = {}
        self.candidates: list[Candidate] = []
        self.cand_seq = 0
        self.baseline_until = 0.0
        self.last_t: Optional[float] = None
        self.last_mono = 0.0
        self.needs_buffer = False
        self.last_event: dict[tuple, float] = {}

    # ------------------------------------------------------------------ events
    def emit(self, camera_state, kind: str, source_id: str, source_label: str, cls: str,
             track: Any, message: str, notify: bool, extra: Optional[dict] = None) -> bool:
        now = time.monotonic()
        key = (kind, source_id)
        last = self.last_event.get(key)
        if last is not None and now - last < conf.ALARM_COOLDOWN_S:
            return False
        self.last_event[key] = now
        trig = {
            "camera": self.camera_id,
            "camera_label": getattr(camera_state, "label", self.camera_id),
            "ts": time.time(),
            "kind": kind,
            "source_id": source_id,
            "source_label": source_label,
            "cls": cls,
            "track_id": int(track) if isinstance(track, int) else None,
            "message": message,
            "notify": bool(notify),
        }
        if extra:
            trig.update(extra)
        return _events.trigger(trig)

    # ------------------------------------------------------------------ main entry
    def update(self, camera_state, detections, w: int, h: int, t: float, zones, settings) -> FrameInfo:
        if w <= 0 or h <= 0:
            return EMPTY
        cfg = _store.camera_config(self.camera_id)
        if self.last_t is None or t - self.last_t > RESTART_GAP_S:
            self.baseline_until = t + ABANDON_BASELINE_S
            self.hist_t.clear()
            for ws in self.wires.values():
                ws.sides.clear()
        self.last_t = t
        self.last_mono = time.monotonic()

        alarms_on = bool(settings.get("alarms", True))
        mst = getattr(camera_state, "esp_motion", None)
        tracks = getattr(mst, "tracks", {}) or {}
        live = [d for d in detections
                if (tr := tracks.get(d.track_key)) is not None and tr.missing_since is None]

        alarm_zones = self._zones(camera_state, cfg, zones, live, w, h, t, alarms_on)
        wires = self._wires(camera_state, cfg, tracks, w, h, t, alarms_on)
        abandoned = self._abandoned(camera_state, cfg, live, w, h, t, settings)

        rules_active = alarms_on and any(r.get("enabled") for r in cfg["zone_rules"].values())
        wire_alarms = alarms_on and any(wd.get("alarm") for wd in cfg["tripwires"])
        self.needs_buffer = rules_active or wire_alarms or bool(settings.get("abandoned", False))
        return FrameInfo(t=t, alarm_zones=alarm_zones, wires=wires, abandoned=abandoned)

    # ------------------------------------------------------------------ zones
    def _zones(self, camera_state, cfg, zones, live, w, h, t, alarms_on) -> frozenset:
        rules = cfg["zone_rules"]
        out = set()
        seen = set()
        local = None
        for zone in zones:
            zid = zone.get("id")
            rule = rules.get(zid)
            if not rule or not rule.get("enabled") or not alarms_on:
                continue
            seen.add(zid)
            zs = self.zones.get(zid)
            if zs is None:
                zs = self.zones[zid] = ZoneState()
            sched = rule.get("schedule") or {}
            if sched.get("mode") == "window":
                if local is None:
                    local = time.localtime()
                if not schedule_active(sched, local.tm_hour * 60 + local.tm_min):
                    zs.enter.clear()
                    zs.cond = False
                    if zs.active_until > t:
                        out.add(zid)
                    continue
            try:
                x1, y1, x2, y2 = (float(v) for v in zone["box"])
            except (KeyError, TypeError, ValueError):
                continue
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            classes = rule.get("classes") or ()
            dwell = float(rule.get("min_dwell_s", 0.0))
            present = {}
            best = None
            best_dwell = -1.0
            for d in live:
                if classes and d.cls_name.lower() not in classes:
                    continue
                if not (x1 <= d.cx <= x2 and y1 <= d.cy <= y2):
                    continue
                enter = zs.enter.get(d.track_key, t)
                present[d.track_key] = enter
                stay = t - enter
                if stay >= dwell and stay > best_dwell:
                    best, best_dwell = d, stay
            zs.enter = present
            if best is not None:
                if not zs.cond:
                    label = str(zone.get("label") or zid)
                    msg = f"{best.cls_name} in {label}"
                    if dwell > 0:
                        msg += f" for {best_dwell:.0f} s"
                    self.emit(camera_state, "zone", zid, label, best.cls_name,
                              best.obj_id if best.obj_id != -1 else None, msg, rule.get("notify", True))
                zs.cond = True
                zs.active_until = t + conf.ALARM_HOLD_S
            else:
                zs.cond = False
            if zs.active_until > t:
                out.add(zid)
        for zid in [z for z in self.zones if z not in seen]:
            del self.zones[zid]
        return frozenset(out)

    # ------------------------------------------------------------------ tripwires
    def _wires(self, camera_state, cfg, tracks, w, h, t, alarms_on) -> tuple:
        wires_cfg = cfg["tripwires"]
        if not wires_cfg:
            self.wires.clear()
            self.hist_t.clear()
            return ()
        history = getattr(camera_state, "track_history", None) or {}
        margin = WIRE_MARGIN * math.hypot(w, h)
        geo = []
        for wd in wires_cfg:
            ws = self.wires.get(wd["id"])
            if ws is None:
                ws = self.wires[wd["id"]] = WireState()
            ax, ay = wd["a"][0] * w, wd["a"][1] * h
            bx, by = wd["b"][0] * w, wd["b"][1] * h
            length = math.hypot(bx - ax, by - ay) or 1.0
            geo.append((wd, ws, ax, ay, bx, by, length))

        for key, pts in history.items():
            if not pts:
                continue
            last = self.hist_t.get(key, -math.inf)
            if pts[-1][2] <= last:
                continue
            tr = tracks.get(key)
            cls = tr.det.cls_name if tr is not None else ""
            for px, py, pt in pts:
                if pt <= last:
                    continue
                for wd, ws, ax, ay, bx, by, length in geo:
                    classes = wd.get("classes") or ()
                    if classes and cls.lower() not in classes:
                        continue
                    dist = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / length
                    if abs(dist) < margin:
                        continue
                    side = 1 if dist > 0 else -1
                    prev = ws.sides.get(key)
                    ws.sides[key] = (side, (px, py))
                    if prev is None or prev[0] == side:
                        continue
                    if not _segments_cross(prev[1], (px, py), (ax, ay), (bx, by)):
                        continue        # went around the end of the line
                    direction = "in" if side > 0 else "out"
                    if direction == "in":
                        ws.n_in += 1
                    else:
                        ws.n_out += 1
                    trig = wd.get("trigger", "both")
                    if alarms_on and wd.get("alarm") and trig in ("both", direction):
                        ws.flash_until = t + conf.ALARM_HOLD_S
                        label = str(wd.get("label") or wd["id"])
                        self.emit(camera_state, "tripwire", wd["id"], label, cls,
                                  tr.det.obj_id if tr is not None and tr.det.obj_id != -1 else None,
                                  f"{cls or 'object'} crossed {label} ({direction})", wd.get("notify", True),
                                  {"direction": direction})
            self.hist_t[key] = pts[-1][2]

        # forget per-track state of vanished tracks
        if len(self.hist_t) > len(history):
            for key in [k for k in self.hist_t if k not in history]:
                del self.hist_t[key]
        ids = set()
        out = []
        for wd, ws, *_ in geo:
            ids.add(wd["id"])
            if len(ws.sides) > len(history):
                for key in [k for k in ws.sides if k not in history]:
                    del ws.sides[key]
            out.append((wd, ws.n_in, ws.n_out, ws.flash_until > t))
        for wid in [k for k in self.wires if k not in ids]:
            del self.wires[wid]
        return tuple(out)

    def reset_counts(self, wire_id: str) -> None:
        ws = self.wires.get(wire_id)
        if ws is not None:
            ws.n_in = ws.n_out = 0

    # ------------------------------------------------------------------ abandoned objects
    def _abandoned(self, camera_state, cfg, live, w, h, t, settings) -> tuple:
        if not settings.get("abandoned", False):
            self.candidates.clear()
            return ()
        try:
            secs = float(settings.get("abandoned_seconds", 30))
        except (TypeError, ValueError):
            secs = 30.0
        for d in live:
            cls = d.cls_name.lower()
            if cls not in conf.ABANDON_CLASSES:
                continue
            box = (d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h)
            c = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
            best, best_iou = None, ABANDON_IOU
            for cand in self.candidates:
                if cand.cls != cls:
                    continue
                iou = _iou(box, cand.box)
                if iou >= best_iou:
                    best, best_iou = cand, iou
            if best is None:
                self.cand_seq += 1
                self.candidates.append(Candidate(self.cand_seq, cls, box, c, t, time.time(), t,
                                                 baseline=t < self.baseline_until))
                continue
            tol = max(0.02, 0.25 * min(box[2] - box[0], box[3] - box[1]))
            if math.hypot(c[0] - best.anchor[0], c[1] - best.anchor[1]) > tol:
                # it moved: someone is carrying it, start over
                best.anchor, best.first_t, best.first_wall = c, t, time.time()
                best.flagged = best.baseline = False
            best.box = box
            best.last_t = t

        out = []
        keep = []
        for cand in self.candidates:
            if t - cand.last_t > ABANDON_GRACE_S:
                continue
            keep.append(cand)
            if cand.baseline:
                continue
            if not cand.flagged and t - cand.first_t >= secs:
                cand.flagged = True
                since = time.strftime("%H:%M", time.localtime(cand.first_wall))
                self.emit(camera_state, "abandoned", f"a{cand.uid}", cand.cls, cand.cls, None,
                          f"{cand.cls} left behind since {since}", cfg.get("abandoned_notify", True),
                          {"since": cand.first_wall})
            if cand.flagged:
                out.append((cand.box, "NEW SINCE " + time.strftime("%H:%M", time.localtime(cand.first_wall))))
        self.candidates = keep
        return tuple(out)

    # ------------------------------------------------------------------ API helpers
    def active(self) -> list[dict]:
        t = time.perf_counter()
        out = [{"kind": "zone", "source_id": zid} for zid, zs in self.zones.items() if zs.active_until > t]
        out += [{"kind": "tripwire", "source_id": wid} for wid, ws in self.wires.items() if ws.flash_until > t]
        return out

    def counts(self) -> dict[str, dict[str, int]]:
        return {wid: {"in": ws.n_in, "out": ws.n_out} for wid, ws in self.wires.items()}


def schedule_active(sched: dict, minute_of_day: int) -> bool:
    if sched.get("mode") != "window":
        return True
    try:
        sh, sm = (int(v) for v in sched["start"].split(":"))
        eh, em = (int(v) for v in sched["end"].split(":"))
    except (KeyError, ValueError, AttributeError):
        return True
    start, end = sh * 60 + sm, eh * 60 + em
    if start == end:
        return True
    if start < end:
        return start <= minute_of_day < end
    return minute_of_day >= start or minute_of_day < end      # wraps past midnight


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1, p2, q1, q2) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 <= 0)


def _iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------- registry
_engines: dict[str, CameraEngine] = {}


def get(camera_id: str) -> CameraEngine:
    eng = _engines.get(camera_id)
    if eng is None:
        eng = _engines.setdefault(camera_id, CameraEngine(camera_id))
    return eng


def peek(camera_id: str) -> Optional[CameraEngine]:
    return _engines.get(camera_id)


def update(camera_state, detections, w, h, t, zones, settings) -> FrameInfo:
    return get(camera_state.camera_id).update(camera_state, detections, w, h, t, zones, settings)


def _wants_buffer(camera_id: str) -> bool:
    eng = _engines.get(camera_id)
    return bool(eng and eng.needs_buffer and time.monotonic() - eng.last_mono < BUFFER_IDLE_S)


clips.wants_buffer = _wants_buffer
