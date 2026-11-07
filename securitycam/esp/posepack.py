"""Pose ESP pack: IFF colors, gaze cones, posture / action labels + fall alarm,
status bars and gesture control (pipeline glue, per-track state, drawing).

Pipeline contract (``pipeline.draw_custom_esp``)::

    ppack = posepack.update(...)            # inside the ESP lock, None when every feature is off
    palettes = ppack.palettes(rc, dets, palettes)
    chams color = ppack.chams(det, default)
    ppack.draw(frame, det, pal, st, t, placed)        # per opaque detection (pass 2)
    ppack.draw_tags(frame, dets, st, placed, bg)       # after the labels

Per-track state is folded once per *new observation* of a track (the motion
model's ``obs_t``), using the observed keypoints; drawing uses the rendered
(smoothed) keypoints. Everything is arithmetic on 17 keypoints per person, so
the per-frame cost stays tiny; no extra models (hand landmarks reuse the
existing MediaPipe hand path, only while gestures are on).

Settings live in the same ESP settings mapping (``PoseSettings`` reads its own
keys so ``EspSettings`` stays untouched).
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np

from .. import config
from . import gestures as G
from . import pose_analysis as PA
from . import renderers as R
from . import trackstats
from .colors import ANTHRACITE, BGR, WHITE, DetectionColors, mix, parse_hex

AA = cv2.LINE_AA
ALARM_RED: BGR = (78, 90, 255)            # #ff5a4e
SPEED_CYAN: BGR = (255, 209, 90)          # #5ad1ff
POSTURE_HOLD_S = 0.6                      # a new posture must persist this long (no flicker)
ACTION_HOLD_S = 0.5
GAZE_TAU_S = 0.25
FALL_WINDOW_S = 1.2                       # upright -> lying within this time ...
FALL_DROP = 0.35                          # ... with the box top dropping by this x the upright height
FALL_CONFIRM_S = 0.3                      # lying must persist this long before the alarm
FALL_FLASH_S = 4.0
SPEED_BAR_MAX = 3.0                       # m/s for a full speed bar
WAVE_AMPLITUDE = 0.18                     # wrist swing, x torso length


def _bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


@dataclass(frozen=True)
class PoseSettings:
    iff_colors: bool
    color_friend: BGR
    color_foe: BGR
    color_unknown: BGR
    gaze: bool
    posture: bool
    fall_alarm: bool
    status_bars: bool
    bar_dwell: bool
    bar_conf: bool
    bar_speed: bool
    gestures: bool
    gesture_wave: bool
    gesture_palm: bool
    gesture_thumbs: bool
    alarms: bool
    loiter_seconds: float

    @property
    def any(self) -> bool:
        return self.iff_colors or self.gaze or self.posture or self.status_bars or self.gestures

    @property
    def needs_state(self) -> bool:
        return self.gaze or self.posture or self.gestures

    @property
    def enabled_gestures(self) -> set:
        return {g for g, on in (("wave", self.gesture_wave), ("palm", self.gesture_palm),
                                ("thumbs_up", self.gesture_thumbs)) if on}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PoseSettings":
        D = config.DEFAULT_ESP_SETTINGS
        g = lambda k: raw.get(k, D.get(k))  # noqa: E731
        b = lambda k: _bool(g(k), bool(D.get(k, False)))  # noqa: E731
        col = lambda k, fb: parse_hex(g(k)) if isinstance(g(k), str) and parse_hex(g(k)) else fb  # noqa: E731
        try:
            loiter = max(5.0, min(3600.0, float(g("loiter_seconds"))))
        except (TypeError, ValueError):
            loiter = 120.0
        return cls(
            iff_colors=b("iff_colors"),
            color_friend=col("iff_color_friend", (161, 224, 126)),
            color_foe=col("iff_color_foe", (90, 90, 255)),
            color_unknown=col("iff_color_unknown", (31, 210, 255)),
            gaze=b("gaze"), posture=b("posture"), fall_alarm=b("fall_alarm"),
            status_bars=b("status_bars"), bar_dwell=b("bar_dwell"), bar_conf=b("bar_conf"), bar_speed=b("bar_speed"),
            gestures=b("gestures"), gesture_wave=b("gesture_wave"), gesture_palm=b("gesture_palm"),
            gesture_thumbs=b("gesture_thumbs"),
            alarms=_bool(g("alarms"), True),
            loiter_seconds=loiter,
        )


# ----------------------------------------------------------------------------- per-track state
class Debounce:
    """A label that only changes after the new value persisted ``hold`` seconds."""

    __slots__ = ("value", "cand", "since")

    def __init__(self):
        self.value: Optional[str] = None
        self.cand: Optional[str] = None
        self.since = 0.0

    def update(self, raw: Optional[str], t: float, hold: float, keep_on_none: bool = True) -> Optional[str]:
        if raw is None and keep_on_none:
            return self.value
        if raw == self.value:
            self.cand = None
        elif raw != self.cand:
            self.cand, self.since = raw, t
        elif t - self.since >= hold:
            self.value, self.cand = raw, None
        return self.value


@dataclass
class TrackPose:
    obs_t: float = -1.0
    posture: Debounce = field(default_factory=Debounce)
    action: Debounce = field(default_factory=Debounce)
    raw_posture: Optional[str] = None
    yaw: Optional[float] = None
    pitch: float = 0.0
    gaze_t: float = 0.0
    looking: bool = False
    both_eyes: bool = False
    head_size: float = 0.0
    waves: Dict[str, PA.WaveDetector] = field(default_factory=lambda: {"l": PA.WaveDetector(), "r": PA.WaveDetector()})
    waving: bool = False
    # fall detection
    upright_t: float = -math.inf
    upright_top: float = 0.0
    upright_h: float = 0.0
    fall_pending: Optional[float] = None
    fall_until: float = 0.0
    fall_t: float = -math.inf
    # gestures
    gestures_now: frozenset = frozenset()
    palm_ref: Optional[tuple] = None


@dataclass
class PackState:
    tracks: Dict[Any, TrackPose] = field(default_factory=dict)
    machine: G.GestureMachine = field(default_factory=G.GestureMachine)
    recent: collections.deque = field(default_factory=lambda: collections.deque(maxlen=8))
    seq: int = 0


def get_state(camera_state) -> PackState:
    st = getattr(camera_state, "esp_posepack", None)
    if not isinstance(st, PackState):
        st = PackState()
        camera_state.esp_posepack = st
    return st


def hand_settings(s, raw: Mapping[str, Any]):
    """Settings for ``sensors.update_hands``: gestures need hand landmarks even when the
    hand skeleton is not drawn (the existing MediaPipe path, every HAND_DETECT_EVERY frames)."""
    if s.hand_skeleton or not _bool(raw.get("gestures", config.DEFAULT_ESP_SETTINGS.get("gestures")), False):
        return s
    return replace(s, hand_skeleton=True)


def _is_person(det) -> bool:
    return det.cls_name.lower() == "person"


def _speed_bh(tr) -> float:
    h = max(1.0, tr.obs[3] - tr.obs[1])
    vx = (tr.vel[0] + tr.vel[2]) * 0.5
    vy = (tr.vel[1] + tr.vel[3]) * 0.5
    return math.hypot(vx, vy) / h


def fold_observation(tp: TrackPose, det, box, t: float, speed_bh: float, ps: PoseSettings) -> Optional[str]:
    """Update posture / action / gaze / fall state from one observation.

    Returns ``"fall"`` when a fall was just confirmed. Pure apart from ``tp``
    (used directly by the tests with synthetic keypoints).
    """
    kps, vis = det.keypoints, det.kp_visible
    event = None
    x1, y1, x2, y2 = (float(v) for v in box)
    bh = max(1.0, y2 - y1)

    raw = PA.classify_posture(kps, vis, box) if kps is not None else (
        "lying" if (x2 - x1) / bh > PA.LYING_ASPECT else None)
    tp.raw_posture = raw
    posture = tp.posture.update(raw, t, POSTURE_HOLD_S)

    # ---- fall: upright -> lying fast, box top dropping
    if raw in PA.UPRIGHT:
        tp.upright_t, tp.upright_top, tp.upright_h = t, y1, bh
        tp.fall_pending = None
    elif raw == "lying":
        if tp.fall_pending is None:
            drop = (y1 - tp.upright_top) / max(1.0, tp.upright_h)
            if t - tp.upright_t <= FALL_WINDOW_S and drop >= FALL_DROP:
                tp.fall_pending = t
        elif t - tp.fall_pending >= FALL_CONFIRM_S:
            tp.fall_pending = None
            tp.upright_t = -math.inf          # one alarm per fall
            tp.fall_t, tp.fall_until = t, t + FALL_FLASH_S
            tp.posture.value, tp.posture.cand = "lying", None
            event = "fall"
    if tp.fall_pending is not None and t - tp.fall_pending > FALL_WINDOW_S + FALL_CONFIRM_S:
        tp.fall_pending = None

    # ---- actions
    waving = False
    b = PA.body(kps, vis) if kps is not None else None
    if kps is not None and b is not None and b.torso > 0:
        raised = {side: w[0] - s[0] for side, w, s in PA.raised_wrists(kps, vis)}
        for side, wd in tp.waves.items():
            x = raised.get(side)
            waving = wd.update(t, x / max(1.0, b.torso) if x is not None else None, WAVE_AMPLITUDE) or waving
    tp.waving = waving
    raw_action = None
    if kps is not None and PA.hands_up(kps, vis):
        raw_action = "hands_up"
    elif waving:
        raw_action = "waving"
    else:
        raw_action = PA.motion_action(speed_bh, posture if raw is None else raw)
    tp.action.update(raw_action or "", t, ACTION_HOLD_S, keep_on_none=False)

    # ---- gaze (EMA on yaw / pitch, time based)
    hp = PA.head_pose(kps, vis, bh) if kps is not None else None
    if hp is None:
        tp.yaw = None
        tp.looking = False
    else:
        a = 1.0 if tp.yaw is None or t - tp.gaze_t > 1.0 else 1.0 - math.exp(-max(0.0, t - tp.gaze_t) / GAZE_TAU_S)
        tp.yaw = hp.yaw if tp.yaw is None else tp.yaw + (hp.yaw - tp.yaw) * a
        tp.pitch += (hp.pitch - tp.pitch) * a
        tp.both_eyes = hp.both_eyes
        tp.head_size = hp.head_size
        tp.looking = PA.looking_at_camera(tp.yaw, hp.both_eyes, tp.looking)
        tp.gaze_t = t
    return event


def gesture_conditions(tp: TrackPose, det, hand_shapes: List[tuple], t: float, landmarks: bool) -> set:
    """Gestures whose condition holds for this person now.

    ``hand_shapes``: [(shape, wrist)] of the hands assigned to this person.
    Without a hand detector (``landmarks`` False) an open palm is approximated
    by a hand held still above the shoulder.
    """
    out = set()
    if tp.waving:
        out.add("wave")
    wave_recent = any(w.recent(t, 1.0) for w in tp.waves.values())
    shapes = {s for s, _ in hand_shapes}
    if "thumbs_up" in shapes:
        out.add("thumbs_up")
    palm = False
    wrist = None
    if landmarks:
        for s, w in hand_shapes:
            if s == "palm":
                palm, wrist = True, w
    elif det.keypoints is not None:
        raised = PA.raised_wrists(det.keypoints, det.kp_visible)
        if len(raised) == 1:
            palm, wrist = True, raised[0][1]
    if palm and not wave_recent and wrist is not None:
        b = PA.body(det.keypoints, det.kp_visible) if det.keypoints is not None else None
        still_r = 0.25 * (b.torso if b is not None and b.torso > 0 else max(1.0, det.height) * 0.3)
        ref = tp.palm_ref
        if ref is None or math.hypot(wrist[0] - ref[0], wrist[1] - ref[1]) > still_r:
            tp.palm_ref = (wrist[0], wrist[1])      # moved: the hold restarts from here
            if ref is not None:
                palm = False
        if palm:
            out.add("palm")
    else:
        tp.palm_ref = None
    return out


def _emit_fall(camera_state, key, det, ps: PoseSettings) -> None:
    if not (ps.fall_alarm and ps.alarms):
        return
    try:
        from ..alerts import engine
        from ..alerts.store import store
        notify = bool(store.camera_config(camera_state.camera_id).get("fall_notify", True))
        tid = det.obj_id if det.obj_id != -1 else None
        engine.get(camera_state.camera_id).emit(
            camera_state, "fall", f"fall-{key}", "Fall", "person", tid,
            f"Possible fall: person #{key}", notify)
    except Exception as exc:
        print(f"[PosePack] fall event failed: {exc.__class__.__name__}: {exc}")


def update(camera_state, mst, hands, t: float) -> Optional["FrameContext"]:
    """Fold new observations (ESP lock held). ``None`` when the pack is off (zero cost)."""
    _install_buffer_hook()
    raw = camera_state.esp_settings
    ps = PoseSettings.from_mapping(raw)
    if not ps.any:
        if getattr(camera_state, "esp_posepack", None) is not None:
            camera_state.esp_posepack = None
        return None
    pst = get_state(camera_state)
    tracks = mst.tracks
    if ps.needs_state:
        persons = []
        for key, tr in tracks.items():
            det = tr.det
            if not _is_person(det):
                continue
            tp = pst.tracks.get(key)
            if tp is None:
                tp = pst.tracks[key] = TrackPose()
            if tr.obs_t == tp.obs_t:
                continue
            tp.obs_t = tr.obs_t
            if fold_observation(tp, det, tr.obs, tr.obs_t, _speed_bh(tr), ps) == "fall":
                _emit_fall(camera_state, key, det, ps)
            if tr.missing_since is None:
                persons.append((key, tp, det, tr))
        if ps.gestures:
            _gestures(camera_state, pst, ps, persons, hands or [], t)
    for key in [k for k in pst.tracks if k not in tracks]:
        del pst.tracks[key]
    stats = trackstats.get_stats(camera_state) if ps.status_bars else {}
    snap = {k: _Snap(tp.posture.value if ps.posture else None,
                     (tp.action.value or None) if ps.posture else None,
                     tp.yaw if ps.gaze else None, tp.pitch, tp.looking and ps.gaze, tp.head_size,
                     tp.fall_until if ps.posture else 0.0, tp.gestures_now)
            for k, tp in pst.tracks.items()}
    bars = {}
    if ps.status_bars:
        for key, tr in tracks.items():
            s = stats.get(key)
            bars[key] = (trackstats.dwell_seconds(s, t) if s is not None else 0.0, float(tr.conf),
                         s.speed_m if s is not None else 0.0)
    return FrameContext(ps, t, snap, bars)


def _gestures(camera_state, pst: PackState, ps: PoseSettings, persons, hands, t: float) -> None:
    landmarks = getattr(camera_state, "hand_detector", None) is not None
    info = [(key, det.keypoints, det.kp_visible, tr.obs) for key, tp, det, tr in persons if det.keypoints is not None]
    owner = PA.assign_hands(hands, info) if hands and info else {}
    by_person: Dict[Any, List[tuple]] = {}
    for hi, key in owner.items():
        shape = PA.hand_gesture(hands[hi])
        if shape is not None:
            by_person.setdefault(key, []).append((shape, hands[hi][0]))
    active = {}
    for key, tp, det, tr in persons:
        conds = gesture_conditions(tp, det, by_person.get(key, []), tr.obs_t, landmarks)
        conds &= ps.enabled_gestures
        tp.gestures_now = frozenset(conds)
        if conds:
            active[key] = conds
    for key, g in pst.machine.update(t, active):
        pst.seq += 1
        G.fire(camera_state, key, g, ps.enabled_gestures, pst.recent, pst.seq)


# ----------------------------------------------------------------------------- rendering
@dataclass(frozen=True)
class _Snap:
    posture: Optional[str]
    action: Optional[str]
    yaw: Optional[float]
    pitch: float
    looking: bool
    head_size: float
    fall_until: float
    gestures: frozenset


class FrameContext:
    def __init__(self, ps: PoseSettings, t: float, snap: Dict[Any, _Snap], bars: Dict[Any, tuple]):
        self.ps, self.t, self.snap, self.bars = ps, t, snap, bars
        self.iff: Dict[Any, str] = {}

    # ---- IFF
    def color_for(self, iff: str) -> BGR:
        return {"friend": self.ps.color_friend, "foe": self.ps.color_foe}.get(iff, self.ps.color_unknown)

    def resolve_iff(self, rc, detections) -> None:
        from ..recognition import overlay as recog
        infos = getattr(rc, "track_info", None) or {}
        for d in detections:
            if not _is_person(d):
                continue
            ti = infos.get(d.obj_id) if d.obj_id != -1 else None
            name = getattr(ti, "name", None) if ti is not None else None
            if not name and rc is not None and getattr(rc, "faces", None):
                name = recog.name_for(rc, d)
            self.iff[d.track_key] = PA.iff_for(name, getattr(ti, "person_id", None),
                                               getattr(ti, "face_uid", None) if ti is not None else None)

    def palettes(self, rc, detections, palettes):
        if not self.ps.iff_colors:
            return palettes
        self.resolve_iff(rc, detections)
        out = list(palettes)
        for i, d in enumerate(detections):
            iff = self.iff.get(d.track_key)
            if iff is None:
                continue
            c = self.color_for(iff)
            out[i] = DetectionColors(box=c, text=mix(c, WHITE, 0.35), fill=c, skeleton=c)
        return out

    def chams(self, det, default: BGR) -> BGR:
        iff = self.iff.get(det.track_key) if self.ps.iff_colors else None
        return self.color_for(iff) if iff is not None else default

    # ---- per detection
    def draw(self, frame, det, pal, st, t: float, placed: list) -> None:
        sn = self.snap.get(det.track_key)
        if self.ps.status_bars:
            self._bars(frame, det, st)
        if sn is not None and sn.fall_until > t:
            self._fall_flash(frame, det, st, t)
        hp = PA.head_pose(det.keypoints, det.kp_visible, det.height) if det.keypoints is not None else None
        if self.ps.gaze and sn is not None and sn.yaw is not None and hp is not None:
            self._gaze(frame, hp, sn, pal.box, st)
        if self.ps.iff_colors:
            iff = self.iff.get(det.track_key)
            if iff is not None:
                self._iff_marker(frame, det, hp, iff, st, placed, sn is not None and sn.looking)
        elif self.ps.gaze and sn is not None and sn.looking and hp is not None:
            self._eye(frame, (int(hp.origin[0] + hp.head_size * 0.75), int(hp.origin[1] - hp.head_size * 0.6)), st, placed)

    def _bars(self, frame, det, st) -> None:
        vals = self.bars.get(det.track_key)
        if vals is None:
            return
        dwell, conf, speed = vals
        ps = self.ps
        items = []
        if ps.bar_dwell:
            f = min(1.0, dwell / ps.loiter_seconds)
            col = mix(WHITE, (31, 210, 255), min(1.0, f * 1.6)) if f < 0.6 else mix((31, 210, 255), ALARM_RED, (f - 0.6) / 0.4)
            items.append((f, col))
        if ps.bar_conf:
            items.append((max(0.0, min(1.0, conf)), WHITE))
        if ps.bar_speed:
            items.append((min(1.0, speed / SPEED_BAR_MAX), SPEED_CYAN))
        if not items:
            return
        h, w = frame.shape[:2]
        bw = max(3, int(round(4 * st.ui)))
        gap = max(2, int(round(2 * st.ui)))
        y0, y1 = det.y1, det.y2
        if y1 - y0 < 24:
            y0 = y1 - 24
        total = len(items) * bw + (len(items) - 1) * gap
        x = det.x1 - st.gap - total
        if x < 0:
            x = det.x2 + st.gap
        x = min(max(0, x), max(0, w - total))
        for f, col in items:
            R.blend_rect(frame, x - 1, y0 - 1, x + bw + 1, y1 + 1, ANTHRACITE, 0.6 * det.alpha)
            top = int(round(y1 - (y1 - y0) * f))
            if top < y1:
                cv2.rectangle(frame, (x, top), (x + bw - 1, y1 - 1), col, -1)
            x += bw + gap

    def _fall_flash(self, frame, det, st, t: float) -> None:
        if int(t * 4.0) % 2 == 0:
            R.blend_rect(frame, det.x1, det.y1, det.x2, det.y2, ALARM_RED, 0.22)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), ALARM_RED, st.t + 2, AA)

    def _gaze(self, frame, hp, sn: _Snap, color: BGR, st) -> None:
        yaw, pitch = math.radians(sn.yaw), math.radians(sn.pitch)
        dx, dy = math.sin(yaw), math.sin(pitch) * 0.8
        mag = math.hypot(dx, dy)
        if mag < 0.25:
            return                                    # looking (roughly) into the camera: no cone
        ux, uy = dx / mag, dy / mag
        length = hp.head_size * (1.5 + 3.0 * min(1.0, mag))
        half = math.radians(18.0)
        ox, oy = hp.origin
        pts = [(ox, oy)]
        for a in (-half, half):
            ca, sa = math.cos(a), math.sin(a)
            pts.append((ox + (ux * ca - uy * sa) * length, oy + (ux * sa + uy * ca) * length))
        poly = np.array(pts, np.int32)
        x0, y0 = poly.min(axis=0) - 2
        x1, y1 = poly.max(axis=0) + 3

        def paint(ov, offx, offy):
            cv2.fillPoly(ov, [poly - (offx, offy)], color, AA)

        R._blend_layer(frame, [(int(x0), int(y0), int(x1), int(y1))], paint, 0.22)
        tip = (int(ox + ux * length), int(oy + uy * length))
        cv2.line(frame, (int(ox), int(oy)), tip, mix(color, WHITE, 0.4), st.thin, AA)

    def _iff_marker(self, frame, det, hp, iff: str, st, placed: list, looking: bool) -> None:
        r = max(6, int(round(8 * st.ui)))
        cx = int(hp.origin[0]) if hp is not None else det.cx
        head_top = det.y1
        if hp is not None:
            head_top = min(det.y1, int(hp.origin[1] - hp.head_size * 0.9))
        cy = head_top - r - st.gap
        h, w = frame.shape[:2]
        cx, cy = min(max(r, cx), w - r - 1), min(max(r, cy), h - r - 1)
        col = self.color_for(iff)
        cv2.circle(frame, (cx, cy), r + 1, ANTHRACITE, -1, AA)
        cv2.circle(frame, (cx, cy), r, col, -1, AA)
        k = max(1, int(round(r * 0.28)))
        if iff == "friend":
            pts = np.array(((cx - r * 0.45, cy), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.4)), np.int32)
            cv2.polylines(frame, [pts], False, ANTHRACITE, k, AA)
        elif iff == "foe":
            cv2.line(frame, (cx, int(cy - r * 0.55)), (cx, int(cy + r * 0.12)), ANTHRACITE, k, AA)
            cv2.circle(frame, (cx, int(cy + r * 0.48)), max(1, k // 2 + 1), ANTHRACITE, -1, AA)
        else:
            scale = r / 15.0
            (tw, th), _ = cv2.getTextSize("?", cv2.FONT_HERSHEY_SIMPLEX, scale, k)
            cv2.putText(frame, "?", (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, ANTHRACITE, k, AA)
        placed.append((cx - r - 1, cy - r - 1, cx + r + 2, cy + r + 2))
        if looking:
            self._eye(frame, (cx + 2 * r + st.gap, cy), st, placed)

    @staticmethod
    def _eye(frame, c, st, placed: list) -> None:
        rx = max(6, int(round(8 * st.ui)))
        ry = max(3, int(rx * 0.55))
        cv2.ellipse(frame, c, (rx + 1, ry + 1), 0, 0, 360, ANTHRACITE, -1, AA)
        cv2.ellipse(frame, c, (rx, ry), 0, 0, 360, WHITE, max(1, st.thin), AA)
        cv2.circle(frame, c, max(2, ry - 1), WHITE, -1, AA)
        placed.append((c[0] - rx - 1, c[1] - ry - 1, c[0] + rx + 2, c[1] + ry + 2))

    # ---- posture / action pills (below the box) and FALL tag
    def draw_tags(self, frame, detections, st, placed: list, background: bool) -> None:
        if not self.ps.posture:
            return
        h, w = frame.shape[:2]
        for d in detections:
            sn = self.snap.get(d.track_key)
            if sn is None or d.alpha < 0.5:
                continue
            falling = sn.fall_until > self.t
            text = "FALL" if falling else (sn.action or sn.posture)
            if not text:
                continue
            accent = ALARM_RED if falling else self.color_for(self.iff[d.track_key]) if d.track_key in self.iff \
                else WHITE
            runs = [(text.replace("_", " ").upper(), True, WHITE if falling else mix(WHITE, ANTHRACITE, 0.08))]
            if sn.action and sn.posture and not falling and sn.action not in ("walking", "running"):
                runs += [("  ", False, WHITE), (sn.posture.upper(), False, mix(WHITE, ANTHRACITE, 0.35))]
            tag = R.make_tag(runs, st, accent, d.alpha, background)
            if tag is None:
                continue
            x = min(max(0, d.x1), max(0, w - tag.w))
            y = d.y2 + st.gap
            if y + tag.h > h:
                y = max(0, d.y2 - tag.h - st.gap)
            for _ in range(4):
                hit = next(((px0, py0, px1, py1) for px0, py0, px1, py1 in placed
                            if x < px1 and x + tag.w > px0 and y < py1 and y + tag.h > py0), None)
                if hit is None:
                    break
                y = hit[3] + 2
            tag.x, tag.y = x, min(y, max(0, h - tag.h))
            placed.append((tag.x, tag.y, tag.x + tag.w, tag.y + tag.h))
            R.draw_tag(frame, tag)


# ----------------------------------------------------------------------------- /targets payload
def decorate_targets(camera_state, payload: dict) -> dict:
    """Additive fields: ``iff`` (persons), ``posture``, ``action``, ``looking_at_camera``,
    ``gaze`` {yaw, pitch}, ``fall``, ``gesture`` (holding now) and top-level ``gestures``."""
    raw = getattr(camera_state, "esp_settings", None) or {}
    ps = PoseSettings.from_mapping(raw)
    pst = getattr(camera_state, "esp_posepack", None)
    tracks = pst.tracks if isinstance(pst, PackState) else {}
    import time as _time
    t = _time.perf_counter()
    for tg in payload.get("targets", []):
        if str(tg.get("label", "")).lower() == "person":
            ident = tg.get("identity") or {}
            tg["iff"] = PA.iff_for(ident.get("name"), ident.get("person_id"), ident.get("face_id"))
        tp = tracks.get(tg.get("id"))
        if tp is None:
            continue
        if ps.posture:
            if tp.posture.value:
                tg["posture"] = tp.posture.value
            if tp.action.value:
                tg["action"] = tp.action.value
            if tp.fall_until > t:
                tg["fall"] = True
        if ps.gaze and tp.yaw is not None:
            tg["gaze"] = {"yaw": round(tp.yaw, 1), "pitch": round(tp.pitch, 1)}
            tg["looking_at_camera"] = bool(tp.looking)
        if ps.gestures and tp.gestures_now:
            tg["gesture"] = sorted(tp.gestures_now)
    if ps.gestures and isinstance(pst, PackState):
        payload["gestures"] = G.recent_payload(pst.recent)
    return payload


# ----------------------------------------------------------------------------- clip pre-roll for fall alarms
def _wants_buffer(camera_id: str) -> bool:
    try:
        from .. import state
        cs = state.get_camera_state(camera_id)
    except Exception:
        return False
    if cs is None or not isinstance(getattr(cs, "esp_posepack", None), PackState):
        return False
    ps = PoseSettings.from_mapping(cs.esp_settings)
    return ps.posture and ps.fall_alarm and ps.alarms


_hook_installed = False


def _install_buffer_hook() -> None:
    global _hook_installed
    if _hook_installed:
        return
    _hook_installed = True
    try:
        from ..alerts import clips, engine  # noqa: F401  (engine installs its hook first)
    except Exception:
        return
    prev = clips.wants_buffer
    if getattr(prev, "_posepack", False):
        return

    def wants(camera_id: str) -> bool:
        return prev(camera_id) or _wants_buffer(camera_id)

    wants._posepack = True
    clips.wants_buffer = wants
