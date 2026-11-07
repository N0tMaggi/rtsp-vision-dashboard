"""Pose pack: pure, per-observation analysis of COCO-17 keypoints and hand landmarks.

No state and no drawing here (see ``posepack.py``); everything takes plain
arrays so it can be tested with synthetic keypoints.

Conventions
-----------
* Image coordinates, y grows downward. COCO "left" is the subject's left.
* ``yaw`` in degrees: 0 = facing the camera, positive = the face turns toward
  the image's right edge, +-90 = profile. ``pitch`` in degrees: positive = down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

POSTURES = ("standing", "sitting", "lying", "crouching")
ACTIONS = ("walking", "running", "waving", "hands_up")
UPRIGHT = frozenset({"standing"})

LYING_ANGLE = 55.0          # torso angle from vertical (deg) above which the person lies
LYING_ASPECT = 1.4          # box w / h above which a person without torso keypoints lies
WALK_SPEED = 0.35           # body heights / s
RUN_SPEED = 1.5
LOOK_ENTER, LOOK_LEAVE = 18.0, 28.0   # |yaw| hysteresis for "looking at camera" (deg)

Point = Tuple[float, float]


def _pt(kps, vis, i: int) -> Optional[Point]:
    if kps is None or vis is None or i >= len(kps) or i >= len(vis) or not bool(vis[i]):
        return None
    return float(kps[i][0]), float(kps[i][1])


def _mid(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is not None and b is not None:
        return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
    return a if a is not None else b


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ----------------------------------------------------------------------------- gaze
@dataclass(frozen=True)
class HeadPose:
    yaw: float                  # deg, + = toward image right
    pitch: float                # deg, + = down
    both_eyes: bool
    head_size: float            # px, rough head width
    origin: Point               # nose (px)


def head_pose(kps, vis, box_h: float = 0.0) -> Optional[HeadPose]:
    """Rough head direction from nose / eyes / ears. ``None`` without a visible nose."""
    nose = _pt(kps, vis, NOSE)
    if nose is None:
        return None
    le, re_ = _pt(kps, vis, L_EYE), _pt(kps, vis, R_EYE)
    lear, rear = _pt(kps, vis, L_EAR), _pt(kps, vis, R_EAR)
    eye_span = abs(le[0] - re_[0]) if le and re_ else 0.0
    ear_span = abs(lear[0] - rear[0]) if lear and rear else 0.0
    one_ear = lear or rear
    head = max(ear_span, eye_span * 2.2, _dist(nose, one_ear) * 1.6 if one_ear and not ear_span else 0.0,
               0.08 * box_h, 6.0)

    if ear_span > 2.0:
        mid = (lear[0] + rear[0]) * 0.5
        yaw = math.degrees(math.asin(_clamp((nose[0] - mid) / (ear_span * 0.5), -1.0, 1.0)))
    elif eye_span > 1.0:
        mid = (le[0] + re_[0]) * 0.5
        yaw = math.degrees(math.atan((nose[0] - mid) / (eye_span * 0.5) * 0.6))
    elif one_ear is not None:
        yaw = 70.0 if nose[0] >= one_ear[0] else -70.0          # profile: the face points away from the ear
    elif le or re_:
        eye = le or re_
        yaw = 45.0 if nose[0] >= eye[0] else -45.0
    else:
        return None

    pitch = 0.0
    eye_mid = _mid(le, re_)
    if eye_mid is not None:
        # nose tip sits ~0.22 head widths below the eye line when level
        d = (nose[1] - eye_mid[1]) / head
        pitch = _clamp((d - 0.22) * 220.0, -60.0, 60.0)
    return HeadPose(yaw=_clamp(yaw, -90.0, 90.0), pitch=pitch, both_eyes=bool(le and re_), head_size=head, origin=nose)


def looking_at_camera(yaw: float, both_eyes: bool, was_looking: bool) -> bool:
    if not both_eyes:
        return False
    return abs(yaw) < (LOOK_LEAVE if was_looking else LOOK_ENTER)


# ----------------------------------------------------------------------------- body
@dataclass(frozen=True)
class Body:
    shoulder: Optional[Point]
    hip: Optional[Point]
    torso: float                # px, shoulder-hip distance (0 if unknown)
    angle: Optional[float]      # torso angle from vertical (deg), None if unknown
    head_y: Optional[float]     # top-most visible head point


def body(kps, vis) -> Body:
    sh = _mid(_pt(kps, vis, L_SHOULDER), _pt(kps, vis, R_SHOULDER))
    hip = _mid(_pt(kps, vis, L_HIP), _pt(kps, vis, R_HIP))
    torso, angle = 0.0, None
    if sh is not None and hip is not None:
        dx, dy = sh[0] - hip[0], sh[1] - hip[1]
        torso = math.hypot(dx, dy)
        if torso > 1.0:
            angle = math.degrees(math.atan2(abs(dx), -dy))        # 0 = upright, 90 = flat, 180 = upside down
    heads = [p[1] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if (p := _pt(kps, vis, i)) is not None]
    return Body(sh, hip, torso, angle, min(heads) if heads else None)


def classify_posture(kps, vis, box: Sequence[float]) -> Optional[str]:
    """standing / sitting / lying / crouching, or None when the keypoints are ambiguous."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    b = body(kps, vis)
    if b.angle is None:
        return "lying" if bw / bh > LYING_ASPECT else None
    if b.angle > LYING_ANGLE or (b.angle > 40.0 and bw / bh > 1.6):
        return "lying"
    torso_v = b.hip[1] - b.shoulder[1]
    if torso_v <= 1.0:
        return None
    knee = _mid(_pt(kps, vis, L_KNEE), _pt(kps, vis, R_KNEE))
    ankle = _mid(_pt(kps, vis, L_ANKLE), _pt(kps, vis, R_ANKLE))
    if knee is None:
        return None
    thigh_v = knee[1] - b.hip[1]
    if ankle is None:
        if thigh_v > 0.55 * torso_v:
            return "standing"
        return "sitting" if thigh_v < 0.35 * torso_v else None
    leg_v = ankle[1] - b.hip[1]
    shin_v = ankle[1] - knee[1]
    if thigh_v > 0.55 * torso_v and leg_v > 1.15 * torso_v:
        return "standing"
    if thigh_v < 0.45 * torso_v:
        return "sitting" if shin_v > 0.55 * torso_v else "crouching"
    if leg_v < 0.9 * torso_v:
        return "crouching"
    return None


def hands_up(kps, vis) -> bool:
    """Both wrists above the head."""
    b = body(kps, vis)
    lw, rw = _pt(kps, vis, L_WRIST), _pt(kps, vis, R_WRIST)
    if lw is None or rw is None:
        return False
    if b.head_y is not None:
        top = b.head_y
    elif b.shoulder is not None and b.torso > 0:
        top = b.shoulder[1] - 0.5 * b.torso
    else:
        return False
    return lw[1] < top and rw[1] < top


def raised_wrists(kps, vis) -> list:
    """[(side, wrist, shoulder)] for wrists clearly above their shoulder (side 'l' / 'r')."""
    b = body(kps, vis)
    scale = b.torso if b.torso > 0 else 0.0
    out = []
    for side, wi, si in (("l", L_WRIST, L_SHOULDER), ("r", R_WRIST, R_SHOULDER)):
        w, s = _pt(kps, vis, wi), _pt(kps, vis, si)
        if w is None or s is None:
            continue
        if w[1] < s[1] - 0.1 * scale:
            out.append((side, w, s))
    return out


def motion_action(speed_bh: float, posture: Optional[str]) -> Optional[str]:
    """walking / running from the speed in body heights per second (upright only)."""
    if posture not in UPRIGHT:
        return None
    if speed_bh >= RUN_SPEED:
        return "running"
    if speed_bh >= WALK_SPEED:
        return "walking"
    return None


class WaveDetector:
    """Lateral wrist oscillation (zig-zag with a minimum amplitude) while the wrist is raised."""

    WINDOW_S = 1.6
    MIN_REVERSALS = 2

    def __init__(self):
        self.extreme: Optional[float] = None
        self.direction = 0
        self.reversals: list = []
        self.last_t = -math.inf

    def update(self, t: float, x: Optional[float], amplitude: float) -> bool:
        """``x``: wrist x relative to its shoulder (None = wrist not raised). Returns waving."""
        if x is None:
            self.extreme, self.direction = None, 0
            self.reversals.clear()
        else:
            if self.extreme is None or t - self.last_t > 0.6:
                self.extreme, self.direction = x, 0
            elif self.direction == 0:
                if abs(x - self.extreme) >= amplitude:
                    self.direction = 1 if x > self.extreme else -1     # first swing
                    self.extreme = x
            elif (x - self.extreme) * self.direction > 0:
                self.extreme = x                                       # still moving the same way
            elif (self.extreme - x) * self.direction >= amplitude:
                self.reversals.append(t)                               # turned around
                self.direction = -self.direction
                self.extreme = x
            self.last_t = t
        cutoff = t - self.WINDOW_S
        while self.reversals and self.reversals[0] < cutoff:
            self.reversals.pop(0)
        return len(self.reversals) >= self.MIN_REVERSALS

    def recent(self, t: float, within: float = 1.0) -> bool:
        return bool(self.reversals) and t - self.reversals[-1] <= within


# ----------------------------------------------------------------------------- hands (MediaPipe 21 points)
def _ext(hand, tip: int, pip: int) -> float:
    w = hand[0]
    return _dist(hand[tip], w) / max(1.0, _dist(hand[pip], w))


def hand_gesture(hand: Sequence[Point]) -> Optional[str]:
    """'palm' (open hand, fingers up) / 'thumbs_up' / None from 21 hand landmarks."""
    if hand is None or len(hand) < 21:
        return None
    hand = [(float(p[0]), float(p[1])) for p in hand]
    wrist = hand[0]
    size = _dist(wrist, hand[9])
    if size < 4.0:
        return None
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))
    ext = [_ext(hand, tip, pip) for tip, pip in fingers]
    if all(e > 1.12 for e in ext) and hand[12][1] < wrist[1] - 0.8 * size and _ext(hand, 4, 2) > 1.1:
        return "palm"
    curled = all(e < 1.0 for e in ext)
    tip, ip, mcp = hand[4], hand[3], hand[2]
    others_top = min(hand[i][1] for i in (6, 8, 10, 12, 14, 16, 18, 20))
    if curled and tip[1] < ip[1] < mcp[1] and tip[1] < others_top - 0.3 * size and tip[1] < wrist[1]:
        return "thumbs_up"
    return None


def assign_hands(hands: Sequence[Sequence[Point]], persons: Sequence[tuple]) -> dict:
    """Map hand index -> person key. ``persons``: [(key, kps, vis, box)].

    A hand belongs to the person whose wrist keypoint is nearest to the hand's
    wrist landmark (within ~a torso length); otherwise to the box containing it.
    """
    out = {}
    for hi, hand in enumerate(hands):
        if not hand:
            continue
        hw = (float(hand[0][0]), float(hand[0][1]))
        best, best_d = None, math.inf
        for key, kps, vis, box in persons:
            b = body(kps, vis)
            reach = max(b.torso, 0.15 * max(1.0, box[3] - box[1]))
            for wi in (L_WRIST, R_WRIST):
                w = _pt(kps, vis, wi)
                if w is not None:
                    d = _dist(w, hw)
                    if d < reach and d < best_d:
                        best, best_d = key, d
        if best is None:
            for key, _, _, box in persons:
                if box[0] <= hw[0] <= box[2] and box[1] <= hw[1] <= box[3]:
                    best = key
                    break
        if best is not None:
            out[hi] = best
    return out


# ----------------------------------------------------------------------------- identity
def iff_for(name: Optional[str], person_id: Optional[str], face_id: Optional[int]) -> str:
    """friend (recognized) / foe (unknown face) / unknown (no face identity)."""
    if name or person_id:
        return "friend"
    if face_id is not None:
        return "foe"
    return "unknown"
