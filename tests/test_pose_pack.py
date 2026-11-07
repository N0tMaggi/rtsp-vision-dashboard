"""Tests for the pose ESP pack: posture / action classification, fall detection,
gaze yaw sign, gesture state machine (holds, release, cooldown), hand shapes,
IFF mapping, settings validation and the ESP / targets integration.

Synthetic keypoints only (COCO-17 and MediaPipe-21 layouts drawn by hand).

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-test-"))
os.environ.setdefault("SECURITYCAM_CAMERAS",
                      "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam import config, state, targets  # noqa: E402
from securitycam.esp import gestures as G  # noqa: E402
from securitycam.esp import motion  # noqa: E402
from securitycam.esp import pose_analysis as PA  # noqa: E402
from securitycam.esp import posepack as PP  # noqa: E402
from securitycam.esp.detections import Detection  # noqa: E402
from securitycam.esp.pipeline import draw_custom_esp  # noqa: E402
from securitycam.validation import sanitize_esp_patch  # noqa: E402

CAM = "front"


# ----------------------------------------------------------------------------- synthetic people
def standing(dx=0.0, dy=0.0):
    pts = {
        0: (150, 130), 1: (156, 124), 2: (144, 124), 3: (164, 128), 4: (136, 128),
        5: (175, 170), 6: (125, 170), 7: (185, 220), 8: (115, 220), 9: (188, 265), 10: (112, 265),
        11: (165, 260), 12: (135, 260), 13: (165, 330), 14: (135, 330), 15: (165, 395), 16: (135, 395),
    }
    return {k: (x + dx, y + dy) for k, (x, y) in pts.items()}, (100 + dx, 100 + dy, 200 + dx, 400 + dy)


def sitting():
    pts, _ = standing(dy=60)
    pts.update({11: (165, 320), 12: (135, 320), 13: (165, 335), 14: (135, 335), 15: (165, 395), 16: (135, 395)})
    return pts, (100, 160, 200, 400)


def crouching():
    pts, _ = standing(dy=80)
    pts.update({11: (165, 330), 12: (135, 330), 13: (172, 345), 14: (128, 345), 15: (165, 375), 16: (135, 375)})
    return pts, (100, 180, 200, 380)


def lying():
    pts = {0: (110, 375), 1: (112, 370), 2: (112, 380), 5: (150, 370), 6: (150, 390),
           11: (240, 372), 12: (240, 388), 13: (300, 375), 14: (300, 385), 15: (360, 378), 16: (360, 384)}
    return pts, (100, 350, 370, 400)


def arrays(pts):
    kps = np.zeros((17, 2), np.int32)
    vis = np.zeros(17, bool)
    for k, (x, y) in pts.items():
        kps[k] = (int(x), int(y))
        vis[k] = True
    return kps, vis


def det_of(pts, box, tid=1):
    kps, vis = arrays(pts)
    x1, y1, x2, y2 = box
    return Detection(index=0, obj_id=tid, cls_id=0, cls_name="person", x1=x1, y1=y1, x2=x2, y2=y2,
                     cx=(x1 + x2) // 2, cy=(y1 + y2) // 2, conf=0.9, keypoints=kps, kp_visible=vis, track_key=tid)


def settings(**kw):
    raw = dict(config.DEFAULT_ESP_SETTINGS)
    raw.update(kw)
    return PP.PoseSettings.from_mapping(raw)


# ----------------------------------------------------------------------------- posture
class Posture(unittest.TestCase):
    def classify(self, maker):
        pts, box = maker()
        kps, vis = arrays(pts)
        return PA.classify_posture(kps, vis, box)

    def test_classes(self):
        self.assertEqual(self.classify(standing), "standing")
        self.assertEqual(self.classify(sitting), "sitting")
        self.assertEqual(self.classify(crouching), "crouching")
        self.assertEqual(self.classify(lying), "lying")

    def test_no_keypoints_wide_box_is_lying(self):
        vis = np.zeros(17, bool)
        self.assertEqual(PA.classify_posture(np.zeros((17, 2)), vis, (0, 0, 300, 100)), "lying")
        self.assertIsNone(PA.classify_posture(np.zeros((17, 2)), vis, (0, 0, 100, 300)))

    def test_hands_up(self):
        pts, _ = standing()
        self.assertFalse(PA.hands_up(*arrays(pts)))
        pts.update({9: (180, 90), 10: (120, 92)})
        self.assertTrue(PA.hands_up(*arrays(pts)))

    def test_motion_action(self):
        self.assertIsNone(PA.motion_action(0.1, "standing"))
        self.assertEqual(PA.motion_action(0.6, "standing"), "walking")
        self.assertEqual(PA.motion_action(2.0, "standing"), "running")
        self.assertIsNone(PA.motion_action(2.0, "lying"))

    def test_debounce_ignores_flicker(self):
        tp = PP.TrackPose()
        ps = settings(posture=True)
        st_pts, st_box = standing()
        si_pts, si_box = sitting()
        t = 0.0
        for _ in range(10):
            PP.fold_observation(tp, det_of(st_pts, st_box), st_box, t, 0.0, ps)
            t += 0.1
        self.assertEqual(tp.posture.value, "standing")
        PP.fold_observation(tp, det_of(si_pts, si_box), si_box, t, 0.0, ps)     # one odd frame
        t += 0.1
        PP.fold_observation(tp, det_of(st_pts, st_box), st_box, t, 0.0, ps)
        self.assertEqual(tp.posture.value, "standing")
        for _ in range(8):
            t += 0.1
            PP.fold_observation(tp, det_of(si_pts, si_box), si_box, t, 0.0, ps)
        self.assertEqual(tp.posture.value, "sitting")

    def test_waving_from_wrist_oscillation(self):
        tp = PP.TrackPose()
        ps = settings(posture=True)
        base, box = standing()
        t = 0.0
        waved = False
        for i in range(30):
            pts = dict(base)
            pts[9] = (175 + 40 * math.sin(2 * math.pi * 1.5 * t), 110)      # left wrist raised, swinging
            pts[7] = (180, 150)
            PP.fold_observation(tp, det_of(pts, box), box, t, 0.0, ps)
            waved = waved or tp.waving
            t += 0.1
        self.assertTrue(waved)
        self.assertEqual(tp.action.value, "waving")
        # held still: no wave
        tp2 = PP.TrackPose()
        pts = dict(base)
        pts[9] = (175, 110)
        for i in range(30):
            PP.fold_observation(tp2, det_of(pts, box), box, i * 0.1, 0.0, ps)
        self.assertFalse(tp2.waving)


# ----------------------------------------------------------------------------- fall
class Fall(unittest.TestCase):
    def run_sequence(self, lie_at, until=3.0):
        tp = PP.TrackPose()
        ps = settings(posture=True)
        events = []
        st_pts, st_box = standing()
        ly_pts, ly_box = lying()
        t = 0.0
        while t <= until:
            pts, box = (st_pts, st_box) if t < lie_at else (ly_pts, ly_box)
            if PP.fold_observation(tp, det_of(pts, box), box, t, 0.0, ps) == "fall":
                events.append(round(t, 2))
            t = round(t + 0.1, 3)
        return tp, events

    def test_fast_fall_raises_one_event(self):
        tp, events = self.run_sequence(lie_at=1.0)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0], 1.0 + PP.FALL_CONFIRM_S + 0.11)
        self.assertEqual(tp.posture.value, "lying")
        self.assertGreater(tp.fall_until, events[0])

    def test_slow_lying_down_is_no_fall(self):
        tp = PP.TrackPose()
        ps = settings(posture=True)
        st_pts, st_box = standing()
        si_pts, si_box = sitting()
        ly_pts, ly_box = lying()
        seq = [(st_pts, st_box)] * 10 + [(si_pts, si_box)] * 30 + [(ly_pts, ly_box)] * 20
        events = [PP.fold_observation(tp, det_of(p, b), b, i * 0.1, 0.0, ps) for i, (p, b) in enumerate(seq)]
        self.assertNotIn("fall", events)
        self.assertEqual(tp.posture.value, "lying")

    def test_fall_emits_alarm_event(self):
        from securitycam.alerts import engine
        cs = SimpleNamespace(camera_id=CAM, label="Front")
        with mock.patch.object(engine.CameraEngine, "emit", return_value=True) as emit:
            PP._emit_fall(cs, 7, det_of(*lying(), tid=7), settings(posture=True))
            self.assertEqual(emit.call_args[0][1], "fall")
            PP._emit_fall(cs, 7, det_of(*lying(), tid=7), settings(posture=True, fall_alarm=False))
            PP._emit_fall(cs, 7, det_of(*lying(), tid=7), settings(posture=True, alarms=False))
            self.assertEqual(emit.call_count, 1)


# ----------------------------------------------------------------------------- gaze
class Gaze(unittest.TestCase):
    def test_yaw_sign(self):
        pts, _ = standing()
        hp = PA.head_pose(*arrays(pts))
        self.assertLess(abs(hp.yaw), 5)
        self.assertTrue(PA.looking_at_camera(hp.yaw, hp.both_eyes, False))
        pts[0] = (160, 130)                     # nose toward the image's right ear
        right = PA.head_pose(*arrays(pts))
        self.assertGreater(right.yaw, 20)
        pts[0] = (140, 130)
        left = PA.head_pose(*arrays(pts))
        self.assertLess(left.yaw, -20)
        self.assertFalse(PA.looking_at_camera(left.yaw, left.both_eyes, False))

    def test_profile_and_fallbacks(self):
        prof = {0: (170, 130), 4: (140, 128), 2: (160, 124)}    # only the right ear / eye, nose to the right
        self.assertGreater(PA.head_pose(*arrays(prof)).yaw, 40)
        self.assertIsNone(PA.head_pose(*arrays({3: (150, 128), 4: (130, 128)})))   # back of the head

    def test_looking_hysteresis(self):
        self.assertFalse(PA.looking_at_camera(22.0, True, False))
        self.assertTrue(PA.looking_at_camera(22.0, True, True))
        self.assertFalse(PA.looking_at_camera(0.0, False, True))


# ----------------------------------------------------------------------------- gestures
class Gestures(unittest.TestCase):
    def test_hold_fires_once(self):
        m = G.GestureMachine()
        fired = []
        t = 0.0
        while t <= 5.0:
            fired += [(round(t, 1), g) for _, g in m.update(t, {1: {"wave"}})]
            t = round(t + 0.1, 3)
        self.assertEqual(fired, [(2.0, "wave")])

    @staticmethod
    def hold(m, t0, t1, gs, key=1):
        fired, t = [], t0
        while t <= t1 + 1e-9:
            fired += [(round(t, 1), g) for _, g in m.update(t, {key: set(gs)})]
            t = round(t + 0.1, 3)
        return fired

    def test_dropout_is_bridged_but_release_restarts(self):
        m = G.GestureMachine(cooldown_s=0.0)
        self.assertEqual(self.hold(m, 0.0, 0.5, ["thumbs_up"]), [])
        self.assertEqual(self.hold(m, 0.6, 0.8, []), [])                   # 0.3 s dropout: bridged
        self.assertEqual(self.hold(m, 0.9, 1.5, ["thumbs_up"]), [(1.0, "thumbs_up")])
        self.assertEqual(self.hold(m, 1.6, 4.0, ["thumbs_up"]), [])         # still held: no repeat
        self.assertEqual(self.hold(m, 4.1, 5.0, []), [])                    # released
        self.assertEqual(self.hold(m, 5.1, 6.5, ["thumbs_up"]), [(6.1, "thumbs_up")])

    def test_long_dropout_restarts_hold(self):
        m = G.GestureMachine(cooldown_s=0.0)
        self.assertEqual(self.hold(m, 0.0, 0.8, ["thumbs_up"]), [])
        self.assertEqual(self.hold(m, 0.9, 1.6, []), [])                    # 0.8 s gap
        self.assertEqual(self.hold(m, 1.7, 2.6, ["thumbs_up"]), [])
        self.assertEqual(self.hold(m, 2.7, 2.7, ["thumbs_up"]), [(2.7, "thumbs_up")])

    def test_cooldown(self):
        m = G.GestureMachine(cooldown_s=6.0)
        fired = []
        for i in range(0, 100):
            t = i * 0.1
            active = {1: {"palm"}} if (t <= 2.0 or t >= 3.0) else {}
            fired += [round(t, 1) for _, g in m.update(t, active)]
        # first at 1.5 s; the second hold (from 3.0 s) is ready at 4.5 s but waits for the cooldown
        self.assertEqual(fired, [1.5, 7.5])

    def test_plan_without_ptz(self):
        cs = SimpleNamespace(camera_id=CAM, esp_locked=None)
        with mock.patch.object(G, "_ptz_ready", return_value=False):
            msg, ok, job = G.plan(cs, 3, "wave", {"wave", "palm", "thumbs_up"})
        self.assertFalse(ok)
        self.assertIsNone(job)
        msg, ok, job = G.plan(cs, 3, "thumbs_up", {"thumbs_up"})
        self.assertTrue(ok)
        msg, ok, job = G.plan(cs, 3, "thumbs_up", set())
        self.assertFalse(ok)

    def test_fire_wave_starts_autotrack_and_logs(self):
        from securitycam.ptz import autotrack
        cs = SimpleNamespace(camera_id=CAM, esp_locked=None, label="Front")
        recent = __import__("collections").deque(maxlen=8)
        old = G.RUN_ASYNC
        G.RUN_ASYNC = False
        try:
            with mock.patch.object(G, "_ptz_ready", return_value=True), \
                    mock.patch.object(autotrack, "set_target", return_value=({}, None)) as st, \
                    mock.patch.object(G, "log_event", return_value=True) as log:
                rec = G.fire(cs, 4, "wave", {"wave"}, recent, 1)
            st.assert_called_once_with(CAM, 4)
            log.assert_called_once()
        finally:
            G.RUN_ASYNC = old
        self.assertEqual(rec["message"], "Auto-track started")
        self.assertEqual(len(G.recent_payload(recent)), 1)
        self.assertEqual(G.recent_payload(recent, now=time.time() + 60), [])

    def test_hand_shapes(self):
        palm = [(100, 200), (80, 190), (70, 175), (60, 165), (52, 155)]
        for x in (85, 95, 105, 115):
            palm += [(x, 160), (x, 140), (x, 125), (x, 110)]
        self.assertEqual(PA.hand_gesture(palm), "palm")
        thumbs = [(100, 200), (95, 185), (95, 165), (96, 145), (97, 125),
                  (110, 165), (128, 160), (125, 175), (115, 182),
                  (112, 172), (130, 168), (127, 182), (117, 188),
                  (112, 180), (128, 177), (125, 190), (116, 194),
                  (110, 188), (124, 186), (122, 196), (114, 198)]
        self.assertEqual(PA.hand_gesture(thumbs), "thumbs_up")
        self.assertIsNone(PA.hand_gesture(thumbs[:10]))

    def test_assign_hands(self):
        a_pts, a_box = standing()
        b_pts, b_box = standing(dx=300)
        ka, va = arrays(a_pts)
        kb, vb = arrays(b_pts)
        hand = [(110, 262)] + [(0, 0)] * 20               # wrist landmark next to person A's right wrist
        far = [(495, 300)] + [(0, 0)] * 20                # inside person B's box, far from the wrists
        owner = PA.assign_hands([hand, far], [("a", ka, va, a_box), ("b", kb, vb, b_box)])
        self.assertEqual(owner, {0: "a", 1: "b"})

    def test_palm_needs_stillness_and_no_wave(self):
        tp = PP.TrackPose()
        d = det_of(*standing())
        self.assertEqual(PP.gesture_conditions(tp, d, [("palm", (180, 100))], 0.0, True), {"palm"})
        self.assertEqual(PP.gesture_conditions(tp, d, [("palm", (240, 100))], 0.1, True), set())   # moved
        tp.waves["l"].reversals.append(0.15)
        self.assertEqual(PP.gesture_conditions(tp, d, [("palm", (240, 100))], 0.2, True), set())   # waving


# ----------------------------------------------------------------------------- identity / settings
class IffAndSettings(unittest.TestCase):
    def test_iff_mapping(self):
        self.assertEqual(PA.iff_for("Anna", "anna", None), "friend")
        self.assertEqual(PA.iff_for(None, "anna", 3), "friend")
        self.assertEqual(PA.iff_for(None, None, 3), "foe")
        self.assertEqual(PA.iff_for(None, None, None), "unknown")

    def test_defaults_off(self):
        ps = PP.PoseSettings.from_mapping(config.DEFAULT_ESP_SETTINGS)
        self.assertFalse(ps.any)
        self.assertEqual(ps.enabled_gestures, {"wave", "palm", "thumbs_up"})

    def test_validation(self):
        out = sanitize_esp_patch({"iff_colors": "on", "iff_color_foe": "#ABCDEF", "iff_color_friend": "green",
                                  "gestures": 1, "gesture_wave": "maybe", "status_bars": True, "bar_speed": "off"})
        self.assertEqual(out, {"iff_colors": True, "iff_color_foe": "#abcdef", "gestures": True,
                               "status_bars": True, "bar_speed": False})

    def test_hand_settings(self):
        from securitycam.esp.settings import EspSettings
        s = EspSettings.from_mapping(config.DEFAULT_ESP_SETTINGS)
        self.assertIs(PP.hand_settings(s, {}), s)
        self.assertTrue(PP.hand_settings(s, {"gestures": True}).hand_skeleton)


# ----------------------------------------------------------------------------- integration
class _Boxes:
    def __init__(self, rows):
        self.data = np.array([[x1, y1, x2, y2, t, 0.9, 0] for t, x1, y1, x2, y2 in rows], np.float32).reshape(-1, 7)
        self.xyxy, self.id, self.conf, self.cls = self.data[:, :4], self.data[:, 4], self.data[:, 5], self.data[:, 6]

    def __len__(self):
        return len(self.data)


def fake_pose_results(people):
    """people: [(track id, pts dict, box)]"""
    rows, xy = [], []
    for tid, pts, box in people:
        rows.append((tid, *box))
        kps, vis = arrays(pts)
        xy.append(np.where(vis[:, None], kps, 0).astype(np.float32))
    kp = SimpleNamespace(xy=np.array(xy, np.float32), conf=np.ones((len(xy), 17), np.float32))
    return [SimpleNamespace(boxes=_Boxes(rows), names={0: "person"}, keypoints=kp)]


class Integration(unittest.TestCase):
    def setUp(self):
        self.cs = state.get_camera_state(CAM)
        self.cs.esp_settings.clear()
        self.cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        self.cs.esp_settings["smoothing"] = 0.0
        self.cs.yolo_enabled = True
        self.cs.esp_motion = motion.MotionState()
        self.cs.esp_posepack = None
        self.cs.view_solo, self.cs.view_hidden = None, []

    def render(self, people, n=6):
        frame = np.zeros((720, 1280, 3), np.uint8)
        out = None
        for i in range(n):
            res = fake_pose_results(people)
            with self.cs.frame_lock:
                self.cs.latest_frame = frame
                self.cs.latest_results = res
            t = time.perf_counter()
            out = draw_custom_esp(self.cs, frame.copy(), res, frame_ts=t, results_ts=t)
            time.sleep(0.12)
        return out

    def test_off_by_default_keeps_no_state(self):
        self.render([(1, *standing())], n=2)
        self.assertIsNone(self.cs.esp_posepack)
        tg = targets.targets_payload(self.cs)["targets"][0]
        self.assertEqual(tg["iff"], "unknown")
        self.assertNotIn("posture", tg)
        self.assertNotIn("gestures", targets.targets_payload(self.cs))

    def test_everything_on_draws_and_exposes_fields(self):
        self.cs.esp_settings.update({"iff_colors": True, "gaze": True, "posture": True, "status_bars": True,
                                     "gestures": True, "chams": True})
        pts, box = standing()
        big = {k: (x * 2, y * 1.5) for k, (x, y) in pts.items()}
        bbox = (box[0] * 2, int(box[1] * 1.5), box[2] * 2, int(box[3] * 1.5))
        with mock.patch("securitycam.esp.sensors.deps.ensure_hand_detector", return_value=None):
            out = self.render([(1, big, bbox)])
        self.assertGreater(int(out.sum()), 0)
        payload = targets.targets_payload(self.cs)
        tg = payload["targets"][0]
        self.assertEqual(tg["iff"], "unknown")
        self.assertEqual(tg["posture"], "standing")
        self.assertIn("gaze", tg)
        self.assertTrue(tg["looking_at_camera"])
        self.assertEqual(payload["gestures"], [])
        # IFF color reaches the box: the unidentified color (yellow) is drawn
        yellow = np.array((31, 210, 255), np.uint8)
        self.assertTrue(np.any(np.all(out == yellow, axis=2)))

    def test_fall_flash_in_payload(self):
        self.cs.esp_settings.update({"posture": True, "fall_alarm": False})
        self.render([(1, *standing())], n=4)
        self.render([(1, *lying())], n=5)
        tg = targets.targets_payload(self.cs)["targets"][0]
        self.assertEqual(tg["posture"], "lying")
        self.assertTrue(tg.get("fall"))


if __name__ == "__main__":
    unittest.main()
