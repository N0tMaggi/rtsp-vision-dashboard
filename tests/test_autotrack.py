"""PTZ auto-track: config / persistence, API routes, aim points, control law,
manual pause, target selection and a closed loop against a laggy simulated camera.

No camera or ONVIF device is contacted (fake backend, injected clock).

Run:  python -m pytest -q tests/test_autotrack.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-autotrack-"))
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS",
                      "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from securitycam import config, ptz, state, targets  # noqa: E402
from securitycam.esp import aim as AIM  # noqa: E402
from securitycam.esp import motion  # noqa: E402
from securitycam.esp.pipeline import draw_custom_esp  # noqa: E402
from securitycam.ptz import autotrack as AT  # noqa: E402
from securitycam.ptz.controller import PtzController  # noqa: E402

CAM = "front"


class Backend:
    def __init__(self, zoom=False):
        self.calls = []
        self.zoom = zoom

    def connect(self):
        return {"pan_tilt": True, "zoom": self.zoom, "relative": True, "presets": False, "home": False}

    def get_presets(self):
        return []

    def continuous_move(self, x, y, z):
        self.calls.append(("move", round(x, 4), round(y, 4), round(z, 4)))

    def stop(self):
        self.calls.append(("stop",))

    def relative_move(self, x, y):
        self.calls.append(("relative", x, y))


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def target(tid, box, aim=None, label="person", identity=None, hidden=False):
    t = {"id": tid, "label": label, "conf": 0.9, "box": list(box)}
    if aim is None:
        aim = AIM._norm(AIM.box_points(box, label == "person"), 1.0, 1.0)
    t["aim"] = aim
    t["aim_src"] = "box"
    if identity:
        t["identity"] = identity
    if hidden:
        t["hidden"] = True
    return t


def at_point(tid, u, v, **kw):
    """A person whose every aim point is (u, v)."""
    box = [u - 0.05, v - 0.2, u + 0.05, v + 0.2]
    return target(tid, box, aim={k: [u, v] for k in AT.AIM_KEYS}, **kw)


class Harness:
    """Fake PTZ controller + injected clock + scripted targets payload."""

    def __init__(self, zoom=False, latency_model=False):
        ptz.reset()
        AT.RUN_LOOP = False
        ptz.available_override = True
        self.clock = Clock()
        AT.clock = self.clock
        AT.wall = self.clock
        self.backend = Backend(zoom)
        self.ctl = PtzController(CAM, lambda: self.backend, ptz.settings_for(CAM), clock=self.clock)
        self.ctl._ensure_thread = lambda: None
        self.ctl._do_connect()
        with ptz._lock:
            ptz._controllers[CAM] = self.ctl
        self.cs = state.get_camera_state(CAM)
        self._yolo = self.cs.yolo_enabled
        self.cs.yolo_enabled = True
        self.cs.esp_locked = None
        self.targets = []
        self.locked = None
        self._patches = [
            mock.patch.object(targets, "targets_payload", self._payload),
            mock.patch.object(targets, "set_lock", self._set_lock),
        ]
        if not latency_model:
            # scripted targets don't react to the camera: test the bare control law
            self._patches += [mock.patch.object(AT, "K_INIT", 0.0), mock.patch.object(AT, "SETTLE_S", 0.0)]
        for p in self._patches:
            p.start()

    def _payload(self, cs):
        if self.locked is not None and self.locked not in {t["id"] for t in self.targets}:
            self.locked = None        # real payload drops a lock whose track is gone
        return {"ts": time.time(), "frame": {"w": 1280, "h": 720}, "locked": self.locked,
                "targets": [dict(t) for t in self.targets]}

    def _set_lock(self, cs, tid):
        if tid is None:
            self.locked = None
            return True
        if tid not in {t["id"] for t in self.targets}:
            return False
        self.locked = tid
        return True

    def close(self):
        for p in self._patches:
            p.stop()
        self.cs.yolo_enabled = self._yolo
        self.cs.esp_locked = None
        ptz.reset()
        ptz.available_override = None
        AT.clock, AT.wall, AT.RUN_LOOP = time.monotonic, time.time, True

    def enable(self, **cfg):
        AT.update_config(CAM, {"enabled": True, **cfg})

    def step(self, dt=0.1):
        self.clock.t += dt
        AT.tracker(CAM).step()
        self.ctl.run_pending()
        return AT.status(CAM)

    def sent(self):
        out, self.backend.calls = self.backend.calls, []
        return out


# ----------------------------------------------------------------------------- config / persistence
class ConfigTests(unittest.TestCase):
    def setUp(self):
        ptz.reset()
        AT.RUN_LOOP = False

    def tearDown(self):
        ptz.reset()
        AT.RUN_LOOP = True

    def test_validation(self):
        self.assertEqual(AT.clean_config({"deadzone": 2, "gain": 0.01, "aim": "head", "mode": "auto"}),
                         {"deadzone": 0.45, "gain": 0.2, "aim": "head", "mode": "auto"})
        for bad in ({"enabled": 1}, {"mode": "follow"}, {"aim": "knee"}, {"gain": "1"}, {"gain": True},
                    {"deadzone": float("nan")}, {"unknown": 1}, [], None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                AT.clean_config(bad)
        self.assertEqual(AT.config_for(CAM), AT.DEFAULT_CONFIG)

    def test_persistence_round_trip(self):
        AT.update_config(CAM, {"aim": "head", "gain": 1.5, "invert_pan": True})
        dumped = json.loads(json.dumps(ptz._dump()))
        self.assertEqual(dumped["autotrack"][CAM], {"aim": "head", "gain": 1.5, "invert_pan": True})
        ptz.reset()
        self.assertEqual(AT.config_for(CAM)["aim"], "chest")
        clean, warnings = ptz._validate(dumped)
        self.assertEqual(warnings, [])
        ptz._load(clean)
        cfg = AT.config_for(CAM)
        self.assertEqual((cfg["aim"], cfg["gain"], cfg["invert_pan"], cfg["deadzone"]), ("head", 1.5, True, 0.12))
        bad, warnings = ptz._validate({"settings": {}, "autotrack": {CAM: {"gain": "x"}, "garage": {"aim": "feet"}}})
        self.assertEqual(bad["autotrack"], {"garage": {"aim": "feet"}})
        self.assertEqual(len(warnings), 1)

    def test_status_always_has_autotrack(self):
        st = ptz.status(CAM)
        self.assertEqual(st["autotrack"]["config"], AT.DEFAULT_CONFIG)
        self.assertEqual(st["autotrack"]["status"]["state"], "off")
        self.assertEqual(set(st["autotrack"]["status"]),
                         {"state", "target_id", "target_label", "aim_point", "error", "velocity", "since", "reason"})


# ----------------------------------------------------------------------------- API
class ApiTests(unittest.TestCase):
    def setUp(self):
        from securitycam.webapp import app
        self.h = Harness()
        self.client = app.test_client()

    def tearDown(self):
        self.h.close()

    def test_patch_route(self):
        r = self.client.patch("/api/cameras/front/ptz/autotrack", json={"aim": "face", "deadzone": 0.2})
        self.assertEqual(r.status_code, 200, r.get_json())
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["autotrack"]["config"]["aim"], "face")
        self.assertEqual(body["autotrack"]["status"]["state"], "off")
        self.assertEqual(self.client.patch("/api/cameras/front/ptz/autotrack", json={"gain": "fast"}).get_json()["error"],
                         "invalid_autotrack")
        self.assertEqual(self.client.patch("/api/cameras/front/ptz/autotrack", json=[1]).status_code, 400)
        self.assertEqual(self.client.patch("/api/cameras/nope/ptz/autotrack", json={}).status_code, 404)
        self.assertEqual(self.client.patch("/api/cameras/front/ptz/autotrack", data="x",
                                           content_type="text/plain").status_code, 415)
        r = self.client.patch("/api/cameras/front/ptz/autotrack", json={"enabled": True})
        self.assertEqual(r.get_json()["autotrack"]["status"]["state"], "searching")
        self.assertEqual(r.get_json()["connect"], "ready")
        self.assertEqual(self.client.get("/api/cameras/front/ptz").get_json()["autotrack"]["config"]["enabled"], True)

    def test_target_route(self):
        self.h.targets = [at_point(5, 0.5, 0.5)]
        r = self.client.post("/api/cameras/front/ptz/autotrack/target", json={"target_id": 9})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.get_json()["error"], "target_not_visible")
        for bad in ({"target_id": "5"}, {"target_id": True}, {}, None):
            self.assertEqual(self.client.post("/api/cameras/front/ptz/autotrack/target", json=bad).status_code, 400)
        self.assertEqual(self.client.post("/api/cameras/nope/ptz/autotrack/target", json={"target_id": 5}).status_code, 404)
        AT.update_config(CAM, {"mode": "auto"})
        r = self.client.post("/api/cameras/front/ptz/autotrack/target", json={"target_id": 5})
        body = r.get_json()
        self.assertEqual(r.status_code, 200, body)
        self.assertEqual(body["locked"], 5)
        self.assertEqual(body["autotrack"]["config"]["mode"], "locked")
        self.assertTrue(body["autotrack"]["config"]["enabled"])
        # a moving camera is stopped when auto-track is switched off with target null
        self.h.targets = [at_point(5, 0.9, 0.5)]
        self.h.step()
        self.assertEqual(self.h.sent()[-1][0], "move")
        r = self.client.post("/api/cameras/front/ptz/autotrack/target", json={"target_id": None})
        self.assertFalse(r.get_json()["autotrack"]["config"]["enabled"])
        self.assertEqual(r.get_json()["autotrack"]["status"]["state"], "off")
        self.h.ctl.run_pending()
        self.assertEqual(self.h.sent(), [("stop",)])

    def test_manual_routes_pause(self):
        self.h.enable(manual_pause_s=2.0)
        self.h.targets = [at_point(5, 0.9, 0.5)]
        self.h.locked = 5
        self.h.step()
        self.assertEqual(AT.status(CAM)["state"], "tracking")
        r = self.client.post("/api/cameras/front/ptz/move", json={"x": -1, "y": 0})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["ptz"]["autotrack"]["status"]["state"], "paused")
        self.h.ctl.run_pending()
        self.h.sent()
        self.assertEqual(self.h.step(0.5)["state"], "paused")
        self.assertEqual(self.h.sent(), [])                       # auto-track keeps its hands off
        self.assertEqual(self.h.step(2.0)["state"], "tracking")
        self.assertEqual(self.h.sent()[0][0], "move")
        for path, body in (("center", {"u": 0.5, "v": 0.5}), ("home", None)):
            self.client.post(f"/api/cameras/front/ptz/{path}", json=body)
            self.assertEqual(AT.status(CAM)["state"], "paused", path)


# ----------------------------------------------------------------------------- aim points
class AimTests(unittest.TestCase):
    def setUp(self):
        self.cs = state.get_camera_state(CAM)
        self.cs.esp_aim = {}

    def _setup(self, rows, kps=None, kp_conf=None, w=1000, h=1000):
        cs = self.cs
        cs.esp_settings.clear()
        cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        cs.esp_settings["smoothing"] = 0.0
        cs.yolo_enabled = True
        names = {0: "person", 2: "car"}
        data = np.array([[x1, y1, x2, y2, t, 0.9, c] for t, c, x1, y1, x2, y2 in rows], np.float32).reshape(-1, 7)
        kp = None
        if kps is not None:
            kp = SimpleNamespace(xy=np.array(kps, np.float32), conf=None if kp_conf is None else np.array(kp_conf, np.float32))

        class Boxes(SimpleNamespace):
            def __len__(self):
                return len(data)

        boxes = Boxes(xyxy=data[:, :4], id=data[:, 4], conf=data[:, 5], cls=data[:, 6], is_track=True)
        res = [SimpleNamespace(boxes=boxes, names=names, keypoints=kp)]
        frame = np.zeros((h, w, 3), np.uint8)
        with cs.frame_lock:
            cs.latest_frame = frame
            cs.latest_results = res
        cs.esp_motion = motion.MotionState()
        cs.view_solo, cs.view_hidden = None, []
        t = time.perf_counter()
        draw_custom_esp(cs, frame.copy(), res, frame_ts=t, results_ts=t)
        return targets.targets_payload(cs)

    def test_box_fallback(self):
        p = self._setup([(1, 0, 400, 100, 600, 900), (2, 2, 0, 0, 200, 100)])
        person = next(t for t in p["targets"] if t["id"] == 1)
        car = next(t for t in p["targets"] if t["id"] == 2)
        self.assertEqual(person["aim_src"], "box")
        self.assertEqual(set(person["aim"]), set(AT.AIM_KEYS))
        self.assertAlmostEqual(person["aim"]["chest"][0], 0.5, places=3)
        self.assertAlmostEqual(person["aim"]["chest"][1], 0.1 + 0.30 * 0.8, places=3)
        self.assertAlmostEqual(person["aim"]["feet"][1], 0.1 + 0.97 * 0.8, places=3)
        self.assertEqual(set(car["aim"]), {"center"})
        self.assertEqual(car["aim_src"], "box")
        np.testing.assert_allclose(car["aim"]["center"], [0.1, 0.05], atol=1e-3)

    def test_keypoints(self):
        kp = np.zeros((17, 2), np.float32)
        kp[0] = (500, 200)                       # nose
        kp[1], kp[2] = (490, 190), (510, 190)    # eyes
        kp[3], kp[4] = (480, 195), (520, 195)    # ears
        kp[5], kp[6] = (460, 300), (540, 300)    # shoulders
        kp[11], kp[12] = (470, 600), (530, 600)  # hips
        kp[15], kp[16] = (480, 880), (520, 880)  # ankles
        p = self._setup([(1, 0, 400, 100, 600, 900)], kps=[kp])
        t = p["targets"][0]
        self.assertEqual(t["aim_src"], "pose")
        a = t["aim"]
        np.testing.assert_allclose(a["face"], [0.5, 0.2], atol=2e-3)
        np.testing.assert_allclose(a["chest"], [0.5, 0.3 + 0.25 * 0.3], atol=2e-3)
        np.testing.assert_allclose(a["belly"], [0.5, 0.3 + 0.65 * 0.3], atol=2e-3)
        np.testing.assert_allclose(a["groin"], [0.5, 0.6 + 0.12 * 0.3], atol=2e-3)
        np.testing.assert_allclose(a["neck"], [0.5, 0.3 - 0.2 * 0.1], atol=2e-3)
        np.testing.assert_allclose(a["feet"], [0.5, 0.88], atol=2e-3)
        self.assertLess(a["head"][1], 0.1925)    # raised above the eye / ear level
        np.testing.assert_allclose(a["center"], [0.5, 0.5], atol=2e-3)

    def test_partial_keypoints_mix(self):
        kp = np.zeros((17, 2), np.float32)       # only the face is visible: torso from the box
        kp[0] = (500, 200)
        p = self._setup([(1, 0, 400, 100, 600, 900)], kps=[kp])
        t = p["targets"][0]
        self.assertEqual(t["aim_src"], "box")
        np.testing.assert_allclose(t["aim"]["face"], [0.5, 0.2], atol=2e-3)
        self.assertAlmostEqual(t["aim"]["chest"][1], 0.1 + 0.30 * 0.8, places=3)

    def test_keypoints_follow_display_box_and_smoothing(self):
        kp = np.zeros((17, 2), np.float32)
        kp[5], kp[6], kp[11], kp[12] = (460, 300), (540, 300), (470, 600), (530, 600)
        tr = SimpleNamespace(det=SimpleNamespace(cls_name="person", keypoints=kp.astype(np.int32),
                                                  kp_visible=(kp[:, 0] > 0)),
                             obs=[400.0, 100.0, 600.0, 900.0])
        aim, src = AIM.compute(tr, [500.0, 100.0, 700.0, 900.0], 1000.0, 1000.0)   # box shifted right by 100 px
        self.assertEqual(src, "pose")
        self.assertAlmostEqual(aim["chest"][0], 0.6, places=3)
        self.cs.esp_aim = {}
        a1 = AIM.smoothed(self.cs, 1, {"center": [0.0, 0.0]}, 10.0)
        a2 = AIM.smoothed(self.cs, 1, {"center": [1.0, 1.0]}, 10.0 + 1 / 12)
        self.assertEqual(a1["center"], [0.0, 0.0])
        self.assertTrue(0.3 < a2["center"][0] < 0.7)          # ~alpha 0.5 at 12 Hz
        a3 = AIM.smoothed(self.cs, 1, {"center": [1.0, 1.0]}, 20.0)   # long gap: restart
        self.assertEqual(a3["center"], [1.0, 1.0])


# ----------------------------------------------------------------------------- control law
class ControlTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def lock(self, t):
        self.h.targets = [t]
        self.h.locked = t["id"]

    def test_waiting_ptz_and_sources(self):
        self.h.enable()
        self.h.ctl.state = "idle"
        st = self.h.step()
        self.assertEqual((st["state"], st["reason"]), ("waiting_ptz", "PTZ not connected"))
        self.h.ctl.state = "ready"
        self.h.ctl.caps["pan_tilt"] = False
        self.assertEqual(self.h.step()["state"], "waiting_ptz")
        self.h.ctl.caps["pan_tilt"] = True
        self.h.cs.source_kind = "file"
        try:
            st = self.h.step()
            self.assertEqual((st["state"], st["reason"]), ("waiting_ptz", "This source has no PTZ"))
        finally:
            self.h.cs.source_kind = "rtsp"
        self.assertEqual(self.h.step()["state"], "searching")
        self.assertEqual(self.h.sent(), [])

    def test_direction_signs_and_min_speed(self):
        self.h.enable(gain=1.0, max_speed=0.6)
        self.lock(at_point(1, 0.8, 0.2))             # right and up
        st = self.h.step()
        self.assertEqual(st["state"], "tracking")
        self.assertEqual(st["target_id"], 1)
        self.assertEqual(st["target_label"], "Person #1")
        np.testing.assert_allclose(st["error"], [0.3, -0.3])
        move = self.h.sent()[-1]
        self.assertEqual(move[0], "move")
        self.assertAlmostEqual(move[1], 0.6)         # 1.0 * 0.3 * 2 = 0.6 (= max)
        self.assertAlmostEqual(move[2], 0.6)         # tilt up is positive
        self.lock(at_point(1, 0.2, 0.8))             # left and down
        self.h.step()
        move = self.h.sent()[-1]
        self.assertLess(move[1], 0)
        self.assertLess(move[2], 0)
        # just outside the deadzone with a low gain: minimum speed so the camera really moves
        self.h.close()
        self.h = Harness()
        self.h.enable(gain=0.2, deadzone=0.05)
        self.lock(at_point(1, 0.56, 0.5))
        self.h.step()
        move = self.h.sent()[-1]
        self.assertAlmostEqual(move[1], AT.MIN_SPEED)
        self.assertEqual(move[2], 0.0)

    def test_speed_setting_not_applied_and_clamped(self):
        self.h.ctl.settings["speed"] = 0.1
        self.h.enable(gain=3.0, max_speed=0.4)
        self.lock(at_point(1, 0.95, 0.5))
        self.h.step()
        self.assertAlmostEqual(self.h.sent()[-1][1], 0.4)

    def test_deadzone_hysteresis_and_single_stop(self):
        self.h.enable(deadzone=0.1)
        self.lock(at_point(1, 0.58, 0.5))            # inside the deadzone: nothing
        self.assertEqual(self.h.step()["state"], "holding")
        self.assertEqual(self.h.sent(), [])
        self.lock(at_point(1, 0.65, 0.5))            # leaves it: start
        self.assertEqual(self.h.step()["state"], "tracking")
        self.assertEqual(self.h.sent()[-1][0], "move")
        self.lock(at_point(1, 0.58, 0.5))            # back inside but > deadzone/2: keep moving
        self.assertEqual(self.h.step()["state"], "tracking")
        self.assertEqual(self.h.sent()[-1][0], "move")
        self.lock(at_point(1, 0.53, 0.5))            # < deadzone/2: stop exactly once
        self.assertEqual(self.h.step()["state"], "holding")
        self.assertEqual(self.h.sent(), [("stop",)])
        self.h.step()
        self.h.step()
        self.assertEqual(self.h.sent(), [])

    def test_inversion(self):
        self.h.enable(invert_pan=True, invert_tilt=True)
        self.lock(at_point(1, 0.8, 0.2))
        st = self.h.step()
        move = self.h.sent()[-1]
        self.assertLess(move[1], 0)
        self.assertLess(move[2], 0)
        self.assertEqual(st["velocity"][:2], [move[1], move[2]])

    def test_rate_limit_and_watchdog_refresh(self):
        self.h.enable()
        self.lock(at_point(1, 0.8, 0.5))
        self.h.step()
        self.assertEqual(len(self.h.sent()), 1)
        for _ in range(5):                            # same velocity for 0.5 s: nothing re-sent
            self.h.step()
        self.assertEqual(self.h.sent(), [])
        self.h.step(0.25)                             # 0.75 s since the last send: refresh
        self.assertEqual(len(self.h.sent()), 1)
        self.assertFalse(any(a[0] == "watchdog" for a in self.h.sent()))
        for _ in range(30):                           # 3 s of constant tracking: watchdog never fires
            self.h.step()
        self.assertTrue(self.h.ctl._moving)
        self.assertNotIn(("stop",), self.h.sent())
        self.lock(at_point(1, 0.7, 0.5))              # smaller error: changed by > 0.04 -> sent now
        self.h.step()
        self.assertEqual(len(self.h.sent()), 1)

    def test_stop_on_lost_and_relock_by_identity(self):
        self.h.enable(lost_timeout_s=2.0)
        self.lock(at_point(1, 0.8, 0.5, identity={"name": "Alice", "person_id": "alice"}))
        st = self.h.step()
        self.assertEqual(st["target_label"], "Alice")
        self.h.sent()
        self.h.targets = []                          # track gone -> lock dropped
        st = self.h.step()
        self.assertEqual((st["state"], st["reason"]), ("lost", "Target lost - holding position"))
        self.assertEqual(self.h.sent(), [("stop",)])
        self.h.step()
        self.assertEqual(self.h.sent(), [])          # stop only once
        # same person comes back with a new track id within the timeout: re-locked
        self.h.targets = [at_point(7, 0.3, 0.5), at_point(8, 0.8, 0.5, identity={"name": "Alice", "person_id": "alice"})]
        st = self.h.step()
        self.assertEqual((st["state"], st["target_id"]), ("tracking", 8))
        self.assertEqual(self.h.locked, 8)
        # lost again; after the timeout a returning Alice is not re-locked
        self.h.targets = []
        self.h.step()
        self.h.step(2.5)
        self.h.targets = [at_point(9, 0.8, 0.5, identity={"name": "Alice", "person_id": "alice"})]
        st = self.h.step()
        self.assertEqual(st["state"], "lost")
        self.assertIsNone(self.h.locked)
        # a new lock resumes
        self.h.locked = 9
        self.assertEqual(self.h.step()["state"], "tracking")

    def test_user_unlock_stops_without_chasing(self):
        self.h.enable()
        self.lock(at_point(1, 0.8, 0.5))
        self.h.step()
        self.h.sent()
        self.h.locked = None
        st = self.h.step()
        self.assertEqual(st["state"], "searching")
        self.assertEqual(self.h.sent(), [("stop",)])
        self.assertIsNone(self.h.locked)

    def test_disable_stops(self):
        self.h.enable()
        self.lock(at_point(1, 0.8, 0.5))
        self.h.step()
        self.h.sent()
        AT.update_config(CAM, {"enabled": False})
        self.h.ctl.run_pending()
        self.assertEqual(self.h.sent(), [("stop",)])
        self.assertEqual(self.h.step()["state"], "off")
        self.assertEqual(self.h.sent(), [])

    def test_aim_selection(self):
        self.h.enable(aim="head")
        t = target(1, [0.4, 0.1, 0.6, 0.9])          # head at y = 0.1 + 0.07 * 0.8
        self.lock(t)
        st = self.h.step()
        self.assertAlmostEqual(st["aim_point"][1], 0.156, places=3)
        self.assertGreater(self.h.sent()[-1][2], 0)  # head above center: tilt up

    def test_auto_zoom(self):
        self.h.close()
        self.h = Harness(zoom=True)
        self.h.enable(auto_zoom=True, zoom_target=0.5)
        self.lock(target(1, [0.48, 0.45, 0.52, 0.65], aim={k: [0.5, 0.5] for k in AT.AIM_KEYS}))   # small box
        st = self.h.step()
        self.assertEqual(st["state"], "tracking")
        self.assertGreater(self.h.sent()[-1][3], 0)
        self.lock(target(1, [0.3, 0.05, 0.7, 0.95], aim={k: [0.5, 0.5] for k in AT.AIM_KEYS}))     # too big
        self.h.step()
        self.assertLess(self.h.sent()[-1][3], 0)
        self.lock(target(1, [0.4, 0.25, 0.6, 0.75], aim={k: [0.5, 0.5] for k in AT.AIM_KEYS}))     # right size
        self.assertEqual(self.h.step()["state"], "holding")
        self.assertEqual(self.h.sent(), [("stop",)])

    def test_manual_pause_expires(self):
        self.h.enable(manual_pause_s=1.0)
        self.lock(at_point(1, 0.8, 0.5))
        self.h.step()
        AT.note_manual(CAM)
        self.h.sent()
        self.assertEqual(self.h.step(0.5)["state"], "paused")
        self.assertEqual(self.h.sent(), [])
        self.assertEqual(self.h.step(0.6)["state"], "tracking")
        self.assertEqual(self.h.sent()[0][0], "move")
        AT.update_config(CAM, {"manual_pause_s": 0})
        AT.note_manual(CAM)
        self.assertEqual(self.h.step()["state"], "tracking")


# ----------------------------------------------------------------------------- auto mode
class AutoModeTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_picks_known_person_then_largest(self):
        self.h.enable(mode="auto")
        big = target(1, [0.1, 0.1, 0.4, 0.9])
        small_known = target(2, [0.6, 0.4, 0.65, 0.6], identity={"name": "Bob", "person_id": "bob"})
        car = target(3, [0.0, 0.0, 1.0, 1.0], label="car")
        self.h.targets = [big, small_known, car]
        st = self.h.step()
        self.assertEqual((st["target_id"], st["target_label"]), (2, "Bob"))
        self.assertEqual(self.h.locked, 2)
        self.h.targets = [big, car]
        st = self.h.step()
        self.assertEqual(st["target_id"], 1)
        self.assertEqual(self.h.locked, 1)
        # current target kept while visible, even if a bigger one appears
        self.h.targets = [big, target(4, [0.0, 0.0, 0.5, 1.0]), car]
        self.assertEqual(self.h.step()["target_id"], 1)
        # hidden targets are never picked; no person: lost, stop
        self.h.targets = [target(5, [0.0, 0.0, 0.5, 1.0], hidden=True), car]
        st = self.h.step()
        self.assertEqual(st["state"], "lost")
        self.assertIsNone(self.h.locked)

    def test_follows_user_lock(self):
        self.h.enable(mode="auto")
        self.h.targets = [target(1, [0.1, 0.1, 0.4, 0.9]), target(2, [0.6, 0.4, 0.65, 0.6])]
        self.assertEqual(self.h.step()["target_id"], 1)
        self.h.locked = 2
        self.assertEqual(self.h.step()["target_id"], 2)


# ----------------------------------------------------------------------------- closed loop with latency
class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(latency_model=True)

    def tearDown(self):
        self.h.close()

    def run_loop(self, target_az, steps=150, latency_s=0.4, deg_per_unit=60.0, fov=60.0, walk=0.0, **cfg):
        """Camera pans at velocity * deg_per_unit deg/s, commands take effect after ``latency_s``."""
        self.h.enable(**cfg)
        pan, queue, active = 0.0, [], (0.0, 0.0, 0.0)
        signs, errors = [], []
        for _ in range(steps):
            target_az += walk * 0.1
            u = 0.5 + (target_az - pan) / fov
            self.lock(at_point(1, min(1.0, max(0.0, u)), 0.5))
            self.h.step(0.1)
            for call in self.h.sent():
                queue.append((self.h.clock.t + latency_s, (call[1], call[2], call[3]) if call[0] == "move" else (0.0, 0.0, 0.0)))
            while queue and queue[0][0] <= self.h.clock.t:
                active = queue.pop(0)[1]
            pan += active[0] * deg_per_unit * 0.1
            errors.append(u - 0.5)
            if active[0]:
                signs.append(1 if active[0] > 0 else -1)
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        return errors, flips

    def lock(self, t):
        self.h.targets = [t]
        self.h.locked = t["id"]

    def test_converges_without_oscillation(self):
        errors, flips = self.run_loop(25.0)            # target 25 deg right (u = 0.92)
        self.assertLess(abs(errors[-1]), 0.12)
        self.assertLessEqual(flips, 1)
        self.assertEqual(AT.status(CAM)["state"], "holding")

    def test_high_gain_still_stable(self):
        errors, flips = self.run_loop(-25.0, gain=2.0, max_speed=1.0)
        self.assertLess(abs(errors[-1]), 0.12)
        self.assertLessEqual(flips, 1)

    def test_latency_mismatch_and_learned_speed(self):
        # picture latency 0.8 s (assumed 0.5 s), camera twice as fast as the initial guess
        errors, flips = self.run_loop(25.0, latency_s=0.8, deg_per_unit=72.0)
        self.assertLess(abs(errors[-1]), 0.12)
        self.assertLessEqual(flips, 2)
        self.assertGreater(AT.tracker(CAM)._k[0], AT.K_INIT)

    def test_settles_after_stop(self):
        self.h.enable()
        self.lock(at_point(1, 0.9, 0.5))
        self.h.step()
        self.lock(at_point(1, 0.5, 0.5))
        self.h.step()                                   # stop
        self.assertEqual(self.h.sent()[-1], ("stop",))
        self.lock(at_point(1, 0.9, 0.5))
        st = self.h.step()                              # 0.1 s later: still settling
        self.assertEqual((st["state"], st["reason"]), ("holding", "Camera stopped - letting the picture settle"))
        self.assertEqual(self.h.sent(), [])
        self.h.step(0.3)
        self.assertEqual(self.h.sent()[-1][0], "move")

    def test_follows_walking_person(self):
        errors, flips = self.run_loop(0.0, walk=6.0, steps=200)    # 6 deg/s to the right
        self.assertLess(max(abs(e) for e in errors[50:]), 0.2)
        self.assertEqual(flips, 0)


if __name__ == "__main__":
    unittest.main()
