"""Tests for persistence, presets/scheduler, settings import/export, grid
downscale feed and ONVIF PTZ (mocked). No camera is ever contacted.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="securitycam-test-")
os.environ["SECURITYCAM_DATA_DIR"] = _TMP
os.environ["SECURITYCAM_SAVE_DEBOUNCE"] = "0.05"
# TEST-NET addresses; nothing is ever connected in these tests.
os.environ["SECURITYCAM_CAMERAS"] = "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1"
os.environ["SECURITYCAM_AUTO_INSTALL_DEPS"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam import config, persistence, presets, ptz, scaled_feed, state, streaming  # noqa: E402
from securitycam.webapp import app  # noqa: E402

presets.SCHEDULER.stop()


def reset_cameras():
    for cs in state.camera_states.values():
        cs.esp_settings.clear()
        cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        cs.yolo_enabled = False
        cs.yolo_model_name = config.YOLO_MODEL_NAME
        cs.esp_zones[:] = []
        cs.esp_zone_seq = 0


def make_store(path: Path, debounce: float = 0.0) -> persistence.Store:
    store = persistence.Store(path, debounce_s=debounce)
    store.register_section("cameras", persistence.dump_cameras, persistence.load_cameras, persistence.validate_cameras)
    return store


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        reset_cameras()
        self.dir = Path(tempfile.mkdtemp(dir=_TMP))
        self.path = self.dir / "state.json"
        self.client = app.test_client()

    def test_round_trip(self):
        store = make_store(self.path)
        store.load()
        self.client.post("/api/cameras/front/update_esp", json={"tracers": True, "confidence_threshold": 0.5})
        self.client.post("/api/cameras/front/toggle_yolo")
        self.client.post("/api/cameras/garage/set_model", json={"model_name": "yolov8s.pt"})
        r = self.client.post("/api/cameras/front/zones", json={"box": [0.1, 0.1, 0.4, 0.5], "label": "Driveway"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(store.save_now())
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(doc["version"], persistence.SCHEMA_VERSION)
        self.assertIn("cameras", doc["sections"])

        reset_cameras()
        store2 = make_store(self.path)
        self.assertTrue(store2.load())
        front, garage = state.camera_states["front"], state.camera_states["garage"]
        self.assertTrue(front.esp_settings["tracers"])
        self.assertEqual(front.esp_settings["confidence_threshold"], 0.5)
        self.assertTrue(front.yolo_enabled)
        self.assertEqual(garage.yolo_model_name, "yolov8s.pt")
        self.assertEqual(front.esp_zones[0]["label"], "Driveway")
        self.assertEqual(front.esp_zone_seq, 1)
        # new zone ids continue after the restored ones
        r = self.client.post("/api/cameras/front/zones", json={"box": [0.5, 0.5, 0.9, 0.9]})
        self.assertEqual(r.get_json()["zone"]["id"], "z2")
        # unchanged state is not rewritten
        store2.save_now()
        self.assertFalse(store2.save_now())

    def test_corrupted_file_keeps_backup(self):
        self.path.write_text("{ not json", encoding="utf-8")
        store = make_store(self.path)
        self.assertFalse(store.load())
        bak = self.path.with_name("state.json.bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_text(encoding="utf-8"), "{ not json")
        self.assertFalse(self.path.exists())
        self.assertIn("unreadable", store.last_error)
        self.assertEqual(state.camera_states["front"].esp_settings, config.DEFAULT_ESP_SETTINGS)
        self.assertTrue(store.save_now())   # defaults can be written afterwards
        self.assertEqual(bak.read_text(encoding="utf-8"), "{ not json")

    def test_wrong_structure_is_corrupt(self):
        self.path.write_text(json.dumps({"sections": {}}), encoding="utf-8")  # no version
        store = make_store(self.path)
        self.assertFalse(store.load())
        self.assertTrue(self.path.with_name("state.json.bak").exists())

    def test_atomic_write_keeps_old_file_on_failure(self):
        store = make_store(self.path)
        store.load()
        store.save_now()
        before = self.path.read_text(encoding="utf-8")
        state.camera_states["front"].esp_settings["tracers"] = True
        with mock.patch("securitycam.persistence.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                store.save_now()
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")], [])
        self.assertTrue(store.save_now())   # retried later
        self.assertNotEqual(self.path.read_text(encoding="utf-8"), before)

    def test_debounced_save_coalesces(self):
        store = make_store(self.path, debounce=0.15)
        store.load()
        writes = []
        original = store._atomic_write
        store._atomic_write = lambda text: (writes.append(1), original(text))
        for i in range(5):
            state.camera_states["front"].esp_settings["line_thickness"] = i + 1
            store.mark_dirty()
            time.sleep(0.02)
        deadline = time.time() + 2
        while store.pending() and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.1)
        self.assertEqual(len(writes), 1)
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(doc["sections"]["cameras"]["front"]["esp_settings"]["line_thickness"], 5)

    def test_unknown_section_and_camera_are_preserved(self):
        self.path.write_text(json.dumps({
            "version": 1,
            "sections": {
                "future_feature": {"x": 1},
                "cameras": {"old-cam": {"esp_settings": {"boxes": False}}, "front": {"esp_settings": {"boxes": False, "bogus": 3}}},
            },
        }), encoding="utf-8")
        store = make_store(self.path)
        store.register_section("late", lambda: {"v": 2}, lambda d: None)
        self.assertTrue(store.load())
        self.assertFalse(state.camera_states["front"].esp_settings["boxes"])
        self.assertNotIn("bogus", state.camera_states["front"].esp_settings)
        store.save_now(force=True)
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(doc["sections"]["future_feature"], {"x": 1})
        self.assertIn("old-cam", doc["sections"]["cameras"])

    def test_late_registration_receives_data(self):
        self.path.write_text(json.dumps({"version": 1, "sections": {"late": {"v": 7}}}), encoding="utf-8")
        store = make_store(self.path)
        store.load()
        got = []
        store.register_section("late", lambda: {"v": 7}, got.append)
        self.assertEqual(got, [{"v": 7}])

    def test_mutating_request_marks_dirty(self):
        with mock.patch.object(persistence.STORE, "mark_dirty") as md:
            self.client.post("/api/cameras/front/update_esp", json={"names": False})
            self.assertTrue(md.called)
            md.reset_mock()
            self.client.get("/api/cameras/front/status")
            self.assertFalse(md.called)


class PresetTests(unittest.TestCase):
    def setUp(self):
        reset_cameras()
        presets.STORE.seed_builtins()
        presets.STORE.last_applied.clear()
        self.client = app.test_client()

    def test_builtins_listed(self):
        data = self.client.get("/api/presets?camera_id=front").get_json()
        names = [p["name"] for p in data["presets"]]
        self.assertEqual(names, ["Day", "Night", "Away", "Minimal"])
        self.assertEqual(len(data["schedules"]), 2)
        self.assertFalse(any(s["enabled"] for s in data["schedules"]))

    def test_save_current_apply_rename_delete(self):
        front = state.camera_states["front"]
        front.esp_settings["tracers"] = True
        front.esp_settings["color_boxes"] = "#123456"
        r = self.client.post("/api/presets", json={"name": "  My   look ", "camera_id": "front"})
        self.assertEqual(r.status_code, 200)
        preset = r.get_json()["preset"]
        self.assertEqual(preset["name"], "My look")
        self.assertEqual(preset["settings"]["color_boxes"], "#123456")
        # camera-bound preset is invisible for the other camera and cannot be applied there
        garage_list = self.client.get("/api/presets?camera_id=garage").get_json()["presets"]
        self.assertNotIn(preset["id"], [p["id"] for p in garage_list])
        r = self.client.post(f"/api/presets/{preset['id']}/apply", json={"camera_id": "garage"})
        self.assertEqual(r.status_code, 400)

        reset_cameras()
        r = self.client.post(f"/api/presets/{preset['id']}/apply", json={"camera_id": "front"})
        self.assertEqual(r.get_json()["cameras"], ["front"])
        self.assertTrue(front.esp_settings["tracers"])
        self.assertEqual(presets.STORE.last_applied["front"]["preset_id"], preset["id"])

        r = self.client.patch(f"/api/presets/{preset['id']}", json={"name": "Renamed"})
        self.assertEqual(r.get_json()["preset"]["name"], "Renamed")
        self.assertEqual(self.client.patch(f"/api/presets/{preset['id']}", json={"name": ""}).status_code, 400)

        r = self.client.post("/api/schedules", json={"preset_id": preset["id"], "time": "08:15", "days": [0, 2]})
        self.assertEqual(r.status_code, 200)
        r = self.client.delete(f"/api/presets/{preset['id']}")
        self.assertEqual(len(r.get_json()["removed_schedules"]), 1)
        self.assertEqual(self.client.delete(f"/api/presets/{preset['id']}").status_code, 404)

    def test_global_apply_to_all_and_builtin_patch_keeps_colors(self):
        for cs in state.camera_states.values():
            cs.esp_settings["color_boxes"] = "#abcdef"
        r = self.client.post("/api/presets/minimal/apply", json={"camera_id": "all"})
        self.assertEqual(sorted(r.get_json()["cameras"]), ["front", "garage"])
        for cs in state.camera_states.values():
            self.assertFalse(cs.esp_settings["names"])
            self.assertEqual(cs.esp_settings["color_boxes"], "#abcdef")
        self.client.post("/api/presets/night/apply", json={"camera_id": "front"})
        self.assertTrue(state.camera_states["front"].yolo_enabled)
        self.assertFalse(state.camera_states["garage"].yolo_enabled)

    def test_schedule_validation(self):
        bad = [
            {"preset_id": "day", "time": "25:00"},
            {"preset_id": "day", "time": "07:00", "days": []},
            {"preset_id": "day", "time": "07:00", "days": [7]},
            {"preset_id": "nope", "time": "07:00"},
            {"preset_id": "day", "time": "07:00", "camera_id": "unknown"},
        ]
        for body in bad:
            self.assertGreaterEqual(self.client.post("/api/schedules", json=body).status_code, 400, body)
        r = self.client.post("/api/schedules", json={"preset_id": "day", "time": "07:30"})
        sid = r.get_json()["schedule"]["id"]
        self.assertEqual(r.get_json()["schedule"]["days"], presets.ALL_DAYS)
        r = self.client.patch(f"/api/schedules/{sid}", json={"enabled": False, "time": "06:45"})
        self.assertEqual(r.get_json()["schedule"]["time"], "06:45")
        self.assertEqual(self.client.delete(f"/api/schedules/{sid}").status_code, 200)

    def test_scheduler_with_fake_clock(self):
        store = presets.STORE
        night = store.create_schedule({"preset_id": "night", "time": "22:00", "days": [0, 1, 2, 3, 4], "camera_id": "front"})
        day = store.create_schedule({"preset_id": "day", "time": "07:00", "days": presets.ALL_DAYS})
        sched = presets.Scheduler(store)
        monday = dt.datetime(2026, 11, 2, 21, 59, 30)       # a Monday
        self.assertEqual(sched.tick(monday), [])             # first tick only anchors
        fired = sched.tick(monday + dt.timedelta(seconds=40))
        self.assertEqual([f["schedule_id"] for f in fired], [night["id"]])
        self.assertTrue(state.camera_states["front"].yolo_enabled)
        self.assertEqual(sched.tick(monday + dt.timedelta(minutes=5)), [])   # no double fire
        # jumps over midnight to 07:00:10 next day -> day preset fires once for both cameras
        fired = sched.tick(dt.datetime(2026, 11, 3, 7, 0, 10))
        self.assertEqual([f["schedule_id"] for f in fired], [day["id"]])
        self.assertEqual(sorted(fired[0]["cameras"]), ["front", "garage"])
        self.assertEqual(store.last_applied["garage"]["source"], "schedule")
        # Saturday 22:00: night schedule is weekdays only
        sched.tick(dt.datetime(2026, 11, 7, 21, 0))
        self.assertEqual(sched.tick(dt.datetime(2026, 11, 7, 22, 1)), [])
        # clock going backwards re-anchors without firing
        self.assertEqual(sched.tick(dt.datetime(2026, 11, 7, 6, 0)), [])
        self.assertEqual(sched.tick(dt.datetime(2026, 11, 7, 7, 0, 1))[0]["schedule_id"], day["id"])
        # disabled schedules never fire
        store.update_schedule(day["id"], {"enabled": False})
        sched.tick(dt.datetime(2026, 11, 8, 6, 59))
        self.assertEqual(sched.tick(dt.datetime(2026, 11, 8, 7, 1)), [])

    def test_next_run(self):
        s = {"enabled": True, "time": "22:00", "days": [5]}
        nxt = presets.next_run(s, dt.datetime(2026, 11, 2, 12, 0))     # Monday
        self.assertEqual(nxt, dt.datetime(2026, 11, 7, 22, 0))

    def test_presets_persist(self):
        p = presets.STORE.create({"name": "Keep me", "settings": {"thermal": True}})
        dumped = presets.STORE.dump()
        clean, warnings = presets.STORE.validate(dumped)
        self.assertEqual(warnings, [])
        presets.STORE.seed_builtins()
        presets.STORE.load(clean)
        self.assertEqual(presets.STORE.get(p["id"])["settings"], {"thermal": True})
        nxt = presets.STORE.create({"name": "Next", "settings": {"thermal": False}})
        self.assertNotEqual(nxt["id"], p["id"])


class ImportExportTests(unittest.TestCase):
    def setUp(self):
        reset_cameras()
        presets.STORE.seed_builtins()
        self.client = app.test_client()

    def test_export_import_preview_and_apply(self):
        r = self.client.get("/api/settings/export")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        doc = r.get_json()
        self.assertEqual(doc["version"], persistence.SCHEMA_VERSION)
        self.assertIn("presets", doc["sections"])
        doc["sections"]["cameras"]["front"]["esp_settings"]["tracers"] = True
        doc["sections"]["cameras"]["front"]["esp_settings"]["line_thickness"] = 99   # clamped
        doc["sections"]["presets"]["presets"][0]["name"] = "Sunny"

        r = self.client.post("/api/settings/import?dry_run=1", data=json.dumps(doc), content_type="application/json")
        body = r.get_json()
        self.assertTrue(body["dry_run"])
        paths = {d["path"]: d for d in body["diff"]}
        self.assertEqual(paths["cameras.front.esp_settings.tracers"]["after"], True)
        self.assertEqual(paths["cameras.front.esp_settings.line_thickness"]["after"], 6)
        self.assertFalse(state.camera_states["front"].esp_settings["tracers"])   # preview changes nothing

        r = self.client.post("/api/settings/import", data=json.dumps(doc), content_type="application/json")
        self.assertTrue(r.get_json()["ok"])
        self.assertTrue(state.camera_states["front"].esp_settings["tracers"])
        self.assertEqual(presets.STORE.get("day")["name"], "Sunny")

    def test_import_rejects_garbage(self):
        for payload in ["nope", json.dumps([1, 2]), json.dumps({"version": 1}), json.dumps({"version": 1, "sections": {"x": {}}}),
                        json.dumps({"app": "other", "version": 1, "sections": {"cameras": {}}})]:
            r = self.client.post("/api/settings/import?dry_run=1", data=payload, content_type="application/json")
            self.assertEqual(r.status_code, 400, payload)

    def test_non_json_bodies_are_rejected(self):
        doc = self.client.get("/api/settings/export").get_data()
        before = dict(state.camera_states["front"].esp_settings)
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
            r = self.client.post("/api/settings/import", data=doc, content_type=ctype)
            self.assertEqual(r.status_code, 415, ctype)
            self.assertEqual(r.get_json()["error"], "unsupported_media_type")
        self.assertEqual(self.client.post("/api/settings/import").status_code, 415)
        self.assertEqual(state.camera_states["front"].esp_settings, before)
        # every body-carrying route of the blueprint
        for method, url in [
            ("post", "/api/presets"), ("patch", "/api/presets/day"), ("post", "/api/presets/day/apply"),
            ("post", "/api/schedules"), ("patch", "/api/schedules/s1"),
            ("post", "/api/cameras/front/ptz/move"), ("post", "/api/cameras/front/ptz/center"),
            ("post", "/api/cameras/front/ptz/presets"), ("patch", "/api/cameras/front/ptz/settings"),
        ]:
            r = getattr(self.client, method)(url, data='{"name": "x", "camera_id": "front"}', content_type="text/plain")
            self.assertEqual(r.status_code, 415, url)
        self.assertEqual(presets.STORE.get("day")["name"], "Day")

    def test_state_endpoint(self):
        data = self.client.get("/api/settings/state").get_json()
        self.assertIn("cameras", data["sections"])
        self.assertIn("presets", data["sections"])
        self.assertIn("ptz", data["sections"])


class FakeStream:
    def __init__(self, w=1920, h=1080):
        self.stopped = False
        self.grabbed = True
        self.lock = threading.Lock()
        self.seq = 1
        self.frame = np.full((h, w, 3), 90, dtype=np.uint8)
        self.frame_ts = time.perf_counter()

    def wait_newer(self, seq, timeout):
        if self.seq > seq:
            return self.seq, self.frame, self.frame_ts
        time.sleep(0.01)
        return None

    def release(self):
        self.stopped = True


class GridFeedTests(unittest.TestCase):
    def setUp(self):
        reset_cameras()
        self.cs = state.camera_states["front"]
        self.feed = streaming.CameraFeed(self.cs, FakeStream())
        streaming._feeds["front"] = self.feed
        scaled_feed._cache.clear()
        self.renders = 0
        real = streaming.render_jpeg

        def counting(*a, **k):
            self.renders += 1
            return real(*a, **k)

        self.patch = mock.patch.object(streaming, "render_jpeg", counting)
        self.patch.start()
        self.worker = mock.patch.object(streaming, "ensure_worker", lambda cs: None)
        self.worker.start()

    def tearDown(self):
        self.patch.stop()
        self.worker.stop()
        streaming._feeds.pop("front", None)

    def test_snap(self):
        self.assertIsNone(scaled_feed.snap_width(0))
        self.assertEqual(scaled_feed.snap_width(100), 320)
        self.assertEqual(scaled_feed.snap_width(600), 640)
        self.assertEqual(scaled_feed.snap_width(5000), 1280)

    def test_shared_renders(self):
        before = scaled_feed.encode_count
        g1 = scaled_feed.gen_frames_scaled("front", 600)
        g2 = scaled_feed.gen_frames_scaled("front", 630)    # same bucket (640)
        part1, part2 = next(g1), next(g2)
        self.assertEqual(part1, part2)
        self.assertEqual(self.renders, 1)                   # overlay rendered once
        self.assertEqual(scaled_feed.encode_count - before, 1)
        g3 = scaled_feed.gen_frames_scaled("front", 300)
        part3 = next(g3)
        self.assertEqual(self.renders, 1)
        self.assertEqual(scaled_feed.encode_count - before, 2)
        jpeg = part1.split(b"\r\n\r\n", 1)[1]
        import cv2
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[1], 640)
        self.assertEqual(img.shape[0], 360)
        self.assertLess(len(part3), len(part1))
        for g in (g1, g2, g3):
            g.close()
        self.assertEqual(self.feed.clients, 0)

    def test_endpoint(self):
        client = app.test_client()
        r = client.get("/video_feed/front?w=480", buffered=False)
        self.assertEqual(r.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
        chunk = next(iter(r.response))
        self.assertTrue(chunk.startswith(b"--frame"))
        r.close()

    def test_overview(self):
        self.cs.stats["fps"] = 12.34
        self.cs.stats["num_detections"] = 3
        self.cs.stats["last_update"] = time.time()
        data = app.test_client().get("/api/overview").get_json()
        front = next(c for c in data["cameras"] if c["id"] == "front")
        self.assertEqual(front["fps"], 12.3)
        self.assertEqual(front["num_detections"], 3)
        self.assertTrue(front["live"])


# ---------------------------------------------------------------------------- PTZ (mocked ONVIF)
class FakeFault(Exception):
    pass


class FakePtzService:
    def __init__(self, cam):
        self.cam = cam
        self.presets = [types.SimpleNamespace(token="1", Name="Gate")]

    def _rec(self, name, params):
        self.cam.calls.append((name, params))

    def GetConfigurationOptions(self, params):
        self._rec("GetConfigurationOptions", params)
        return types.SimpleNamespace(Spaces=types.SimpleNamespace(
            ContinuousPanTiltVelocitySpace=[types.SimpleNamespace(URI="generic")],
            ContinuousZoomVelocitySpace=[],
            RelativePanTiltTranslationSpace=[types.SimpleNamespace(URI="http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationGenericSpace")],
        ))

    def GetPresets(self, params):
        self._rec("GetPresets", params)
        return list(self.presets)

    def ContinuousMove(self, params):
        self._rec("ContinuousMove", params)
        if FakeOnvif.fail_commands_auth:
            raise FakeFault("Sender not Authorized")

    def Stop(self, params):
        self._rec("Stop", params)

    def RelativeMove(self, params):
        self._rec("RelativeMove", params)

    def GotoPreset(self, params):
        self._rec("GotoPreset", params)

    def SetPreset(self, params):
        self._rec("SetPreset", params)
        token = str(len(self.presets) + 1)
        self.presets.append(types.SimpleNamespace(token=token, Name=params["PresetName"]))
        return token

    def RemovePreset(self, params):
        self._rec("RemovePreset", params)
        self.presets = [p for p in self.presets if p.token != params["PresetToken"]]


class FakeOnvif:
    instances = []
    fail_auth = False
    fail_network = False
    fail_commands_auth = False

    def __init__(self, host, port, user, password, transport=None):
        FakeOnvif.instances.append(self)
        self.host, self.port = host, port
        self.calls = []
        if FakeOnvif.fail_auth:
            raise FakeFault("wsse:FailedAuthentication The security token could not be authenticated")
        if FakeOnvif.fail_network:
            raise ConnectionError("timed out")
        self._ptz = FakePtzService(self)

    def create_media_service(self):
        profile = types.SimpleNamespace(token="prof1", PTZConfiguration=types.SimpleNamespace(token="ptzcfg"))
        return types.SimpleNamespace(GetProfiles=lambda: [profile])

    def create_ptz_service(self):
        return self._ptz


def wait_for(pred, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class PtzTests(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType("onvif")
        fake.ONVIFCamera = FakeOnvif
        self.mod = mock.patch.dict(sys.modules, {"onvif": fake})
        self.mod.start()
        FakeOnvif.instances = []
        FakeOnvif.fail_auth = FakeOnvif.fail_network = FakeOnvif.fail_commands_auth = False
        ptz.reset()
        ptz.available_override = None
        streaming._connect_failures.clear()
        self.client = app.test_client()

    def tearDown(self):
        ptz.reset()
        ptz.available_override = None
        self.mod.stop()

    def connect(self):
        r = self.client.post("/api/cameras/front/ptz/connect")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(wait_for(lambda: ptz.status("front")["state"] in ("ready", "backoff", "unsupported")))
        return ptz.status("front")

    def calls(self, name):
        return [p for n, p in FakeOnvif.instances[-1].calls if n == name]

    def test_unavailable_without_library(self):
        ptz.available_override = False
        st = self.client.get("/api/cameras/front/ptz").get_json()
        self.assertFalse(st["available"])
        self.assertEqual(st["install_hint"], "pip install onvif-zeep")
        self.assertEqual(self.client.post("/api/cameras/front/ptz/move", json={"x": 1}).status_code, 501)

    def test_status_does_not_connect(self):
        st = self.client.get("/api/cameras/front/ptz").get_json()
        self.assertTrue(st["available"])
        self.assertEqual(st["state"], "idle")
        self.assertEqual(st["port"], 2020)
        self.assertEqual(FakeOnvif.instances, [])

    def test_connect_move_stop_presets_center(self):
        st = self.connect()
        self.assertEqual(st["state"], "ready")
        self.assertEqual(FakeOnvif.instances[0].port, 2020)
        self.assertTrue(st["caps"]["pan_tilt"])
        self.assertFalse(st["caps"]["zoom"])
        self.assertEqual(st["presets"], [{"token": "1", "name": "Gate"}])

        self.client.post("/api/cameras/front/ptz/move", json={"x": 1, "y": -1, "zoom": 1})
        self.assertTrue(wait_for(lambda: self.calls("ContinuousMove")))
        vel = self.calls("ContinuousMove")[0]["Velocity"]
        self.assertEqual(vel, {"PanTilt": {"x": 0.5, "y": -0.5}})   # speed 0.5, zoom unsupported
        self.client.post("/api/cameras/front/ptz/stop")
        self.assertTrue(wait_for(lambda: self.calls("Stop")))

        r = self.client.post("/api/cameras/front/ptz/center", json={"u": 1.0, "v": 0.5})
        tr = r.get_json()["translation"]
        # (1.0-0.5) * 100 deg FOV = 50 deg of 180 deg half-range
        self.assertAlmostEqual(tr["x"], 50 / 180, places=3)
        self.assertEqual(tr["y"], 0)
        self.assertTrue(wait_for(lambda: self.calls("RelativeMove")))
        self.assertEqual(self.client.post("/api/cameras/front/ptz/center", json={"u": "a"}).status_code, 400)

        self.client.post("/api/cameras/front/ptz/presets", json={"name": "Door"})
        self.assertTrue(wait_for(lambda: len(ptz.status("front")["presets"]) == 2))
        self.client.post("/api/cameras/front/ptz/presets/2/goto")
        self.assertTrue(wait_for(lambda: self.calls("GotoPreset")))
        self.client.delete("/api/cameras/front/ptz/presets/2")
        self.assertTrue(wait_for(lambda: len(ptz.status("front")["presets"]) == 1))

    def test_watchdog_stops_movement(self):
        self.connect()
        ctl = ptz.get_controller("front")
        with mock.patch("securitycam.ptz.controller.MOVE_WATCHDOG_S", 0.1):
            self.client.post("/api/cameras/front/ptz/move", json={"x": 1})
            self.assertTrue(wait_for(lambda: self.calls("ContinuousMove")))
            self.assertTrue(wait_for(lambda: self.calls("Stop"), timeout=2.0))
        self.assertFalse(ctl.status()["moving"])

    def test_command_before_connect_triggers_connect(self):
        r = self.client.post("/api/cameras/front/ptz/move", json={"x": 1})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.get_json()["connect"], "queued")
        self.assertTrue(wait_for(lambda: ptz.status("front")["state"] == "ready"))

    def test_auth_failure_backs_off(self):
        FakeOnvif.fail_auth = True
        st = self.connect()
        self.assertEqual(st["state"], "backoff")
        self.assertEqual(st["error_kind"], "auth")
        self.assertGreater(st["retry_in"], 50)
        # further attempts are refused without touching the camera
        r = self.client.post("/api/cameras/front/ptz/connect")
        self.assertEqual(r.status_code, 429)
        self.client.post("/api/cameras/front/ptz/move", json={"x": 1})
        time.sleep(0.1)
        self.assertEqual(len(FakeOnvif.instances), 1)
        # second auth failure doubles the wait
        ctl = ptz.get_controller("front")
        ctl.retry_at = 0
        self.connect()
        self.assertEqual(len(FakeOnvif.instances), 2)
        self.assertGreater(ptz.status("front")["retry_in"], 110)

    def test_network_failure_short_backoff(self):
        FakeOnvif.fail_network = True
        st = self.connect()
        self.assertEqual(st["error_kind"], "network")
        self.assertLessEqual(st["retry_in"], 5.0)

    def test_auth_error_on_command_drops_session(self):
        self.connect()
        FakeOnvif.fail_commands_auth = True
        self.client.post("/api/cameras/front/ptz/move", json={"x": 1})
        self.assertTrue(wait_for(lambda: ptz.status("front")["state"] == "backoff"))
        self.assertEqual(ptz.status("front")["error_kind"], "auth")

    def test_rtsp_backoff_blocks_onvif_login(self):
        streaming._connect_failures["front"] = (2, time.monotonic() + 30)
        st = self.connect()
        self.assertEqual(st["state"], "backoff")
        self.assertEqual(st["error_kind"], "rtsp")
        self.assertEqual(FakeOnvif.instances, [])

    def test_settings_validation_and_persistence(self):
        r = self.client.patch("/api/cameras/front/ptz/settings", json={"fov_h": 90, "speed": 5})
        self.assertEqual(r.get_json()["settings"]["speed"], 1.0)
        self.assertEqual(self.client.patch("/api/cameras/front/ptz/settings", json={"fov_h": "x"}).status_code, 400)
        dumped = ptz._dump()
        self.assertEqual(dumped["settings"]["front"]["fov_h"], 90)
        ptz.reset()
        clean, _ = ptz._validate(dumped)
        ptz._load(clean)
        self.assertEqual(ptz.settings_for("front")["fov_h"], 90)


if __name__ == "__main__":
    unittest.main()
