"""Tests for the depth package: scale calibration from person box heights,
ground projection (fixed and PTZ cameras), compute workload registration and
Auto resolution, settings validation, worker lifecycle and the targets payload.
The depth model is faked; nothing is downloaded, compiled or contacted.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="securitycam-depth-test-")
os.environ.setdefault("SECURITYCAM_DATA_DIR", _TMP)
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS", "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ["SECURITYCAM_AUTO_INSTALL_DEPS"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam import config, depth, presets, state  # noqa: E402
from securitycam.compute import advice, assignments  # noqa: E402
from securitycam.depth import bench, calib, geometry, model, render, worker  # noqa: E402
from securitycam.geo import Lens, Pose  # noqa: E402
from securitycam.validation import sanitize_esp_patch  # noqa: E402
from securitycam.webapp import app  # noqa: E402

presets.SCHEDULER.stop()


class FakeBackend:
    """Returns a fixed disparity map (relative inverse depth) of the input's size."""

    def __init__(self, dev: str, shape, value: float = 2.0):
        self.device = dev
        self.shape = tuple(shape)
        self.label = f"fake {dev}"
        self.value = value
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return np.full((1, self.shape[2], self.shape[3]), self.value, np.float32)


def fake_camera(cid: str = "cam", **settings) -> SimpleNamespace:
    s = dict(config.DEFAULT_ESP_SETTINGS)
    s.update(settings)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    return SimpleNamespace(camera_id=cid, esp_settings=s, latest_frame=frame, latest_frame_ts=time.perf_counter(),
                           frame_lock=threading.Lock(), closed=False)


def make_worker(cs, value: float = 2.0, tracks=None):
    made = []

    def factory(path, dev, shape):
        be = FakeBackend(dev, shape, value)
        made.append(be)
        return be

    w = worker.DepthWorker(cs, backend_factory=factory, model_path=lambda: Path("fake.onnx"))
    w._tracks = lambda: list(tracks or [])          # no ESP / motion state needed
    w.lens_pose = lambda fw, fh, s=None: (geometry.lens_for_fov(90, fw, fh), Pose(), False)
    return w, made


# ---------------------------------------------------------------------------- calibration
class CalibrationTests(unittest.TestCase):
    def test_scale_is_robust_median_of_box_height_samples(self):
        c = calib.ScaleCalibrator()
        rng = np.random.default_rng(1)
        for i in range(30):
            ref = 2.0 + i * 0.3                       # meters from the box height
            disp = 12.0 / ref * (1 + rng.normal(0, 0.03))   # true scale 12
            self.assertTrue(c.add(disp, ref, t=float(i)))
        c.add(100.0, 5.0, t=99.0)                      # one wild outlier
        cal = c.current()
        self.assertEqual(cal.mode, "auto")
        self.assertAlmostEqual(cal.scale, 12.0, delta=0.5)
        self.assertTrue(cal.ready)
        self.assertAlmostEqual(calib.depth_m(cal.scale, 12.0 / 4.0), 4.0, delta=0.2)

    def test_not_ready_with_few_or_scattered_samples(self):
        c = calib.ScaleCalibrator()
        for i in range(3):
            c.add(2.0, 5.0, t=float(i))
        self.assertFalse(c.current().ready)
        c2 = calib.ScaleCalibrator()
        rng = np.random.default_rng(2)
        for i in range(40):
            c2.add(float(rng.uniform(0.2, 8.0)), 5.0, t=float(i))
        self.assertLess(c2.current().confidence, calib.MIN_CONFIDENCE)
        self.assertFalse(c2.current().ready)

    def test_user_scale_overrides_auto(self):
        c = calib.ScaleCalibrator()
        for i in range(20):
            c.add(2.0, 5.0, t=float(i))
        cal = c.current(user_scale=30.0)
        self.assertEqual((cal.mode, cal.scale, cal.ready), ("user", 30.0, True))
        self.assertAlmostEqual(calib.user_scale_for(4.0, 2.5), 10.0)
        self.assertIsNone(calib.user_scale_for(4.0, 0.0))

    def test_one_sample_per_track_and_second(self):
        c = calib.ScaleCalibrator()
        self.assertTrue(c.add(2.0, 5.0, t=0.0, track=1))
        self.assertFalse(c.add(2.0, 5.0, t=0.5, track=1))
        self.assertTrue(c.add(2.0, 5.0, t=0.5, track=2))
        self.assertTrue(c.add(2.0, 5.0, t=1.2, track=1))

    def test_person_filter(self):
        ok = calib.person_ok
        self.assertTrue(ok("person", 0.8, [0.4, 0.2, 0.5, 0.8], 60, 300))
        self.assertFalse(ok("car", 0.8, [0.4, 0.2, 0.5, 0.8], 60, 300))           # not a person
        self.assertFalse(ok("person", 0.3, [0.4, 0.2, 0.5, 0.8], 60, 300))        # unsure
        self.assertFalse(ok("person", 0.8, [0.4, 0.0, 0.5, 0.8], 60, 300))        # cut off at the top
        self.assertFalse(ok("person", 0.8, [0.4, 0.2, 0.5, 0.8], 200, 220))       # sitting / squat box
        self.assertFalse(ok("person", 0.8, [0.4, 0.2, 0.5, 0.8], 8, 20))          # too small

    def test_worker_calibrates_from_tracked_persons(self):
        cs = fake_camera(depth=True)
        t = cs.latest_frame_ts
        # person box 400 px tall in a 1280x720 frame -> box-height distance 1000/400 = 2.5 m
        tracks = [(7, "person", 0.9, [600.0, 100.0, 700.0, 500.0], 1280, 720, t, True)]
        w, _ = make_worker(cs, value=4.0, tracks=tracks)
        for i in range(8):
            cs.latest_frame = cs.latest_frame.copy()        # a new frame each step
            w._clock = lambda i=i: 1000.0 + i * 1.5          # > SAMPLE_GAP_S apart
            self.assertTrue(w.step())
        cal = w.calibration()
        self.assertEqual(cal.mode, "auto")
        self.assertAlmostEqual(cal.scale, 4.0 * 2.5, places=3)
        self.assertTrue(cal.ready)
        self.assertTrue(w.latest().scan, "floor scan once calibrated")


# ---------------------------------------------------------------------------- geometry
class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.lens = Lens(90.0, 90.0)

    def test_center_point_is_straight_ahead(self):
        x, z = geometry.ground_point(0.5, 0.5, 5.0, self.lens)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(z, 5.0, places=6)

    def test_frame_edge_at_half_fov(self):
        x, z = geometry.ground_point(1.0, 0.5, 4.0, self.lens)     # 45 deg right
        self.assertAlmostEqual(x, 4.0, places=6)
        self.assertAlmostEqual(z, 4.0, places=6)
        self.assertAlmostEqual(geometry.bearing_deg(x, z), 45.0, places=6)

    def test_ptz_pan_rotates_world_position_only(self):
        pose = Pose(pan=90.0, tilt=0.0, zoom=1.0)
        cx, cz = geometry.ground_point(0.5, 0.5, 3.0, self.lens, pose, world=False)
        wx, wz = geometry.ground_point(0.5, 0.5, 3.0, self.lens, pose, world=True)
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cz, 3.0, places=6)
        self.assertAlmostEqual(wx, 3.0, places=6)        # panned right by 90 deg -> +x in the world
        self.assertAlmostEqual(wz, 0.0, places=6)

    def test_same_person_same_world_spot_while_panning(self):
        # a person at world bearing 20 deg, 6 m away; camera pans from 0 to 30 deg
        lens = Lens(60.0, 34.0)
        spots = []
        for pan in (0.0, 15.0, 30.0):
            rel = math.radians(20.0 - pan)
            fx, _ = lens.focal(1.0)
            u = 0.5 + math.tan(rel) * fx
            z_cam = 6.0 * math.cos(rel)                  # z-depth along the optical axis
            spots.append(geometry.ground_point(u, 0.5, z_cam, lens, Pose(pan, 0.0, 1.0), world=True))
        for sx, sz in spots:
            self.assertAlmostEqual(sx, 6.0 * math.sin(math.radians(20)), places=4)
            self.assertAlmostEqual(sz, 6.0 * math.cos(math.radians(20)), places=4)

    def test_tilt_shortens_horizontal_range(self):
        x, z = geometry.ground_point(0.5, 0.5, 10.0, self.lens, Pose(0.0, -30.0, 1.0))
        self.assertAlmostEqual(z, 10.0 * math.cos(math.radians(30)), places=4)

    def test_lens_for_fov_keeps_square_pixels(self):
        lens = geometry.lens_for_fov(90, 1280, 720)
        fx, fy = lens.focal()
        self.assertAlmostEqual(fx * 1280, fy * 720, places=3)

    def test_scan_points_and_region_median(self):
        d = np.full((27, 48), 5.0, np.float32)
        pts = geometry.scan_points(d, geometry.lens_for_fov(90, 48, 27))
        self.assertTrue(pts)
        self.assertTrue(all(abs(p[1] - 5.0) < 0.31 for p in pts))
        disp = np.zeros((100, 100), np.float32)
        disp[30:70, 40:60] = 3.0
        self.assertAlmostEqual(geometry.region_median(disp, [0.3, 0.1, 0.7, 0.9]), 3.0)

    def test_input_shape_is_static_multiple_of_14(self):
        self.assertEqual(model.input_shape(364, 1280, 720), (210, 364))
        self.assertEqual(model.input_shape(364, 720, 1280), (364, 210))
        h, w = model.input_shape(500, 1920, 1080)
        self.assertEqual((h % 14, w % 14), (0, 0))

    def test_colorize_and_blend(self):
        disp = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        col = render.colorize(disp, "inferno")
        self.assertEqual(col.shape, (10, 10, 3))
        frame = np.zeros((10, 10, 3), np.uint8)
        render.blend(frame, col, 0.5)
        self.assertGreater(int(frame.sum()), 0)


# ---------------------------------------------------------------------------- compute workload
class WorkloadTests(unittest.TestCase):
    def tearDown(self):
        model.clear_auto_failures()
        assignments.reset()

    def test_registered_with_all_devices(self):
        meta = assignments.WORKLOADS["depth"]
        self.assertEqual(set(meta["devices"]), {"cuda", "npu", "igpu", "cpu"})
        self.assertEqual(assignments.assignment("depth"), "auto")
        self.assertTrue(assignments.is_default())

    def test_auto_prefers_npu_then_gpu_then_igpu_then_cpu(self):
        avail = lambda have: (lambda d: d in have)  # noqa: E731
        self.assertEqual(model.auto_device(avail({"npu", "cuda", "igpu"})), "npu")
        self.assertEqual(model.auto_device(avail({"cuda", "igpu"})), "cuda")
        self.assertEqual(model.auto_device(avail({"igpu"})), "igpu")
        self.assertEqual(model.auto_device(avail(set())), "cpu")
        model.mark_auto_failed("npu", "boom")
        self.assertEqual(model.auto_device(avail({"npu", "cuda"})), "cuda")

    def test_resolve_user_choice_and_auto(self):
        with mock.patch.object(assignments, "effective", return_value={
                "requested": "igpu", "source": "global", "device": "igpu", "fallback_reason": None, "error": None}):
            r = model.resolve()
        self.assertEqual((r["device"], r["source"]), ("igpu", "user"))
        with mock.patch.object(model, "auto_device", return_value="npu"):
            r = model.resolve()
        self.assertEqual((r["device"], r["source"]), ("npu", "auto"))

    def test_user_device_failure_falls_back_to_auto(self):
        cs = fake_camera(depth=True)
        w = worker.DepthWorker(cs, backend_factory=lambda *a: (_ for _ in ()).throw(RuntimeError("no npu")),
                               model_path=lambda: Path("x"))
        with mock.patch.object(model, "resolve", return_value={"requested": "npu", "device": "npu", "source": "user",
                                                                "fallback_reason": None, "error": None}), \
                mock.patch.object(assignments, "mark_failed") as mf:
            self.assertFalse(w.step())
        mf.assert_called_once()
        self.assertEqual(mf.call_args[0][:2], ("depth", "npu"))

    def test_notices_for_every_device(self):
        for dev in assignments.WORKLOADS["depth"]["devices"]:
            texts = " ".join(n["text"] for n in advice.notices("depth", dev))
            self.assertIn("Depth takes", texts, dev)
        self.assertEqual(advice.notices("depth", "auto"), [])

    def test_benchmark_runner_times_every_row(self):
        job = {"rows": [{"device": d} for d in ("auto", "npu", "cpu")]}
        rows = {}

        def set_row(dev, **kw):
            rows.setdefault(dev, {}).update(kw)

        with mock.patch("securitycam.deps.asset_path", return_value=Path("fake.onnx")):
            bench.run(job, set_row, lambda fn: (fn(), 12.5)[1], factory=lambda p, d, s: FakeBackend(d, s))
        self.assertEqual({d: r["status"] for d, r in rows.items()}, {"auto": "done", "npu": "done", "cpu": "done"})
        self.assertEqual(rows["npu"]["ms"], 12.5)


# ---------------------------------------------------------------------------- settings
class SettingsTests(unittest.TestCase):
    def test_defaults_are_off(self):
        d = config.DEFAULT_ESP_SETTINGS
        self.assertIs(d["depth"], False)
        self.assertIs(d["depth_view"], False)
        self.assertEqual(d["depth_scale"], 0.0)

    def test_validation(self):
        out = sanitize_esp_patch({"depth": "on", "depth_view": 1, "depth_rate": 99, "depth_view_palette": "rainbow",
                                  "depth_size": 100, "depth_fov": 200, "depth_view_opacity": -1, "depth_scale": "x"})
        self.assertEqual(out["depth"], True)
        self.assertEqual(out["depth_view"], True)
        self.assertEqual(out["depth_rate"], 10.0)
        self.assertEqual(out["depth_size"], 196)
        self.assertEqual(out["depth_fov"], 170)
        self.assertEqual(out["depth_view_opacity"], 0.0)
        self.assertNotIn("depth_view_palette", out)
        self.assertNotIn("depth_scale", out)
        self.assertEqual(sanitize_esp_patch({"depth_view_palette": "INFERNO"}), {"depth_view_palette": "inferno"})

    def test_snapshot_snaps_size(self):
        s = worker.DepthSettings.of(fake_camera(depth=True, depth_size=300, depth_rate="abc"))
        self.assertEqual(s.size % 14, 0)
        self.assertEqual(s.rate, 3.0)


# ---------------------------------------------------------------------------- lifecycle
class LifecycleTests(unittest.TestCase):
    def tearDown(self):
        worker.stop_all()

    def test_no_thread_when_disabled(self):
        cs = fake_camera("off-cam", depth=False)
        started = []
        with mock.patch.object(worker, "WORKER_FACTORY", side_effect=lambda c: started.append(c)):
            depth.on_render(cs)
            self.assertIsNone(worker.ensure(cs))
        self.assertEqual(started, [])
        self.assertIsNone(worker.get("off-cam"))
        names = [t.name for t in threading.enumerate()]
        self.assertFalse(any(n == "DepthWorker-off-cam" for n in names))

    def test_starts_once_when_enabled_and_pings_afterwards(self):
        cs = fake_camera("on-cam", depth=True)

        class Fake:
            starts = 0

            def __init__(self, c):
                self.camera_id = c.camera_id
                self.pings = 0

            def ping(self):
                self.pings += 1

            def start(self):
                Fake.starts += 1

            def alive(self):
                return Fake.starts > 0

            def stop(self):
                pass

        with mock.patch.object(worker, "WORKER_FACTORY", Fake):
            w1 = worker.ensure(cs)
            w2 = worker.ensure(cs)
        self.assertIs(w1, w2)
        self.assertEqual(Fake.starts, 1)
        self.assertEqual(w1.pings, 2)

    def test_worker_stops_when_unwatched_or_disabled(self):
        cs = fake_camera("idle-cam", depth=True)
        now = [100.0]
        w = worker.DepthWorker(cs, backend_factory=lambda p, d, s: FakeBackend(d, s), model_path=lambda: Path("x"),
                               clock=lambda: now[0])
        self.assertTrue(w.should_run())
        now[0] += worker.IDLE_STOP_S + 0.1
        self.assertFalse(w.should_run())          # nobody renders the camera any more
        w.ping()
        self.assertTrue(w.should_run())
        cs.esp_settings["depth"] = False
        self.assertFalse(w.should_run())

    def test_real_thread_runs_and_exits(self):
        cs = fake_camera("thread-cam", depth=True, depth_rate=10)
        w, made = make_worker(cs)
        with mock.patch.object(worker, "WORKER_FACTORY", lambda c: w):
            worker.ensure(cs)
            deadline = time.time() + 3
            while w.latest() is None and time.time() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(w.latest())
            self.assertEqual(w.state, "running")
            cs.esp_settings["depth"] = False
            w.thread.join(timeout=3)
        self.assertFalse(w.thread.is_alive())
        self.assertIsNone(worker.get("thread-cam"))

    def test_downloading_state_while_model_missing(self):
        cs = fake_camera(depth=True)
        w = worker.DepthWorker(cs, backend_factory=lambda p, d, s: FakeBackend(d, s), model_path=lambda: None)
        self.assertFalse(w.step())
        self.assertEqual(w.state, "downloading")


# ---------------------------------------------------------------------------- payload / API
class PayloadTests(unittest.TestCase):
    def tearDown(self):
        worker.stop_all()

    def _payload(self):
        return {"ts": 0, "frame": {"w": 1280, "h": 720}, "locked": None,
                "targets": [{"id": 3, "label": "person", "conf": 0.9, "box": [0.45, 0.2, 0.55, 0.8], "distance": 7.0}]}

    def test_unchanged_when_disabled(self):
        cs = fake_camera("p-off", depth=False)
        p = depth.decorate_targets(cs, self._payload())
        self.assertNotIn("depth", p)
        self.assertNotIn("ground", p["targets"][0])

    def test_box_distance_ground_before_calibration(self):
        cs = fake_camera("p-box", depth=True)
        p = depth.decorate_targets(cs, self._payload())
        tg = p["targets"][0]
        self.assertEqual(tg["ground_src"], "box")
        self.assertAlmostEqual(tg["ground"][1], 7.0, delta=0.01)
        self.assertEqual(p["depth"]["note"], depth.NOTE)
        self.assertNotIn("depth_m", tg)

    def test_depth_distance_and_ground_when_calibrated(self):
        cs = fake_camera("p-cal", depth=True, depth_scale=10.0)      # user calibration
        w, _ = make_worker(cs, value=2.0)                             # disparity 2 -> 5 m
        self.assertTrue(w.step())
        with worker._reg_lock:
            worker._workers["p-cal"] = w
        p = depth.decorate_targets(cs, self._payload())
        tg = p["targets"][0]
        self.assertAlmostEqual(tg["depth_m"], 5.0, places=2)
        self.assertEqual(tg["distance"], tg["depth_m"])
        self.assertEqual(tg["distance_src"], "depth")
        self.assertEqual(tg["distance_box"], 7.0)
        self.assertEqual(tg["ground_src"], "depth")
        self.assertAlmostEqual(tg["ground"][0], 0.0, delta=0.01)
        self.assertAlmostEqual(tg["ground"][1], 5.0, delta=0.01)
        st = p["depth"]
        self.assertEqual(st["state"], "running")
        self.assertEqual(st["calibration"]["mode"], "user")
        self.assertTrue(st["scan"]["pts"])

    def test_api_status_and_calibration(self):
        client = app.test_client()
        cs = state.get_camera_state("front")
        r = client.get("/api/cameras/front/depth")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["depth"]["enabled"])
        self.assertEqual(client.get("/api/cameras/nope/depth").status_code, 404)
        r = client.post("/api/cameras/front/depth/calibrate", json={"target_id": 1, "distance_m": 4})
        self.assertEqual(r.status_code, 409)                    # depth not running
        r = client.post("/api/cameras/front/depth/calibrate", json={"target_id": 1, "distance_m": 9999})
        self.assertEqual(r.get_json()["error"], "invalid_distance")
        cs.esp_settings["depth_scale"] = 12.0
        r = client.post("/api/cameras/front/depth/calibrate", json={"reset": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(cs.esp_settings["depth_scale"], 0.0)

    def test_compute_payload_lists_depth(self):
        d = app.test_client().get("/api/compute").get_json()
        dw = next(w for w in d["workloads"] if w["id"] == "depth")
        self.assertEqual(dw["assignment"], "auto")
        self.assertIn(dw["auto"]["device"], ("npu", "cuda", "igpu", "cpu"))
        self.assertIn(dw["state"], ("idle", "active", "preparing"))


if __name__ == "__main__":
    unittest.main()
