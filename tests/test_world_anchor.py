"""World-anchored PTZ overlays: projection math, pose fusion on a synthetic
panorama camera, alarm robustness during pans, persistence migration, API.

No camera or ONVIF device is ever contacted (ONVIF is faked).

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-anchor-"))
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS",
                      "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from securitycam import anchor, config, geo, persistence, ptz, state  # noqa: E402
from securitycam.alerts import engine as alert_engine  # noqa: E402
from securitycam.alerts.store import AlertStore  # noqa: E402
from securitycam.esp import motion  # noqa: E402
from securitycam.geo import Lens, Pose  # noqa: E402
from securitycam.ptz import pose as ptz_pose  # noqa: E402
from synthetic_ptz import SyntheticPtz, draw_person, make_panorama  # noqa: E402

LENS = Lens(100.0, 56.0)
_PANO = None


def panorama():
    global _PANO
    if _PANO is None:
        _PANO = make_panorama()
    return _PANO


def zone_error_px(cam: SyntheticPtz, rays, true: Pose, est: Pose) -> float:
    a, fa = cam.project_px(rays, true)
    b, fb = cam.project_px(rays, est)
    ok = fa & fb
    return float(np.max(np.linalg.norm(a[ok] - b[ok], axis=1))) if ok.any() else 0.0


# ----------------------------------------------------------------------------- geometry
class GeoTests(unittest.TestCase):
    def test_round_trip_many_poses(self):
        rng = np.random.default_rng(1)
        uv = rng.random((50, 2))
        for pose in (Pose(), Pose(37.0, -12.0, 1.0), Pose(-170.0, 40.0, 2.5), Pose(359.0, -60.0, 0.7)):
            rays = geo.image_to_rays(uv, pose, LENS)
            back, front = geo.rays_to_image(rays, pose, LENS)
            self.assertTrue(front.all())
            np.testing.assert_allclose(back, uv, atol=1e-9)
            azel = geo.image_to_world(uv, pose, LENS)
            back2, _ = geo.world_to_image(azel, pose, LENS)
            np.testing.assert_allclose(back2, uv, atol=1e-9)

    def test_center_is_pose_axis_and_edges_match_fov(self):
        pose = Pose(25.0, 10.0, 1.0)
        az, el = geo.image_to_world(((0.5, 0.5),), pose, LENS)[0]
        self.assertAlmostEqual(az, 25.0, places=9)
        self.assertAlmostEqual(el, 10.0, places=9)
        # at tilt 0 the left/right edge centers are exactly +-fov/2 away
        l, r = geo.image_to_world(((0.0, 0.5), (1.0, 0.5)), Pose(), LENS)
        self.assertAlmostEqual(l[0], -50.0, places=9)
        self.assertAlmostEqual(r[0], 50.0, places=9)
        t, b = geo.image_to_world(((0.5, 0.0), (0.5, 1.0)), Pose(), LENS)
        self.assertAlmostEqual(t[1], 28.0, places=9)
        self.assertAlmostEqual(b[1], -28.0, places=9)

    def test_zoom_changes_fov(self):
        fh, fv = LENS.effective_fov(2.0)
        self.assertAlmostEqual(math.tan(math.radians(fh / 2)), math.tan(math.radians(50)) / 2, places=9)
        # a point seen at zoom 1 lands twice as far from the center at zoom 2
        uv, _ = geo.transfer(((0.6, 0.45),), Pose(), Pose(0, 0, 2.0), LENS)
        np.testing.assert_allclose(uv[0], (0.7, 0.4), atol=1e-9)

    def test_pan_moves_points_left_and_wraps(self):
        uv, front = geo.transfer(((0.5, 0.5),), Pose(179.0, 0, 1), Pose(-179.0, 0, 1), LENS)
        self.assertTrue(front[0])
        self.assertLess(uv[0, 0], 0.5)          # camera turned right by 2 deg across the seam
        self.assertAlmostEqual(geo.angle_between(Pose(179.0), Pose(-179.0)), 2.0, places=6)

    def test_polygon_clipping(self):
        rays = geo.image_to_rays(geo.box_corners([0.3, 0.3, 0.7, 0.7]), Pose(), LENS)
        full = geo.project_polygon(rays, Pose(), LENS)
        np.testing.assert_allclose(np.array(full), geo.box_corners([0.3, 0.3, 0.7, 0.7]), atol=1e-9)
        # partly visible after a pan: clipped to the frame, still a valid polygon
        part = geo.project_polygon(rays, Pose(40.0), LENS)
        self.assertGreaterEqual(len(part), 3)
        arr = np.array(part)
        self.assertTrue((arr >= -1e-9).all() and (arr <= 1 + 1e-9).all())
        self.assertAlmostEqual(arr[:, 0].min(), 0.0, places=9)
        # behind the camera: nothing (no mirrored wrap-around)
        self.assertEqual(geo.project_polygon(rays, Pose(180.0), LENS), [])
        # straddling the camera plane (wide zone, camera turned ~90 deg): only the front part
        wide = geo.image_to_rays(geo.box_corners([0.0, 0.2, 1.0, 0.8]), Pose(), LENS)
        side = geo.project_polygon(wide, Pose(95.0), LENS)
        if side:
            arr = np.array(side)
            self.assertTrue((arr >= -1e-9).all() and (arr <= 1 + 1e-9).all())
        seg = geo.project_segment(geo.image_to_rays(((0.2, 0.5), (0.8, 0.5)), Pose(), LENS), Pose(170.0), LENS)
        self.assertIsNone(seg)

    def test_homography_matches_transfer(self):
        p0, p1 = Pose(10.0, 5.0, 1.0), Pose(14.0, 3.0, 1.3)
        hmat = geo.homography(p0, p1, LENS, (320, 180), (320, 180))
        pts = np.array([[10.0, 20.0], [160.0, 90.0], [300.0, 170.0]])
        hp = np.c_[pts, np.ones(3)] @ hmat.T
        hp = hp[:, :2] / hp[:, 2:]
        uv = (pts + 0.5) / (320, 180)
        t, _ = geo.transfer(uv, p0, p1, LENS)
        np.testing.assert_allclose(hp, t * (320, 180) - 0.5, atol=1e-6)

    def test_point_in_polygon(self):
        poly = [(0, 0), (10, 0), (12, 8), (-2, 8)]
        self.assertTrue(geo.point_in_polygon(5, 4, poly))
        self.assertFalse(geo.point_in_polygon(13, 4, poly))


# ----------------------------------------------------------------------------- pose fusion (synthetic camera)
class FusionHarness:
    def __init__(self, size=(640, 360), estimate_zoom=False, onvif_noise=None, onvif_every=8, seed=3,
                 people=False, onvif_scale=(1.0, 1.0)):
        self.cam = SyntheticPtz(panorama(), size=size, lens=LENS)
        self.f = ptz_pose.PoseFusion("synthetic", clock=lambda: self.t)
        self.f.lens_override = LENS
        self.f.ptz_known = True
        self.f.estimate_zoom = estimate_zoom
        self.f._ranges = lambda: (360.0, 114.0, 10.0)
        self.t = 100.0
        self.i = 0
        self.rng = np.random.default_rng(seed)
        self.onvif_noise = onvif_noise
        self.onvif_every = onvif_every
        self.onvif_scale = onvif_scale
        self.people = people
        self.true = Pose()

    def step(self, pose: Pose, dt=0.04):
        self.t += dt
        self.i += 1
        self.true = pose
        frame = self.cam.render(pose)
        boxes = []
        if self.people:
            # two people walking across the picture (foreground, not part of the room)
            w, h = self.cam.size
            for k, speed in ((0, 3.0), (1, -2.2)):
                x = int((self.i * speed * 4 + k * 300) % w)
                boxes.append(draw_person(frame, x, int(h * (0.75 + 0.1 * k)), scale=h / 360.0))
        self.f.process(frame, self.t, boxes)
        if self.onvif_noise is not None and self.i % self.onvif_every == 0:
            n = self.rng.normal(0.0, self.onvif_noise, 2)
            self.f.on_onvif_position({"x": (pose.pan * self.onvif_scale[0] + n[0]) / 180.0,
                                      "y": (pose.tilt * self.onvif_scale[1] + n[1]) / 57.0, "zoom": None})
        return self.f.current()

    def move(self, target: Pose, frames: int, command=True):
        start = self.true
        if command:
            self.f.on_command("motion", (1.0, 0.0, 0.0))
        for k in range(1, frames + 1):
            a = k / frames
            self.step(Pose(start.pan + (target.pan - start.pan) * a, start.tilt + (target.tilt - start.tilt) * a,
                           start.zoom * (target.zoom / start.zoom) ** a))
        if command:
            self.f.on_command("stop")

    def hold(self, frames: int):
        for _ in range(frames):
            self.step(self.true)


class PoseFusionTests(unittest.TestCase):
    def assert_zone_sticks(self, h: FusionHarness, rays, tol_px: float, label: str):
        err = zone_error_px(h.cam, rays, h.true, h.f.current())
        self.assertLess(err, tol_px, f"{label}: zone off by {err:.2f}px (est {h.f.current()}, true {h.true})")
        return err

    def test_vision_only_pan_tilt(self):
        h = FusionHarness()
        h.hold(3)
        rays = geo.image_to_rays(geo.box_corners([0.35, 0.4, 0.6, 0.75]), h.f.current(), LENS)
        h.move(Pose(30.0, 0.0), 40)
        h.hold(40)
        self.assert_zone_sticks(h, rays, 3.0, "after pan right")
        h.move(Pose(30.0, 15.0), 30)
        h.hold(40)
        self.assert_zone_sticks(h, rays, 3.0, "after tilt up")
        h.move(Pose(-5.0, -8.0), 60)
        h.hold(40)
        self.assert_zone_sticks(h, rays, 4.0, "after pan left / tilt down")
        h.move(Pose(0.0, 0.0), 30)
        h.hold(40)
        self.assert_zone_sticks(h, rays, 4.0, "back home")

    def test_errors_during_motion_stay_small(self):
        h = FusionHarness()
        h.hold(3)
        rays = geo.image_to_rays(geo.box_corners([0.3, 0.3, 0.7, 0.7]), Pose(), LENS)
        worst = 0.0
        h.f.on_command("motion", (1.0, 0.0, 0.0))
        for k in range(1, 75):
            h.step(Pose(k * 0.6, k * 0.1))       # 15 deg/s pan, 2.5 deg/s tilt
            worst = max(worst, zone_error_px(h.cam, rays, h.true, h.f.current()))
        self.assertLess(worst, 4.0)

    def test_zoom(self):
        h = FusionHarness(estimate_zoom=True)
        h.hold(3)
        rays = geo.image_to_rays(geo.box_corners([0.4, 0.4, 0.6, 0.6]), Pose(), LENS)
        h.move(Pose(5.0, 2.0, 2.0), 40)
        h.hold(30)
        self.assert_zone_sticks(h, rays, 4.0, "zoomed in")
        self.assertAlmostEqual(h.f.current().zoom, 2.0, delta=0.05)
        h.move(Pose(0.0, 0.0, 1.0), 40)
        h.hold(30)
        self.assert_zone_sticks(h, rays, 3.0, "zoomed out")

    def test_moving_foreground_people(self):
        h = FusionHarness(people=True)
        h.hold(3)
        rays = geo.image_to_rays(geo.box_corners([0.35, 0.3, 0.65, 0.6]), Pose(), LENS)
        h.move(Pose(25.0, 6.0), 50)
        h.hold(30)
        self.assert_zone_sticks(h, rays, 4.0, "pan with walking people")

    def test_idle_camera_does_not_drift(self):
        h = FusionHarness(people=True)
        h.hold(5)
        p0 = h.f.current()
        h.hold(150)                              # 6 s still, people walking through
        p1 = h.f.current()
        self.assertEqual(p0, p1, "pose must be frozen while the camera is idle")

    def test_noisy_onvif_corrects_vision_drift(self):
        h = FusionHarness(onvif_noise=0.15)
        h.hold(20)                               # first readings align the ONVIF frame
        self.assertIsNotNone(h.f.offset)
        rays = geo.image_to_rays(geo.box_corners([0.35, 0.4, 0.6, 0.75]), h.f.current(), LENS)
        h.move(Pose(35.0, 8.0), 40)
        h.hold(10)
        # inject a vision drift of 1.5 deg (as many keyframe switches could)
        p = h.f.current()
        with h.f.lock:
            h.f._shift(Pose(p.pan + 1.5, p.tilt - 0.5, p.zoom))
        h.hold(150)                              # 6 s settled: ONVIF pulls it back
        err = self.assert_zone_sticks(h, rays, 4.0, "ONVIF corrected")
        self.assertEqual(h.f.source(), "both")
        self.assertLess(h.f.drift_estimate()[0], 0.6)
        self.assertLess(err, 4.0)

    def test_inverted_and_frozen_onvif_are_detected(self):
        inv = FusionHarness(onvif_noise=0.05, onvif_scale=(-1.0, 1.0))
        inv.hold(20)
        inv.move(Pose(20.0, 0.0), 30)
        inv.hold(40)
        self.assertEqual(inv.f.sign[0], -1.0)
        inv.move(Pose(-10.0, 3.0), 40)
        inv.hold(60)
        rays = geo.image_to_rays(geo.box_corners([0.3, 0.3, 0.7, 0.7]), Pose(), LENS)
        self.assertLess(zone_error_px(inv.cam, rays, inv.true, inv.f.current()), 5.0)

        frozen = FusionHarness(onvif_noise=0.0, onvif_scale=(0.0, 0.0))
        frozen.hold(20)
        for target in (Pose(15.0, 0.0), Pose(-5.0, 4.0), Pose(10.0, 0.0)):
            frozen.move(target, 30)
            frozen.hold(40)
        self.assertTrue(frozen.f.onvif_stale)
        self.assertEqual(frozen.f.source(), "vision")

    def test_blind_vision_follows_onvif_and_command_prior(self):
        h = FusionHarness(onvif_noise=0.05, onvif_every=3)
        h.hold(20)
        blank = np.full((360, 640, 3), 90, np.uint8)
        h.f.on_command("relative", (12.0, 0.0))
        for k in range(1, 30):
            h.t += 0.04
            h.true = Pose(12.0 * k / 29, 0.0)
            h.f.process(blank, h.t)
            if k % 3 == 0 or k == 29:
                h.f.on_onvif_position({"x": h.true.pan / 180.0, "y": 0.0})
        h.t += 0.04
        h.f.process(blank, h.t)
        self.assertEqual(h.f.vision_state, "lost")
        self.assertAlmostEqual(h.f.current().pan, 12.0, delta=0.5)

        # no ONVIF at all: the commanded relative move is applied once settled
        g = FusionHarness()
        g.hold(5)
        g.f.on_command("relative", (8.0, -2.0))
        for _ in range(60):
            g.t += 0.04
            g.f.process(blank, g.t)
        self.assertAlmostEqual(g.f.current().pan, 8.0, delta=1e-6)
        self.assertAlmostEqual(g.f.current().tilt, -2.0, delta=1e-6)

    def test_pose_history_lookup(self):
        f = ptz_pose.PoseFusion("hist")
        for i in range(10):
            f.pose = Pose(i * 1.0)
            f._record(float(i))
        self.assertAlmostEqual(f.pose_at(3.5).pan, 3.5)
        self.assertAlmostEqual(f.pose_at(100.0).pan, 9.0)
        self.assertAlmostEqual(f.speed_at(5.0), 1.0, places=6)

    def test_perf_1080p(self):
        cam = SyntheticPtz(panorama(), size=(1920, 1080), lens=LENS)
        frames = [cam.render(Pose(i * 0.6, i * 0.1)) for i in range(30)]
        f = ptz_pose.PoseFusion("perf")
        f.lens_override, f.ptz_known, f.commanding = LENS, True, True
        times = []
        for i, fr in enumerate(frames):
            a = time.perf_counter()
            f.process(fr, 10.0 + i * 0.04)
            times.append(time.perf_counter() - a)
        ms = float(np.median(times[2:]) * 1000)
        print(f"\n[perf] pose fusion at 1080p while moving: median {ms:.2f} ms/frame "
              f"(idle: every {ptz_pose.IDLE_EVERY}th frame)")
        self.assertLess(ms, 8.0)                 # generous for slow CI machines; ~2 ms here


# ----------------------------------------------------------------------------- alarms during pans
def _cam_state(camera_id="front"):
    return types.SimpleNamespace(
        camera_id=camera_id, label="Front", esp_settings=dict(config.DEFAULT_ESP_SETTINGS, abandoned=True),
        esp_motion=motion.MotionState(), track_history={}, esp_zones=[])


class AlarmRobustnessTests(unittest.TestCase):
    W, H = 1280, 720

    def setUp(self):
        ptz_pose.reset()
        self.tmp = Path(tempfile.mkdtemp())
        self.store = AlertStore(self.tmp / "alerts.json")
        self._orig_store = alert_engine._store
        alert_engine._store = self.store
        alert_engine._engines.clear()
        self.events = []

    def tearDown(self):
        alert_engine._store = self._orig_store
        alert_engine._engines.clear()
        ptz_pose.reset()

    def _fusion(self, cs):
        f = ptz_pose.get(cs.camera_id)
        f.lens_override, f.ptz_known = LENS, True
        return f

    def _engine(self, cs):
        eng = alert_engine.get(cs.camera_id)
        eng.emit = lambda *a, **k: self.events.append(a[1:5]) or True
        return eng

    def _run(self, cs, f, pan_of_t, people, seconds=3.0, fps=25, yolo_every=3):
        """people: list of (key, fn(t) -> world ray). Returns nothing; events in self.events."""
        eng = self._engine(cs)
        t = 1000.0
        n = int(seconds * fps)
        for i in range(n + 1):
            t = 1000.0 + i / fps
            pose = Pose(pan_of_t(i / fps), 0.0)
            f.pose = pose
            f._record(t)
            if i % yolo_every == 0:
                for key, ray_fn in people:
                    uv, front = geo.rays_to_image(ray_fn(i / fps)[None, :], pose, LENS)
                    if not front[0]:
                        continue
                    px, py = uv[0, 0] * self.W, uv[0, 1] * self.H
                    cs.track_history.setdefault(key, []).append((int(px), int(py), t))
                    cs.esp_motion.tracks[key] = types.SimpleNamespace(
                        det=types.SimpleNamespace(cls_name="person", obj_id=key), missing_since=None)
            eng.update(cs, [], self.W, self.H, t, [], cs.esp_settings)
        return eng

    def _wire(self, cs, anchor_pose=Pose()):
        return self.store.add_tripwire(cs.camera_id, {"a": [0.5, 0.2], "b": [0.5, 0.95], "alarm": True,
                                                      "anchor": anchor_pose.as_dict()})

    def test_no_false_crossing_while_panning(self):
        cs = _cam_state()
        f = self._fusion(cs)
        self._wire(cs)
        # a person standing still in the room, just right of the wire
        still = geo.image_to_rays(((0.56, 0.7),), Pose(), LENS)[0]
        eng = self._run(cs, f, lambda s: 10.0 * s, [(1, lambda s: still)])
        self.assertEqual(self.events, [])
        self.assertEqual(eng.counts()["t1"], {"in": 0, "out": 0})

    def test_fast_pan_no_false_crossing(self):
        cs = _cam_state()
        f = self._fusion(cs)
        self._wire(cs)
        still = geo.image_to_rays(((0.53, 0.7),), Pose(), LENS)[0]
        self._run(cs, f, lambda s: 40.0 * s, [(1, lambda s: still)], seconds=1.5)
        self.assertEqual(self.events, [])

    def test_image_space_would_have_crossed(self):
        """Control: the same pan without anchoring produces the false alarm the user saw."""
        cs = _cam_state()
        cs.esp_settings["world_anchor"] = False
        f = self._fusion(cs)
        self._wire(cs)
        still = geo.image_to_rays(((0.56, 0.7),), Pose(), LENS)[0]
        self._run(cs, f, lambda s: 10.0 * s, [(1, lambda s: still)])
        self.assertEqual(len(self.events), 1)

    def test_real_crossing_during_pan_counts_once(self):
        cs = _cam_state()
        f = self._fusion(cs)
        self._wire(cs)
        a = geo.image_to_rays(((0.35, 0.7),), Pose(), LENS)[0]
        b = geo.image_to_rays(((0.65, 0.7),), Pose(), LENS)[0]

        def walker(s):
            k = min(1.0, s / 3.0)
            r = a * (1 - k) + b * k
            return r / np.linalg.norm(r)
        eng = self._run(cs, f, lambda s: -6.0 * s, [(7, walker)])
        self.assertEqual(len(self.events), 1)
        c = eng.counts()["t1"]
        self.assertEqual(c["in"] + c["out"], 1)

    def test_wire_drawn_at_other_pose(self):
        """A wire drawn while the camera looked 20 deg right is evaluated in that view."""
        cs = _cam_state()
        f = self._fusion(cs)
        self._wire(cs, Pose(20.0, 0.0))
        still = geo.image_to_rays(((0.56, 0.7),), Pose(20.0, 0.0), LENS)[0]
        self._run(cs, f, lambda s: 10.0 * s, [(1, lambda s: still)])
        self.assertEqual(self.events, [])

    def test_zone_dwell_survives_pan_and_abandoned_object_stays(self):
        cs = _cam_state()
        f = self._fusion(cs)
        cs.esp_zones = [{"id": "z1", "label": "Gate", "box": [0.3, 0.4, 0.7, 0.95], "color": "#ffd21f",
                         "anchor": Pose().as_dict()}]
        self.store.set_rule(cs.camera_id, "z1", {"enabled": True, "min_dwell_s": 2.0})
        eng = self._engine(cs)
        person = geo.image_to_rays(((0.5, 0.8),), Pose(), LENS)[0]
        bag = geo.image_to_rays(geo.box_corners([0.42, 0.5, 0.47, 0.56]), Pose(), LENS)
        from securitycam.esp.detections import Detection
        first_t = None
        for i in range(0, 100):
            t = 2000.0 + i * 0.04
            pose = Pose(0.25 * i, 0.0)         # 6 deg/s pan
            f.pose = pose
            f._record(t)
            view = anchor.view_for(cs, t)
            zones = anchor.project_zones(view, cs.esp_zones, self.W, self.H)
            uv, _ = geo.rays_to_image(person[None, :], pose, LENS)
            cx, cy = int(uv[0, 0] * self.W), int(uv[0, 1] * self.H)
            buv, _ = geo.rays_to_image(bag, pose, LENS)
            bx1, by1 = buv.min(axis=0) * (self.W, self.H)
            bx2, by2 = buv.max(axis=0) * (self.W, self.H)
            dets = [Detection(0, 1, 0, "person", cx - 20, cy - 60, cx + 20, cy, cx, cy - 30, 0.9, track_key=1),
                    Detection(1, 2, 24, "backpack", int(bx1), int(by1), int(bx2), int(by2),
                              int((bx1 + bx2) / 2), int((by1 + by2) / 2), 0.9, track_key=2)]
            dets[0] = Detection(0, 1, 0, "person", cx - 20, cy - 60, cx + 20, cy, cx, cy, 0.9, track_key=1)
            for d in dets:
                cs.esp_motion.tracks[d.track_key] = types.SimpleNamespace(
                    det=d, missing_since=None)
            eng.update(cs, dets, self.W, self.H, t, zones, cs.esp_settings, view)
            if first_t is None and eng.candidates:
                first_t = eng.candidates[0].first_t
        zone_events = [e for e in self.events if e[0] == "zone"]
        self.assertEqual(len(zone_events), 1, self.events)
        self.assertEqual(len(eng.candidates), 1)
        self.assertEqual(eng.candidates[0].first_t, first_t, "pan must not restart the abandoned timer")


# ----------------------------------------------------------------------------- persistence / API
class PersistenceMigrationTests(unittest.TestCase):
    def setUp(self):
        ptz_pose.reset()
        for cs in state.camera_states.values():
            cs.esp_zones[:] = []
            cs.esp_zone_seq = 0
            cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)

    def tearDown(self):
        ptz_pose.reset()

    def test_old_zones_get_anchor_at_load_pose(self):
        f = ptz_pose.get("front")
        f.ptz_known = True
        f.pose = Pose(12.5, -3.0, 1.0)
        clean, warnings = persistence.validate_cameras(
            {"front": {"zones": [{"id": "z1", "label": "Old", "box": [0.1, 0.1, 0.3, 0.4], "color": "#ffd21f"}],
                       "zone_seq": 1}})
        persistence.load_cameras(clean)
        z = state.camera_states["front"].esp_zones[0]
        self.assertEqual(z["anchor"], {"pan": 12.5, "tilt": -3.0, "zoom": 1.0})
        dumped = persistence.dump_cameras()["front"]["zones"][0]
        self.assertEqual(dumped["anchor"], z["anchor"])
        self.assertEqual(dumped["box"], [0.1, 0.1, 0.3, 0.4])
        # unknown pose (no PTZ ever seen): identity
        ptz_pose.reset()
        persistence.load_cameras(clean)
        self.assertEqual(state.camera_states["front"].esp_zones[0]["anchor"], Pose().as_dict())

    def test_old_alerts_json_tripwires_migrated(self):
        tmp = Path(tempfile.mkdtemp()) / "alerts.json"
        tmp.write_text(json.dumps({"version": 1, "cameras": {"front": {
            "zone_rules": {}, "tripwires": [{"id": "t1", "a": [0.2, 0.5], "b": [0.8, 0.5], "label": "Door"}],
            "tripwire_seq": 1}}}), encoding="utf-8")
        f = ptz_pose.get("front")
        f.pose = Pose(-7.0, 2.0, 1.0)
        st = AlertStore(tmp)
        wire = st.camera_config("front")["tripwires"][0]
        self.assertEqual(wire["anchor"], {"pan": -7.0, "tilt": 2.0, "zoom": 1.0})
        self.assertEqual(wire["a"], [0.2, 0.5])
        saved = json.loads(tmp.read_text(encoding="utf-8"))
        self.assertEqual(saved["cameras"]["front"]["tripwires"][0]["anchor"]["pan"], -7.0)

    def test_ptz_section_round_trip(self):
        f = ptz_pose.get("front")
        f.ptz_known = True
        f.pose = Pose(33.0, -4.0, 1.5)
        f.offset = (1.25, -0.5)
        f.preset_poses["2"] = Pose(90.0, 0.0, 1.0)
        dumped = ptz._dump()
        ptz_pose.reset()
        clean, warnings = ptz._validate(json.loads(json.dumps(dumped)))
        ptz._load(clean)
        g = ptz_pose.peek("front")
        self.assertTrue(g.ptz_known)
        self.assertEqual(g.current(), Pose(33.0, -4.0, 1.5))
        self.assertEqual(g.offset, (1.25, -0.5))
        self.assertEqual(g.preset_poses["2"].pan, 90.0)
        bad, _ = ptz._validate({"settings": {}, "pose": {"front": {"pose": {"pan": "x"}}}})
        self.assertNotIn("pose", bad)


class ApiTests(unittest.TestCase):
    def setUp(self):
        from securitycam.webapp import app
        ptz_pose.reset()
        self.client = app.test_client()
        self.cs = state.camera_states["front"]
        self.cs.esp_zones[:] = []
        self.cs.esp_zone_seq = 0
        self.cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)

    def tearDown(self):
        ptz_pose.reset()

    def test_fixed_camera_unchanged(self):
        r = self.client.post("/api/cameras/front/zones", json={"box": [0.1, 0.2, 0.4, 0.6]})
        z = r.get_json()["zone"]
        self.assertEqual(z["box"], [0.1, 0.2, 0.4, 0.6])
        self.assertTrue(z["in_view"])
        self.assertFalse(z["anchored"])
        self.assertEqual(len(z["polygon"]), 4)
        self.assertEqual(self.client.get("/api/cameras/front/ptz/pose").get_json()["active"], False)
        self.assertEqual(self.client.post("/api/cameras/front/ptz/pose/reanchor").status_code, 409)

    def test_zone_projection_follows_pose(self):
        f = ptz_pose.get("front")
        f.ptz_known = True
        f.pose = Pose(0.0)
        z = self.client.post("/api/cameras/front/zones", json={"box": [0.4, 0.3, 0.6, 0.7]}).get_json()["zone"]
        self.assertEqual(z["anchor"], {"pan": 0.0, "tilt": 0.0, "zoom": 1.0})
        self.assertEqual(len(z["world"]), 4)
        f.pose = Pose(10.0, 0.0)
        z2 = self.client.get("/api/cameras/front/zones").get_json()["zones"][0]
        self.assertTrue(z2["anchored"] and z2["in_view"])
        self.assertLess(z2["box"][0], 0.4 - 0.05)          # moved left in the picture
        self.assertEqual(z2["drawn_box"], [0.4, 0.3, 0.6, 0.7])
        self.assertEqual(z2["world"], z["world"])
        f.pose = Pose(150.0, 0.0)
        z3 = self.client.get("/api/cameras/front/zones").get_json()["zones"][0]
        self.assertFalse(z3["in_view"])
        self.assertEqual(z3["polygon"], [])
        # drawing while panned anchors at that pose
        f.pose = Pose(30.0, 5.0)
        z4 = self.client.post("/api/cameras/front/zones", json={"box": [0.4, 0.3, 0.6, 0.7]}).get_json()["zone"]
        self.assertEqual(z4["anchor"]["pan"], 30.0)
        self.assertEqual(z4["box"], [0.4, 0.3, 0.6, 0.7])
        # world_anchor off: back to plain image boxes
        self.client.post("/api/cameras/front/esp", json={"world_anchor": False})
        self.cs.esp_settings["world_anchor"] = False
        z5 = self.client.get("/api/cameras/front/zones").get_json()["zones"][0]
        self.assertEqual(z5["box"], [0.4, 0.3, 0.6, 0.7])

    def test_tripwire_api_and_pose_endpoints(self):
        f = ptz_pose.get("front")
        f.ptz_known = True
        f.pose = Pose(5.0)
        r = self.client.post("/api/cameras/front/tripwires", json={"a": [0.3, 0.5], "b": [0.7, 0.5]})
        self.assertEqual(r.status_code, 200, r.get_json())
        wire = r.get_json()["tripwire"]
        self.assertEqual(wire["anchor"]["pan"], 5.0)
        self.assertIn("world", wire)
        f.pose = Pose(-5.0)
        cfg = self.client.get("/api/cameras/front/alerts").get_json()
        w = next(x for x in cfg["tripwires"] if x["id"] == wire["id"])
        self.assertTrue(w["in_view"])
        self.assertGreater(w["a"][0], 0.3 + 0.05)          # camera turned left: wire moved right
        self.assertEqual(w["drawn_a"], [0.3, 0.5])
        flipped = self.client.patch(f"/api/cameras/front/tripwires/{wire['id']}", json={"flip": True}).get_json()
        self.assertEqual(flipped["tripwire"]["anchor"]["pan"], 5.0)
        pose = self.client.get("/api/cameras/front/ptz/pose").get_json()
        self.assertTrue(pose["active"])
        self.assertEqual(pose["pose"]["pan"], -5.0)
        self.assertIn(pose["source"], ("none", "vision", "onvif", "both"))
        r = self.client.post("/api/cameras/front/ptz/pose/reanchor")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.post("/api/cameras/front/ptz/pose/sync").status_code, 409)
        self.client.delete(f"/api/cameras/front/tripwires/{wire['id']}")

    def test_targets_payload_has_pose_for_radar(self):
        f = ptz_pose.get("front")
        f.ptz_known = True
        f.pose = Pose(42.0, 3.0)
        body = self.client.get("/api/cameras/front/targets").get_json()
        self.assertEqual(body["pose"]["pan"], 42.0)
        self.assertIn("fov_h", body["pose"])

    def test_world_anchor_setting_validated(self):
        from securitycam.validation import sanitize_esp_patch
        self.assertEqual(sanitize_esp_patch({"world_anchor": "off"}), {"world_anchor": False})
        self.assertEqual(sanitize_esp_patch({"world_anchor": 3j}), {})


class RenderTests(unittest.TestCase):
    def setUp(self):
        ptz_pose.reset()

    def tearDown(self):
        ptz_pose.reset()

    def test_render_projected_zone_wire_and_heatmap(self):
        from securitycam.esp import draw_custom_esp, heatmap
        cs = state.camera_states["garage"]
        cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        cs.esp_settings.update({"heatmap": True, "zones": True})
        cs.esp_zones[:] = [{"id": "z1", "label": "Drive", "box": [0.3, 0.3, 0.7, 0.8], "color": "#ffd21f",
                            "anchor": Pose().as_dict()}]
        f = ptz_pose.get("garage")
        f.ptz_known, f.lens_override = True, LENS
        frame0 = np.full((360, 640, 3), 60, np.uint8)
        f.pose = Pose(0.0)
        f._record(10.0)
        out0 = draw_custom_esp(cs, frame0.copy(), None, frame_ts=10.0)
        f.pose = Pose(15.0, 4.0)
        f._record(10.5)
        out1 = draw_custom_esp(cs, frame0.copy(), None, frame_ts=10.5)
        # the zone fill moved left (it was drawn at pan 0)
        col0 = np.nonzero((out0 != 60).any(axis=2).any(axis=0))[0]
        col1 = np.nonzero((out1 != 60).any(axis=2).any(axis=0))[0]
        self.assertGreater(col0.min() - col1.min(), 40)
        # heatmap world accumulation + view sampling
        hs = heatmap.get_state(cs)
        hs.world.ensure(heatmap.WINDOWS)
        ray = geo.image_to_rays(((0.5, 0.7),), Pose(15.0, 4.0), LENS)
        for _ in range(20):
            from securitycam.esp import heatmap_world as HW
            HW.accumulate(hs.world, heatmap.WINDOWS, ray, 0.1)
        view = anchor.view_for(cs, 10.5)
        heat = heatmap.normalized(cs, "1h", view)
        self.assertIsNotNone(heat)
        gy, gx = np.unravel_index(np.argmax(heat), heat.shape)
        self.assertAlmostEqual(gx / heat.shape[1], 0.5, delta=0.06)
        self.assertAlmostEqual(gy / heat.shape[0], 0.7, delta=0.08)
        out2 = draw_custom_esp(cs, frame0.copy(), None, frame_ts=10.5)
        self.assertEqual(out2.shape, frame0.shape)
        cs.esp_zones[:] = []
        cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        heatmap.reset(cs)


class ControllerPollingTests(unittest.TestCase):
    def test_position_polling_and_command_notifications(self):
        from securitycam.ptz.controller import PtzController

        class Backend:
            def __init__(self):
                self.n = 0

            def connect(self):
                return {"pan_tilt": True, "zoom": False, "relative": True, "presets": False, "home": False}

            def get_presets(self):
                return []

            def get_position(self):
                self.n += 1
                return {"x": 0.1, "y": -0.2, "zoom": None, "moving": False}

            def relative_move(self, x, y):
                pass

        clock = [0.0]
        ctl = PtzController("front", Backend, {"pan_range": 360.0, "tilt_range": 114.0, "speed": 0.5},
                            clock=lambda: clock[0])
        got, cmds = [], []
        ctl.position_listener = got.append
        ctl.command_listener = lambda kind, *a: cmds.append((kind, a))
        ctl._do_connect()
        self.assertEqual(cmds[0][0], "connected")
        self.assertEqual(ctl._next_action(), ("poll",))
        ctl._execute(("poll",))
        self.assertEqual(got[-1]["x"], 0.1)
        self.assertTrue(ctl.position_supported)
        ctl.relative_degrees(18.0, 0.0)
        ctl.run_pending()
        self.assertEqual(cmds[-1][0], "relative")
        self.assertAlmostEqual(cmds[-1][1][0][0], 18.0)
        # a camera without GetStatus stops being polled
        ctl2 = PtzController("front", lambda: types.SimpleNamespace(
            connect=lambda: {"pan_tilt": True}, get_presets=lambda: []), {}, clock=lambda: clock[0])
        ctl2.position_listener = got.append
        ctl2._do_connect()
        ctl2._execute(("poll",))
        self.assertFalse(ctl2.position_supported)
        self.assertIsNone(ctl2._poll_wait_locked())


if __name__ == "__main__":
    unittest.main()
