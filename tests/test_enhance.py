"""Tests for the Enhance view (securitycam/enhance).

Synthetic frames only; the upscaler is faked where a real model would be needed,
no camera is contacted and no model is downloaded.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-test-"))
os.environ.setdefault("SECURITYCAM_CAMERAS",
                      "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from securitycam import deps, state  # noqa: E402
from securitycam.enhance import conf, jobs, marking, models  # noqa: E402
from securitycam.enhance.region import EnhanceInputError, crop_box, parse_options, parse_region  # noqa: E402
from securitycam.webapp import app  # noqa: E402

CAM = "front"


def test_frame(w=1280, h=720):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    cv2.putText(img, "AB 123", (500, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img


def fake_enhance(img, model_id, scale, denoise, device, progress, cancelled, stage):
    stage("running")
    progress(0.5)
    out = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    progress(1.0)
    return models.Result(out, models.CATALOG[model_id], "Fake device", None, {"infer_s": 0.01})


class RegionMath(unittest.TestCase):
    def test_default_is_full_frame(self):
        self.assertEqual(parse_region(None), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(crop_box((0.0, 0.0, 1.0, 1.0), 1920, 1080), (0, 0, 1920, 1080))

    def test_clamps_and_orders_corners(self):
        self.assertEqual(parse_region([0.8, 1.4, -0.2, 0.25]), (0.0, 0.25, 0.8, 1.0))

    def test_rounds_outward(self):
        # 0.1001 * 1000 = 100.1 -> 100 ; 0.2999 * 1000 = 299.9 -> 300
        self.assertEqual(crop_box((0.1001, 0.1001, 0.2999, 0.2999), 1000, 1000), (100, 100, 300, 300))
        # exact zoom viewport (2x zoom, centred) on 1280x720
        self.assertEqual(crop_box((0.25, 0.25, 0.75, 0.75), 1280, 720), (320, 180, 960, 540))

    def test_tiny_region_grows_inside_frame(self):
        x1, y1, x2, y2 = crop_box((0.9999, 0.9999, 1.0, 1.0), 640, 360)
        self.assertEqual((x2 - x1, y2 - y1), (8, 8))
        self.assertEqual((x2, y2), (640, 360))

    def test_invalid_regions(self):
        for bad in ([0, 0, 1], "0,0,1,1", [0, 0, float("nan"), 1], [0.5, 0.5, 0.5, 0.9], [True, 0, 1, 1], {"x": 1}):
            with self.assertRaises(EnhanceInputError, msg=repr(bad)):
                parse_region(bad)

    def test_options(self):
        ids = models.MODEL_IDS
        o = parse_options({"model": "lanczos", "scale": 2, "denoise": 3}, ids, models.DEFAULT_MODEL)
        self.assertEqual((o["model"], o["scale"], o["denoise"], o["device"]), ("lanczos", 2, 1.0, "auto"))
        for bad in ({"model": "../x"}, {"scale": 3}, {"scale": True}, {"device": "tpu"}, {"denoise": "a"},
                    {"include_overlay": "yes"}, [1, 2]):
            with self.assertRaises(EnhanceInputError, msg=repr(bad)):
                parse_options(bad, ids, models.DEFAULT_MODEL)


class Marking(unittest.TestCase):
    def test_png_text_roundtrip_and_band(self):
        img = test_frame(320, 180)
        png, band = marking.mark_export(img, "Real-ESRGAN general-x4v3", 4,
                                        {"Software": marking.SOFTWARE, "Description": marking.DESCRIPTION}, ai=True)
        text = marking.read_png_text(png)
        self.assertEqual(text["Software"], "SecurityCam Enhance")
        self.assertIn("details may be invented", text["Description"])
        decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (180 + band, 320))
        self.assertGreaterEqual(band, 30)
        # the band is anthracite with a yellow chip on the left
        self.assertTrue(np.allclose(decoded[180 + band - 3, -3], (39, 36, 34), atol=3))
        self.assertTrue(np.allclose(decoded[180 + band - 3, 2], (31, 210, 255), atol=3))
        # the corner tag changed pixels inside the picture
        self.assertFalse(np.array_equal(decoded[6, 6], img[6, 6]))


class RunnerLock(unittest.TestCase):
    def test_cuda_tiles_run_inside_gpu_lock(self):
        held = {"n": 0, "depth": 0}

        class RecordingLock:
            def __enter__(self):
                held["depth"] += 1
                return self

            def __exit__(self, *exc):
                held["depth"] -= 1

        class FakeCudaRunner(models.TorchRunner):
            def __init__(self):     # no torch / GPU needed
                self.cuda = True
                self.native_scale = 4
                self.device_label = "fake cuda"

            def infer(self, tile):
                assert held["depth"] > 0, "CUDA work outside gpu.LOCK"
                held["n"] += 1
                return np.repeat(np.repeat(tile, 4, axis=2), 4, axis=3)

        img = test_frame(100, 70)
        with mock.patch.object(models.gpu, "LOCK", RecordingLock()), mock.patch.object(conf, "TILE", 32):
            out = FakeCudaRunner().run(img, lambda f: None, lambda: False)
        self.assertEqual(held["n"], 4 * 3)            # every tile, each one inside the lock
        self.assertEqual(out.shape, (280, 400, 3))
        # tiled result == untiled nearest upscale (overlap stitching is exact)
        ref = cv2.resize(img, (400, 280), interpolation=cv2.INTER_NEAREST)
        self.assertLessEqual(int(np.abs(out.astype(int) - ref.astype(int)).max()), 1)

    def test_cancel_between_tiles(self):
        r = models.Runner()
        r.infer = lambda t: np.repeat(np.repeat(t, 4, axis=2), 4, axis=3)
        calls = []
        with mock.patch.object(conf, "TILE", 16), self.assertRaises(models.Cancelled):
            r.run(test_frame(64, 64), calls.append, lambda: len(calls) >= 2)

    def test_fallback_when_download_fails(self):
        with mock.patch.object(deps, "asset_path", return_value=None), \
             mock.patch.dict(deps.asset_status, {"esrgan_general_x4v3": "failed"}):
            res = models.enhance(test_frame(40, 30), "realesr-general-x4v3", 2, 0.5, "auto",
                                 lambda f: None, lambda: False, lambda s: None)
        self.assertEqual(res.model.id, "lanczos")
        self.assertEqual(res.fallback, "model_unavailable")
        self.assertEqual(res.image.shape, (60, 80, 3))


class Api(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.cs = state.get_camera_state(CAM)
        with self.cs.frame_lock:
            self.cs.latest_frame = test_frame()
            self.cs.latest_frame_ts = time.perf_counter()
        p = mock.patch.object(jobs.manager, "_enhance", fake_enhance)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        jobs.manager.wait_idle()
        with self.cs.frame_lock:
            self.cs.latest_frame = None

    def start(self, body=None, **kw):
        return self.client.post(f"/api/cameras/{CAM}/enhance", json=body if body is not None else {}, **kw)

    def fetch(self, url):
        with self.client.get(url) as r:     # closes the file handle (Windows can't delete open files)
            self.assertEqual(r.status_code, 200, url)
            return r.data

    def wait(self, job_id, want=("done", "failed", "canceled")):
        for _ in range(300):
            j = self.client.get(f"/api/enhance/{job_id}").get_json()
            if j and j["status"] in want:
                return j
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_lifecycle_files_and_metadata(self):
        r = self.start({"region": [0.25, 0.25, 0.75, 0.75], "model": "realesr-general-x4v3", "scale": 4})
        self.assertEqual(r.status_code, 202, r.get_json())
        job = r.get_json()["job"]
        self.assertEqual(job["region_px"], [320, 180, 960, 540])
        self.assertEqual(job["crop_size"], [640, 360])
        done = self.wait(job["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["output_size"], [2560, 1440])
        self.assertTrue(done["ai_enhanced"])
        self.assertEqual(done["device_used"], "Fake device")
        self.assertIn("not as evidence", done["notice"])

        enh = self.fetch(done["urls"]["enhanced"])
        text = marking.read_png_text(enh)
        self.assertEqual(text["Title"], "AI ENHANCED")
        self.assertEqual(text["AI-Enhanced"], "true")
        self.assertEqual(text["Scale"], "4x")
        self.assertEqual(text["Model"], "Real-ESRGAN General v3")
        self.assertIn("Front (front)", text["Source-Camera"])
        self.assertIn("pixels [320, 180, 960, 540] of 1280x720", text["Region"])
        self.assertTrue(text["Source-Timestamp"])
        self.assertIn("not evidence", text["Description"])
        img = cv2.imdecode(np.frombuffer(enh, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (1440 + done["band_px"], 2560))

        orig = self.fetch(done["urls"]["original"])
        o = cv2.imdecode(np.frombuffer(orig, np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(o, test_frame()[180:540, 320:960]))     # raw pixels, untouched
        self.assertEqual(marking.read_png_text(orig)["AI-Enhanced"], "false")
        with self.client.get(done["urls"]["enhanced"] + "?download=1") as dl:
            self.assertIn("attachment", dl.headers["Content-Disposition"])

        listing = self.client.get("/api/enhance").get_json()
        self.assertIn(done["id"], [j["id"] for j in listing["jobs"]])
        self.assertIn("lanczos", [m["id"] for m in listing["models"]])

        self.assertEqual(self.client.delete(f"/api/enhance/{done['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/enhance/{done['id']}").status_code, 404)
        self.assertFalse((conf.ENHANCED_DIR / done["id"]).exists())

    def test_fallback_marks_no_ai(self):
        with mock.patch.object(jobs.manager, "_enhance", models.enhance), \
             mock.patch.object(deps, "asset_path", return_value=None), \
             mock.patch.dict(deps.asset_status, {"esrgan_general_x4v3": "failed"}):
            job = self.start({"region": [0, 0, 0.1, 0.1], "scale": 2}).get_json()["job"]
            done = self.wait(job["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["fallback"], "model_unavailable")
        self.assertFalse(done["ai_enhanced"])
        text = marking.read_png_text(self.fetch(done["urls"]["enhanced"]))
        self.assertEqual(text["Title"], "UPSCALED (NO AI)")

    def test_paused_video_source_metadata(self):
        class FakeSource:
            label = "clip.mp4"

            def playback_payload(self):
                return {"playing": False, "position": 12.48, "frame": 312}

        with mock.patch.object(self.cs, "source", FakeSource()), mock.patch.object(self.cs, "source_kind", "file"):
            job = self.start({"model": "lanczos", "scale": 2}).get_json()["job"]
        self.assertEqual(job["source"]["position_s"], 12.48)
        self.assertEqual(job["source"]["frame"], 312)
        self.assertFalse(job["source"]["playing"])
        done = self.wait(job["id"])
        self.assertIn("video position 12.48 s, frame 312",
                      marking.read_png_text(self.fetch(done["urls"]["enhanced"]))["Source-Timestamp"])

    def test_queue_and_cancel(self):
        release = threading.Event()

        def slow(img, model_id, scale, denoise, device, progress, cancelled, stage):
            stage("running")
            while not cancelled() and not release.is_set():
                time.sleep(0.01)
            if cancelled():
                raise models.Cancelled()
            return fake_enhance(img, model_id, scale, denoise, device, progress, cancelled, stage)

        with mock.patch.object(jobs.manager, "_enhance", slow):
            a = self.start({"model": "lanczos"}).get_json()["job"]
            b = self.start({"model": "lanczos"}).get_json()["job"]
            self.wait(a["id"], want=("running",))
            self.assertEqual(self.client.get(f"/api/enhance/{b['id']}").get_json()["status"], "queued")
            # cancel the queued one, then the running one
            self.assertEqual(self.client.post(f"/api/enhance/{b['id']}/cancel", json={}).status_code, 200)
            self.assertEqual(self.client.get(f"/api/enhance/{b['id']}").get_json()["status"], "canceled")
            self.assertEqual(self.client.post(f"/api/enhance/{a['id']}/cancel", json={}).status_code, 200)
            self.assertEqual(self.wait(a["id"])["status"], "canceled")
            jobs.manager.wait_idle()
        self.assertFalse((conf.ENHANCED_DIR / a["id"]).exists())
        self.assertNotIn(a["id"], [j["id"] for j in self.client.get("/api/enhance").get_json()["jobs"]])
        self.assertEqual(self.client.post(f"/api/enhance/{a['id']}/cancel", json={}).status_code, 409)

    def test_retention(self):
        with mock.patch.object(conf, "MAX_COUNT", 2):
            ids = []
            for _ in range(3):
                ids.append(self.start({"model": "lanczos", "scale": 2}).get_json()["job"]["id"])
                self.wait(ids[-1])
                time.sleep(0.01)
            jobs.manager.wait_idle()
        self.assertFalse((conf.ENHANCED_DIR / ids[0]).exists())
        self.assertTrue((conf.ENHANCED_DIR / ids[2]).exists())

    def test_rejections(self):
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/enhance", data={"region": "x"}).status_code, 415)
        self.assertEqual(self.start({"model": "nope"}).status_code, 400)
        self.assertEqual(self.start({"region": [0, 0, 0, 0]}).status_code, 400)
        self.assertEqual(self.client.post("/api/cameras/nope/enhance", json={}).status_code, 404)
        r = self.client.post(f"/api/cameras/{CAM}/enhance", json={}, headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        with mock.patch.object(conf, "MAX_OUTPUT_MP", 1.0):
            self.assertEqual(self.start({"scale": 4}).get_json()["error"], "region_too_large")
        with self.cs.frame_lock:
            self.cs.latest_frame = None
        self.assertEqual(self.start({}).status_code, 409)

    def test_traversal_attempts(self):
        job = self.start({"model": "lanczos", "scale": 2}).get_json()["job"]
        self.wait(job["id"])
        jid = job["id"]
        secret = Path(os.environ["SECURITYCAM_DATA_DIR"]) / "secret.txt"
        secret.write_text("x")
        for url in (
            f"/api/enhance/{jid}/..%2F..%2Fsecret.txt",
            f"/api/enhance/{jid}/../../secret.txt",
            f"/api/enhance/..%2F{jid}/meta.json",
            "/api/enhance/..%5C..%5Csecret.txt",
            f"/api/enhance/{jid}/meta.json.tmp",
            f"/api/enhance/{jid}/original.png%00.txt",
            f"/api/enhance/{jid.upper()}/original.png",
            "/api/enhance/../../secret.txt",
        ):
            with self.client.get(url) as r:
                self.assertIn(r.status_code, (404, 308), url)
                self.assertNotEqual(r.data, b"x", url)
        self.assertIn(self.client.delete("/api/enhance/..%2F..").status_code, (404, 405))
        self.assertEqual(self.client.delete(f"/api/enhance/{jid}x").status_code, 404)
        self.fetch(f"/api/enhance/{jid}/meta.json")
        self.assertIsNone(jobs.manager.file_path("../" + jid, "meta.json"))
        self.assertIsNone(jobs.manager.file_path(jid, "../secret.txt"))
        self.assertTrue(secret.exists())


if __name__ == "__main__":
    unittest.main()
