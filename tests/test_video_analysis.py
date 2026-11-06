"""Tests for video analysis: library upload/list/rename/delete, file sources as
virtual cameras, real-time pacing, playback control, track-state reset, YOLO
worker and alarms on a video camera. Uses synthetic clips written with
cv2.VideoWriter; no camera is ever contacted.

Run:  python -m unittest discover -s tests
Optional real-model test: set SECURITYCAM_TEST_MODELS_DIR to a folder with yolov8n.pt.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="securitycam-video-test-")
os.environ.setdefault("SECURITYCAM_DATA_DIR", _TMP)
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS", "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ["SECURITYCAM_AUTO_INSTALL_DEPS"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from securitycam import persistence, presets, sources, state, streaming  # noqa: E402
from securitycam.videos import api as videos_api  # noqa: E402
from securitycam.videos import library  # noqa: E402
from securitycam.webapp import app  # noqa: E402

presets.SCHEDULER.stop()

_MEDIA = Path(tempfile.mkdtemp(prefix="securitycam-media-"))


def make_clip(path: Path, fps: float = 25.0, frames: int = 75, size=(320, 240), background=None) -> Path:
    """Moving yellow rectangle on anthracite (or panning over ``background``)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "cv2.VideoWriter could not open mp4v"
    for i in range(frames):
        if background is not None:
            bh, bw = background.shape[:2]
            x0 = int((bw - w) * (i / max(1, frames - 1)))
            img = np.ascontiguousarray(background[(bh - h) // 2:(bh - h) // 2 + h, x0:x0 + w])
        else:
            img = np.full((h, w, 3), (39, 36, 34), np.uint8)
            x = int((w - 60) * i / max(1, frames - 1))
            cv2.rectangle(img, (x, 90), (x + 60, 150), (31, 210, 255), -1)
        writer.write(img)
    writer.release()
    return path


CLIP = make_clip(_MEDIA / "clip25.mp4")


def upload(client, data: bytes, filename: str, **headers):
    return client.post(
        "/api/videos",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        headers=headers,
    )


class LibraryCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=_TMP))
        self._orig = (library.LIBRARY.root, library.LIBRARY.incoming, library.LIBRARY.max_total)
        library.LIBRARY.root = self.root
        library.LIBRARY.incoming = self.root / ".incoming"
        self.client = app.test_client()

    def tearDown(self):
        sources.close_all()
        library.LIBRARY.root, library.LIBRARY.incoming, library.LIBRARY.max_total = self._orig
        shutil.rmtree(self.root, ignore_errors=True)

    def enable_urls(self, http: dict | None = None):
        """Turn URL sources on and answer HTTP from ``http`` {url: (status, headers, body)}."""
        self.http = dict(http or {})
        self.fetched: list = []

        def fake_fetch(url, want_body):
            self.fetched.append(url)
            return self.http.get(url, (200, {"content-type": "video/mp4"}, b"\x00" * 16))

        for target, value in ((sources, ("URLS_ENABLED", True)), (sources, ("_http_fetch", fake_fetch))):
            patcher = mock.patch.object(target, value[0], value[1])
            patcher.start()
            self.addCleanup(patcher.stop)

    def add_clip(self, name="clip.mp4", path: Path = CLIP) -> dict:
        r = upload(self.client, path.read_bytes(), name)
        self.assertEqual(r.status_code, 200, r.get_json())
        return r.get_json()["video"]

    def incoming_files(self):
        inc = self.root / ".incoming"
        return list(inc.glob("*")) if inc.is_dir() else []


class UploadTests(LibraryCase):
    def test_valid_upload_metadata_thumb_and_list(self):
        v = self.add_clip("Parking lot.mp4")
        self.assertEqual(v["name"], "Parking lot.mp4")
        self.assertEqual((v["width"], v["height"]), (320, 240))
        self.assertAlmostEqual(v["fps"], 25.0, delta=0.5)
        self.assertAlmostEqual(v["duration"], 3.0, delta=0.2)
        self.assertGreater(v["size"], 0)
        self.assertTrue(v["thumb_url"])
        thumb = self.client.get(v["thumb_url"])
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.mimetype, "image/jpeg")
        thumb.close()
        listed = self.client.get("/api/videos").get_json()
        self.assertEqual([x["id"] for x in listed["videos"]], [v["id"]])
        self.assertEqual(listed["total_bytes"], v["size"])
        self.assertIn("mp4", listed["extensions"])
        self.assertTrue((self.root / v["id"] / "meta.json").is_file())
        self.assertEqual(self.incoming_files(), [])

    def test_large_multichunk_upload(self):
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        clip = make_clip(_MEDIA / "noise.avi", fps=10, frames=20, size=(640, 480), background=noise)
        self.assertGreater(clip.stat().st_size, 2 * 1024 * 1024)
        v = self.add_clip("noise.avi", clip)
        self.assertEqual(v["size"], clip.stat().st_size)
        self.assertEqual(v["frames"], 20)

    def test_too_big(self):
        with mock.patch.object(videos_api, "MAX_UPLOAD_BYTES", 1000):
            r = upload(self.client, CLIP.read_bytes(), "big.mp4")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["error"], "too_large")
        self.assertEqual(self.incoming_files(), [])
        self.assertEqual(self.client.get("/api/videos").get_json()["videos"], [])

    def test_too_big_streamed_without_length_check(self):
        # Content-Length passes the early check, the streaming sink still enforces the limit.
        with mock.patch.object(videos_api, "MAX_UPLOAD_BYTES", 4000), mock.patch.object(videos_api, "_FORM_OVERHEAD", 10**9):
            r = upload(self.client, CLIP.read_bytes(), "big.mp4")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(self.incoming_files(), [])

    def test_wrong_type(self):
        r = upload(self.client, CLIP.read_bytes(), "clip.exe")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(r.get_json()["error"], "unsupported_type")
        r = upload(self.client, b"hello", "notes.txt")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(self.incoming_files(), [])

    def test_corrupt_file(self):
        r = upload(self.client, b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096), "broken.mp4")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(r.get_json()["error"], "invalid_video")
        r = upload(self.client, b"<?php echo 1; ?>", "script.mkv")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(list(p for p in self.root.iterdir() if p.name != ".incoming"), [])
        self.assertEqual(self.incoming_files(), [])

    def test_traversal_names_and_ids(self):
        v = self.add_clip("../../..\\evil/..\\pwn\x07.mp4")
        self.assertEqual(v["name"], "pwn.mp4")
        files = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())
        self.assertEqual(files, [f"{v['id']}/meta.json", f"{v['id']}/thumb.jpg", f"{v['id']}/video.mp4"])
        for bad in ("..", "..%2f..%2fetc", "ABCDEF123456", "abc", "../" + v["id"]):
            r = self.client.get(f"/api/videos/{bad}")
            self.assertIn(r.status_code, (404,), bad)
            r = self.client.delete(f"/api/videos/{bad}")
            self.assertIn(r.status_code, (404, 405), bad)
        self.assertTrue((self.root / v["id"]).is_dir())
        r = self.client.patch(f"/api/videos/{v['id']}", json={"name": "../../x/Gate cam"})
        self.assertEqual(r.get_json()["video"]["name"], "Gate cam")

    def test_requires_multipart_and_json(self):
        r = self.client.post("/api/videos", data=CLIP.read_bytes(), content_type="application/octet-stream")
        self.assertEqual(r.status_code, 415)
        v = self.add_clip()
        r = self.client.patch(f"/api/videos/{v['id']}", data="name=x", content_type="application/x-www-form-urlencoded")
        self.assertEqual(r.status_code, 415)
        r = self.client.post("/api/videos", data={"other": "x"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "missing_file")

    def test_cross_site_upload_rejected(self):
        r = upload(self.client, CLIP.read_bytes(), "clip.mp4", Origin="http://evil.example")
        self.assertEqual(r.status_code, 403)
        r = upload(self.client, CLIP.read_bytes(), "clip.mp4", **{"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.client.get("/api/videos").get_json()["videos"], [])

    def test_rename_and_delete(self):
        v = self.add_clip()
        r = self.client.patch(f"/api/videos/{v['id']}", json={"name": "  Night  shift  "})
        self.assertEqual(r.get_json()["video"]["name"], "Night shift")
        self.assertEqual(self.client.patch(f"/api/videos/{v['id']}", json={"name": "   "}).status_code, 400)
        self.assertEqual(self.client.patch(f"/api/videos/{v['id']}", json={"name": 5}).status_code, 400)
        r = self.client.delete(f"/api/videos/{v['id']}")
        self.assertEqual(r.status_code, 200)
        self.assertFalse((self.root / v["id"]).exists())
        self.assertEqual(self.client.delete(f"/api/videos/{v['id']}").status_code, 404)

    def test_retention_prunes_oldest(self):
        a = self.add_clip("a.mp4")
        b = self.add_clip("b.mp4")
        library.LIBRARY.max_total = a["size"] * 2 + 10
        c = self.add_clip("c.mp4")
        ids = [x["id"] for x in self.client.get("/api/videos").get_json()["videos"]]
        self.assertEqual(sorted(ids), sorted([b["id"], c["id"]]))
        self.assertNotIn(a["id"], ids)
        # an open video is never pruned; if nothing else can go the upload is refused
        self.client.post(f"/api/videos/{b['id']}/open", json={})
        self.client.post(f"/api/videos/{c['id']}/open", json={})
        r = upload(self.client, CLIP.read_bytes(), "d.mp4")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["error"], "library_full")


class SourceTests(LibraryCase):
    def open_clip(self, **body):
        v = self.add_clip()
        r = self.client.post(f"/api/videos/{v['id']}/open", json=body)
        self.assertEqual(r.status_code, 200, r.get_json())
        cid = r.get_json()["camera_id"]
        return v, cid, state.get_camera_state(cid)

    def test_open_close_and_camera_lists(self):
        v, cid, cs = self.open_clip()
        self.assertEqual(cid, f"video-{v['id']}")
        self.assertEqual(cs.source_kind, "file")
        cams = {c["id"]: c for c in self.client.get("/api/cameras").get_json()["cameras"]}
        self.assertEqual(cams[cid]["source_kind"], "file")
        self.assertTrue(cams[cid]["has_stream"])
        self.assertEqual(cams["front"]["source_kind"], "rtsp")
        self.assertIn(cid, [c["id"] for c in self.client.get("/api/overview").get_json()["cameras"]])
        self.assertEqual(self.client.get(f"/api/cameras/{cid}/status").status_code, 200)
        # idempotent open
        again = self.client.post(f"/api/videos/{v['id']}/open", json={}).get_json()
        self.assertEqual(again["camera_id"], cid)
        listed = self.client.get("/api/videos").get_json()["videos"][0]
        self.assertEqual(listed["camera_id"], cid)
        # never persisted
        self.assertNotIn(cid, persistence.dump_cameras())
        r = self.client.post(f"/api/sources/{cid}/close", json={})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(state.get_camera_state(cid))
        self.assertNotIn(cid, [c["id"] for c in self.client.get("/api/cameras").get_json()["cameras"]])
        self.assertEqual(self.client.post(f"/api/sources/{cid}/close", json={}).status_code, 404)
        self.assertEqual(self.client.post("/api/sources/front/close", json={}).status_code, 404)

    def test_copy_settings_from_current_camera(self):
        front = state.get_camera_state("front")
        front.esp_settings["thermal"] = True
        try:
            _, _, cs = self.open_clip(copy_settings_from="front")
            self.assertTrue(cs.esp_settings["thermal"])
            self.assertIsNot(cs.esp_settings, front.esp_settings)
        finally:
            front.esp_settings["thermal"] = False
        _, _, fresh = self.open_clip()
        self.assertFalse(fresh.esp_settings["thermal"])
        self.assertTrue(fresh.yolo_enabled)          # AI on by default for video analysis

    def test_delete_open_video_closes_it(self):
        v, cid, cs = self.open_clip()
        feed = streaming._acquire_feed(cs)
        self.assertIsNotNone(feed)
        r = self.client.delete(f"/api/videos/{v['id']}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["closed"], cid)
        self.assertTrue(feed.stream.stopped)
        self.assertFalse((self.root / v["id"]).exists())

    def test_mjpeg_stream_from_video(self):
        _, cid, cs = self.open_clip()
        cs.yolo_enabled = False
        gen = streaming.gen_frames(cid)
        parts = [next(gen) for _ in range(5)]
        gen.close()
        self.assertTrue(all(p.startswith(b"--frame") for p in parts))
        self.assertEqual(cs.stream_error, "")
        self.assertEqual(cs.stats["resolution"], "320x240")

    def _feed(self, cs):
        feed = streaming._acquire_feed(cs)
        self.assertIsNotNone(feed)
        self.addCleanup(streaming._release_feed, cs, feed)
        return feed

    def test_realtime_pacing(self):
        _, _, cs = self.open_clip()
        cs.yolo_enabled = False
        feed = self._feed(cs)
        seq0, frame0 = feed.stream.seq, cs.source.playback.frame
        time.sleep(2.0)
        produced = feed.stream.seq - seq0
        advanced = cs.source.playback.frame - frame0
        # 25 fps clip -> ~50 new frames in 2 s (not hundreds)
        self.assertGreaterEqual(produced, 40, produced)
        self.assertLessEqual(produced, 60, produced)
        self.assertAlmostEqual(advanced, 50, delta=10)

    def test_pause_seek_speed_step(self):
        _, cid, cs = self.open_clip()
        cs.yolo_enabled = False
        feed = self._feed(cs)
        post = lambda body: self.client.post(f"/api/sources/{cid}/playback", json=body)  # noqa: E731
        r = post({"action": "pause"})
        self.assertFalse(r.get_json()["playback"]["playing"])
        time.sleep(0.15)
        f0, s0 = cs.source.playback.frame, feed.stream.seq
        time.sleep(0.6)
        self.assertEqual(cs.source.playback.frame, f0)          # position frozen
        self.assertGreaterEqual(feed.stream.seq - s0, 3)        # still frame re-published for the ESP
        # seek in seconds / fraction while paused
        post({"action": "seek", "seconds": 2.0})
        time.sleep(0.2)
        self.assertEqual(cs.source.playback.frame, 50)
        self.assertAlmostEqual(self.client.get(f"/api/sources/{cid}/playback").get_json()["position"], 2.0, delta=0.05)
        post({"action": "seek", "fraction": 0.5})
        time.sleep(0.2)
        self.assertEqual(cs.source.playback.frame, 37)
        post({"action": "seek", "delta": -1.0})
        time.sleep(0.2)
        self.assertEqual(cs.source.playback.frame, 12)
        # frame step
        post({"action": "step", "frames": 1})
        time.sleep(0.2)
        self.assertEqual(cs.source.playback.frame, 13)
        post({"action": "step", "frames": -1})
        time.sleep(0.2)
        self.assertEqual(cs.source.playback.frame, 12)
        # speed 2x: ~50 frames per second of wall time
        post({"action": "seek", "seconds": 0})
        r = post({"action": "play", "speed": 2.0})
        self.assertEqual(r.get_json()["playback"]["speed"], 2.0)
        time.sleep(0.2)
        f1 = cs.source.playback.frame
        time.sleep(1.0)
        self.assertAlmostEqual(cs.source.playback.frame - f1, 50, delta=10)
        # validation
        self.assertEqual(post({"action": "seek", "seconds": -1}).status_code, 400)
        self.assertEqual(post({"speed": 10}).status_code, 400)
        self.assertEqual(post({"speed": True}).status_code, 400)
        self.assertEqual(post({"action": "explode"}).status_code, 400)
        self.assertEqual(post({"loop": "yes"}).status_code, 400)
        self.assertEqual(self.client.get("/api/sources/front/playback").status_code, 404)

    def test_loop_resets_track_state(self):
        _, cid, cs = self.open_clip()
        cs.yolo_enabled = False
        self._feed(cs)
        pb = cs.source.playback
        cuts0, epoch0 = pb.cuts, cs.source_epoch
        self.client.post(f"/api/sources/{cid}/playback", json={"action": "seek", "seconds": 2.6})
        time.sleep(0.1)
        cuts1 = pb.cuts
        self.assertEqual(cuts1, cuts0 + 1)           # a 2.6 s jump is a cut
        cs.track_history[7] = [(1, 1, 0.0), (2, 2, 0.1)]
        cs.esp_motion = object()
        cs.esp_locked = 7
        with cs.frame_lock:
            cs.latest_results = ["stale"]
        time.sleep(1.0)                              # runs past the end and loops
        self.assertEqual(pb.cuts, cuts1 + 1)
        self.assertGreater(cs.source_epoch, epoch0)
        self.assertLess(pb.frame, 40)
        self.assertEqual(cs.track_history, {})
        self.assertIsNone(cs.esp_motion)
        self.assertIsNone(cs.esp_locked)
        self.assertTrue(pb.playing)

    def test_loop_off_ends_and_play_restarts(self):
        _, cid, cs = self.open_clip(loop=False)
        cs.yolo_enabled = False
        self._feed(cs)
        pb = cs.source.playback
        self.assertFalse(pb.loop)
        self.client.post(f"/api/sources/{cid}/playback", json={"action": "seek", "seconds": 2.7})
        time.sleep(0.8)
        p = self.client.get(f"/api/sources/{cid}/playback").get_json()
        self.assertTrue(p["ended"])
        self.assertFalse(p["playing"])
        self.assertGreaterEqual(p["frame"], 70)
        self.client.post(f"/api/sources/{cid}/playback", json={"action": "play"})
        time.sleep(0.3)
        self.assertTrue(pb.playing)
        self.assertLess(pb.frame, 20)
        self.client.post(f"/api/sources/{cid}/playback", json={"loop": True})
        self.assertTrue(pb.loop)

    def test_restart(self):
        _, cid, cs = self.open_clip()
        cs.yolo_enabled = False
        self._feed(cs)
        time.sleep(0.5)
        self.client.post(f"/api/sources/{cid}/playback", json={"action": "restart"})
        time.sleep(0.1)
        self.assertLess(cs.source.playback.frame, 6)

    def test_yolo_worker_on_video_source_with_mocked_model(self):
        _, cid, cs = self.open_clip()
        cs.esp_settings["tracking"] = False
        calls = []

        class FakeModel:
            def __call__(self, frame, **kw):
                calls.append(frame.shape)
                return []

        cs.yolo_model = FakeModel()
        cs.yolo_enabled = True
        with mock.patch.object(state, "YOLO_AVAILABLE", True):
            self._feed(cs)
            streaming.ensure_worker(cs)
            time.sleep(1.0)
            self.assertGreaterEqual(len(calls), 15)       # one inference per new frame (~25/s)
            self.assertLessEqual(len(calls), 40)
            self.assertEqual(calls[0], (240, 320, 3))
            worker = cs.worker_thread
            self.client.post(f"/api/sources/{cid}/close", json={})
            worker.join(timeout=3.0)
            self.assertFalse(worker.is_alive())           # closing stops the worker thread

    def test_zones_and_alarm_rules_on_video_camera(self):
        _, cid, cs = self.open_clip()
        r = self.client.post(f"/api/cameras/{cid}/zones", json={"box": [0.1, 0.1, 0.9, 0.9], "label": "Lot"})
        self.assertEqual(r.status_code, 200, r.get_json())
        zid = r.get_json()["zone"]["id"]
        r = self.client.put(f"/api/cameras/{cid}/zones/{zid}/rule", json={"enabled": True, "classes": ["person"], "notify": False})
        self.assertEqual(r.status_code, 200, r.get_json())
        r = self.client.post(f"/api/cameras/{cid}/tripwires", json={"a": [0.5, 0.0], "b": [0.5, 1.0]})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(len(self.client.get(f"/api/cameras/{cid}/alerts").get_json()["tripwires"]), 1)
        self.assertEqual(self.client.get(f"/api/events?camera={cid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/cameras/{cid}/targets").status_code, 200)
        self.assertEqual(self.client.post(f"/api/cameras/{cid}/update_esp", json={"heatmap": True}).status_code, 200)

    def test_url_source_validation(self):
        self.enable_urls()
        for bad, code in (("file:///etc/passwd", "unsupported_scheme"), ("javascript:alert(1)", "unsupported_scheme"),
                          ("ftp://x/y.mp4", "unsupported_scheme"), ("http://", "invalid_url"), (5, "invalid_url"),
                          ("http://a b/x", "invalid_url")):
            r = self.client.post("/api/sources/url", json={"url": bad})
            self.assertEqual(r.status_code, 400, bad)
            self.assertEqual(r.get_json()["error"], code, bad)
        # a public host (resolution mocked): the source opens lazily, nothing is contacted here
        with fake_dns({"cams.example.com": ["93.184.216.34"]}):
            r = self.client.post("/api/sources/url", json={"url": "http://user:secret@cams.example.com:9/cam.mjpg", "label": "Test"})
        self.assertEqual(r.status_code, 200, r.get_json())
        cid = r.get_json()["camera_id"]
        listed = self.client.get("/api/sources").get_json()["sources"]
        self.assertEqual(listed[0]["url"], "http://user:***@cams.example.com:9/cam.mjpg")
        self.assertNotIn("secret", json.dumps(self.client.get("/api/cameras").get_json()))
        self.assertEqual(self.client.post(f"/api/sources/{cid}/playback", json={"action": "pause"}).status_code, 409)
        self.assertFalse(self.client.get(f"/api/sources/{cid}/playback").get_json()["seekable"])


import socket  # noqa: E402
from contextlib import contextmanager  # noqa: E402


@contextmanager
def fake_dns(table: dict):
    """Resolve host names from ``table``; everything else goes to the real resolver."""
    real = socket.getaddrinfo

    def fake(host, port, *args, **kwargs):
        if host in table:
            out = []
            for ip in table[host]:
                fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
                sa = (ip, port, 0, 0) if fam == socket.AF_INET6 else (ip, port)
                out.append((fam, socket.SOCK_STREAM, 6, "", sa))
            return out
        return real(host, port, *args, **kwargs)

    with mock.patch.object(socket, "getaddrinfo", fake):
        yield


class UrlPolicyTests(LibraryCase):
    """SSRF policy for URL sources, all through the HTTP API."""

    def setUp(self):
        super().setUp()
        self.enable_urls()

    def post(self, url):
        return self.client.post("/api/sources/url", json={"url": url})

    def assertRefused(self, url, codes=("blocked_address",)):
        before = len(sources.list_sources())
        r = self.post(url)
        self.assertEqual(r.status_code, 400, f"{url}: {r.get_json()}")
        self.assertIn(r.get_json()["error"], codes, url)
        self.assertEqual(len(sources.list_sources()), before, url)

    def test_loopback_and_literal_tricks_blocked(self):
        for url in ("http://localhost/x.mp4", "http://127.0.0.1/x", "rtsp://[::1]:554/s",
                    "http://[::ffff:127.0.0.1]/x", "http://[::ffff:7f00:1]/x", "http://0.0.0.0/x",
                    "http://[::]/x"):
            self.assertRefused(url)
        # decimal / octal / hex spellings either resolve to loopback or don't resolve at all
        for url in ("http://127.1/x", "http://0x7f000001/x", "http://2130706433/x", "http://0177.0.0.1/x", "http://017700000001/x"):
            self.assertRefused(url, ("blocked_address", "unresolvable_host"))

    def test_metadata_link_local_multicast_reserved_blocked(self):
        for url in ("http://169.254.169.254/latest/meta-data/", "http://[fd00:ec2::254]/", "http://[fe80::1]/x",
                    "rtsp://224.0.0.1/s", "http://239.255.255.250/x", "http://[ff02::1]/x", "http://240.0.0.1/x",
                    "http://255.255.255.255/x", "http://[2002:7f00:1::]/x", "http://[64:ff9b::7f00:1]/x"):
            self.assertRefused(url)
        # even the private opt-in and an allowlist never open metadata / loopback
        with mock.patch.object(sources, "URL_ALLOW_PRIVATE", True), \
                mock.patch.object(sources, "URL_ALLOW_CIDRS", (sources.ipaddress.ip_network("0.0.0.0/0"),)):
            self.assertRefused("http://169.254.169.254/latest/meta-data/")
            self.assertRefused("http://[fd00:ec2::254]/")
            self.assertRefused("http://127.0.0.1/x")

    def test_private_ranges_need_opt_in(self):
        private = ("http://10.0.0.5/x.mp4", "http://192.168.1.20/cam.mjpg", "rtsp://172.16.0.9/stream1",
                   "http://[fd12:3456::1]/x", "http://100.64.1.1/x")
        for url in private:
            self.assertRefused(url, ("private_address",))
        with mock.patch.object(sources, "URL_ALLOW_PRIVATE", True):
            for url in private[:4]:
                r = self.post(url)
                self.assertEqual(r.status_code, 200, f"{url}: {r.get_json()}")
        sources.close_all()
        nets = (sources.ipaddress.ip_network("192.168.1.0/24"),)
        with mock.patch.object(sources, "URL_ALLOW_CIDRS", nets):
            self.assertEqual(self.post("http://192.168.1.20/cam.mjpg").status_code, 200)
            self.assertRefused("http://192.168.2.20/cam.mjpg", ("private_address",))

    def test_hostname_resolution_is_checked(self):
        with fake_dns({"evil.example": ["127.0.0.1"]}):
            self.assertRefused("http://evil.example/x.m3u8")
        # one bad address among several is enough to refuse
        with fake_dns({"mixed.example": ["93.184.216.34", "10.0.0.1"]}):
            self.assertRefused("http://mixed.example/x", ("private_address",))
        with fake_dns({"v6.example": ["::ffff:169.254.169.254"]}):
            self.assertRefused("http://v6.example/x")
        self.assertRefused("http://does-not-exist.invalid/x", ("unresolvable_host",))
        self.http["https://cams.example.org/live/index.m3u8"] = (200, {}, b"#EXTM3U\n#EXTINF:4,\nseg1.ts\n")
        with fake_dns({"cams.example.org": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]}):
            r = self.post("https://cams.example.org/live/index.m3u8")
        self.assertEqual(r.status_code, 200, r.get_json())

    def test_master_switch(self):
        with mock.patch.object(sources, "URLS_ENABLED", False):
            with fake_dns({"cams.example.com": ["93.184.216.34"]}):
                r = self.post("http://cams.example.com/x.mp4")
            self.assertEqual(r.status_code, 403)
            self.assertEqual(r.get_json()["error"], "urls_disabled")
            self.assertFalse(self.client.get("/api/videos").get_json()["allow_urls"])
            self.assertEqual(self.fetched, [])                 # nothing fetched while disabled
        # uploads are unaffected by the switch
        with mock.patch.object(sources, "URLS_ENABLED", False):
            self.add_clip()

    def test_url_sources_off_by_default(self):
        self.assertIn("SECURITYCAM_VIDEO_URLS", Path(sources.__file__).read_text(encoding="utf-8"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SECURITYCAM_VIDEO_URLS", None)
            os.environ.pop("SECURITYCAM_VIDEO_ALLOW_URLS", None)
            self.assertFalse(sources._env_flag("SECURITYCAM_VIDEO_URLS", sources._env_flag("SECURITYCAM_VIDEO_ALLOW_URLS", False)))

    PUBLIC = {"start.example": ["93.184.216.34"], "cdn.example": ["93.184.216.35"], "lan.example": ["192.168.1.7"]}

    def redirect(self, to):
        return (302, {"location": to}, b"")

    def test_redirects_to_internal_addresses_rejected(self):
        for target, code in (("http://127.0.0.1/x.mp4", "blocked_address"),
                             ("http://169.254.169.254/latest/meta-data/", "blocked_address"),
                             ("http://192.168.1.10/cam.mjpg", "private_address"),
                             ("http://lan.example/cam.mjpg", "private_address"),
                             ("/relative-but-fine.mp4", None),
                             ("file:///etc/passwd", "bad_redirect")):
            self.http = {"http://start.example/v.mp4": self.redirect(target)}
            with fake_dns(self.PUBLIC):
                r = self.post("http://start.example/v.mp4")
            if code is None:
                self.assertEqual(r.status_code, 200, r.get_json())
            else:
                self.assertEqual(r.status_code, 400, target)
                self.assertEqual(r.get_json()["error"], code, target)
            self.assertNotIn(target, self.fetched)            # the bad hop is never requested
        sources.close_all()

    def test_redirect_chain_limit(self):
        chain = {f"http://start.example/{i}": self.redirect(f"http://cdn.example/{i + 1}") for i in range(0, 6, 2)}
        chain.update({f"http://cdn.example/{i}": self.redirect(f"http://start.example/{i + 1}") for i in range(1, 7, 2)})
        self.http = chain                                    # 0 -> 1 -> ... -> 6: six redirects
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/0")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "too_many_redirects")
        self.http = {k: v for k, v in chain.items() if k not in ("http://cdn.example/5",)}   # 5 hops -> fine
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/0")
        self.assertEqual(r.status_code, 200, r.get_json())
        cid = r.get_json()["camera_id"]
        # FFmpeg only ever sees the final, validated URL
        seen = {}
        real_capture = cv2.VideoCapture

        def fake_capture(*args, **kwargs):
            if args:
                seen["url"] = args[0]
            return real_capture()

        with fake_dns(self.PUBLIC), mock.patch.object(sources.cv2, "VideoCapture", fake_capture):
            sources.UrlVideoStream(state.get_camera_state(cid).source.url, state.get_camera_state(cid)).release()
        self.assertEqual(seen["url"], "http://cdn.example/5")

    HLS_HEAD = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n"

    def test_hls_with_private_segment_rejected(self):
        media = self.HLS_HEAD + "#EXTINF:4,\nseg1.ts\n#EXTINF:4,\nhttp://lan.example/seg2.ts\n"
        self.http = {"http://start.example/live.m3u8": (200, {"content-type": "application/vnd.apple.mpegurl"}, media.encode())}
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/live.m3u8")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "private_address")
        # a key on the metadata address, a playlist served without .m3u8 / content type
        media = self.HLS_HEAD + '#EXT-X-KEY:METHOD=AES-128,URI="http://169.254.169.254/key"\n#EXTINF:4,\nseg1.ts\n'
        self.http = {"http://start.example/play": (200, {"content-type": "text/plain"}, media.encode())}
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/play")
        self.assertEqual(r.get_json()["error"], "blocked_address")

    def test_hls_master_variant_checked(self):
        master = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=800000\nhttp://cdn.example/low/index.m3u8\n" \
                 '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="en",URI="audio/index.m3u8"\n'
        good = self.HLS_HEAD + "#EXTINF:4,\nseg1.ts\n#EXTINF:4,\nhttps://cdn.example/seg2.ts\n"
        bad = self.HLS_HEAD + "#EXTINF:4,\nhttp://127.0.0.1:8080/seg.ts\n"
        self.http = {
            "http://start.example/master.m3u8": (200, {"content-type": "application/x-mpegURL"}, master.encode()),
            "http://cdn.example/low/index.m3u8": (200, {}, good.encode()),
            "http://start.example/audio/index.m3u8": (200, {}, bad.encode()),
        }
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/master.m3u8")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "blocked_address")
        self.http["http://start.example/audio/index.m3u8"] = (200, {}, good.encode())
        with fake_dns(self.PUBLIC):
            r = self.post("http://start.example/master.m3u8")
        self.assertEqual(r.status_code, 200, r.get_json())

    def test_clean_public_hls_passes(self):
        media = self.HLS_HEAD + "#EXTINF:4,\nseg1.ts\n#EXTINF:4,\nhttps://cdn.example/seg2.ts\n#EXT-X-ENDLIST\n"
        self.http = {"https://start.example/live.m3u8": (200, {"content-type": "application/vnd.apple.mpegurl"}, media.encode())}
        with fake_dns(self.PUBLIC):
            r = self.post("https://start.example/live.m3u8")
        self.assertEqual(r.status_code, 200, r.get_json())

    def test_fetch_refuses_automatic_redirects(self):
        # the real urllib path: a 302 is reported, not followed (local server, check bypassed)
        import http.server

        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(302)
                self.send_header("Location", "/second")
                self.end_headers()

            def log_message(self, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            mock.patch.stopall()
            status, headers, _ = sources._http_fetch(f"http://127.0.0.1:{srv.server_port}/first", True)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(status, 302)
        self.assertEqual(headers["location"], "/second")
        self.assertEqual(hits, ["/first"])

    def test_reader_rechecks_and_whitelists_protocols(self):
        with fake_dns({"cams.example.com": ["93.184.216.34"]}):
            cid = self.post("http://cams.example.com/x.mp4").get_json()["camera_id"]
        cs = state.get_camera_state(cid)
        seen = {}
        real_capture = cv2.VideoCapture

        def fake_capture(*args, **kwargs):
            if args:
                seen["options"] = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
                seen["args"] = args
            return real_capture()

        before = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        # DNS now points somewhere else (rebinding): the reader refuses before FFmpeg connects
        with fake_dns({"cams.example.com": ["127.0.0.1"]}), mock.patch.object(sources.cv2, "VideoCapture", fake_capture):
            stream = sources.UrlVideoStream(cs.source.url, cs)
        self.assertNotIn("options", seen)
        self.assertFalse(stream.grabbed)
        self.assertIn("blocked", cs.stream_error)
        stream.release()
        with fake_dns({"cams.example.com": ["93.184.216.34"]}), mock.patch.object(sources.cv2, "VideoCapture", fake_capture):
            stream = sources.UrlVideoStream(cs.source.url, cs)
        stream.release()
        self.assertIn("protocol_whitelist;http,https,tcp,tls,rtsp,rtp,udp,crypto", seen["options"])
        self.assertNotIn("file", seen["options"].split("protocol_whitelist;")[1])
        self.assertEqual(seen["args"][1], cv2.CAP_FFMPEG)
        self.assertEqual(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"), before)   # restored


@unittest.skipUnless(os.getenv("SECURITYCAM_TEST_MODELS_DIR"), "set SECURITYCAM_TEST_MODELS_DIR for the real-model test")
class RealModelTests(LibraryCase):
    def test_detects_and_tracks_people_in_a_clip(self):
        from securitycam import config

        if not state.YOLO_AVAILABLE:
            self.skipTest("ultralytics not available")
        import ultralytics

        bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        clip = make_clip(_MEDIA / "bus.mp4", fps=25, frames=100, size=(640, 480), background=cv2.resize(bus, (960, 1280)[::-1]))
        v = self.add_clip("bus.mp4", clip)
        cid = self.client.post(f"/api/videos/{v['id']}/open", json={}).get_json()["camera_id"]
        cs = state.get_camera_state(cid)
        cs.esp_settings.update({"tracking": True, "tracers": True})
        with mock.patch.object(config, "MODELS_DIR", Path(os.environ["SECURITYCAM_TEST_MODELS_DIR"])):
            gen = streaming.gen_frames(cid)
            stop = time.time() + 8.0
            labels: set = set()
            while time.time() < stop:
                next(gen)
                t = self.client.get(f"/api/cameras/{cid}/targets").get_json()
                labels |= {x["label"] for x in t.get("targets", [])}
                if "person" in labels and "bus" in labels and cs.track_history:
                    break
            self.assertIn("person", labels)
            self.assertTrue(cs.track_history)
            ids_before = set(cs.track_history)
            # a cut (seek back) rebuilds the tracker; tracking resumes with fresh ids
            self.client.post(f"/api/sources/{cid}/playback", json={"action": "seek", "seconds": 0.0})
            stop = time.time() + 6.0
            while time.time() < stop:
                next(gen)
                if cs.track_history and not (set(cs.track_history) & ids_before):
                    break
            gen.close()
        self.assertTrue(cs.track_history)
        self.assertFalse(set(cs.track_history) & ids_before)


if __name__ == "__main__":
    unittest.main()
