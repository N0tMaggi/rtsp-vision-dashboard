"""Tests for the SSE push channel (securitycam/live.py) and the cached /status GPU sampler.

No camera is contacted: the dummy cameras use TEST-NET addresses and the
stream is read through Flask's test client (the generator runs in this thread).

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-live-"))
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS",
                      "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from securitycam import deps, live, presets, state  # noqa: E402
from securitycam import status as status_mod  # noqa: E402
from securitycam.webapp import app  # noqa: E402

presets.SCHEDULER.stop()

STATIC_GPU = {"name": "TestGPU", "util": 1, "mem_used": 2, "mem_total": 3}


def parse_frames(text: str) -> list[dict]:
    """Split SSE text into frames: {"event", "data", "comment", "retry"}."""
    frames = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        frame: dict = {"event": None, "data": None, "comments": [], "retry": None}
        data_lines = []
        for line in block.split("\n"):
            if line.startswith(":"):
                frame["comments"].append(line[1:].strip())
            elif line.startswith("event: "):
                frame["event"] = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("retry: "):
                frame["retry"] = int(line[7:])
        if data_lines:
            frame["data"] = json.loads("\n".join(data_lines))
        frames.append(frame)
    return frames


class StreamReader:
    """Pulls chunks from a non-buffered test response."""

    def __init__(self, response):
        self.response = response
        self.it = iter(response.response)

    def chunk(self) -> str:
        c = next(self.it)
        return c.decode("utf-8") if isinstance(c, bytes) else c

    def until(self, pred, timeout: float = 3.0) -> list[dict]:
        """Read frames until ``pred(frame)`` is true for one of them; returns all frames read."""
        got: list[dict] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frames = parse_frames(self.chunk())
            got += frames
            if any(pred(f) for f in frames):
                return got
        raise AssertionError(f"condition not met; got {[f['event'] or f['comments'] for f in got]}")

    def read_for(self, seconds: float) -> list[dict]:
        got: list[dict] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            got += parse_frames(self.chunk())
        return got

    def close(self):
        self.response.close()


class LiveStreamTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        live._SNAPSHOTS.clear()
        for cs in state.camera_states.values():
            cs.yolo_enabled = False
        self.patches = [
            mock.patch.object(live, "HEARTBEAT_S", 0.1),
            mock.patch.object(status_mod.GPU, "snapshot", return_value=dict(STATIC_GPU)),
        ]
        for p in self.patches:
            p.start()
        self.readers: list[StreamReader] = []

    def tearDown(self):
        for r in self.readers:
            r.close()
        for p in self.patches:
            p.stop()
        for cs in state.camera_states.values():
            cs.yolo_enabled = False

    def open(self, query: str = "camera=front") -> StreamReader:
        resp = self.client.get(f"/api/stream?{query}", buffered=False)
        self.assertEqual(resp.status_code, 200)
        r = StreamReader(resp)
        self.readers.append(r)
        return r

    # ------------------------------------------------------------------ framing
    def test_headers_and_initial_snapshot(self):
        r = self.open()
        h = r.response.headers
        self.assertTrue(h["Content-Type"].startswith("text/event-stream"))
        self.assertIn("no-cache", h["Cache-Control"])
        self.assertEqual(h["X-Accel-Buffering"], "no")
        first = parse_frames(r.chunk())
        self.assertEqual(first[0]["retry"], live.RETRY_MS)
        frames = r.until(lambda f: f["event"] == "plates")
        by_name = {f["event"]: f["data"] for f in frames if f["event"]}
        for name in ("status", "targets", "events", "plates", "ptz"):
            self.assertIn(name, by_name)
        self.assertNotIn("overview", by_name)
        self.assertNotIn("playback", by_name)       # not a video source
        # same shapes as the GET endpoints
        st = by_name["status"]
        self.assertEqual(st["camera"]["id"], "front")
        self.assertEqual(set(st), set(self.client.get("/api/cameras/front/status").get_json()))
        self.assertNotIn("rtsp_url", json.dumps(st))
        self.assertEqual(set(by_name["targets"]), set(self.client.get("/api/cameras/front/targets").get_json()))
        self.assertEqual(by_name["targets"]["targets"], [])
        self.assertEqual(set(by_name["events"]), {"events", "active", "now"})
        self.assertIn("recent", by_name["plates"])
        self.assertIn("version", by_name["plates"])

    def test_format_event_splits_lines(self):
        self.assertEqual(live.format_event("x", '{"a":1}'), 'event: x\ndata: {"a":1}\n\n')
        self.assertEqual(live.format_event("x", "a\nb"), "event: x\ndata: a\ndata: b\n\n")

    # ------------------------------------------------------------------ change-only + heartbeat
    def test_only_changes_are_sent_then_heartbeats(self):
        r = self.open()
        r.until(lambda f: f["event"] == "plates")
        later = r.read_for(1.6)     # > status / events / ptz intervals: they are recomputed but unchanged
        self.assertTrue(later)
        self.assertEqual([f for f in later if f["event"]], [])
        self.assertTrue(all("ping" in f["comments"] for f in later))

    def test_heartbeat_interval(self):
        with mock.patch.object(live, "HEARTBEAT_S", 0.3):
            r = self.open()
            r.until(lambda f: f["event"] == "plates")
            t0 = time.monotonic()
            frames = r.until(lambda f: "ping" in f["comments"], timeout=2.0)
            dt = time.monotonic() - t0
        self.assertGreaterEqual(dt, 0.2)
        self.assertLess(dt, 1.5)
        self.assertEqual([f for f in frames if f["event"]], [])

    def test_write_pushes_status_immediately(self):
        r = self.open()
        r.until(lambda f: f["event"] == "plates")
        t0 = time.monotonic()
        resp = self.client.post("/api/cameras/front/toggle_yolo")
        self.assertEqual(resp.status_code, 200)
        frames = r.until(lambda f: f["event"] == "status" and f["data"]["yolo_enabled"] is True, timeout=2.0)
        self.assertLess(time.monotonic() - t0, 0.6)    # not waiting for the 1 s tick
        self.assertTrue(any(f["event"] == "status" for f in frames))

    def test_status_change_from_worker_is_pushed(self):
        r = self.open()
        r.until(lambda f: f["event"] == "plates")
        cs = state.camera_states["front"]
        old = cs.stream_error
        try:
            cs.stream_error = "Camera stream is unavailable."
            r.until(lambda f: f["event"] == "status" and f["data"]["stream_error"] == cs.stream_error, timeout=2.5)
        finally:
            cs.stream_error = old

    def test_targets_follow_rendered_frames(self):
        cs = state.camera_states["front"]
        r = self.open()
        r.until(lambda f: f["event"] == "plates")
        fake = {"ts": 0, "frame": {"w": 640, "h": 480}, "locked": None,
                "targets": [{"id": 1, "label": "person", "conf": 0.9, "box": [0.1, 0.1, 0.2, 0.3]}]}
        calls = []

        def payload(c):
            calls.append(time.monotonic())
            return dict(fake, ts=time.time())

        with mock.patch.object(live.targets, "targets_payload", side_effect=payload):
            cs.yolo_enabled = True
            live.notify_change()
            r.until(lambda f: f["event"] == "targets" and f["data"]["targets"], timeout=2.0)
            n = len(calls)
            # no new frame -> no recompute, nothing sent
            quiet = r.read_for(0.4)
            self.assertEqual(len(calls), n)
            self.assertFalse([f for f in quiet if f["event"] == "targets"])
            # new frames, but identical boxes (only ``ts`` differs) -> recomputed, still not sent
            cs.stats["total_frames"] += 1
            quiet = r.read_for(0.3)
            self.assertGreater(len(calls), n)
            self.assertFalse([f for f in quiet if f["event"] == "targets"])
            # a moved box on the next frame is sent
            fake["targets"] = [dict(fake["targets"][0], box=[0.2, 0.1, 0.3, 0.3])]
            cs.stats["total_frames"] += 1
            r.until(lambda f: f["event"] == "targets" and f["data"]["targets"][0]["box"][0] == 0.2, timeout=1.0)

    def test_targets_rate_is_capped(self):
        cs = state.camera_states["front"]
        cs.yolo_enabled = True
        stop = threading.Event()

        def frames():   # a fast renderer: far more frames than the cap
            while not stop.is_set():
                cs.stats["total_frames"] += 1
                time.sleep(0.002)

        n = {"i": 0}

        def payload(c):
            n["i"] += 1
            return {"ts": time.time(), "frame": {"w": 1, "h": 1}, "locked": None,
                    "targets": [{"id": 1, "label": "person", "conf": 0.5, "box": [0, 0, 0.1 + n["i"] * 1e-4, 0.1]}]}

        t = threading.Thread(target=frames, daemon=True)
        with mock.patch.object(live.targets, "targets_payload", side_effect=payload):
            r = self.open()
            r.until(lambda f: f["event"] == "targets")
            t.start()
            try:
                got = r.read_for(1.0)
            finally:
                stop.set()
                t.join()
        sent = [f for f in got if f["event"] == "targets"]
        self.assertGreaterEqual(len(sent), 6)       # smooth: at least ~ today's 120 ms polling
        self.assertLessEqual(len(sent), 14)         # capped at ~12 Hz

    def test_overview_only_on_request(self):
        r = self.open("camera=front&overview=1")
        frames = r.until(lambda f: f["event"] == "overview")
        ov = next(f["data"] for f in frames if f["event"] == "overview")
        self.assertEqual({c["id"] for c in ov["cameras"]}, set(state.camera_states))

    def test_playback_for_video_sources(self):
        cs = state.camera_states["garage"]
        pb = {"kind": "file", "seekable": True, "playing": False, "ended": False, "speed": 1.0, "loop": True,
              "fps": 25.0, "frame": 0, "frames": 250, "position": 0.0, "duration": 10.0, "server_time": 0}
        src = SimpleNamespace(playback_payload=lambda: dict(pb, server_time=time.time()))
        with mock.patch.object(cs, "source", src):
            r = self.open("camera=garage")
            frames = r.until(lambda f: f["event"] == "playback")
            self.assertEqual(next(f["data"] for f in frames if f["event"] == "playback")["position"], 0.0)
            pb["position"] = 4.0
            r.until(lambda f: f["event"] == "playback" and f["data"]["position"] == 4.0, timeout=2.0)

    # ------------------------------------------------------------------ lifecycle
    def test_camera_switch_and_unknown_camera(self):
        a = self.open("camera=front")
        fa = a.until(lambda f: f["event"] == "status")
        self.assertEqual(next(f["data"] for f in fa if f["event"] == "status")["camera"]["id"], "front")
        a.close()
        b = self.open("camera=garage")
        fb = b.until(lambda f: f["event"] == "status")
        self.assertEqual(next(f["data"] for f in fb if f["event"] == "status")["camera"]["id"], "garage")
        self.assertEqual(self.client.get("/api/stream?camera=nope").status_code, 404)
        self.assertEqual(self.client.get("/api/stream").status_code, 404)

    def test_disconnect_frees_the_slot(self):
        base = live.client_count()
        r = self.open()
        r.chunk()
        self.assertEqual(live.client_count(), base + 1)
        r.close()       # what werkzeug does when the browser goes away (GeneratorExit in the generator)
        self.assertEqual(live.client_count(), base)
        # a generator that was never started holds no slot (client gone before the first byte)
        gen = live.client_stream(state.camera_states["front"])
        self.assertEqual(live.client_count(), base)
        gen.close()
        self.assertEqual(live.client_count(), base)
        # many connect / disconnect cycles leave nothing behind
        threads = threading.active_count()
        for _ in range(20):
            resp = self.client.get("/api/stream?camera=front", buffered=False)
            next(iter(resp.response))
            resp.close()
        self.assertEqual(live.client_count(), base)
        self.assertLessEqual(threading.active_count(), threads + 1)   # (+1: GPU sampler may start)

    def test_closed_virtual_camera_ends_stream(self):
        cs = state.camera_states["garage"]
        r = self.open("camera=garage")
        r.until(lambda f: f["event"] == "status")
        base = live.client_count()
        with mock.patch.object(cs, "closed", True):
            frames = r.until(lambda f: f["event"] == "closed")
            self.assertEqual(frames[-1]["data"]["camera"], "garage")
            with self.assertRaises(StopIteration):
                r.chunk()
        self.assertEqual(live.client_count(), base - 1)

    def test_too_many_streams(self):
        with mock.patch.object(live, "MAX_CLIENTS", live.client_count()):
            resp = self.client.get("/api/stream?camera=front")
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.get_json()["error"], "too_many_streams")

    def test_snapshots_shared_between_clients(self):
        calls = []

        def payload():
            calls.append(1)
            return {"x": 1}

        a = live._SNAPSHOTS.get(("t",), 5.0, payload)
        b = live._SNAPSHOTS.get(("t",), 5.0, payload)
        self.assertIs(a, b)
        self.assertEqual(len(calls), 1)
        live.notify_change()        # a write invalidates
        live._SNAPSHOTS.get(("t",), 5.0, payload)
        self.assertEqual(len(calls), 2)

    def test_snapshot_digest_ignores_volatile_fields(self):
        a = live._snapshot({"ts": 1, "v": [1]}, ("ts",))
        b = live._snapshot({"ts": 2, "v": [1]}, ("ts",))
        c = live._snapshot({"ts": 2, "v": [2]}, ("ts",))
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.data, b.data)
        self.assertNotEqual(b.digest, c.digest)


class StatusCacheTests(unittest.TestCase):
    def test_status_does_not_wait_for_nvidia_smi(self):
        sampler = status_mod.GpuSampler(interval_s=0.05, idle_s=0.3)
        calls = []

        def slow_snapshot():
            calls.append(1)
            time.sleep(0.25)        # nvidia-smi under load
            return {"name": "SlowGPU", "util": 42, "mem_used": 1, "mem_total": 2}

        client = app.test_client()
        with mock.patch.object(status_mod, "GPU", sampler), mock.patch.object(deps, "get_gpu_snapshot", slow_snapshot):
            try:
                times = []
                for _ in range(10):
                    t = time.perf_counter()
                    self.assertEqual(client.get("/status").status_code, 200)
                    times.append(time.perf_counter() - t)
                self.assertLess(sorted(times)[len(times) // 2], 0.05)
                self.assertTrue(sampler.running())
                deadline = time.monotonic() + 2.0
                while sampler.samples == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(client.get("/status").get_json()["gpu"]["name"], "SlowGPU")
                # no readers -> the sampler thread ends by itself
                deadline = time.monotonic() + 3.0
                while sampler.running() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(sampler.running())
                n = len(calls)
                time.sleep(0.3)
                self.assertEqual(len(calls), n)
                # and a new reader starts it again
                client.get("/status")
                self.assertTrue(sampler.running())
            finally:
                sampler.stop()
                state.gpu_stats.update({"name": state.DEVICE, "util": 0, "mem_used": 0, "mem_total": 0})
        self.assertFalse(sampler.running())

    def test_sampler_survives_errors(self):
        sampler = status_mod.GpuSampler(interval_s=0.02, idle_s=1.0)
        with mock.patch.object(deps, "get_gpu_snapshot", side_effect=RuntimeError("nvml")):
            try:
                sampler.snapshot()
                time.sleep(0.1)
                self.assertTrue(sampler.running())
            finally:
                sampler.stop()


if __name__ == "__main__":
    unittest.main()
