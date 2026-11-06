"""Tests for the persistent license-plate log (securitycam/recognition/platelog*.py).

Synthetic frames only (a car-ish shape with a plate drawn by cv2.putText); the
plate reader is mocked, no camera is contacted and no model is downloaded.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
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

from securitycam import state  # noqa: E402
from securitycam.recognition import common, platelog, worker  # noqa: E402
from securitycam.recognition import plates as P  # noqa: E402
from securitycam.recognition.common import PlateRead, TrackInfo  # noqa: E402
from securitycam.webapp import app  # noqa: E402

CAM = "front"
API = "/api/recognition/platelog"


def car_frame(text: str = "AB 123 CD", w: int = 1280, h: int = 720):
    """A grey scene with a boxy car and a white plate. Returns (frame, vehicle box, plate box)."""
    img = np.full((h, w, 3), (92, 96, 100), np.uint8)
    cv2.rectangle(img, (0, int(h * 0.75)), (w, h), (60, 62, 64), -1)                 # road
    x1, y1, x2, y2 = 380, 260, 900, 600
    cv2.rectangle(img, (x1, y1 + 90), (x2, y2 - 40), (140, 60, 30), -1)              # body
    cv2.rectangle(img, (x1 + 90, y1), (x2 - 90, y1 + 100), (120, 50, 25), -1)        # cabin
    cv2.rectangle(img, (x1 + 110, y1 + 15), (x2 - 110, y1 + 85), (200, 190, 170), -1)  # window
    for cx in (x1 + 100, x2 - 100):
        cv2.circle(img, (cx, y2 - 40), 48, (25, 25, 25), -1)                          # wheels
    px1, py1, px2, py2 = 540, 470, 740, 515
    cv2.rectangle(img, (px1, py1), (px2, py2), (245, 245, 245), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (20, 20, 20), 2)
    cv2.putText(img, text, (px1 + 12, py2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    return img, (x1, y1, x2, y2), (px1, py1, px2, py2)


class FakeReader:
    """Stands in for plates.PlateReader: returns scripted reads for the drawn plate."""

    def __init__(self, texts, plate_box, conf=0.9):
        self.texts = list(texts)
        self.box = plate_box
        self.conf = conf

    def read_vehicle(self, frame, box):
        if not self.texts:
            return None
        return PlateRead(text=self.texts.pop(0), conf=self.conf, box=self.box)


def new_log(**kw) -> platelog.PlateLog:
    root = Path(tempfile.mkdtemp(prefix="platelog-"))
    log = platelog.PlateLog(root, **kw)
    platelog.set_log(log)
    return log


def observe(log, plate="AB123CD", tid=7, reads=3, share=0.8, conf=0.9, frame=None, cam=CAM, now=None, pbox=None):
    f, vbox, box = car_frame() if frame is None else frame
    read = PlateRead(text=plate, conf=conf, box=pbox or box)
    return log.observe(cam, tid, "car", plate, share, reads, read, f, vbox, now=now)


class TextMatching(unittest.TestCase):
    def test_fold_and_similar(self):
        self.assertEqual(platelog.fold("ab-123 cd"), platelog.fold("A8 l23 C0"))
        self.assertTrue(platelog.similar("AB123CD", "A8123CD"))
        self.assertTrue(platelog.similar("AB123CD", "AB128CD"))      # one real substitution
        self.assertFalse(platelog.similar("AB123CD", "XY987ZZ"))

    def test_match_score(self):
        self.assertEqual(platelog.match_score("123", "AB123CD"), 3)
        self.assertEqual(platelog.match_score("a8 l23", "AB123CD"), 2)      # 8/B and l/1 confusions
        self.assertEqual(platelog.match_score("0B12", "OB12XY"), 2)
        self.assertEqual(platelog.match_score("AB124C", "AB123CD"), 1)     # one typo
        self.assertEqual(platelog.match_score("XYZ99", "AB123CD"), 0)
        self.assertEqual(platelog.match_score("", "AB123CD"), 3)


class LogWrites(unittest.TestCase):
    def setUp(self):
        self.log = new_log()

    def test_needs_confirmation(self):
        self.assertIsNone(observe(self.log, reads=1))
        self.assertIsNone(observe(self.log, reads=4, share=0.3))
        self.assertIsNone(observe(self.log, tid=None, conf=0.6))        # untracked needs a confident read
        self.log.flush()
        self.assertEqual(self.log.summary()["total"], 0)
        eid = observe(self.log, tid=None, conf=0.92)
        self.assertIsNotNone(eid)

    def test_entry_images_and_boxes(self):
        eid = observe(self.log)
        self.assertTrue(self.log.flush())
        e = self.log.get(eid)
        self.assertEqual(e["plate"], "AB123CD")
        self.assertEqual(e["vehicle"], "car")
        self.assertEqual(e["track_id"], 7)
        self.assertEqual(e["images"], {"plate": True, "vehicle": True, "frame": True})
        self.assertAlmostEqual(e["plate_box"][0], 540 / 1280, places=3)
        self.assertAlmostEqual(e["vehicle_box"][3], 600 / 720, places=3)
        plate = cv2.imread(str(self.log.image_path(eid, "plate")))
        self.assertGreaterEqual(plate.shape[1], platelog.PLATE_CROP_W)   # upscaled
        frame = cv2.imread(str(self.log.image_path(eid, "frame")))
        self.assertEqual(frame.shape[1], 1280)
        vehicle = cv2.imread(str(self.log.image_path(eid, "vehicle")))
        self.assertLessEqual(max(vehicle.shape[:2]), platelog.VEHICLE_MAX_SIDE)

    def test_large_frame_is_downscaled(self):
        f, vbox, box = car_frame(w=2560, h=1440)
        eid = observe(self.log, frame=(f, vbox, box))
        self.log.flush()
        frame = cv2.imread(str(self.log.image_path(eid, "frame")))
        self.assertEqual(frame.shape[1], platelog.FRAME_MAX_W)

    def test_one_entry_per_track_updated_while_alive(self):
        eid = observe(self.log, reads=2, share=0.6, conf=0.7)
        self.log.flush()
        first = self.log.get(eid)
        st = next(iter(self.log._tracks.values()))
        st["img_mono"] = st["meta_mono"] = -1e9        # skip the rate limits
        self.assertEqual(observe(self.log, reads=5, share=0.9, conf=0.95), eid)
        self.log.flush()
        e = self.log.get(eid)
        self.assertEqual(self.log.summary()["total"], 1)
        self.assertEqual(e["reads"], 5)
        self.assertEqual(e["conf"], 0.9)
        self.assertGreater(e["quality"], first["quality"])
        self.assertEqual(e["first_seen"], first["first_seen"])
        self.assertGreaterEqual(e["last_seen"], first["last_seen"])
        # a vote that settles on a confusable variant keeps the entry
        st["meta_mono"] = -1e9
        self.assertEqual(observe(self.log, plate="A8123CD", reads=6), eid)
        # a different plate on the same track id (id reused by another car) -> new entry
        other = observe(self.log, plate="XY987ZZ", reads=6)
        self.assertNotEqual(other, eid)
        self.log.flush()
        self.assertEqual(self.log.summary()["total"], 2)

    def test_track_gap_starts_new_entry(self):
        eid = observe(self.log)
        self.log.flush()
        for st in self.log._tracks.values():
            st["mono"] -= platelog.TRACK_GAP_S + 1
        self.assertNotEqual(observe(self.log), eid)

    def test_reload_from_sqlite(self):
        eid = observe(self.log)
        self.log.flush()
        again = platelog.PlateLog(self.log.root)
        self.assertEqual(again.get(eid)["plate"], "AB123CD")

    def test_retention_count(self):
        log = new_log(max_count=3)
        ids = []
        for i in range(5):
            ids.append(observe(log, plate=f"AB{i}{i}{i}CD", tid=100 + i, now=time.time() + i))
            log.flush()
        self.assertEqual(log.summary()["total"], 3)
        self.assertIsNone(log.get(ids[0]))
        self.assertFalse((log.root / CAM / ids[0]).exists())
        self.assertIsNotNone(log.get(ids[4]))

    def test_retention_age(self):
        log = new_log(max_age_days=1)
        old = observe(log, plate="OLD1234", tid=1, now=time.time() - 3 * 86400)
        fresh = observe(log, plate="NEW1234", tid=2)
        log.flush()
        reloaded = platelog.PlateLog(log.root, max_age_days=1)
        self.assertIsNone(reloaded.get(old))
        self.assertIsNotNone(reloaded.get(fresh))
        self.assertFalse((log.root / CAM / old).exists())

    def test_delete_and_clear(self):
        a = observe(self.log, plate="AB123CD", tid=1)
        b = observe(self.log, plate="XY987ZZ", tid=2, cam="garage")
        self.log.flush()
        self.assertTrue(self.log.delete(a))
        self.assertFalse((self.log.root / CAM / a).exists())
        self.assertFalse(self.log.delete(a))
        self.assertEqual(self.log.clear("garage"), 1)
        self.assertIsNone(self.log.get(b))
        self.assertEqual(platelog.PlateLog(self.log.root).summary()["total"], 0)

    def test_invalid_camera_ignored(self):
        self.assertIsNone(observe(self.log, cam="../evil"))


class VotingIntegration(unittest.TestCase):
    """worker.run_plates -> vote -> plate log (mocked reader, synthetic frame)."""

    def test_votes_into_one_entry(self):
        log = new_log()
        frame, vbox, pbox = car_frame("AB 123 CD")
        rt = common.RecogRuntime(camera_id=CAM)
        reader = FakeReader(["AB123CD", "A8123CD", "AB123CD", "AB123CD"], pbox)
        dets = [(42, "car", vbox, 0.9), (-1, "person", (10, 10, 60, 200), 0.8)]
        with mock.patch.object(P, "get_reader", return_value=reader), \
                mock.patch.object(worker, "PLATE_RETRY_S", 0.0):
            worker.run_plates(rt, frame, dets, 0.0)
            log.flush()
            self.assertEqual(log.summary()["total"], 0)       # one read is not a confirmed vote
            for _ in range(3):
                worker.run_plates(rt, frame, dets, 0.0)
        log.flush()
        items = log.query()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["plate"], "AB123CD")
        self.assertEqual(items[0]["track_id"], 42)
        self.assertEqual(items[0]["reads"], 4)
        self.assertEqual(rt.track_info[42].plate_best, "AB123CD")
        # the existing in-memory "Recent Plates" list keeps working
        self.assertEqual(rt.plate_log[0]["plate"], "AB123CD")


class Api(unittest.TestCase):
    def setUp(self):
        self.log = new_log()
        self.client = app.test_client()
        self.a = observe(self.log, plate="AB123CD", tid=1, now=time.time() - 7200)
        self.b = observe(self.log, plate="KL555MN", tid=2)
        self.c = observe(self.log, plate="A8123CD", tid=3, cam="garage")
        self.log.flush()

    def test_list_filters_and_pagination(self):
        r = self.client.get(API).get_json()
        self.assertEqual(r["total"], 3)
        self.assertEqual(r["items"][0]["id"], self.c)          # newest first
        self.assertTrue(r["items"][0]["plate_url"].startswith(f"{API}/{self.c}/plate.jpg"))
        self.assertEqual(r["items"][0]["camera_label"], "Garage")
        r = self.client.get(f"{API}?q=ab123").get_json()        # fuzzy: also finds A8123CD
        self.assertEqual({i["id"] for i in r["items"]}, {self.a, self.c})
        self.assertEqual(r["items"][0]["match"], 3)             # exact first
        r = self.client.get(f"{API}?q=ab123&camera=front").get_json()
        self.assertEqual([i["id"] for i in r["items"]], [self.a])
        r = self.client.get(f"{API}?since={time.time() - 600}").get_json()
        self.assertEqual({i["id"] for i in r["items"]}, {self.b, self.c})
        r = self.client.get(f"{API}?until={time.time() - 3600}").get_json()
        self.assertEqual([i["id"] for i in r["items"]], [self.a])
        r = self.client.get(f"{API}?limit=1&offset=1").get_json()
        self.assertEqual((len(r["items"]), r["total"]), (1, 3))
        self.assertEqual(self.client.get(f"{API}?since=abc").status_code, 400)
        self.assertEqual(self.client.get(f"{API}?camera=../x").status_code, 400)

    def test_summary(self):
        r = self.client.get(f"{API}/summary?seen={time.time() - 600}").get_json()
        self.assertEqual((r["total"], r["new"]), (3, 2))

    def test_get_and_images(self):
        r = self.client.get(f"{API}/{self.a}")
        self.assertEqual(r.get_json()["plate"], "AB123CD")
        for name in ("plate", "vehicle", "frame"):
            r = self.client.get(f"{API}/{self.a}/{name}.jpg")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.mimetype, "image/jpeg")
            self.assertEqual(r.data[:2], b"\xff\xd8")
            r.close()
        self.assertEqual(self.client.get(f"{API}/20990101-000000-abcdef").status_code, 404)

    def test_traversal_attempts(self):
        bad = [
            f"{API}/{self.a}/meta.jpg",
            f"{API}/{self.a}/..%2f..%2fplatelog.jpg",
            f"{API}/..%2f..%2f..%2fplatelog/plate.jpg",
            f"{API}/{self.a}%2f..%2f..%2f/plate.jpg",
            f"{API}/{self.a}/..%5c..%5cwatchlist.jpg",
            f"{API}/..%5cfront/plate.jpg",
            f"{API}/{self.a}/plate.jpg/../../../platelog.sqlite3",
            f"{API}/plate.jpg",
        ]
        for url in bad:
            r = self.client.get(url)
            self.assertIn(r.status_code, (400, 404, 405), url)
            self.assertNotEqual(r.mimetype, "image/jpeg", url)
        self.assertIsNone(self.log.image_path("../" + self.a, "plate"))
        self.assertIsNone(self.log.image_path(self.a, "../plate"))

    def test_delete_and_clear(self):
        self.assertEqual(self.client.delete(f"{API}/{self.a}").status_code, 200)
        self.assertEqual(self.client.delete(f"{API}/{self.a}").status_code, 404)
        self.assertEqual(self.client.get(f"{API}/{self.a}/plate.jpg").status_code, 404)
        r = self.client.delete(f"{API}?camera=garage").get_json()
        self.assertEqual(r["deleted"], 1)
        r = self.client.delete(API).get_json()
        self.assertEqual(r["deleted"], 1)
        self.assertEqual(self.client.get(API).get_json()["total"], 0)

    def test_csrf_guard(self):
        r = self.client.delete(f"{API}/{self.a}", headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.client.post(f"{API}/watchlist", json={"plate": "AB123CD"}, headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        self.assertIsNotNone(self.log.get(self.a))

    def test_export(self):
        r = self.client.get(f"{API}/export?format=csv&q=ab123")
        self.assertIn("attachment", r.headers["Content-Disposition"])
        rows = list(csv.DictReader(io.StringIO(r.data.decode("utf-8-sig"))))
        self.assertEqual({row["plate"] for row in rows}, {"AB123CD", "A8123CD"})
        self.assertTrue(rows[0]["last_seen_iso"])
        r = self.client.get(f"{API}/export?format=json")
        data = json.loads(r.data)
        self.assertEqual(data["count"], 3)
        self.assertIn("plate_url", data["items"][0])
        self.assertEqual(self.client.get(f"{API}/export?format=xml").status_code, 400)

    def test_csv_formula_guard(self):
        with mock.patch("securitycam.recognition.platelog_api._labels", return_value={CAM: "=HYPERLINK(1)"}):
            r = self.client.get(f"{API}/export?format=csv&camera=front")
        self.assertIn("'=HYPERLINK", r.data.decode("utf-8-sig"))

    def test_sightings_grouped(self):
        r = self.client.get(f"{API}/plates").get_json()
        groups = {g["plate_key"]: g for g in r["plates"]}
        g = groups[platelog.fold("AB123CD")]
        self.assertEqual(g["count"], 2)                         # AB123CD + A8123CD (OCR confusion)
        self.assertEqual(sorted(g["cameras"]), ["front", "garage"])
        self.assertEqual(g["variants"], ["A8123CD", "AB123CD"])
        r = self.client.get(f"{API}/plates/AB-123-CD").get_json()
        self.assertEqual(r["count"], 2)
        self.assertEqual(r["by_camera"], {"front": 1, "garage": 1})
        self.assertEqual(self.client.get(f"{API}/plates/x").status_code, 400)

    def test_watchlist_raises_alarm(self):
        hits = []
        self.log.on_watch_hit = lambda e, w: hits.append((e["plate"], w["note"]))
        r = self.client.post(f"{API}/watchlist", json={"plate": "zz 999 zz", "note": "Delivery van"})
        self.assertEqual(r.status_code, 200)
        wid = r.get_json()["item"]["id"]
        self.assertEqual(self.client.post(f"{API}/watchlist", json={"plate": "-"}).status_code, 400)
        eid = observe(self.log, plate="ZZ999ZZ", tid=9)
        observe(self.log, plate="ZZ999ZZ", tid=9)               # same entry: no second alarm
        self.log.flush()
        self.assertEqual(hits, [("ZZ999ZZ", "Delivery van")])
        self.assertEqual(self.log.get(eid)["watch"]["note"], "Delivery van")
        r = self.client.get(f"{API}?watch=1").get_json()
        self.assertEqual([i["id"] for i in r["items"]], [eid])
        self.assertEqual(self.client.delete(f"{API}/watchlist/{wid}").status_code, 200)
        self.assertEqual(self.client.get(f"{API}/watchlist").get_json()["plates"], [])

    def test_watch_alarm_goes_through_alerts_engine(self):
        from securitycam.alerts import engine as alert_engine

        entry = {"id": self.b, "camera": CAM, "plate": "KL555MN", "vehicle": "car", "track_id": 2}
        with mock.patch.object(alert_engine.CameraEngine, "emit", return_value=True) as emit:
            platelog.raise_watch_alarm(entry, {"plate": "KL555MN", "note": "Neighbour"})
        args = emit.call_args.args
        self.assertEqual(args[1:4], ("plate", "plate-K1555MN", "KL555MN"))
        self.assertIn("Neighbour", args[6])
        self.assertEqual(emit.call_args.kwargs["extra"]["plate_log_id"], self.b)


class LiveTarget(unittest.TestCase):
    def setUp(self):
        self.log = new_log()
        self.client = app.test_client()
        self.eid = observe(self.log, plate="AB123CD", tid=5)
        self.log.flush()
        self.rt = common.runtime(state.get_camera_state(CAM))
        with self.rt.lock:
            self.rt.track_info[5] = TrackInfo(plate_best="AB123CD", last_seen=time.monotonic())

    def tearDown(self):
        with self.rt.lock:
            self.rt.track_info.clear()

    def payload(self, tid=5, label="car"):
        return {"ts": time.time(), "frame": {"w": 1280, "h": 720}, "locked": None,
                "targets": [{"id": tid, "label": label, "conf": 0.9, "box": [0.3, 0.36, 0.7, 0.83]}]}

    def test_live_and_lock(self):
        with mock.patch("securitycam.targets.targets_payload", return_value=self.payload()):
            r = self.client.get(f"{API}/{self.eid}/live").get_json()
            self.assertTrue(r["live"])
            self.assertEqual((r["camera"], r["target_id"]), (CAM, 5))
            with mock.patch("securitycam.targets.set_lock", return_value=True) as lock:
                r = self.client.post(f"{API}/{self.eid}/live")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(lock.call_args.args[1], 5)

    def test_not_live_when_track_gone_or_reused(self):
        with mock.patch("securitycam.targets.targets_payload", return_value=self.payload(tid=6)):
            self.assertFalse(self.client.get(f"{API}/{self.eid}/live").get_json()["live"])
        with mock.patch("securitycam.targets.targets_payload", return_value=self.payload(label="person")):
            self.assertFalse(self.client.get(f"{API}/{self.eid}/live").get_json()["live"])
        with self.rt.lock:
            self.rt.track_info[5].plate_best = "XY987ZZ"       # id now belongs to another car
        with mock.patch("securitycam.targets.targets_payload", return_value=self.payload()):
            self.assertFalse(self.client.get(f"{API}/{self.eid}/live").get_json()["live"])
            self.assertEqual(self.client.post(f"{API}/{self.eid}/live").status_code, 409)
        with self.rt.lock:
            self.rt.track_info[5].plate_best = "AB123CD"
            self.rt.track_info[5].last_seen = time.monotonic() - 60  # stale
        with mock.patch("securitycam.targets.targets_payload", return_value=self.payload()):
            self.assertFalse(self.client.get(f"{API}/{self.eid}/live").get_json()["live"])


if __name__ == "__main__":
    unittest.main()
