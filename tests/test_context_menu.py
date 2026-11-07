"""Tests for the right-click menu backends: tags (identity follow-through, persistence),
solo / hidden (payload + ESP renderer), manual plate annotation and automatic face ids.

Synthetic data only: fake YOLO results, a drawn car with a plate, random unit
vectors as face embeddings. No camera, no model download (the plate reader is
mocked), nothing leaves the machine.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import shutil
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from securitycam import config, state, tags, targets  # noqa: E402
from securitycam.esp import motion  # noqa: E402
from securitycam.esp import renderers as R  # noqa: E402
from securitycam.esp.pipeline import draw_custom_esp  # noqa: E402
from securitycam.recognition import common, gallery, manual_plates, overlay, platelog, worker  # noqa: E402
from securitycam.recognition import unknown_faces as U  # noqa: E402
from securitycam.recognition.common import FaceHit, TrackInfo  # noqa: E402
from securitycam.webapp import app  # noqa: E402

CAM = "front"
NAMES = {0: "person", 2: "car"}


class _Boxes:
    def __init__(self, rows):
        self.data = np.array([[x1, y1, x2, y2, t, 0.9, c] for t, c, x1, y1, x2, y2 in rows], np.float32).reshape(-1, 7)
        self.xyxy, self.id, self.conf, self.cls = self.data[:, :4], self.data[:, 4], self.data[:, 5], self.data[:, 6]
        self.is_track = True

    def __len__(self):
        return len(self.data)


def fake_results(rows):
    """rows: [(track id, cls, x1, y1, x2, y2)] -> object shaped like ultralytics results."""
    return [SimpleNamespace(boxes=_Boxes(rows), names=NAMES, keypoints=None)]


def setup_camera(rows, w=1280, h=720):
    cs = state.get_camera_state(CAM)
    cs.esp_settings.clear()
    cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
    cs.esp_settings["smoothing"] = 0.0
    cs.yolo_enabled = True
    frame = np.zeros((h, w, 3), np.uint8)
    res = fake_results(rows)
    with cs.frame_lock:
        cs.latest_frame = frame
        cs.latest_results = res
    cs.esp_motion = motion.MotionState()
    cs.view_solo, cs.view_hidden = None, []
    t = time.perf_counter()
    draw_custom_esp(cs, frame.copy(), res, frame_ts=t, results_ts=t)
    return cs, frame, res


def reset_tags():
    with tags.STORE._lock:
        tags.STORE._tags.clear()
        tags.STORE._touch.clear()
        tags.STORE._bound.clear()
        tags.STORE._publish_locked()


def unit(rng, n=128):
    v = rng.normal(size=n).astype(np.float32)
    return v / np.linalg.norm(v)


def near(rng, v, noise=0.12):
    w = v + rng.normal(size=v.size).astype(np.float32) * noise / np.sqrt(v.size)
    return w / np.linalg.norm(w)


ROWS = [(1, 0, 100, 100, 260, 600), (2, 0, 600, 120, 760, 620), (3, 2, 900, 300, 1250, 650)]


class TagsAndView(unittest.TestCase):
    def setUp(self):
        reset_tags()
        self.cs, self.frame, self.res = setup_camera(ROWS)
        self.rt = common.runtime(self.cs)
        with self.rt.lock:
            self.rt.track_info.clear()
        self.client = app.test_client()

    def tearDown(self):
        with self.rt.lock:
            self.rt.track_info.clear()
        self.cs.view_solo, self.cs.view_hidden = None, []
        reset_tags()

    def payload(self):
        return {t["id"]: t for t in targets.targets_payload(self.cs)["targets"]}

    def test_targets_visible(self):
        self.assertEqual(sorted(self.payload()), [1, 2, 3])
        self.assertEqual(targets.targets_payload(self.cs)["view"], {"solo": None, "hidden": []})

    def test_tag_add_remove_and_validation(self):
        url = f"/api/cameras/{CAM}/targets/1/tags"
        r = self.client.post(url, json={"label": "  Mail   carrier ", "color": "#FF5A5A"})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["owner"], "track")
        self.assertEqual(self.payload()[1]["tags"], [{"label": "Mail carrier", "color": "#ff5a5a"}])
        self.assertNotIn("tags", self.payload()[2])
        # same label again (case-insensitive) updates, does not duplicate; color defaults to the catalog's
        self.client.post(f"/api/cameras/{CAM}/targets/2/tags", json={"label": "mail CARRIER"})
        self.assertEqual(self.payload()[2]["tags"][0]["color"], "#ff5a5a")
        cat = self.client.get("/api/tags").get_json()["tags"]
        self.assertEqual(cat[0]["count"], 2)
        for bad in ({"label": "x" * 25}, {"label": ""}, {"label": 5}, {"label": "ok", "color": "red"}, {}):
            self.assertEqual(self.client.post(url, json=bad).status_code, 400, bad)
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/targets/99/tags", json={"label": "a"}).status_code, 404)
        self.assertEqual(self.client.post("/api/cameras/nope/targets/1/tags", json={"label": "a"}).status_code, 404)
        # CSRF guard
        r = self.client.post(url, json={"label": "evil"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.client.delete(url, json={"label": "MAIL carrier"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("tags", self.payload()[1])
        self.assertEqual(self.client.delete(url, json={"label": "missing"}).status_code, 404)

    def test_tag_follows_identity_and_persists(self):
        self.client.post(f"/api/cameras/{CAM}/targets/1/tags", json={"label": "VIP"})
        self.assertNotIn("tags", tags.STORE.dump()["tags"].get("p:alice", []))
        self.assertEqual(tags.STORE.dump()["tags"], {})          # track tags are not persisted
        with self.rt.lock:
            self.rt.track_info[1] = TrackInfo(name="Alice", person_id="alice", last_seen=time.monotonic())
        t = self.payload()[1]
        self.assertEqual(t["tags"][0]["label"], "VIP")
        self.assertEqual(t["identity"]["name"], "Alice")
        dumped = tags.STORE.dump()
        self.assertEqual(list(dumped["tags"]), ["p:alice"])     # moved to the person
        # next session: state.json round trip, a new track id recognized as the same person
        clean, warnings = tags.TagStore.validate(dumped)
        self.assertEqual(warnings, [])
        reset_tags()
        tags.STORE.load(clean)
        with self.rt.lock:
            self.rt.track_info.clear()
            self.rt.track_info[2] = TrackInfo(name="Alice", person_id="alice", last_seen=time.monotonic())
        p = self.payload()
        self.assertEqual(p[2]["tags"][0]["label"], "VIP")
        self.assertNotIn("tags", p[1])
        # face ids and plates are durable identities too
        with self.rt.lock:
            self.rt.track_info[1] = TrackInfo(face_uid=7, last_seen=time.monotonic())
            self.rt.track_info[3] = TrackInfo(plate_best="AB123CD", plate_reads=5, last_seen=time.monotonic())
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/targets/1/tags", json={"label": "Neighbor"})
                         .get_json()["owner"], "face")
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/targets/3/tags", json={"label": "Delivery"})
                         .get_json()["owner"], "plate")
        self.assertIn("f:7", tags.STORE.dump()["tags"])
        self.assertIn("v:A8123C0", tags.STORE.dump()["tags"])   # OCR-confusion folded plate key

    def test_tags_survive_recognition_changes_on_a_live_track(self):
        now = time.monotonic()
        with self.rt.lock:
            self.rt.track_info[3] = TrackInfo(plate_best="62518", plate_reads=5, last_seen=now)
            self.rt.track_info[1] = TrackInfo(name="Alice", person_id="alice", last_seen=now)
        self.client.post(f"/api/cameras/{CAM}/targets/3/tags", json={"label": "Delivery"})
        self.client.post(f"/api/cameras/{CAM}/targets/1/tags", json={"label": "VIP"})
        with self.rt.lock:
            self.rt.track_info[3].plate_best = "B25786"          # plate reading corrected (vote flipped)
            self.rt.track_info[1].name, self.rt.track_info[1].person_id = "Bob", "bob"   # face flip
        p = self.payload()
        self.assertEqual([t["label"] for t in p[3]["tags"]], ["Delivery"])
        self.assertEqual([t["label"] for t in p[1]["tags"]], ["VIP"])   # still shown on this track
        dumped = tags.STORE.dump()["tags"]
        self.assertIn("v:825786", dumped)                         # moved with the corrected plate
        self.assertNotIn("v:62518", dumped)
        self.assertIn("p:alice", dumped)                          # a misrecognition never re-labels people
        self.assertNotIn("p:bob", dumped)
        _, labels = self.render()
        self.assertTrue(any("#VIP" in runs for runs in labels), labels)
        r = self.client.delete(f"/api/cameras/{CAM}/targets/1/tags", json={"label": "vip"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("p:alice", tags.STORE.dump()["tags"])

    def test_validate_drops_garbage(self):
        clean, warnings = tags.TagStore.validate({"tags": {
            "p:bob": [{"label": "ok", "color": "#123456"}, {"label": "x" * 40}],
            "../etc": [{"label": "bad"}], "t:front:0:1": [{"label": "track"}]}})
        self.assertEqual(list(clean["tags"]), ["p:bob"])
        self.assertEqual(len(clean["tags"]["p:bob"]), 1)
        self.assertTrue(warnings)
        with self.assertRaises(ValueError):
            tags.TagStore.validate([])

    def test_track_tag_does_not_jump_to_reused_id(self):
        self.client.post(f"/api/cameras/{CAM}/targets/1/tags", json={"label": "Courier"})
        self.assertIn("tags", self.payload()[1])
        with self.rt.lock:
            self.rt.track_epoch += 1                              # tracker restarted its ids
        self.assertNotIn("tags", self.payload()[1])

    def test_solo_hide_payload(self):
        r = self.client.post(f"/api/cameras/{CAM}/view", json={"action": "solo", "target_id": 2})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["view"]["solo"]["id"], 2)
        p = self.payload()
        self.assertTrue(p[1].get("hidden") and p[3].get("hidden"))
        self.assertNotIn("hidden", p[2])
        self.client.post(f"/api/cameras/{CAM}/view", json={"action": "unsolo"})
        self.client.post(f"/api/cameras/{CAM}/view", json={"action": "hide", "target_id": 3})
        p = self.payload()
        self.assertTrue(p[3].get("hidden"))
        self.assertFalse(p[1].get("hidden") or p[2].get("hidden"))
        self.assertEqual([h["id"] for h in targets.targets_payload(self.cs)["view"]["hidden"]], [3])
        # solo follows the durable identity across a new track id
        with self.rt.lock:
            self.rt.track_info[1] = TrackInfo(name="Alice", person_id="alice", last_seen=time.monotonic())
        self.client.post(f"/api/cameras/{CAM}/view", json={"action": "solo", "target_id": 1})
        with self.rt.lock:
            self.rt.track_info[2] = self.rt.track_info.pop(1)
        p = self.payload()
        self.assertNotIn("hidden", p[2])                      # same person, new track id
        self.assertTrue(p[3].get("hidden"))
        r = self.client.post(f"/api/cameras/{CAM}/view", json={"action": "clear"})
        self.assertEqual(r.get_json()["view"], {"solo": None, "hidden": []})
        for bad in ({"action": "solo"}, {"action": "solo", "target_id": "2"}, {"action": "nuke"},
                    {"action": "hide", "target_id": True}):
            self.assertEqual(self.client.post(f"/api/cameras/{CAM}/view", json=bad).status_code, 400, bad)
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/view", json={"action": "solo", "target_id": 42})
                         .status_code, 404)

    def render(self):
        drawn, labels = [], []
        real_box, real_tag = R.draw_box, R.make_tag

        def box(frame, det, *a, **k):
            drawn.append(det.track_key)
            return real_box(frame, det, *a, **k)

        def tag(runs, *a, **k):
            labels.append([r[0] for r in runs])
            return real_tag(runs, *a, **k)

        with mock.patch.object(R, "draw_box", box), mock.patch.object(R, "make_tag", tag):
            t = time.perf_counter()
            draw_custom_esp(self.cs, self.frame.copy(), self.res, frame_ts=t, results_ts=t - 0.01)
        return drawn, labels

    def test_renderer_solo_hide_and_tag_runs(self):
        drawn, _ = self.render()
        self.assertEqual(sorted(drawn), [1, 2, 3])
        tags.set_solo(self.cs, {"id": 2, "label": "person"})
        drawn, _ = self.render()
        self.assertEqual(drawn, [2])
        tags.set_solo(self.cs, None)
        tags.hide(self.cs, {"id": 1, "label": "person"})
        drawn, _ = self.render()
        self.assertEqual(sorted(drawn), [2, 3])
        tags.clear_view(self.cs)
        tags.STORE.add(tags.target_keys(self.cs, 3), "Delivery", "#5ad1ff")
        _, labels = self.render()
        self.assertTrue(any("#DELIVERY" in runs for runs in labels), labels)
        # alarms / zones keep counting everything: zone count sees all three
        zone = {"id": "z1", "label": "All", "box": [0.0, 0.0, 1.0, 1.0], "color": "#ffd21f"}
        self.cs.esp_zones[:] = [zone]
        tags.set_solo(self.cs, {"id": 2, "label": "person"})
        counts = []
        real_zone = R.draw_zone
        with mock.patch.object(R, "draw_zone", lambda f, b, c, label, count, *a: counts.append(count) or real_zone(f, b, c, label, count, *a)):
            t = time.perf_counter()
            draw_custom_esp(self.cs, self.frame.copy(), self.res, frame_ts=t, results_ts=t - 0.01)
        self.cs.esp_zones[:] = []
        self.assertEqual(counts[-1], 3)


# ----------------------------------------------------------------------------- manual plates
def car_frame(text="AB 123 CD", w=1280, h=720):
    img = np.full((h, w, 3), (92, 96, 100), np.uint8)
    x1, y1, x2, y2 = 380, 260, 900, 600
    cv2.rectangle(img, (x1, y1 + 90), (x2, y2 - 40), (140, 60, 30), -1)
    px1, py1, px2, py2 = 540, 470, 740, 515
    cv2.rectangle(img, (px1, py1), (px2, py2), (245, 245, 245), -1)
    cv2.putText(img, text, (px1 + 12, py2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    return img, (x1, y1, x2, y2), (px1, py1, px2, py2)


class FakeOcr:
    """Plate reader stand-in: 'reads' the plate only when the crop is mostly white (the drawn plate)."""

    def __init__(self, text="AB123CD"):
        self.text = text
        self.calls = 0

    def read(self, img):
        self.calls += 1
        white = float((img > 200).all(axis=2).mean()) if img.ndim == 3 else 0.0
        return (self.text, 0.5 + 0.45 * white) if white > 0.2 else ("", 0.0)

    def detect(self, img):
        return []

    def read_vehicle(self, frame, box):
        return common.PlateRead(text=self.text, conf=0.91, box=(540, 470, 740, 515))


class ManualPlates(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="platelog-"))
        self.log = platelog.PlateLog(self.root)
        platelog.set_log(self.log)
        self.cs = state.get_camera_state(CAM)
        self.frame, self.vbox, self.pbox = car_frame()
        with self.cs.frame_lock:
            self.cs.latest_frame = self.frame
        self.client = app.test_client()
        self.ocr = FakeOcr()
        self.p_reader = mock.patch.object(manual_plates, "get_reader", return_value=self.ocr)
        self.p_reader.start()
        self.rt = common.runtime(self.cs)
        with self.rt.lock:
            self.rt.track_info[9] = TrackInfo(last_seen=time.monotonic())
        w, h = 1280, 720
        self.car = {"id": 9, "label": "car", "conf": 0.9,
                    "box": [self.vbox[0] / w, self.vbox[1] / h, self.vbox[2] / w, self.vbox[3] / h]}
        self.p_targets = mock.patch("securitycam.targets.targets_payload", return_value={
            "ts": time.time(), "frame": {"w": w, "h": h}, "locked": None, "targets": [self.car]})
        self.p_targets.start()

    def tearDown(self):
        self.p_reader.stop()
        self.p_targets.stop()
        with self.rt.lock:
            self.rt.track_info.clear()
        platelog.set_log(None)
        shutil.rmtree(self.root, ignore_errors=True)

    def norm(self, box):
        return [box[0] / 1280, box[1] / 720, box[2] / 1280, box[3] / 720]

    def test_end_to_end(self):
        r = self.client.post(f"/api/cameras/{CAM}/freeze", json={})
        self.assertEqual(r.status_code, 200, r.get_json())
        fz = r.get_json()
        self.assertEqual((fz["w"], fz["h"]), (1280, 720))
        self.assertEqual([v["id"] for v in fz["vehicles"]], [9])
        img = self.client.get(fz["url"])
        self.assertEqual((img.status_code, img.mimetype), (200, "image/jpeg"))
        # the camera moves on; the frozen frame stays what the user is drawing on
        with self.cs.frame_lock:
            self.cs.latest_frame = np.zeros_like(self.frame)
        box = self.norm(self.pbox)
        r = self.client.post(f"/api/cameras/{CAM}/manual_plate/read", json={"freeze_id": fz["id"], "box": box})
        self.assertEqual(r.status_code, 200, r.get_json())
        read = r.get_json()
        self.assertEqual(read["text"], "AB123CD")
        self.assertGreater(read["conf"], 0.8)
        self.assertGreaterEqual(read["tried"], 15)                       # padded + rotated variants
        self.assertTrue(read["crop"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(read["vehicle"]["id"], 9)
        # user corrects one character and confirms
        r = self.client.post(f"/api/cameras/{CAM}/manual_plate",
                             json={"freeze_id": fz["id"], "box": box, "plate": "ab-123-cx", "ocr_conf": read["conf"]})
        self.assertEqual(r.status_code, 200, r.get_json())
        e = r.get_json()["entry"]
        self.assertEqual((e["plate"], e["source"], e["track_id"], e["vehicle"]), ("AB123CX", "manual", 9, "car"))
        self.assertTrue(all(e["images"].values()))
        listed = self.client.get("/api/recognition/platelog").get_json()["items"]
        self.assertEqual([(i["id"], i["source"]) for i in listed], [(e["id"], "manual")])
        self.assertEqual(self.client.get(e["plate_url"]).status_code, 200)
        with self.rt.lock:
            self.assertEqual(self.rt.track_info[9].plate_best, "AB123CX")   # fed into the track vote
        # re-OCR the stored crop, then apply the correction
        r = self.client.post(f"/api/recognition/platelog/{e['id']}/reocr", json={})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["text"], "AB123CD")
        r = self.client.patch(f"/api/recognition/platelog/{e['id']}", json={"plate": "AB123CD"})
        self.assertEqual(r.get_json()["entry"]["plate"], "AB123CD")
        self.assertTrue(r.get_json()["entry"]["edited"])

    def test_box_outside_vehicle_and_typed_plate(self):
        fz = self.client.post(f"/api/cameras/{CAM}/freeze", json={}).get_json()
        box = [0.02, 0.02, 0.12, 0.06]                                     # no plate there, no vehicle
        read = self.client.post(f"/api/cameras/{CAM}/manual_plate/read",
                                json={"freeze_id": fz["id"], "box": box}).get_json()
        self.assertEqual(read["text"], "")
        self.assertIsNone(read["vehicle"])
        e = self.client.post(f"/api/cameras/{CAM}/manual_plate",
                             json={"freeze_id": fz["id"], "box": box, "plate": "M AB 1234"}).get_json()["entry"]
        self.assertEqual((e["plate"], e["track_id"], e["source"]), ("MAB1234", None, "manual"))

    def test_validation(self):
        fz = self.client.post(f"/api/cameras/{CAM}/freeze", json={}).get_json()
        url = f"/api/cameras/{CAM}/manual_plate"
        good = self.norm(self.pbox)
        cases = [
            ({"freeze_id": "deadbeefdeadbeef", "box": good, "plate": "AB12"}, 404),
            ({"freeze_id": "../../etc/passwd", "box": good, "plate": "AB12"}, 404),
            ({"freeze_id": fz["id"], "box": [0, 0, 1], "plate": "AB12"}, 400),
            ({"freeze_id": fz["id"], "box": ["a", 0, 1, 1], "plate": "AB12"}, 400),
            ({"freeze_id": fz["id"], "box": [0.5, 0.5, 0.5001, 0.5001], "plate": "AB12"}, 400),
            ({"freeze_id": fz["id"], "box": good, "plate": "!"}, 400),
            ({"freeze_id": fz["id"], "box": good, "plate": "X" * 30}, 400),
        ]
        for body, code in cases:
            self.assertEqual(self.client.post(url, json=body).status_code, code, body)
        self.assertEqual(self.client.get(f"/api/cameras/{CAM}/freeze/{'0' * 16}.jpg").status_code, 404)
        self.assertEqual(self.client.get(f"/api/cameras/garage/freeze/{fz['id']}.jpg").status_code, 404)  # other camera
        self.assertEqual(self.client.post(url, json={"freeze_id": fz["id"], "box": good, "plate": "AB12"},
                                          headers={"Origin": "http://evil.example"}).status_code, 403)
        self.assertEqual(self.client.patch("/api/recognition/platelog/20250101-000000-abcdef",
                                           json={"plate": "AB12"}).status_code, 404)

    def test_read_plate_on_target(self):
        r = self.client.post(f"/api/cameras/{CAM}/targets/9/read_plate", json={})
        self.assertEqual(r.status_code, 200, r.get_json())
        out = r.get_json()
        self.assertTrue(out["found"])
        self.assertEqual(out["text"], "AB123CD")
        self.assertTrue(out["freeze"]["id"])
        with self.rt.lock:
            self.assertEqual(self.rt.track_info[9].plate_best, "AB123CD")
        self.assertEqual(self.client.post(f"/api/cameras/{CAM}/targets/4/read_plate", json={}).status_code, 404)

    def test_variant_ranking(self):
        class Rot:
            def __init__(self):
                self.n = 0

            def read(self, img):
                self.n += 1
                return ("AB123CD", 0.9) if self.n == 7 else ("A8123", 0.6)

            def detect(self, img):
                return []

        out = manual_plates.read_variants(Rot(), self.frame, self.pbox)
        self.assertEqual(out["text"], "AB123CD")
        self.assertEqual([c["text"] for c in out["candidates"]], ["AB123CD", "A8123"])


# ----------------------------------------------------------------------------- automatic face ids
class FaceIds(unittest.TestCase):
    TH = 0.45

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="faces-"))
        self.p_faces = mock.patch.object(gallery, "FACES_DIR", self.tmp)
        self.p_faces.start()
        gallery._invalidate()
        self.store = U.UnknownFaces(self.tmp / "unknown")
        U.set_store(self.store)
        self.rng = np.random.default_rng(7)
        self.client = app.test_client()
        reset_tags()

    def tearDown(self):
        U.set_store(None)
        self.p_faces.stop()
        gallery._invalidate()
        shutil.rmtree(self.tmp, ignore_errors=True)
        reset_tags()

    def feed(self, v, n=3, t0=1000.0, **kw):
        out = None
        for i in range(n):
            out = self.store.observe(near(self.rng, v), self.TH, thumb=np.full((112, 112, 3), 128, np.uint8),
                                     quality=0.5, now=t0 + i, **kw)
        return out

    def test_same_face_same_id_different_face_new_id(self):
        a, b = unit(self.rng), unit(self.rng)
        self.assertIsNone(self.store.observe(a, self.TH, now=1.0))         # pending first
        self.assertIsNone(self.store.observe(near(self.rng, a), self.TH, now=1.5))
        uid_a = self.store.observe(near(self.rng, a), self.TH, now=2.0)
        self.assertEqual(uid_a, 1)
        for i in range(5):
            self.assertEqual(self.store.observe(near(self.rng, a), self.TH, now=3.0 + i), 1)
        uid_b = self.feed(b, t0=10.0)
        self.assertEqual(uid_b, 2)
        self.assertEqual(self.store.observe(near(self.rng, b), self.TH, now=20.0), 2)
        self.assertEqual(self.store.observe(near(self.rng, a), self.TH, now=21.0), 1)
        self.assertEqual(len(self.store), 2)
        # low-quality faces match but never create
        c = unit(self.rng)
        for i in range(5):
            self.assertIsNone(self.store.observe(c, self.TH, learn=False, now=30.0 + i))
        self.assertEqual(len(self.store), 2)

    def test_sightings_and_persistence(self):
        a = unit(self.rng)
        uid = self.feed(a, t0=0.0)
        self.feed(a, n=1, t0=500.0)                                           # > SIGHTING_GAP later
        self.store.save(force=True)
        again = U.UnknownFaces(self.tmp / "unknown")
        self.assertEqual(again.observe(near(self.rng, a), self.TH, now=520.0), uid)
        item = again.list()[0]
        self.assertEqual((item["uid"], item["label"], item["sightings"]), (uid, f"Face #{uid}", 2))
        self.assertGreaterEqual(item["samples"], 1)
        self.assertEqual(again._next_uid, uid + 1)

    def test_auto_merge_when_centroids_converge(self):
        a = unit(self.rng)
        r = unit(self.rng)
        r = r - (r @ a) * a
        r /= np.linalg.norm(r)
        b = 0.4 * a + np.sqrt(1 - 0.4 ** 2) * r                             # cos(a, b) = 0.4 < threshold
        ua, ub = self.feed(a, t0=0.0), self.feed(b, t0=10.0)
        self.assertNotEqual(ua, ub)
        mid = (a + b) / np.linalg.norm(a + b)
        for i in range(60):
            self.store.observe(mid, self.TH, now=20.0 + i)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.list()[0]["uid"], min(ua, ub))
        self.assertIn((max(ua, ub), min(ua, ub)), self.store.merges)

    def test_prune_and_cap(self):
        a, b = unit(self.rng), unit(self.rng)
        ua = self.feed(a, t0=0.0)
        self.feed(b, t0=0.0)
        self.feed(b, n=1, t0=1000.0)                                          # b: two sightings
        removed = self.store.prune(now=10 * 86400.0)
        self.assertEqual(removed, [ua])
        self.assertEqual(len(self.store), 1)

    def test_api_merge_name_delete(self):
        a, b, c = unit(self.rng), unit(self.rng), unit(self.rng)
        ua, ub, uc = self.feed(a, t0=0.0), self.feed(b, t0=10.0), self.feed(c, t0=20.0)
        faces = self.client.get("/api/recognition/unknown").get_json()["faces"]
        self.assertEqual(sorted(f["uid"] for f in faces), [ua, ub, uc])
        self.assertEqual(self.client.get(f"/api/recognition/unknown/{ua}/thumb").status_code, 200)
        self.assertEqual(self.client.get("/api/recognition/unknown/999/thumb").status_code, 404)
        # tags on an unknown identity follow merge and naming
        tags.STORE.add([f"f:{ub}"], "Gardener")
        r = self.client.post(f"/api/recognition/unknown/{ub}/merge", json={"into": ua})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(tags.STORE.get([f"f:{ua}"])[0]["label"], "Gardener")
        self.assertEqual(self.store.observe(near(self.rng, b), self.TH, now=50.0), ua)  # b now resolves to a
        for bad in ({"into": ua}, {"into": "x"}, {"into": True}, {}):
            self.assertIn(self.client.post(f"/api/recognition/unknown/{ua}/merge", json=bad).status_code, (400, 404))
        rt = common.runtime(state.get_camera_state(CAM))
        with rt.lock:
            rt.track_info[4] = TrackInfo(face_uid=ua, last_seen=time.monotonic())
        r = self.client.post(f"/api/recognition/unknown/{ua}/name", json={"name": "Gerd"})
        self.assertEqual(r.status_code, 200, r.get_json())
        person = r.get_json()["person"]
        self.assertEqual(person["name"], "Gerd")
        self.assertGreaterEqual(person["samples"], 2)                     # samples of a and b moved over
        name, pid, _ = gallery.match(near(self.rng, a), 0.36)
        self.assertEqual((name, pid), ("Gerd", person["id"]))
        self.assertEqual(tags.STORE.get([f"p:{person['id']}"])[0]["label"], "Gardener")
        with rt.lock:
            self.assertEqual(rt.track_info[4].name, "Gerd")
            rt.track_info.clear()
        self.assertNotIn(ua, [f["uid"] for f in self.store.list()])
        self.assertEqual(self.client.post(f"/api/recognition/unknown/{uc}/name", json={"name": ""}).status_code, 400)
        self.assertEqual(self.client.post(f"/api/recognition/unknown/{ua}/name", json={"name": "X"}).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/recognition/unknown/{uc}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/recognition/unknown/{uc}").status_code, 404)
        self.assertEqual(self.client.post(f"/api/recognition/unknown/{ub}/name", json={"name": "Y"},
                                          headers={"Origin": "http://evil.example"}).status_code, 403)
        # the reserved folder can't be deleted through the people API
        self.assertEqual(self.client.delete("/api/recognition/people/unknown").status_code, 404)

    def test_worker_assigns_ids_and_overlay_shows_them(self):
        cs = state.get_camera_state(CAM)
        rt = common.runtime(cs)
        s = common.RecogSettings.from_mapping(dict(config.DEFAULT_ESP_SETTINGS, known_faces=True))
        self.assertTrue(s.face_auto_ids)
        a = unit(self.rng)
        frame = np.zeros((720, 1280, 3), np.uint8)
        cv2.randu(frame, 0, 255)                                           # sharp texture for the learn check
        row = np.zeros(15, np.float32)
        row[:4] = (500, 200, 120, 120)
        row[14] = 0.95
        dets = [(5, "person", (420, 150, 720, 700), 0.9)]

        def fake_recognize(frame_, threshold):
            return [(row, FaceHit(box=(500, 200, 120, 120), name=None, person_id=None, score=0.1,
                                  recognizable=True, embedding=near(self.rng, a)))]

        with mock.patch.object(worker, "recognize_frame", fake_recognize):
            for i in range(4):
                worker.run_faces(rt, s, frame, dets, time.perf_counter())
        with rt.lock:
            hit = rt.faces[0]
            self.assertEqual(hit.face_uid, 1)
            self.assertEqual(rt.track_info[5].face_uid, 1)
        ctx = overlay.FrameContext(s=s, k=1.0, faces=[hit])
        texts = []
        real = R.make_tag
        with mock.patch.object(R, "make_tag", lambda runs, *a, **k: texts.append(runs[0][0]) or real(runs, *a, **k)):
            overlay.draw_faces(ctx, frame.copy(), [(500, 200, 120, 120)], SimpleNamespace(
                color_face=(255, 0, 0), color_text=(255, 255, 255), label_background=True),
                R.style_for(1280, 720, 2), [])
        self.assertEqual(texts, ["FACE #1"])
        with rt.lock:
            rt.track_info.clear()
            rt.faces = []


if __name__ == "__main__":
    unittest.main()
