"""Custom face-cover images ("image" face blur mode).

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-test-"))
os.environ.setdefault("SECURITYCAM_CAMERAS", "front|Front|192.0.2.10|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from securitycam import face_masks  # noqa: E402
from securitycam.esp import renderers as R  # noqa: E402
from securitycam.validation import sanitize_esp_patch  # noqa: E402


def _png(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class FaceMaskTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._orig = face_masks.MASK_DIR
        face_masks.MASK_DIR = Path(self._dir.name) / "face_masks"
        face_masks.reset_cache()

    def tearDown(self):
        face_masks.MASK_DIR = self._orig
        face_masks.reset_cache()
        self._dir.cleanup()

    def test_add_list_rename_delete(self):
        img = np.zeros((900, 600, 4), np.uint8)
        img[:, :, 2] = 255
        img[:, :, 3] = 255
        m = face_masks.add_mask("  Red   square ", _png(img))
        self.assertEqual(m["name"], "Red square")
        self.assertEqual(max(m["w"], m["h"]), face_masks.MAX_SIDE)
        self.assertFalse(m["alpha"])
        self.assertEqual([x["id"] for x in face_masks.list_masks()], [m["id"]])
        self.assertEqual(face_masks.rename_mask(m["id"], "Red")["name"], "Red")
        self.assertTrue(face_masks.delete_mask(m["id"]))
        self.assertEqual(face_masks.list_masks(), [])

    def test_rejects_garbage_and_huge(self):
        with self.assertRaises(face_masks.MaskError):
            face_masks.add_mask("x", "data:image/png;base64,bm90IGFuIGltYWdl")
        with self.assertRaises(face_masks.MaskError):
            face_masks.add_mask("x", "%%%")
        with self.assertRaises(face_masks.MaskError) as e:
            face_masks.add_mask("x", "A" * (face_masks.MAX_BYTES * 2))
        self.assertEqual(e.exception.status, 413)

    def test_image_mode_covers_face_and_underlay_hides_transparent_parts(self):
        img = np.zeros((100, 100, 4), np.uint8)
        img[30:70, 30:70] = (0, 255, 255, 255)        # opaque yellow center, transparent rest
        m = face_masks.add_mask("dot", _png(img))
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        orig = frame.copy()
        R.anonymize_faces(frame, [(100, 60, 80, 80)], "image", 70, image=m["id"], fit="contain", underlay=True)
        self.assertEqual(tuple(int(v) for v in frame[100, 140]), (0, 255, 255))
        corner = frame[52:60, 82:90].astype(int)
        self.assertGreater(np.abs(corner - orig[52:60, 82:90].astype(int)).mean(), 20)
        self.assertTrue((frame[0:40, 0:60] == orig[0:40, 0:60]).all())

    def test_missing_image_falls_back_to_black(self):
        frame = np.full((120, 120, 3), 200, np.uint8)
        R.anonymize_faces(frame, [(40, 40, 40, 40)], "image", 70, image="aaaaaaaaaaaa")
        self.assertTrue((frame[60, 60] == 0).all())

    def test_settings_validation(self):
        out = sanitize_esp_patch({"face_blur_mode": "image", "face_blur_image": "0123456789ab",
                                  "face_blur_image_fit": "contain", "face_blur_image_underlay": False})
        self.assertEqual(out, {"face_blur_mode": "image", "face_blur_image": "0123456789ab",
                               "face_blur_image_fit": "contain", "face_blur_image_underlay": False})
        self.assertEqual(sanitize_esp_patch({"face_blur_image": "../../etc/passwd"}), {})
        self.assertEqual(sanitize_esp_patch({"face_blur_image": ""}), {"face_blur_image": ""})

    def test_api_roundtrip(self):
        from securitycam.webapp import app

        c = app.test_client()
        r = c.post("/api/face-masks", json={"name": "Gray", "data": _png(np.full((50, 50, 3), 128, np.uint8))})
        self.assertEqual(r.status_code, 200, r.get_json())
        mid = r.get_json()["mask"]["id"]
        img = c.get(f"/api/face-masks/{mid}.png")
        self.assertEqual(img.status_code, 200)
        img.close()
        self.assertEqual(c.get("/api/face-masks").get_json()["masks"][0]["id"], mid)
        self.assertEqual(c.patch(f"/api/face-masks/{mid}", json={"name": "G"}).get_json()["mask"]["name"], "G")
        self.assertEqual(c.post("/api/face-masks", json={"data": "xx"}).status_code, 400)
        self.assertEqual(c.delete(f"/api/face-masks/{mid}").status_code, 200)
        self.assertEqual(c.delete(f"/api/face-masks/{mid}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
