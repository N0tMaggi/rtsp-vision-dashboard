"""Silhouette chams: segment models in the catalog, mask crops, motion warp, render modes.

Synthetic masks only (no model download, no GPU).

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SECURITYCAM_DATA_DIR", tempfile.mkdtemp(prefix="securitycam-test-"))
os.environ.setdefault("SECURITYCAM_CAMERAS", "front|Front|192.0.2.10|user|pw|stream1")
os.environ.setdefault("SECURITYCAM_AUTO_INSTALL_DEPS", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from securitycam import config, state  # noqa: E402
from securitycam.esp import masks as M  # noqa: E402
from securitycam.esp import motion  # noqa: E402
from securitycam.esp import renderers as R  # noqa: E402
from securitycam.esp import silhouette as S  # noqa: E402
from securitycam.esp.colors import DetectionColors  # noqa: E402
from securitycam.esp.detections import Detection, extract_detections  # noqa: E402
from securitycam.esp.pipeline import draw_custom_esp  # noqa: E402
from securitycam.esp.settings import EspSettings  # noqa: E402
from securitycam.recognition import engines, reid  # noqa: E402
from securitycam.validation import sanitize_esp_patch  # noqa: E402

YELLOW = (31, 210, 255)   # BGR of #ffd21f


def _settings(**kw) -> EspSettings:
    raw = dict(config.DEFAULT_ESP_SETTINGS)
    raw.update({"chams": True, "chams_opacity": 0.6, "smoothing": 0.0})
    raw.update(kw)
    return EspSettings.from_mapping(raw)


def _ellipse_crop(w=80, h=160) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 6, h // 2 - 6), 0, 0, 360, 255, -1)
    return m


def _det(x1, y1, x2, y2, mask=None, alpha=1.0, key=1) -> Detection:
    return Detection(index=0, obj_id=key, cls_id=0, cls_name="person", x1=x1, y1=y1, x2=x2, y2=y2,
                     cx=(x1 + x2) // 2, cy=(y1 + y2) // 2, conf=0.9, alpha=alpha, track_key=key, mask=mask)


def _pal(s: EspSettings) -> DetectionColors:
    return DetectionColors(box=s.color_boxes, text=s.color_text, fill=s.color_secondary, skeleton=s.color_skeleton)


class _Boxes:
    def __init__(self, rows):
        self.data = np.array([[x1, y1, x2, y2, t, 0.9, 0] for t, x1, y1, x2, y2 in rows], np.float32).reshape(-1, 7)
        self.xyxy, self.id, self.conf, self.cls = self.data[:, :4], self.data[:, 4], self.data[:, 5], self.data[:, 6]

    def __len__(self):
        return len(self.data)


def _seg_results(rows, frame_hw=(720, 1280), mask_hw=(384, 640), with_masks=True):
    """Fake ultralytics seg result; each box gets an ellipse mask in letterboxed mask space."""
    oh, ow = frame_hw
    mh, mw = mask_hw
    gain = min(mh / oh, mw / ow)
    px, py = round((mw - ow * gain) / 2 - 0.1), round((mh - oh * gain) / 2 - 0.1)
    data = np.zeros((len(rows), mh, mw), np.uint8)
    for i, (_t, x1, y1, x2, y2) in enumerate(rows):
        cx, cy = ((x1 + x2) / 2) * gain + px, ((y1 + y2) / 2) * gain + py
        ax, ay = (x2 - x1) * gain / 2 - 3, (y2 - y1) * gain / 2 - 3
        cv2.ellipse(data[i], (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1, -1)
    masks = SimpleNamespace(data=data, orig_shape=frame_hw) if with_masks else None
    return [SimpleNamespace(boxes=_Boxes(rows), names={0: "person"}, keypoints=None, masks=masks,
                            orig_shape=frame_hw)]


class CatalogTests(unittest.TestCase):
    def test_seg_models_in_catalog(self):
        for size in "nsmlx":
            self.assertIn(f"yolo26{size}-seg.pt", config.AVAILABLE_MODELS)
        info = engines.model_info("yolo26m-seg.pt")
        self.assertEqual((info["family"], info["task"], info["size"]), ("YOLO26", "segment", "M"))
        self.assertEqual(info["label"], "YOLO26 M Seg")
        self.assertTrue(info["masks"] and info["tracking"])
        self.assertFalse(info["keypoints"])
        self.assertEqual(engines.model_info("yolo11s-seg.pt")["task"], "segment")
        # detect / pose stay unchanged
        self.assertEqual(engines.model_info("yolo26s-pose.pt")["task"], "pose")
        self.assertEqual(engines.model_info("yolo26s.pt")["label"], "YOLO26 S")
        seg = [m for m in engines.catalog() if m["task"] == "segment"]
        self.assertGreaterEqual(len(seg), 5)

    def test_reid_reuses_seg_features(self):
        model = SimpleNamespace(ckpt_path="models/yolo26n-seg.pt", task="segment")
        self.assertTrue(reid.native_features(model))

    def test_is_seg_model(self):
        self.assertTrue(S.is_seg_model("yolo26n-seg.pt"))
        self.assertTrue(S.is_seg_model("yolo26n-seg.engine"))
        self.assertFalse(S.is_seg_model("yolo26n-pose.pt"))
        self.assertFalse(S.is_seg_model(None))


class SettingsTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(sanitize_esp_patch({"chams_mode": " XRay "}), {"chams_mode": "xray"})
        self.assertEqual(sanitize_esp_patch({"chams_mode": "wireframe"}), {})
        self.assertEqual(sanitize_esp_patch({"outline_glow": 99}), {"outline_glow": 24})
        self.assertEqual(sanitize_esp_patch({"outline_glow": "3"}), {"outline_glow": 3})
        self.assertEqual(sanitize_esp_patch({"outline_glow": True}), {})

    def test_defaults_and_snapshot(self):
        self.assertEqual(config.DEFAULT_ESP_SETTINGS["chams_mode"], "box")
        s = EspSettings.from_mapping({"chams_mode": "bogus", "outline_glow": -4})
        self.assertEqual((s.chams_mode, s.outline_glow), ("box", 0))
        s = EspSettings.from_mapping({"chams": True, "chams_mode": "outline"})
        self.assertTrue(S.uses_masks(s))
        self.assertFalse(S.uses_masks(EspSettings.from_mapping({"chams": False, "chams_mode": "outline"})))


class MaskCropTests(unittest.TestCase):
    def test_letterbox_crop(self):
        res = _seg_results([(1, 400, 200, 600, 600)])
        src = M.prepare(res[0])
        self.assertIsNotNone(src)
        self.assertAlmostEqual(src.gain, 0.5)
        self.assertEqual((src.pad_x, src.pad_y), (0, 12))
        crop = M.crop(src, 0, (400, 200, 600, 600))
        # box in mask px is 100 x 200; the long side is capped at MASK_MAX_SIDE
        self.assertEqual(crop.shape, (M.MASK_MAX_SIDE, 100))
        self.assertEqual(crop[80, 50], 255)            # body center
        self.assertEqual(crop[2, 2], 0)                # box corner, outside the ellipse
        # symmetric: the ellipse is centered in the crop (no letterbox offset error)
        rows = np.where(crop[:, 50] > 127)[0]
        self.assertLessEqual(abs((rows[0] + rows[-1]) / 2 - 79.5), 1.5)
        cols = np.where(crop[80] > 127)[0]
        self.assertLessEqual(abs((cols[0] + cols[-1]) / 2 - 49.5), 1.5)

    def test_no_masks(self):
        self.assertIsNone(M.prepare(_seg_results([(1, 0, 0, 10, 10)], with_masks=False)[0]))
        dets = extract_detections(_seg_results([(1, 400, 200, 600, 600)], with_masks=False), _settings())
        self.assertIsNone(dets[0].mask)

    def test_extract_attaches_crops(self):
        dets = extract_detections(_seg_results([(1, 100, 100, 300, 500), (2, 700, 200, 900, 600)]), _settings())
        self.assertEqual(len(dets), 2)
        for d in dets:
            self.assertIsNotNone(d.mask)
            self.assertLessEqual(max(d.mask.shape), M.MASK_MAX_SIDE)


class MotionWarpTests(unittest.TestCase):
    def test_mask_follows_predicted_box(self):
        st = motion.MotionState()
        hist = {}
        crop = _ellipse_crop()
        motion.ingest(st, [_det(100, 100, 300, 500, crop)], 0.0, 0.0, hist, 30)
        motion.ingest(st, [_det(140, 100, 340, 500, crop)], 0.1, 0.1, hist, 30)   # 400 px/s to the right
        out = motion.step(st, 0.2, 0.5, hist, False)
        self.assertEqual(len(out), 1)
        d = out[0]
        self.assertIs(d.mask, crop)                     # same box-relative crop ...
        self.assertGreater(d.x1, 140)                   # ... on the motion-predicted box
        frame = np.zeros((720, 1280, 3), np.uint8)
        s = _settings(chams_mode="silhouette", chams_opacity=1.0)
        n = S.draw(frame, [d], [_pal(s)], s, R.style_for(1280, 720, 2))
        self.assertEqual(n, 1)
        ys, xs = np.nonzero(frame.any(axis=2))
        self.assertLessEqual(abs(xs.mean() - (d.x1 + d.x2) / 2), 2.0)
        self.assertLessEqual(abs(ys.mean() - (d.y1 + d.y2) / 2), 2.0)
        self.assertGreaterEqual(xs.min(), d.x1)
        self.assertLessEqual(xs.max(), d.x2)


class RenderModeTests(unittest.TestCase):
    W, H = 640, 480
    BOX = (200, 80, 360, 400)

    def _draw(self, mode, frame=None, mask=True, **kw):
        frame = np.zeros((self.H, self.W, 3), np.uint8) if frame is None else frame
        s = _settings(chams_mode=mode, **kw)
        x1, y1, x2, y2 = self.BOX
        det = _det(x1, y1, x2, y2, _ellipse_crop() if mask else None)
        n = S.draw(frame, [det], [_pal(s)], s, R.style_for(self.W, self.H, 2))
        return frame, n

    def test_silhouette_fills_body_only(self):
        f, n = self._draw("silhouette", chams_opacity=1.0)
        self.assertEqual(n, 1)
        self.assertEqual(tuple(f[240, 280]), YELLOW)          # body center
        self.assertEqual(f[84, 204].sum(), 0)                 # box corner (outside the body)
        # anti-aliased edge: intermediate values along the contour, not just on/off
        row = f[240, 200:280, 1]
        self.assertTrue(np.any((row > 10) & (row < 200)))

    def test_outline_and_glow(self):
        f, _ = self._draw("outline", outline_glow=8)
        self.assertEqual(f[240, 280].sum(), 0)                # interior untouched
        row = f[240, :, 1].astype(int)
        edge_x = int(np.argmax(row[:280]))                    # strongest response left of center
        self.assertLess(abs(edge_x - 212), 4)                 # the ellipse's left contour (6/80 of 160 px in)
        self.assertGreater(row[edge_x], 150)
        # glow: dimmer light outside the contour, fading with distance
        self.assertTrue(0 < row[edge_x - 6] < row[edge_x])
        f0, _ = self._draw("outline", outline_glow=0)
        self.assertEqual(f0[240, edge_x - 8].sum(), 0)

    def test_silhouette_outline(self):
        f, _ = self._draw("silhouette_outline", chams_opacity=0.5)
        self.assertGreater(int(f[240, 280, 1]), 50)           # translucent fill
        self.assertLess(int(f[240, 280, 1]), int(f[240, :280, 1].max()))   # outline brighter

    def test_xray_darkens_around_people(self):
        base = np.full((self.H, self.W, 3), 200, np.uint8)
        f, _ = self._draw("xray", frame=base.copy(), outline_glow=0)
        self.assertLess(int(f[20, 20].mean()), 80)            # background dimmed
        self.assertGreater(int(f[240, 280].mean()), 150)      # person stays bright
        self.assertGreater(int(f[240, 280, 1]), int(f[240, 280, 0]))   # faint yellow tint

    def test_fallback_to_box_without_mask(self):
        f, n = self._draw("silhouette", mask=False, chams_opacity=1.0)
        self.assertEqual(n, 0)
        self.assertEqual(tuple(f[84, 204]), YELLOW)           # whole box filled (classic chams)
        base = np.full((self.H, self.W, 3), 200, np.uint8)
        f, _ = self._draw("xray", frame=base.copy(), mask=False)
        self.assertEqual(int(f[20, 20].mean()), 200)          # no darkening without masks

    def test_no_outline_along_frame_edge(self):
        # body cut off by the bottom of the picture: outlined on the sides, not along the cut
        crop = np.zeros((160, 80), np.uint8)
        crop[20:, 10:70] = 255
        s = _settings(chams_mode="outline", outline_glow=0)
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        S.draw(frame, [_det(200, 160, 360, self.H, crop)], [_pal(s)], s, R.style_for(self.W, self.H, 2))
        self.assertEqual(frame[self.H - 2, 240:320].sum(), 0)    # bottom edge stays clean
        self.assertGreater(frame[300, 210:250, 1].max(), 150)   # left side outlined

    def test_class_color_overrides_chams_color(self):
        s = _settings(chams_mode="silhouette", chams_opacity=1.0)
        self.assertEqual(S.chams_color(s, s.color_boxes), s.chams_color)
        self.assertEqual(S.chams_color(s, (1, 2, 3)), (1, 2, 3))


class PipelineTests(unittest.TestCase):
    def _cam(self, **settings):
        cs = state.get_camera_state("front")
        cs.esp_settings.clear()
        cs.esp_settings.update(config.DEFAULT_ESP_SETTINGS)
        cs.esp_settings.update({"smoothing": 0.0, "boxes": False, "names": False, "conf": False,
                                "skeleton": False, "chams": True, "chams_opacity": 1.0, "zones": False,
                                "alarms": False, "tripwires": False})
        cs.esp_settings.update(settings)
        cs.esp_motion = motion.MotionState()
        cs.esp_locked = None
        cs.view_solo, cs.view_hidden = None, []
        return cs

    def test_end_to_end_silhouette_and_fallback(self):
        rows = [(1, 400, 200, 600, 600)]
        cs = self._cam(chams_mode="silhouette")
        t = time.perf_counter()
        res = _seg_results(rows)
        draw_custom_esp(cs, np.zeros((720, 1280, 3), np.uint8), res, frame_ts=t, results_ts=t)
        # second frame after the fade-in, same results (no new inference)
        out = draw_custom_esp(cs, np.zeros((720, 1280, 3), np.uint8), res, frame_ts=t + 0.3, results_ts=t)
        self.assertEqual(tuple(out[400, 500]), YELLOW)
        self.assertEqual(out[206, 406].sum(), 0)

        cs = self._cam(chams_mode="silhouette")
        res = _seg_results(rows, with_masks=False)
        draw_custom_esp(cs, np.zeros((720, 1280, 3), np.uint8), res, frame_ts=t, results_ts=t)
        out = draw_custom_esp(cs, np.zeros((720, 1280, 3), np.uint8), res, frame_ts=t + 0.3, results_ts=t)
        self.assertEqual(tuple(out[206, 406]), YELLOW)       # box chams fallback


class PerfTests(unittest.TestCase):
    def test_render_cost_1080p_five_people(self):
        w, h = 1920, 1080
        rows = [(i + 1, 120 + i * 360, 220, 120 + i * 360 + 260, 220 + 700) for i in range(5)]
        dets = [d for d in extract_detections(_seg_results(rows, frame_hw=(h, w)), _settings())]
        self.assertEqual(len(dets), 5)
        st = R.style_for(w, h, 2)
        base = np.random.default_rng(1).integers(0, 255, (h, w, 3), np.uint8)
        results = {}
        for mode in ("silhouette", "outline", "silhouette_outline", "xray"):
            s = _settings(chams_mode=mode)
            pals = [_pal(s)] * len(dets)
            times = []
            for _ in range(15):
                f = base.copy()
                t0 = time.perf_counter()
                S.draw(f, dets, pals, s, st)
                times.append((time.perf_counter() - t0) * 1000)
            results[mode] = float(np.median(times))
        print("\n[perf] silhouette chams 1080p, 5 people: " +
              ", ".join(f"{k} {v:.2f} ms" for k, v in results.items()))
        self.assertLess(max(results.values()), 40.0)


if __name__ == "__main__":
    unittest.main()
