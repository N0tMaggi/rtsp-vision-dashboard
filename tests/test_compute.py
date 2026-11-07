"""Tests for the compute package: device discovery (with and without OpenVINO),
assignments + persistence, fallback on failures, warning texts and the
/api/compute endpoints. Hardware is faked; nothing is exported or compiled
and no camera is ever contacted.

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
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="securitycam-compute-test-")
os.environ.setdefault("SECURITYCAM_DATA_DIR", _TMP)
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ.setdefault("SECURITYCAM_CAMERAS", "front|Front|192.0.2.10|user|pw|stream1;garage|Garage|192.0.2.11|user|pw|stream1")
os.environ["SECURITYCAM_AUTO_INSTALL_DEPS"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam import compute, persistence, presets, state, streaming  # noqa: E402
from securitycam.compute import advice, assignments, benchmark, devices, ov_runtime, small  # noqa: E402
from securitycam.compute import yolo as cy  # noqa: E402
from securitycam.webapp import app  # noqa: E402

presets.SCHEDULER.stop()


class FakeCore:
    """Stands in for openvino.Core: the user's machine (Core Ultra + RTX via OpenCL)."""

    def __init__(self, devices_=None):
        self.available_devices = devices_ or ["CPU", "GPU.0", "GPU.1", "NPU"]
        self.props = {
            "CPU": {"FULL_DEVICE_NAME": "Intel(R) Core(TM) Ultra 7 265K", "DEVICE_TYPE": "Type.INTEGRATED"},
            "GPU.0": {"FULL_DEVICE_NAME": "Intel(R) Graphics (iGPU)", "DEVICE_TYPE": "Type.INTEGRATED"},
            "GPU.1": {"FULL_DEVICE_NAME": "NVIDIA GeForce RTX 5080 (dGPU)", "DEVICE_TYPE": "Type.DISCRETE"},
            "NPU": {"FULL_DEVICE_NAME": "Intel(R) AI Boost", "DEVICE_TYPE": "Type.INTEGRATED"},
        }

    def get_property(self, dev, name):
        return self.props[dev][name]

    def set_property(self, *a, **kw):
        pass


def fake_discovery(devices_=None, installed=True):
    """Run discovery synchronously against a FakeCore; returns the patchers to stop."""
    patches = [mock.patch.object(devices, "openvino_installed", return_value=installed),
               mock.patch.object(devices, "CORE_FACTORY", lambda: FakeCore(devices_))]
    for p in patches:
        p.start()
    devices.refresh(sync=True)
    return patches


def stop_all(patches):
    for p in patches:
        p.stop()


class FakeModel:
    """Minimal Ultralytics-like model."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self._securitycam_device = "intel:npu"
        self._securitycam_backend = "openvino"

    def __call__(self, frame, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Exception from src/plugins/intel_npu: device lost")
        return []

    track = __call__


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


class ComputeTestBase(unittest.TestCase):
    def setUp(self):
        cy.idle(5)
        self.patches = fake_discovery()
        assignments.reset()
        cy._active.clear()
        cy._prepared.clear()
        cy._jobs.clear()
        self.client = app.test_client()
        self.cs = state.get_camera_state("front")
        self._saved = (self.cs.yolo_model, self.cs.yolo_model_name, self.cs.yolo_enabled)
        self.cs.yolo_model = None
        self.cs.yolo_enabled = False
        # never export / compile anything for real
        self.prep = mock.patch.object(cy, "prepare_model", side_effect=lambda name, dev, on_stage=None, camera_state=None: FakeModel())
        self.prep.start()

    def tearDown(self):
        cy.idle(5)
        self.prep.stop()
        assignments.reset()
        cy._active.clear()
        cy._prepared.clear()
        cy._jobs.clear()
        self.cs.yolo_model, self.cs.yolo_model_name, self.cs.yolo_enabled = self._saved
        stop_all(self.patches)
        with mock.patch.object(devices, "openvino_installed", return_value=False):
            devices.refresh(sync=True)


# ---------------------------------------------------------------------------- discovery
class DiscoveryTests(unittest.TestCase):
    def tearDown(self):
        with mock.patch.object(devices, "openvino_installed", return_value=False):
            devices.refresh(sync=True)

    def test_without_openvino_intel_devices_show_install_command(self):
        with mock.patch.object(devices, "openvino_installed", return_value=False):
            devices.refresh(sync=True)
            snap = devices.snapshot()
            by_id = {d["id"]: d for d in snap["devices"]}
            self.assertFalse(snap["openvino"]["installed"])
            self.assertEqual(snap["openvino"]["install_cmd"], "python -m pip install openvino")
            for dev in ("npu", "igpu"):
                self.assertFalse(by_id[dev]["available"])
                self.assertIn("python -m pip install openvino", by_id[dev]["reason"])
            self.assertFalse(by_id["cpu"]["available"])          # the OpenVINO CPU row
            self.assertIn("cpu_torch", by_id)
            # "CPU" stays selectable: PyTorch / OpenCV run on the CPU without OpenVINO
            self.assertEqual(devices.availability("cpu"), (True, None))
            ok, reason = assignments.supports("yolo", "npu")
            self.assertFalse(ok)
            self.assertIn("OpenVINO", reason)

    def test_with_openvino_picks_intel_gpu_not_the_nvidia_opencl_device(self):
        patches = fake_discovery()
        try:
            self.assertEqual(devices.ov_device("npu"), "NPU")
            self.assertEqual(devices.ov_device("igpu"), "GPU.0")
            self.assertEqual(devices.ov_device("cpu"), "CPU")
            snap = devices.snapshot()
            by_id = {d["id"]: d for d in snap["devices"]}
            self.assertTrue(by_id["npu"]["available"])
            self.assertEqual(by_id["npu"]["name"], "Intel(R) AI Boost")
            self.assertEqual(by_id["igpu"]["name"], "Intel(R) Graphics (iGPU)")
            self.assertTrue(snap["ready"])
            self.assertEqual(ov_runtime.ultralytics_device("igpu"), "intel:gpu.0")
            self.assertEqual(ov_runtime.ultralytics_device("npu"), "intel:npu")
        finally:
            stop_all(patches)

    def test_missing_npu_is_unavailable_with_reason(self):
        patches = fake_discovery(["CPU"])
        try:
            ok, reason = devices.availability("npu")
            self.assertFalse(ok)
            self.assertIn("No Intel NPU", reason)
            self.assertFalse(devices.availability("igpu")[0])
            self.assertTrue(devices.availability("cpu")[0])
        finally:
            stop_all(patches)

    def test_openvino_core_failure_does_not_raise(self):
        def boom():
            raise RuntimeError("driver exploded")

        with mock.patch.object(devices, "openvino_installed", return_value=True), \
                mock.patch.object(devices, "CORE_FACTORY", boom):
            devices.refresh(sync=True)
            ok, reason = devices.availability("npu")
            self.assertFalse(ok)
            self.assertIn("driver exploded", reason)
            self.assertIn("error", devices.snapshot()["openvino"])

    def test_snapshot_never_waits_for_discovery(self):
        gate = threading.Event()

        def slow():
            gate.wait(5)
            return FakeCore()

        with mock.patch.object(devices, "openvino_installed", return_value=True), \
                mock.patch.object(devices, "CORE_FACTORY", slow):
            devices.refresh()
            t0 = time.perf_counter()
            snap = devices.snapshot()
            self.assertLess(time.perf_counter() - t0, 0.5)
            self.assertFalse(snap["ready"])
            self.assertEqual(devices.availability("npu")[1], "Detecting hardware…")
            gate.set()
            self.assertTrue(devices.wait(5))


# ---------------------------------------------------------------------------- defaults
class DefaultBehaviourTests(ComputeTestBase):
    def test_nothing_configured_changes_nothing(self):
        self.assertTrue(assignments.is_default())
        self.assertIsNone(compute.load_yolo(self.cs, "yolo26s.pt"))
        self.assertEqual(compute.yolo_device(self.cs, object()), state.DEVICE)
        self.assertEqual(compute.small_key("plates"), "auto")
        self.assertIsNone(compute.faces_detect(np.zeros((32, 32, 3), np.uint8), 0.7, 0.3, 50))
        self.assertIsNone(compute.face_feature(np.zeros((112, 112, 3), np.uint8)))
        self.assertIsNone(compute.plate_nets("d", "o", 384, (1, 70, 140, 1)))
        self.assertFalse(compute.on_yolo_error(self.cs, RuntimeError("x")))
        cy.prepare_model.assert_not_called()

    def test_nvidia_choice_uses_the_regular_cuda_path(self):
        with mock.patch.object(state, "CUDA_AVAILABLE", True):
            assignments.apply_patch({"workloads": {"yolo": "cuda"}})
            self.assertIsNone(compute.load_yolo(self.cs, "yolo26s.pt"))
            self.assertEqual(compute.yolo_device(self.cs, object()), state.DEVICE)
            cy.prepare_model.assert_not_called()


# ---------------------------------------------------------------------------- assignments + persistence
class AssignmentTests(ComputeTestBase):
    def test_validate_patch(self):
        known = {"front", "garage"}
        clean = assignments.validate_patch({"workloads": {"yolo": "npu", "faces": "igpu"},
                                            "cameras": {"garage": "cpu", "front": None}}, known)
        self.assertEqual(clean["workloads"], {"yolo": "npu", "faces": "igpu"})
        self.assertEqual(clean["cameras"], {"garage": "cpu", "front": None})
        cases = [
            ({"workloads": {"gpu": "npu"}}, "invalid_workload"),
            ({"workloads": {"yolo": "tpu"}}, "invalid_device"),
            ({"workloads": {"faces": "cuda"}}, "device_unavailable"),
            ({"workloads": {"plates": "npu"}}, "device_unavailable"),   # NPU breaks the plate NMS
            ({"cameras": {"nope": "npu"}}, "unknown_camera"),
            ({}, "empty_patch"),
            ([1], "invalid_body"),
        ]
        for body, code in cases:
            with self.assertRaises(assignments.AssignmentError) as ctx:
                assignments.validate_patch(body, known)
            self.assertEqual(ctx.exception.code, code, body)

    def test_camera_override_wins_and_can_be_cleared(self):
        assignments.apply_patch({"workloads": {"yolo": "npu"}, "cameras": {"garage": "auto"}})
        self.assertEqual(assignments.effective("yolo", "front")["device"], "npu")
        self.assertEqual(assignments.effective("yolo", "garage")["device"], "auto")
        self.assertEqual(assignments.effective("yolo", "garage")["source"], "camera")
        assignments.apply_patch({"cameras": {"garage": None}})
        self.assertEqual(assignments.effective("yolo", "garage")["device"], "npu")

    def test_dump_validate_load_roundtrip(self):
        assignments.apply_patch({"workloads": {"yolo": "igpu", "faces": "npu"}, "cameras": {"front": "cpu"}})
        assignments.store_benchmark("yolo", "yolo26s.pt", {"workload": "yolo", "model": "yolo26s.pt", "finished": 1.0,
                                                           "rows": [{"device": "npu", "label": "Intel NPU", "ms": 12.0}]})
        dumped = json.loads(json.dumps(assignments.dump()))
        assignments.reset()
        clean, warnings = assignments.validate(dumped)
        self.assertEqual(warnings, [])
        assignments.load(clean)
        self.assertEqual(assignments.assignment("yolo"), "igpu")
        self.assertEqual(assignments.assignment("faces"), "npu")
        self.assertEqual(assignments.camera_override("front"), "cpu")
        self.assertEqual(assignments.benchmark("yolo", "yolo26s.pt")["rows"][0]["ms"], 12.0)

    def test_validate_drops_garbage(self):
        clean, warnings = assignments.validate({"workloads": {"yolo": "warp", "faces": "npu", "x": "cpu"},
                                                "cameras": {"front": "nope", "garage": "igpu"},
                                                "benchmarks": {"yolo": {"m": {"rows": "x"}}}})
        self.assertEqual(clean["workloads"], {"faces": "npu"})
        self.assertEqual(clean["cameras"], {"garage": "igpu"})
        self.assertEqual(clean["benchmarks"], {})
        self.assertEqual(len(warnings), 3)
        with self.assertRaises(ValueError):
            assignments.validate([])

    def test_persisted_through_the_state_file(self):
        path = Path(tempfile.mkdtemp()) / "state.json"
        store = persistence.Store(path, debounce_s=0.0)
        store.register_section("compute", assignments.dump, assignments.load, assignments.validate)
        store.load()
        assignments.apply_patch({"workloads": {"yolo": "npu"}, "cameras": {"garage": "cpu"}})
        store.mark_dirty()
        store.flush()
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["sections"]["compute"]["workloads"]["yolo"], "npu")
        assignments.reset()
        store2 = persistence.Store(path, debounce_s=0.0)
        store2.register_section("compute", assignments.dump, assignments.load, assignments.validate)
        store2.load()
        self.assertEqual(assignments.assignment("yolo"), "npu")
        self.assertEqual(assignments.camera_override("garage"), "cpu")

    def test_stored_device_that_vanished_falls_back_to_auto(self):
        assignments.load({"workloads": {"yolo": "npu"}, "cameras": {}, "benchmarks": {}})
        stop_all(self.patches)
        self.patches = fake_discovery(["CPU"])
        eff = assignments.effective("yolo", "front")
        self.assertEqual(eff["requested"], "npu")
        self.assertEqual(eff["device"], "auto")
        self.assertIn("No Intel NPU", eff["fallback_reason"])
        self.assertIsNone(compute.load_yolo(self.cs, "yolo26s.pt"))
        cy.prepare_model.assert_not_called()


# ---------------------------------------------------------------------------- YOLO hand-over + fallback
class YoloFallbackTests(ComputeTestBase):
    def test_prepared_model_is_handed_over_with_its_device(self):
        self.cs.yolo_model_name = "yolo26s.pt"
        assignments.apply_patch({"workloads": {"yolo": "npu"}})
        self.assertIsNone(compute.load_yolo(self.cs, "yolo26s.pt"))   # auto model meanwhile
        self.assertTrue(wait_for(lambda: (cy.job("front", "yolo26s.pt", "npu") or {}).get("status") == "ready"))
        model = compute.load_yolo(self.cs, "yolo26s.pt")
        self.assertIsInstance(model, FakeModel)
        self.assertEqual(compute.yolo_device(self.cs, model), "intel:npu")
        self.assertEqual(compute.yolo_device(self.cs, object()), state.DEVICE)
        self.assertEqual(cy.camera_state_info("front")["state"], "active")

    def test_inference_error_falls_back_to_auto(self):
        model = FakeModel(fail=True)
        self.cs.yolo_model = model
        self.cs.yolo_model_name = "yolo26s.pt"
        assignments.apply_patch({"workloads": {"yolo": "npu"}})
        cy._active["front"] = {"ref": __import__("weakref").ref(model), "device": "npu", "name": "yolo26s.pt",
                               "ul_device": "intel:npu", "backend": "openvino"}
        self.assertTrue(compute.on_yolo_error(self.cs, RuntimeError("device lost")))
        self.assertIsNone(self.cs.yolo_model)
        eff = assignments.effective("yolo", "front")
        self.assertEqual(eff["device"], "auto")
        self.assertEqual(eff["fallback_reason"], "failed")
        self.assertIn("device lost", eff["error"])
        notes = advice.notices("yolo", "npu", "yolo26s.pt", eff)
        self.assertEqual(notes[0]["level"], "error")
        self.assertIn("running on Auto", notes[0]["text"])
        # a second error from an auto model is not ours
        self.cs.yolo_model = FakeModel()
        self.assertFalse(compute.on_yolo_error(self.cs, RuntimeError("cuda")))
        # picking the device again retries
        assignments.apply_patch({"workloads": {"yolo": "npu"}})
        self.assertIsNone(assignments.effective("yolo", "front")["fallback_reason"])

    def test_worker_falls_back_without_crashing(self):
        bad = FakeModel(fail=True)
        good = FakeModel()
        self.cs.yolo_model_name = "yolo26s.pt"
        assignments.apply_patch({"workloads": {"yolo": "npu"}})
        cy._active["front"] = {"ref": __import__("weakref").ref(bad), "device": "npu", "name": "yolo26s.pt",
                               "ul_device": "intel:npu", "backend": "openvino"}

        def get_model(cs):
            if cs.yolo_model is None:
                cs.yolo_model = good
            return cs.yolo_model

        cam = state.CameraRuntime(camera_id="front", label="Front", ip="192.0.2.10", stream_path="s", rtsp_url="")
        cam.yolo_model, cam.yolo_model_name, cam.yolo_enabled = bad, "yolo26s.pt", True
        cam.esp_settings["tracking"] = False
        with mock.patch.object(state, "YOLO_AVAILABLE", True), \
                mock.patch.object(streaming.deps, "get_yolo_model", side_effect=get_model), \
                mock.patch.object(streaming.recognition, "after_inference", side_effect=lambda cs, f, r, ts: r):
            t = threading.Thread(target=streaming.yolo_worker, args=(cam,), daemon=True)
            t.start()

            def feed():
                with cam.frame_lock:
                    cam.latest_frame = np.zeros((48, 64, 3), np.uint8)
                return good.calls > 0
            ok = wait_for(feed, 5)
            cam.closed = True
            t.join(2)
        self.assertTrue(ok, "worker never switched to the auto model")
        self.assertEqual(bad.calls, 1)
        self.assertEqual(assignments.effective("yolo", "front")["fallback_reason"], "failed")

    def test_preparation_failure_marks_device_failed(self):
        self.cs.yolo_model_name = "yolo26m.pt"
        cy.prepare_model.side_effect = RuntimeError("export exploded")
        assignments.apply_patch({"workloads": {"yolo": "igpu"}})
        compute.load_yolo(self.cs, "yolo26m.pt")
        self.assertTrue(wait_for(lambda: assignments.failure("yolo", "igpu") is not None))
        self.assertEqual(cy.job("front", "yolo26m.pt", "igpu")["status"], "failed")
        self.assertEqual(assignments.effective("yolo", "front")["device"], "auto")
        self.assertIsNone(compute.load_yolo(self.cs, "yolo26m.pt"))

    def test_reset_drops_compute_models(self):
        model = FakeModel()
        self.cs.yolo_model = model
        assignments.apply_patch({"workloads": {"yolo": "npu"}})
        cy._active["front"] = {"ref": __import__("weakref").ref(model), "device": "npu", "name": "x",
                               "ul_device": "intel:npu", "backend": "openvino"}
        assignments.reset()
        self.assertIsNone(self.cs.yolo_model)
        self.assertNotIn("front", cy._active)


# ---------------------------------------------------------------------------- small models
class SmallModelTests(ComputeTestBase):
    def test_compile_failure_falls_back(self):
        assignments.apply_patch({"workloads": {"faces": "npu"}})
        pool = ov_runtime.NetPool()

        class BadCore(FakeCore):
            def read_model(self, path):
                raise RuntimeError("cannot read")

        with mock.patch.object(devices, "core", lambda: BadCore()):
            self.assertIsNone(pool.get("faces", Path("x.onnx"), "npu", (1, 3, 32, 32), on_fail=small._fail))
            self.assertTrue(wait_for(lambda: assignments.failure("faces", "npu") is not None))
        self.assertEqual(pool.status("faces")["failed"], 1)
        self.assertEqual(assignments.effective("faces")["device"], "auto")
        self.assertIsNone(small.active_device("faces"))
        self.assertEqual(compute.small_key("faces"), "auto")

    def test_ovnet_reports_runtime_errors_and_reraises(self):
        seen = []

        class Req:
            def infer(self, _):
                raise RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST")

        class Compiled:
            outputs = []

            def create_infer_request(self):
                return Req()

        net = ov_runtime.OVNet(Compiled(), "npu", on_error=seen.append)
        with self.assertRaises(RuntimeError):
            net(np.zeros((1, 3, 4, 4), np.float32))
        self.assertEqual(len(seen), 1)

    def test_yunet_decode_matches_opencv_layout(self):
        # one anchor at stride 8, cell (r=1, c=2), centered offsets -> box around (20, 12)
        pw, ph = 32, 32
        out = {}
        for s in small.YUNET_STRIDES:
            n = (pw // s) * (ph // s)
            out[f"cls_{s}"] = np.zeros((1, n, 1), np.float32)
            out[f"obj_{s}"] = np.zeros((1, n, 1), np.float32)
            out[f"bbox_{s}"] = np.zeros((1, n, 4), np.float32)
            out[f"kps_{s}"] = np.zeros((1, n, 10), np.float32)
        idx = 1 * 4 + 2
        out["cls_8"][0, idx, 0] = 0.81
        out["obj_8"][0, idx, 0] = 1.0
        out["bbox_8"][0, idx] = [0.5, 0.5, np.log(2.0), np.log(2.0)]
        rows = small._yunet_decode(out, pw, ph, 0.7, 0.3, 50)
        self.assertEqual(rows.shape, (1, 15))
        x, y, w, h = rows[0, :4]
        self.assertAlmostEqual(float(w), 16.0, places=4)
        self.assertAlmostEqual(float(x + w / 2), (2 + 0.5) * 8, places=4)
        self.assertAlmostEqual(float(y + h / 2), (1 + 0.5) * 8, places=4)
        self.assertAlmostEqual(float(rows[0, 14]), 0.9, places=4)
        self.assertAlmostEqual(float(rows[0, 4]), 2 * 8, places=4)      # landmark x = (0 + c) * stride


# ---------------------------------------------------------------------------- warnings
class AdviceTests(ComputeTestBase):
    def test_npu_medium_on_a_cuda_machine(self):
        with mock.patch.object(state, "CUDA_AVAILABLE", True), \
                mock.patch.object(devices, "cuda_name", return_value="NVIDIA GeForce RTX 5080"):
            texts = " ".join(n["text"] for n in advice.notices("yolo", "npu", "yolo26m.pt"))
        self.assertIn("about 1.9x slower than your NVIDIA GeForce RTX 5080 for YOLO26 M", texts)
        self.assertIn("free the GPU", texts)
        self.assertIn("Larger models than S are not recommended on the NPU", texts)
        self.assertIn("Core Ultra 7 265K", texts)            # static hint is labelled as reference
        self.assertIn("Not the recommended setting", texts)

    def test_cpu_medium_is_choppy(self):
        texts = " ".join(n["text"] for n in advice.notices("yolo", "cpu", "yolo26m.pt"))
        self.assertIn("~91 ms per frame", texts)
        self.assertIn("expect choppy detection", texts)

    def test_measured_numbers_win(self):
        assignments.store_benchmark("yolo", "yolo26s.pt", {
            "workload": "yolo", "model": "yolo26s.pt", "finished": time.time(),
            "rows": [{"device": "npu", "ms": 50.0}, {"device": "cuda", "ms": 10.0}]})
        with mock.patch.object(state, "CUDA_AVAILABLE", True):
            texts = " ".join(n["text"] for n in advice.notices("yolo", "npu", "yolo26s.pt"))
        self.assertIn("5.0x slower", texts)
        self.assertIn("measured here", texts)

    def test_auto_has_no_notices_and_every_other_choice_has_some(self):
        self.assertEqual(advice.notices("yolo", "auto", "yolo26s.pt"), [])
        for workload in ("yolo", "faces", "plates"):
            for dev in assignments.WORKLOADS[workload]["devices"]:
                self.assertTrue(advice.notices(workload, dev, "yolo26n.pt" if workload == "yolo" else None),
                                (workload, dev))

    def test_faces_on_npu_mentions_numbers(self):
        texts = " ".join(n["text"] for n in advice.notices("faces", "npu"))
        self.assertIn("ms per call on Intel NPU", texts)


# ---------------------------------------------------------------------------- API
class ApiTests(ComputeTestBase):
    def test_get_shape(self):
        r = self.client.get("/api/compute")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual({w["id"] for w in d["workloads"]}, {"yolo", "faces", "plates", "depth", "sound"})
        self.assertFalse(d["non_default"])
        self.assertEqual({c["id"] for c in d["cameras"]} >= {"front", "garage"}, True)
        yolo = next(w for w in d["workloads"] if w["id"] == "yolo")
        opts = {o["id"]: o for o in yolo["options"]}
        self.assertTrue(opts["auto"]["recommended"])
        self.assertTrue(opts["npu"]["available"])
        faces = next(w for w in d["workloads"] if w["id"] == "faces")
        self.assertFalse({o["id"]: o for o in faces["options"]}["cuda"]["available"])
        self.assertEqual(r.headers.get("Cache-Control"), "no-store")

    def test_patch_reset_and_camera_override(self):
        r = self.client.patch("/api/compute", json={"workloads": {"faces": "igpu"}, "cameras": {"garage": "cpu"}},
                              headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 200, r.get_json())
        d = r.get_json()
        self.assertTrue(d["non_default"])
        garage = next(c for c in d["cameras"] if c["id"] == "garage")
        self.assertEqual(garage["badge"], "CPU")
        self.assertTrue(garage["warnings"])
        faces = next(w for w in d["workloads"] if w["id"] == "faces")
        self.assertEqual(faces["assignment"], "igpu")
        self.assertTrue(faces["warnings"])
        r = self.client.post("/api/compute/reset", json={})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["non_default"])
        self.assertTrue(assignments.is_default())

    def test_validation_errors(self):
        for body, code in [({"workloads": {"yolo": "quantum"}}, "invalid_device"),
                           ({"workloads": {"plates": "npu"}}, "device_unavailable"),
                           ({"cameras": {"nope": "npu"}}, "unknown_camera"),
                           ({"workloads": {"bogus": "npu"}}, "invalid_workload")]:
            r = self.client.patch("/api/compute", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertEqual(r.get_json()["error"], code)
        self.assertTrue(assignments.is_default())

    def test_writes_must_be_json_and_same_origin(self):
        r = self.client.patch("/api/compute", data="workloads=yolo", content_type="application/x-www-form-urlencoded")
        self.assertEqual(r.status_code, 415)
        r = self.client.patch("/api/compute", data='{"workloads": {"yolo": "npu"}}', content_type="text/plain")
        self.assertEqual(r.status_code, 415)
        r = self.client.patch("/api/compute", json={"workloads": {"yolo": "npu"}},
                              headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/api/compute/reset", json={}, headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        self.assertTrue(assignments.is_default())

    def test_benchmark_job(self):
        def runner(job, set_row):
            for row in job["rows"]:
                set_row(row["device"], ms=4.2, status="done")

        with mock.patch.object(benchmark, "_default_runner", runner):
            r = self.client.post("/api/compute/benchmark", json={"workload": "faces"})
            self.assertEqual(r.status_code, 202, r.get_json())
            self.assertTrue(wait_for(lambda: self.client.get("/api/compute/benchmark").get_json()["job"]["status"] == "done"))
        job = self.client.get("/api/compute/benchmark").get_json()["job"]
        self.assertEqual({r["device"] for r in job["rows"]}, {"auto", "npu", "igpu", "cpu"})
        self.assertTrue(all(r["ms"] == 4.2 for r in job["rows"]))
        self.assertEqual(assignments.benchmark("faces", "_")["rows"][0]["ms"], 4.2)
        r = self.client.post("/api/compute/benchmark", json={"workload": "nope"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/compute/benchmark", json={"workload": "yolo", "model": "evil.pt"})
        self.assertEqual(r.get_json()["error"], "invalid_model")

    def test_benchmark_busy(self):
        gate = threading.Event()
        with mock.patch.object(benchmark, "_default_runner", lambda job, set_row: gate.wait(5)):
            self.assertEqual(self.client.post("/api/compute/benchmark", json={"workload": "plates"}).status_code, 202)
            r = self.client.post("/api/compute/benchmark", json={"workload": "faces"})
            self.assertEqual(r.status_code, 409)
            gate.set()
            wait_for(lambda: benchmark.status()["status"] != "running")


if __name__ == "__main__":
    unittest.main()
