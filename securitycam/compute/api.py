"""HTTP API of the compute package (``/api/compute``). Writes are JSON only."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from .. import config, state
from . import advice, assignments, benchmark, devices, small
from . import yolo as cy

blueprint = Blueprint("compute", __name__, url_prefix="/api/compute")
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


@blueprint.record_once
def _on_register(setup_state) -> None:
    from . import register

    register()


@blueprint.before_request
def _require_json_body():
    # Same rule as the ops API: a body on a mutating route must be JSON, which a
    # cross-site page can't send without a CORS preflight.
    if request.method in _MUTATING and not request.is_json:
        if request.content_length or request.headers.get("Transfer-Encoding"):
            return _err("unsupported_media_type", 415)
    return None


def _err(code: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": code, **extra}), status


def _auto_device(workload: str) -> Dict[str, str]:
    if workload == "yolo":
        if state.CUDA_AVAILABLE:
            return {"device": "cuda", "label": f"{devices.cuda_name() or 'NVIDIA GPU'} (PyTorch / TensorRT)"}
        return {"device": "cpu", "label": "CPU (PyTorch)"}
    if workload == "faces":
        return {"device": "cpu", "label": "CPU (OpenCV)"}
    if workload == "depth":    # depth package: NPU > NVIDIA GPU > Intel GPU > CPU
        from ..depth import bench as depth_bench, model as depth_model
        dev = depth_model.auto_device()
        return {"device": dev, "label": depth_bench.row_label(dev)}
    return {"device": "cpu", "label": "CPU (onnxruntime / OpenCV)"}


def _options(workload: str) -> list:
    out = [{"id": "auto", "label": "Auto", "available": True, "reason": None, "recommended": True}]
    for dev in devices.DEVICE_IDS:
        ok, reason = assignments.supports(workload, dev)
        out.append({"id": dev, "label": devices.LABELS[dev], "short": devices.SHORT[dev],
                    "available": ok, "reason": reason, "recommended": False})
    return out


def _main_model() -> Optional[str]:
    cams = state.list_camera_states()
    if not cams:
        return config.YOLO_MODEL_NAME
    enabled = [c.yolo_model_name for c in cams if c.yolo_enabled]
    pool = enabled or [c.yolo_model_name for c in cams]
    return Counter(pool).most_common(1)[0][0]


def _small_state(workload: str, eff: Dict[str, Any]) -> Dict[str, Any]:
    from .ov_runtime import POOL

    if eff["device"] == "auto":
        return {"state": "auto"}
    if small.active_device(workload) is None:
        return {"state": "active"}       # e.g. CPU without OpenVINO = the usual CPU path
    st = POOL.status(workload)
    if st["compiling"]:
        return {"state": "preparing"}
    if st["ready"]:
        return {"state": "active"}
    return {"state": "idle"}            # compiled on first use (when the feature is switched on)


def _depth_state() -> Dict[str, Any]:
    from ..depth import worker as depth_worker
    states = [w.state for w in (depth_worker.get(cs.camera_id) for cs in state.list_camera_states()) if w]
    if "running" in states:
        return {"state": "active"}
    if "loading" in states or "downloading" in states:
        return {"state": "preparing"}
    return {"state": "idle"}


def _badge(dev: str) -> Optional[str]:
    return devices.SHORT.get(dev) if dev not in ("auto",) else None


def payload() -> Dict[str, Any]:
    hw = devices.snapshot()
    model = _main_model()
    workloads = []
    for name, meta in assignments.WORKLOADS.items():
        req = assignments.assignment(name)
        eff = assignments.effective(name)
        wl_model = model if name == "yolo" else None
        item: Dict[str, Any] = {
            "id": name, "label": meta["label"], "description": meta["description"],
            "per_camera": meta["per_camera"], "assignment": req, "effective": eff,
            "auto": _auto_device(name), "options": _options(name),
            "model": wl_model,
            "warnings": advice.notices(name, req, wl_model, eff),
            "benchmark": assignments.benchmark(name, wl_model or "_"),
        }
        if name == "depth":       # depth package: state of the running depth workers
            item.update(_depth_state())
        elif name != "yolo":
            item.update(_small_state(name, eff))
        workloads.append(item)

    cams = []
    for cs in state.list_camera_states():
        override = assignments.camera_override(cs.camera_id)
        eff = assignments.effective("yolo", cs.camera_id)
        req = eff["requested"]
        info = cy.camera_state_info(cs.camera_id)
        cams.append({
            "id": cs.camera_id, "label": cs.label, "source_kind": cs.source_kind,
            "model": cs.yolo_model_name, "yolo_enabled": cs.yolo_enabled,
            "override": override, "effective": eff,
            "badge": _badge(req) if req != "auto" else None,
            "warnings": advice.notices("yolo", req, cs.yolo_model_name, eff) if override else [],
            **info,
        })
    return {
        "ok": True,
        "hardware": hw,
        "workloads": workloads,
        "cameras": cams,
        "non_default": not assignments.is_default(),
        "benchmark_job": benchmark.status(),
        "reference_machine": advice.REFERENCE_MACHINE,
    }


@blueprint.get("")
def get_compute():
    resp = jsonify(payload())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.patch("")
def patch_compute():
    known = {c.camera_id for c in state.list_camera_states()}
    try:
        clean = assignments.validate_patch(request.get_json(silent=True), known)
    except assignments.AssignmentError as exc:
        return _err(exc.code, detail=exc.detail)
    assignments.apply_patch(clean)
    return jsonify(payload())


@blueprint.post("/reset")
def reset_compute():
    assignments.reset()
    return jsonify(payload())


@blueprint.post("/refresh")
def refresh_devices():
    devices.refresh()
    return jsonify({"ok": True})


@blueprint.post("/benchmark")
def start_benchmark():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("invalid_body")
    workload = body.get("workload")
    model = body.get("model")
    if workload == "yolo" and model is None:
        cam = state.get_camera_state(body.get("camera") or "") if body.get("camera") else None
        model = cam.yolo_model_name if cam else _main_model()
    try:
        job = benchmark.start(workload, model)
    except benchmark.BenchmarkError as exc:
        return _err(exc.code, 409 if exc.code == "busy" else 400)
    return jsonify({"ok": True, "job": job}), 202


@blueprint.get("/benchmark")
def benchmark_status():
    resp = jsonify({"ok": True, "job": benchmark.status()})
    resp.headers["Cache-Control"] = "no-store"
    return resp
