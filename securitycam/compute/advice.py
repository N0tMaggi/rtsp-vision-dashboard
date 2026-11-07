"""Notices for non-default device choices, backed by numbers where possible.

Numbers come from the last "Test speed" run on this machine when there is
one for the model, otherwise from a reference table measured on an Intel Core
Ultra 7 265K + RTX 5080 (pure inference per 640x640 frame; the RTX value is
eager PyTorch FP16, a TensorRT engine is much faster).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import assignments, devices

REFERENCE_MACHINE = "Core Ultra 7 265K + RTX 5080"

# ms per frame, YOLO26 by size (L/X extrapolated conservatively from the M ratios)
YOLO_REF: Dict[str, Dict[str, float]] = {
    "N": {"npu": 6.1, "igpu": 5.6, "cpu": 10.5, "cuda": 16.4},
    "S": {"npu": 12.0, "igpu": 12.7, "cpu": 31.0, "cuda": 12.7},
    "M": {"npu": 27.1, "igpu": 35.2, "cpu": 91.4, "cuda": 14.0},
    "L": {"npu": 36.0, "igpu": 47.0, "cpu": 120.0, "cuda": 17.0},
    "X": {"npu": 75.0, "igpu": 95.0, "cpu": 250.0, "cuda": 24.0},
}
# faces: YuNet 640 + SFace 112; plates: detector 384 (OCR not included)
SMALL_REF: Dict[str, Dict[str, float]] = {
    "faces": {"npu": 5.5, "igpu": 3.1, "cpu": 5.0},
    "plates": {"igpu": 3.1, "cpu": 5.7},
    # Sound ESP: YAMNet, one 0.975 s window (measured with OpenVINO 2026.4)
    "sound": {"npu": 1.6, "igpu": 1.5, "cpu": 2.6},
}
CHOPPY_MS = 66.0        # below ~15 inferences per second detection looks choppy
SLOW_MS = 33.0


def _label(dev: str) -> str:
    return devices.LABELS.get(dev, dev)


def _model_bits(name: Optional[str]) -> tuple[str, str]:
    if not name:
        return "the model", "S"
    from ..recognition.engines import model_info

    info = model_info(name)
    return info["label"], (info["size"] if info["size"] in YOLO_REF else "M")


def _ms(workload: str, dev: str, model: Optional[str]) -> tuple[Optional[float], str]:
    """(ms, source) for a device: "measured" from the last Test speed run, else "reference"."""
    key = model if workload == "yolo" else "_"
    b = assignments.benchmark(workload, key or "_")
    if b:
        for row in b.get("rows", []):
            if row.get("device") == dev and row.get("ms"):
                return float(row["ms"]), "measured"
    if workload == "yolo":
        _, size = _model_bits(model)
        ref = YOLO_REF[size]
        if model and "pose" in model:
            ref = {k: v * 1.08 for k, v in ref.items()}
        v = ref.get(dev)
        if v is not None and model and not model.startswith("yolo26"):
            v *= 1.15     # older families are a little slower on every device
        return v, "reference"
    return SMALL_REF.get(workload, {}).get(dev), "reference"


def _fmt(ms: float) -> str:
    return f"{ms:.1f}" if ms < 10 else f"{ms:.0f}"


def _src(source: str) -> str:
    return "measured here" if source == "measured" else f"reference: {REFERENCE_MACHINE}"


def notices(workload: str, requested: str, model: Optional[str] = None,
            effective: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """List of {level: info|warn|error, text} for one workload/device choice."""
    out: List[Dict[str, str]] = []
    if requested == "auto":
        return out
    eff = effective or {}
    if eff.get("fallback_reason") == "failed":
        out.append({"level": "error",
                    "text": f"{_label(requested)} failed, running on Auto instead: {eff.get('error') or 'unknown error'}"})
    elif eff.get("fallback_reason"):
        out.append({"level": "error", "text": f"{_label(requested)} is not available: {eff['fallback_reason']} Running on Auto."})

    if workload == "yolo":
        out.extend(_yolo(requested, model))
    else:
        out.extend(_small(workload, requested))
    out.append({"level": "info",
                "text": "Not the recommended setting. Auto picks the fastest backend known to work here; "
                        "switch back with Reset to recommended if detection gets slower or unstable."})
    return out


def _yolo(dev: str, model: Optional[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    label, size = _model_bits(model)
    ms, src = _ms("yolo", dev, model)
    cuda_ok = devices.availability("cuda")[0]
    gpu_name = devices.cuda_name() or "NVIDIA GPU"
    if dev == "cuda":
        out.append({"level": "info", "text": "Same as Auto on this machine: PyTorch on the NVIDIA GPU, "
                                             "or its TensorRT engine when you built one."})
        return out
    gpu_ms, gsrc = _ms("yolo", "cuda", model) if cuda_ok else (None, "")
    if dev in ("npu", "igpu") and ms is not None:
        name = _label(dev)
        if cuda_ok and gpu_ms:
            ratio = ms / gpu_ms
            if ratio >= 1.3:
                out.append({"level": "warn", "text":
                            f"{name} is about {ratio:.1f}x slower than your {gpu_name} for {label} "
                            f"(~{_fmt(ms)} ms vs ~{_fmt(gpu_ms)} ms per frame, {_src(src)}). "
                            "Use it to free the GPU (e.g. while gaming), not for speed."})
            else:
                out.append({"level": "info", "text":
                            f"{name} runs {label} at ~{_fmt(ms)} ms per frame ({_src(src)}), close to the "
                            f"{gpu_name} in PyTorch; a TensorRT engine on the GPU is still faster. "
                            "Good for freeing the GPU (e.g. while gaming)."})
        else:
            out.append({"level": "info", "text":
                        f"{name} runs {label} at ~{_fmt(ms)} ms per frame ({_src(src)})."})
        if dev == "npu" and size in ("M", "L", "X"):
            out.append({"level": "warn", "text": "Larger models than S are not recommended on the NPU."})
        if dev == "igpu":
            out.append({"level": "info", "text": "The Intel GPU shares power and memory bandwidth with the CPU; "
                                                 "the first start compiles the model (~10 s)."})
    if dev == "cpu":
        backend = "OpenVINO" if devices.openvino_installed() else "PyTorch"
        if ms is not None and backend == "PyTorch":
            ms *= 2.5      # PyTorch on the CPU is much slower than OpenVINO
        if ms is not None and ms >= CHOPPY_MS:
            out.append({"level": "warn", "text":
                        f"CPU ({backend}) runs {label} at ~{_fmt(ms)} ms per frame ({_src(src)}) "
                        "— expect choppy detection."})
        elif ms is not None:
            out.append({"level": "info" if ms < SLOW_MS else "warn", "text":
                        f"CPU ({backend}) runs {label} at ~{_fmt(ms)} ms per frame ({_src(src)})."})
        if cuda_ok:
            out.append({"level": "warn", "text": f"Your {gpu_name} is idle for detection with this setting."})
    if dev in ("npu", "igpu", "cpu") and devices.openvino_installed():
        out.append({"level": "info", "text": "Tracking and pose keep working. ReID uses a small extra classifier "
                                             "(yolo26n-cls) on the CPU with OpenVINO models."})
    return out


def _measured_auto(workload: str) -> Optional[float]:
    b = assignments.benchmark(workload, "_")
    for row in (b or {}).get("rows", []):
        if row.get("device") == "auto" and row.get("ms"):
            return float(row["ms"])
    return None


def _small(workload: str, dev: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    what = "Face models" if workload == "faces" else "Plate models"
    if dev == "cpu" and not devices.openvino_installed():
        out.append({"level": "info", "text": f"{what} already run on the CPU in Auto; without OpenVINO this is the same."})
        return out
    ms, src = _ms(workload, dev, None)
    auto_ms = _measured_auto(workload)
    cpu_ms, _ = _ms(workload, "cpu", None)
    if ms is not None and auto_ms:
        out.append({"level": "info", "text": f"{what} take ~{_fmt(ms)} ms per call on {_label(dev)} vs ~{_fmt(auto_ms)} ms "
                                             f"with Auto ({_src(src)})."})
    elif ms is not None:
        line = f"{what} take ~{_fmt(ms)} ms per call on {_label(dev)} ({_src(src)})"
        if dev != "cpu" and cpu_ms:
            line += f", ~{_fmt(cpu_ms)} ms on the CPU with OpenVINO"
        out.append({"level": "info", "text": line + "."})
    if ms is not None and auto_ms and ms < auto_ms * 0.7:
        out.append({"level": "info", "text": "Faster than Auto on this machine. The trade-off: OpenVINO compiles the "
                                             "models on first use (a few seconds) and is less tested here than Auto."})
    else:
        out.append({"level": "info", "text": f"{what} are small and run fine on the CPU; moving them mainly helps "
                                             "when the CPU is busy. The first use compiles them (a few seconds)."})
    if workload == "plates":
        out.append({"level": "info", "text": "The plate OCR has a dynamic batch size; it is fixed to one plate per "
                                             "call for OpenVINO, so readings match Auto."})
    return out
