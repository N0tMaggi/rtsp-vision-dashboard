"""Face (YuNet + SFace) and license plate models on OpenVINO devices.

Same contracts as the OpenCV / onnxruntime code in ``recognition.faces`` and
``recognition.plates``; every function returns ``None`` when the workload is
on auto, the OpenVINO network is still compiling, or it failed, so the caller
simply runs its usual code path.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from . import assignments, devices
from .ov_runtime import POOL, OVNet

YUNET_STRIDES = (8, 16, 32)


def active_device(workload: str) -> Optional[str]:
    """OpenVINO device id the workload should use now, or None for the usual (auto) path."""
    eff = assignments.effective(workload)
    dev = eff["device"]
    if dev in ("auto", "cuda"):
        return None
    if not devices.openvino_installed():
        return None          # "cpu" without OpenVINO = the usual CPU path
    return dev


def _fail(workload: str, msg: str) -> None:
    dev = assignments.effective(workload)["requested"]
    if dev != "auto":
        assignments.mark_failed(workload, dev, msg)


def net(workload: str, path, shape: Sequence[int]) -> Optional[OVNet]:
    dev = active_device(workload)
    if dev is None or path is None:
        return None
    return POOL.get(workload, path, dev, shape, on_fail=_fail, on_ready=assignments._changed)


def key(workload: str) -> str:
    """Cache key for thread-local readers: changes whenever the backend in use changes."""
    dev = active_device(workload)
    return "auto" if dev is None else f"{dev}@{assignments.version()}"


# ----------------------------------------------------------------------------- YuNet
def _pad32(v: int) -> int:
    return (int((v - 1) / 32) + 1) * 32


def _asset(name: str):
    from .. import deps

    return deps.asset_path(name)


def yunet_detect(img: np.ndarray, score_thr: float, nms_thr: float, top_k: int) -> Optional[np.ndarray]:
    """YuNet on ``img`` (already scaled) -> (N, 15) float32 rows like cv2.FaceDetectorYN, or None."""
    if active_device("faces") is None:
        return None
    path = _asset("face_yunet")
    ih, iw = img.shape[:2]
    ph, pw = _pad32(ih), _pad32(iw)
    n = net("faces", path, (1, 3, ph, pw))
    if n is None:
        return None
    pad = cv2.copyMakeBorder(img, 0, ph - ih, 0, pw - iw, cv2.BORDER_CONSTANT, value=0)
    blob = pad.transpose(2, 0, 1)[None].astype(np.float32)
    try:
        out = n.outputs(blob)
        return _yunet_decode(out, pw, ph, score_thr, nms_thr, top_k)
    except Exception as exc:          # inference errors already reported by OVNet; decode errors here
        _fail("faces", f"{type(exc).__name__}: {exc}")
        return None


def _yunet_decode(out, pw: int, ph: int, score_thr: float, nms_thr: float, top_k: int) -> np.ndarray:
    boxes, scores, rows = [], [], []
    for stride in YUNET_STRIDES:
        cols, nrows = pw // stride, ph // stride
        cls = np.clip(out[f"cls_{stride}"].reshape(-1), 0, 1)
        obj = np.clip(out[f"obj_{stride}"].reshape(-1), 0, 1)
        score = np.sqrt(cls * obj)
        idx = np.nonzero(score >= score_thr)[0]
        if idx.size == 0:
            continue
        bbox = out[f"bbox_{stride}"].reshape(-1, 4)[idx]
        kps = out[f"kps_{stride}"].reshape(-1, 10)[idx]
        r, c = idx // cols, idx % cols
        cx = (c + bbox[:, 0]) * stride
        cy = (r + bbox[:, 1]) * stride
        w = np.exp(bbox[:, 2]) * stride
        h = np.exp(bbox[:, 3]) * stride
        x1, y1 = cx - w / 2, cy - h / 2
        lm = np.empty((idx.size, 10), np.float32)
        lm[:, 0::2] = (kps[:, 0::2] + c[:, None]) * stride
        lm[:, 1::2] = (kps[:, 1::2] + r[:, None]) * stride
        block = np.concatenate([np.stack([x1, y1, w, h], 1), lm, score[idx, None]], 1).astype(np.float32)
        rows.append(block)
        boxes.extend([[int(a), int(b), int(cw), int(ch)] for a, b, cw, ch in block[:, :4]])
        scores.extend(block[:, 14].tolist())
    if not rows:
        return np.zeros((0, 15), np.float32)
    allrows = np.concatenate(rows, 0)
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr, top_k=top_k)
    keep = np.asarray(keep, np.int64).reshape(-1)
    return allrows[keep] if keep.size else np.zeros((0, 15), np.float32)


# ----------------------------------------------------------------------------- SFace
def sface_feature(aligned: np.ndarray) -> Optional[np.ndarray]:
    """Raw 128-d SFace feature for a 112x112 aligned crop (BGR), or None."""
    if aligned is None or aligned.ndim != 3 or active_device("faces") is None:
        return None
    n = net("faces", _asset("face_sface"), (1, 3, 112, 112))
    if n is None:
        return None
    img = aligned if aligned.shape[:2] == (112, 112) else cv2.resize(aligned, (112, 112))
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)   # swapRB like OpenCV
    try:
        return np.asarray(n(blob), np.float32).reshape(1, -1)
    except Exception:
        return None


# ----------------------------------------------------------------------------- plates
def plate_nets(det_path, ocr_path, det_size: int, ocr_shape: Tuple[int, int, int, int]) -> Optional[Tuple[OVNet, OVNet]]:
    """(detector, ocr) OpenVINO nets when plates run on an OpenVINO device and both are compiled."""
    if active_device("plates") is None:
        return None
    det = net("plates", det_path, (1, 3, det_size, det_size))
    ocr = net("plates", ocr_path, ocr_shape)      # fixes the OCR's dynamic batch dim to 1
    if det is None or ocr is None:
        return None
    return det, ocr


def ms_hint(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{math.floor(v + 0.5):.0f}"
