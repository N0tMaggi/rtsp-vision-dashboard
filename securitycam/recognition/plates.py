"""License plates: YOLOv9-t plate detector + fast-plate-ocr CCT model (both ONNX, MIT).

Inference runs through OpenCV's DNN module (no extra package needed with
OpenCV >= 5). If OpenCV can't run a model, ``onnxruntime`` is used when it is
installed. Models are downloaded into models/ via ``deps.asset_path``.
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import deps
from .common import PlateRead

DET_CONF = 0.35
MIN_PLATE_CHARS = 4
MIN_OCR_CONF = 0.55
ORT_INSTALL_CMD = "python -m pip install onnxruntime"

_local = threading.local()
_status: Dict[str, str] = {}          # "detector" / "ocr" -> "" or error text


def _ort_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


class _Net:
    """Single-input/single-output ONNX model via cv2.dnn, falling back to onnxruntime."""

    def __init__(self, path: str, dummy: np.ndarray, check_finite: bool = False):
        self.backend = None
        self._net = self._sess = None
        try:
            net = cv2.dnn.readNet(path)
            net.setInput(dummy)
            out = np.asarray(net.forward())
            if check_finite and not np.all(np.isfinite(out)):
                raise ValueError("non-finite output")    # some transformer ops are wrong in OpenCV DNN
            self._net, self.backend = net, "opencv"
            return
        except (cv2.error, ValueError) as exc:
            err = str(exc).splitlines()[0][:160]
        if _ort_available():
            import onnxruntime as ort

            self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._in = self._sess.get_inputs()[0].name
            self.backend = "onnxruntime"
            return
        raise RuntimeError(f"OpenCV DNN cannot run {path}: {err}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._net is not None:
            self._net.setInput(x)
            return np.asarray(self._net.forward())
        return np.asarray(self._sess.run(None, {self._in: x})[0])


def _load_yaml(path) -> Dict[str, Any]:
    """Tiny parser for the flat fast-plate-ocr config (avoids a hard PyYAML dependency)."""
    try:
        import yaml  # type: ignore

        with open(path, encoding="utf-8") as fh:
            return dict(yaml.safe_load(fh) or {})
    except ImportError:
        pass
    out: Dict[str, Any] = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if val.startswith("'") and val.endswith("'") or val.startswith('"') and val.endswith('"'):
            out[key.strip()] = val[1:-1]
        elif val.lower() in ("true", "false"):
            out[key.strip()] = val.lower() == "true"
        else:
            try:
                out[key.strip()] = int(val)
            except ValueError:
                out[key.strip()] = val
    return out


class PlateReader:
    def __init__(self, det_path, ocr_path, cfg_path):
        cfg = _load_yaml(cfg_path)
        self.alphabet: str = str(cfg["alphabet"])
        self.pad_char: str = str(cfg.get("pad_char", "_"))
        self.slots = int(cfg["max_plate_slots"])
        self.h, self.w = int(cfg["img_height"]), int(cfg["img_width"])
        self.rgb = str(cfg.get("image_color_mode", "grayscale")).lower() == "rgb"
        self.det_size = 384
        self.det = _Net(str(det_path), np.zeros((1, 3, self.det_size, self.det_size), np.float32))
        dummy = np.full((1, self.h, self.w, 3 if self.rgb else 1), 128, np.uint8)
        self.ocr = _Net(str(ocr_path), dummy, check_finite=True)

    # ---- detection -------------------------------------------------------------
    def _letterbox(self, img: np.ndarray):
        s = self.det_size
        h, w = img.shape[:2]
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw, dh = (s - nw) / 2, (s - nh) / 2
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return x[None], r, (dw, dh)

    def detect(self, img: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Plate boxes (x1, y1, x2, y2) in ``img`` pixels with confidence, best first."""
        if img is None or img.size == 0:
            return []
        x, r, (dw, dh) = self._letterbox(img)
        try:
            pred = self.det(np.ascontiguousarray(x))
        except cv2.error:
            return []    # end-to-end NMS raises on frames without any plate in OpenCV DNN
        pred = pred.reshape(-1, pred.shape[-1]) if pred.ndim >= 2 else pred.reshape(1, -1)
        if pred.shape[-1] < 7:
            return []
        h, w = img.shape[:2]
        out = []
        for row in pred:
            score = float(row[6])
            if not np.isfinite(score) or score < DET_CONF:
                continue
            x1 = int(max(0, (row[1] - dw) / r))
            y1 = int(max(0, (row[2] - dh) / r))
            x2 = int(min(w, (row[3] - dw) / r))
            y2 = int(min(h, (row[4] - dh) / r))
            if x2 - x1 >= 8 and y2 - y1 >= 4:
                out.append(((x1, y1, x2, y2), score))
        out.sort(key=lambda b: -b[1])
        return out

    # ---- OCR -------------------------------------------------------------------
    def read(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """(text, confidence) for a cropped plate image (BGR)."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        if self.rgb:
            img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[:, :, None]
        try:
            out = self.ocr(np.ascontiguousarray(img[None].astype(np.uint8)))
        except cv2.error:
            return "", 0.0
        probs = out.reshape(self.slots, -1)
        if not np.all(np.isfinite(probs)):
            return "", 0.0
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        chars = [self.alphabet[i] if i < len(self.alphabet) else self.pad_char for i in idx]
        keep = [(c, p) for c, p in zip(chars, conf) if c != self.pad_char]
        if not keep:
            return "", 0.0
        text = "".join(c for c, _ in keep)
        return text, float(np.mean([p for _, p in keep]))

    def read_vehicle(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[PlateRead]:
        """Find + read the best plate inside a vehicle box (frame pixels)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * 0.04), int((y2 - y1) * 0.04)
        x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
        if x2 - x1 < 48 or y2 - y1 < 32:
            return None
        crop = frame[y1:y2, x1:x2]
        found = self.detect(crop)
        if not found:
            return None
        (bx1, by1, bx2, by2), _ = found[0]
        # small margin around the plate helps the OCR model
        mx, my = int((bx2 - bx1) * 0.06), int((by2 - by1) * 0.12)
        bx1, by1 = max(0, bx1 - mx), max(0, by1 - my)
        bx2, by2 = min(crop.shape[1], bx2 + mx), min(crop.shape[0], by2 + my)
        text, conf = self.read(crop[by1:by2, bx1:bx2])
        if len(text) < MIN_PLATE_CHARS or conf < MIN_OCR_CONF:
            return None
        return PlateRead(text=text, conf=conf, box=(x1 + bx1, y1 + by1, x1 + bx2, y1 + by2))


def get_reader() -> Optional[PlateReader]:
    """Thread-local reader, or None while models download / when they can't run."""
    reader = getattr(_local, "reader", None)
    if reader is not None:
        return reader
    if getattr(_local, "failed", False):
        return None
    paths = [deps.asset_path(k) for k in ("plate_detector", "plate_ocr", "plate_ocr_config")]
    if any(p is None for p in paths):
        return None
    try:
        reader = PlateReader(*paths)
    except Exception as exc:   # cv2.error, RuntimeError, KeyError (bad config) ...
        _status["error"] = str(exc)
        _local.failed = True
        print(f"[RECOG] License plate models unavailable: {exc}")
        return None
    _status["error"] = ""
    _status["backend"] = reader.det.backend or ""
    _local.reader = reader
    return reader


def status() -> Dict[str, Any]:
    names = ("plate_detector", "plate_ocr", "plate_ocr_config")
    assets = {n: deps.asset_status.get(n, "missing") for n in names}
    for n in names:
        if (deps.config.MODELS_DIR / deps.ASSETS[n][0]).is_file():
            assets[n] = "ready"
    error = _status.get("error", "")
    return {
        "available": not error,
        "backend": _status.get("backend") or None,
        "assets": assets,
        "error": error or None,
        "install_cmd": ORT_INSTALL_CMD if error else None,
    }


# ----------------------------------------------------------------------------- voting
def normalize_plate(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalnum())


def vote(votes: Dict[str, float], text: str, conf: float) -> Tuple[Optional[str], float]:
    """Add one read to ``votes`` (text -> summed confidence) and return (best, share).

    Similar reads (same length, <= 1 differing character) also support each
    other with half weight, so one misread character doesn't split the vote.
    """
    text = normalize_plate(text)
    if len(text) >= MIN_PLATE_CHARS and conf > 0:
        votes[text] = votes.get(text, 0.0) + float(conf)
    if not votes:
        return None, 0.0

    def support(t: str) -> float:
        s = votes[t]
        for other, w in votes.items():
            if other != t and len(other) == len(t) and sum(a != b for a, b in zip(other, t)) <= 1:
                s += 0.5 * w
        return s

    best = max(votes, key=lambda t: (support(t), votes[t], t))
    total = sum(votes.values())
    return best, votes[best] / total if total > 0 else 0.0
