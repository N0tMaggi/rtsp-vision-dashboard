"""Upscaling models for the Enhance view.

* Real-ESRGAN (Xintao Wang et al., BSD-3-Clause), official PyTorch weights from
  the GitHub release assets, run with a minimal re-implementation of the two
  network architectures (SRVGGNetCompact, RRDBNet) - no basicsr/realesrgan pip
  packages needed.
* "Lanczos + sharpen": classic resampling, no AI, no download. Also the
  fallback when a model download fails.

Devices: CUDA (FP16, every tile inside ``gpu.LOCK``), OpenVINO on the CPU
(model converted from PyTorch in memory), or plain PyTorch on the CPU.
Large crops are processed in overlapping tiles so VRAM/RAM stays bounded.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .. import deps, gpu, state
from . import conf

# --- model files (downloaded on demand into models/ via deps.asset_path) -----------
_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
ASSETS = {
    "esrgan_general_x4v3": ("realesr-general-x4v3.pth", f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth"),
    "esrgan_general_wdn_x4v3": ("realesr-general-wdn-x4v3.pth", f"{_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth"),
    "esrgan_x4plus": ("RealESRGAN_x4plus.pth", f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth"),
}
for _name, _entry in ASSETS.items():   # registered next to the face/plate models (additive)
    deps.ASSETS.setdefault(_name, _entry)

ESRGAN_LICENSE = "BSD-3-Clause (Real-ESRGAN, Copyright (c) 2021 Xintao Wang)"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    short: str                       # used in the watermark band
    ai: bool
    native_scale: int
    arch: str                        # srvgg | rrdb | lanczos
    assets: tuple[str, ...] = ()
    denoise: bool = False            # supports the denoise-strength slider
    license: str = ""
    source: str = ""
    size_mb: float = 0.0
    note: str = ""


CATALOG: dict[str, ModelSpec] = {
    m.id: m
    for m in (
        ModelSpec(
            id="realesr-general-x4v3", label="Real-ESRGAN General v3", short="Real-ESRGAN general-x4v3",
            ai=True, native_scale=4, arch="srvgg",
            assets=("esrgan_general_x4v3", "esrgan_general_wdn_x4v3"), denoise=True,
            license=ESRGAN_LICENSE, source="https://github.com/xinntao/Real-ESRGAN (release v0.2.5.0)",
            size_mb=9.8, note="Fast general model for real-world photos, adjustable denoise.",
        ),
        ModelSpec(
            id="realesrgan-x4plus", label="Real-ESRGAN x4plus", short="Real-ESRGAN x4plus",
            ai=True, native_scale=4, arch="rrdb", assets=("esrgan_x4plus",),
            license=ESRGAN_LICENSE, source="https://github.com/xinntao/Real-ESRGAN (release v0.1.0)",
            size_mb=64.0, note="Larger model, sharper textures, much slower.",
        ),
        ModelSpec(
            id="lanczos", label="Lanczos + sharpen (no AI)", short="Lanczos + sharpen",
            ai=False, native_scale=1, arch="lanczos", license="-", source="OpenCV",
            note="Classic resampling. Invents nothing; baseline and download-free fallback.",
        ),
    )
}
DEFAULT_MODEL = "realesr-general-x4v3"
MODEL_IDS = tuple(CATALOG)


def catalog_payload() -> list[dict[str, Any]]:
    out = []
    for m in CATALOG.values():
        files = [deps.ASSETS[a][0] for a in m.assets]
        ready = all((deps.config.MODELS_DIR / f).is_file() for f in files)
        out.append({
            "id": m.id, "label": m.label, "ai": m.ai, "denoise": m.denoise, "license": m.license,
            "source": m.source, "size_mb": m.size_mb, "note": m.note, "ready": ready,
            "downloading": any(deps.asset_status.get(a) == "downloading" for a in m.assets),
        })
    return out


# --- architectures (re-implemented from Real-ESRGAN / BasicSR, BSD-3-Clause) ------
def _build_srvgg(torch):
    nn = torch.nn
    F = torch.nn.functional

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

    return SRVGGNetCompact()


def _build_rrdb(torch, scale=4):
    nn = torch.nn
    F = torch.nn.functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch *= 4
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = F.pixel_unshuffle(x, 2) if self.scale == 2 else x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet(scale=scale)


def _load_weights(torch, path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("params_ema", "params"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def build_torch_model(spec: ModelSpec, paths: list[Path], denoise: float):
    """CPU float32 nn.Module with the official weights (denoise blends the two general-v3 models)."""
    torch = _torch()
    if spec.arch == "srvgg":
        net = _build_srvgg(torch)
        sd = _load_weights(torch, paths[0])
        if spec.denoise and len(paths) > 1 and denoise < 0.999:
            # Real-ESRGAN "dni": strength 1 = general-x4v3 (strong denoise), 0 = wdn model (keeps noise/detail)
            weak = _load_weights(torch, paths[1])
            sd = {k: v * denoise + weak[k] * (1.0 - denoise) for k, v in sd.items()}
    elif spec.arch == "rrdb":
        net = _build_rrdb(torch, scale=4)
        sd = _load_weights(torch, paths[0])
    else:
        raise ValueError(spec.arch)
    net.load_state_dict(sd, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _torch():
    if state.torch is not None:
        return state.torch
    import torch  # noqa: PLC0415 - only when an AI model actually runs

    return torch


# --- device choice -------------------------------------------------------------------
def _openvino_ok() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("openvino") is not None
    except Exception:
        return False


def pick_device(requested: str) -> str:
    """'cuda' | 'openvino' | 'cpu' for a request (auto honours SECURITYCAM_ENHANCE_DEVICE)."""
    want = requested if requested != "auto" else conf.DEVICE
    cuda_ok = bool(state.CUDA_AVAILABLE) and str(state.DEVICE).startswith("cuda")
    if want == "cuda":
        return "cuda" if cuda_ok else ("openvino" if _openvino_ok() else "cpu")
    if want == "openvino":
        return "openvino" if _openvino_ok() else "cpu"
    if want == "cpu":
        return "cpu"
    if cuda_ok:
        return "cuda"
    return "openvino" if _openvino_ok() else "cpu"


# --- runners ---------------------------------------------------------------------------
class Cancelled(Exception):
    pass


ProgressFn = Callable[[float], None]
CancelFn = Callable[[], bool]


class Runner:
    """Upscales one BGR uint8 image by the model's native scale, tile by tile."""

    device_label = "cpu"
    native_scale = 4

    def infer(self, tile: np.ndarray) -> np.ndarray:   # float32 1x3xHxW RGB 0..1 -> same layout, scaled
        raise NotImplementedError

    def lock(self):
        import contextlib

        return contextlib.nullcontext()

    def run(self, img: np.ndarray, progress: ProgressFn, cancelled: CancelFn) -> np.ndarray:
        h, w = img.shape[:2]
        s = self.native_scale
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        out = np.zeros((h * s, w * s, 3), np.float32)
        tile, pad = conf.TILE, conf.TILE_PAD
        boxes = [(ty, tx) for ty in range(0, h, tile) for tx in range(0, w, tile)]
        for i, (ty, tx) in enumerate(boxes):
            if cancelled():
                raise Cancelled()
            y2, x2 = min(h, ty + tile), min(w, tx + tile)
            py1, px1 = max(0, ty - pad), max(0, tx - pad)
            py2, px2 = min(h, y2 + pad), min(w, x2 + pad)
            with self.lock():          # one tile per lock hold: camera inference interleaves
                res = self.infer(x[:, :, py1:py2, px1:px2])
            oy, ox = (ty - py1) * s, (tx - px1) * s
            out[ty * s:y2 * s, tx * s:x2 * s] = res[0].transpose(1, 2, 0)[oy:oy + (y2 - ty) * s, ox:ox + (x2 - tx) * s]
            progress((i + 1) / len(boxes))
        out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        pass


class TorchRunner(Runner):
    def __init__(self, net, device: str):
        torch = _torch()
        self.torch = torch
        self.device = device
        self.cuda = device == "cuda"
        self.device_label = "CPU (PyTorch)"
        self.native_scale = 4
        if self.cuda:
            with gpu.LOCK:
                self.device_label = f"CUDA ({_cuda_name()})"
                self.net = net.half().to("cuda")
        else:
            self.net = net.float()

    def lock(self):
        return gpu.LOCK if self.cuda else super().lock()

    def infer(self, tile):
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(tile))
            if self.cuda:
                t = t.to("cuda").half()
                y = self.net(t).float().cpu().numpy()
                torch.cuda.synchronize()
                return y
            return self.net(t).numpy()

    def close(self) -> None:
        if self.cuda:
            with gpu.LOCK:
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass


class OpenVinoRunner(Runner):
    def __init__(self, net):
        import openvino as ov

        torch = _torch()
        self.device_label = "OpenVINO (CPU)"
        self.native_scale = 4
        with torch.inference_mode():
            model = ov.convert_model(net, example_input=torch.zeros(1, 3, 64, 64),
                                     input=[ov.PartialShape([1, 3, -1, -1])])
        try:
            from ..compute import devices

            core = devices.core()
        except Exception:
            core = ov.Core()
        self.compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        self._req = self.compiled.create_infer_request()

    def infer(self, tile):
        self._req.infer({0: np.ascontiguousarray(tile)})
        return np.asarray(self._req.get_output_tensor(0).data).copy()


def _cuda_name() -> str:
    try:
        return _torch().cuda.get_device_name(0)
    except Exception:
        return "GPU"


_cache: "OrderedDict[tuple, Runner]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 2


def get_runner(spec: ModelSpec, paths: list[Path], denoise: float, device: str) -> Runner:
    """Cached runner (only touched from the single enhance worker thread)."""
    key = (spec.id, round(denoise, 2) if spec.denoise else None, device)
    with _cache_lock:
        r = _cache.get(key)
        if r is not None:
            _cache.move_to_end(key)
            return r
    net = build_torch_model(spec, paths, denoise)
    if device == "cuda":
        r = TorchRunner(net, "cuda")
    elif device == "openvino":
        r = OpenVinoRunner(net)
    else:
        r = TorchRunner(net, "cpu")
    with _cache_lock:
        _cache[key] = r
        while len(_cache) > _CACHE_MAX:
            _, old = _cache.popitem(last=False)
            old.close()
    return r


def lanczos(img: np.ndarray, scale: int, denoise: float = 0.0) -> np.ndarray:
    """Lanczos upscale + gentle unsharp mask (optional bilateral pre-denoise). No AI."""
    src = img
    if denoise > 0.05:
        d = int(3 + round(denoise * 4))
        src = cv2.bilateralFilter(img, d, 20 + 40 * denoise, 3 + 4 * denoise)
    up = cv2.resize(src, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0 * scale / 2)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)


def wait_for_assets(spec: ModelSpec, timeout: float, cancelled: CancelFn,
                    on_wait: Callable[[], None] | None = None) -> list[Path] | None:
    """Model files, triggering background downloads; None when a download failed or timed out."""
    deadline = time.monotonic() + timeout
    while True:
        paths = [deps.asset_path(a) for a in spec.assets]
        if all(p is not None for p in paths):
            return [p for p in paths if p is not None]
        if any(deps.asset_status.get(a) == "failed" for a in spec.assets):
            return None
        if time.monotonic() > deadline:
            return None
        if cancelled():
            raise Cancelled()
        if on_wait is not None:
            on_wait()
        time.sleep(0.25)


@dataclass
class Result:
    image: np.ndarray
    model: ModelSpec
    device: str
    fallback: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


def enhance(img: np.ndarray, model_id: str, scale: int, denoise: float, device: str,
            progress: ProgressFn, cancelled: CancelFn, on_stage: Callable[[str], None]) -> Result:
    """Run one enhancement. Falls back to Lanczos (marked as such) when the AI model is unavailable."""
    spec = CATALOG[model_id]
    timings: dict[str, float] = {}
    if not spec.ai:
        t0 = time.perf_counter()
        out = lanczos(img, scale, denoise)
        timings["infer_s"] = time.perf_counter() - t0
        progress(1.0)
        return Result(out, spec, "CPU (OpenCV)", None, timings)

    t0 = time.perf_counter()
    on_stage("downloading")
    paths = wait_for_assets(spec, conf.DOWNLOAD_WAIT_S, cancelled)
    timings["download_s"] = time.perf_counter() - t0
    if paths is None:
        out = lanczos(img, scale, denoise * 0.5)
        progress(1.0)
        return Result(out, CATALOG["lanczos"], "CPU (OpenCV)", "model_unavailable", timings)

    dev = pick_device(device)
    on_stage("loading")
    t1 = time.perf_counter()
    fallback = None
    try:
        runner = get_runner(spec, paths, denoise, dev)
    except Exception as exc:   # OpenVINO conversion / CUDA init failed -> PyTorch CPU
        print(f"[ENHANCE] {spec.id} on {dev} unavailable ({type(exc).__name__}: {exc}); using PyTorch CPU.")
        dev, fallback = "cpu", f"{dev}_failed"
        runner = get_runner(spec, paths, denoise, dev)
    timings["load_s"] = time.perf_counter() - t1

    on_stage("running")
    t2 = time.perf_counter()
    try:
        out = runner.run(img, progress, cancelled)
    finally:
        if dev == "cuda":
            runner.close()
    timings["infer_s"] = time.perf_counter() - t2
    if scale != runner.native_scale:
        out = cv2.resize(out, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_AREA)
    return Result(out, spec, runner.device_label, fallback, timings)
