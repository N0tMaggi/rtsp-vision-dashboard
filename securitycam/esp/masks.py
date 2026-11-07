"""Instance-segmentation masks -> small box-relative alpha crops.

Segment models (``*-seg.pt``) return one binary mask per detection at the
letterboxed inference resolution (e.g. 384x640 for a 16:9 frame at imgsz 640).
The worker already moves it to host memory inside the CUDA-locked section
(``Results.cpu()``), so nothing here touches the GPU.

Per detection we keep only the part of the mask under its box, resampled into
a small uint8 crop (<= ``MASK_MAX_SIDE`` px, values 0..255, slightly blurred so
the staircase of the binary mask turns into a smooth contour). The crop is in
*box-relative* coordinates: drawing it means resizing it onto whatever box the
motion model displays for this frame - the same offset + scale transform
``motion._snapshot`` applies to keypoints - so the silhouette sticks to the
moving person at render fps between inferences.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

MASK_MAX_SIDE = 160        # crops never exceed this many px per side (memory / extraction cost)
MASK_MIN_SIDE = 2


class MaskSource:
    """Mask stack of one result plus its letterbox transform (frame px -> mask px)."""

    __slots__ = ("data", "gain", "pad_x", "pad_y")

    def __init__(self, data: np.ndarray, gain: float, pad_x: float, pad_y: float):
        self.data = data
        self.gain = gain
        self.pad_x = pad_x
        self.pad_y = pad_y


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def prepare(result: Any) -> Optional[MaskSource]:
    """``MaskSource`` for an Ultralytics result with masks, else ``None`` (detect / pose models)."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    data = _to_numpy(getattr(masks, "data", None))
    if data is None or data.ndim != 3 or data.shape[0] == 0:
        return None
    orig = getattr(result, "orig_shape", None) or getattr(masks, "orig_shape", None)
    mh, mw = data.shape[1:]
    if not orig:
        oh, ow = mh, mw
    else:
        oh, ow = int(orig[0]), int(orig[1])
    if oh <= 0 or ow <= 0:
        return None
    # Same letterbox math as ultralytics.utils.ops.scale_boxes (retina masks: gain 1, no pad).
    gain = min(mh / oh, mw / ow)
    pad_x = round((mw - ow * gain) / 2 - 0.1)
    pad_y = round((mh - oh * gain) / 2 - 0.1)
    return MaskSource(data, gain, pad_x, pad_y)


def crop(src: MaskSource, index: int, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Box-relative uint8 alpha crop of mask ``index`` for ``box`` (frame px x1, y1, x2, y2)."""
    if src is None or index < 0 or index >= src.data.shape[0]:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    g = src.gain
    mx1, my1 = x1 * g + src.pad_x, y1 * g + src.pad_y
    mx2, my2 = x2 * g + src.pad_x, y2 * g + src.pad_y
    bw, bh = mx2 - mx1, my2 - my1
    if not (bw > 0.5 and bh > 0.5):
        return None
    mh, mw = src.data.shape[1:]
    # integer window around the box (+1 px for the bilinear taps)
    ix0, iy0 = max(0, int(mx1) - 1), max(0, int(my1) - 1)
    ix1, iy1 = min(mw, int(np.ceil(mx2)) + 1), min(mh, int(np.ceil(my2)) + 1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    window = src.data[index, iy0:iy1, ix0:ix1]
    window = (window > 0.5).astype(np.uint8) * np.uint8(255)
    if not window.any():
        return None
    cw = int(min(MASK_MAX_SIDE, max(MASK_MIN_SIDE, round(bw))))
    ch = int(min(MASK_MAX_SIDE, max(MASK_MIN_SIDE, round(bh))))
    sx, sy = bw / cw, bh / ch
    # crop px (u, v) center -> mask index coords (pixel i covers [i, i+1))
    m = np.array([[sx, 0.0, mx1 - ix0 + 0.5 * sx - 0.5],
                  [0.0, sy, my1 - iy0 + 0.5 * sy - 0.5]], np.float32)
    out = cv2.warpAffine(window, m, (cw, ch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if min(cw, ch) >= 6:
        out = cv2.GaussianBlur(out, (3, 3), 0.8)
    return out
