"""Honesty marking for enhanced images.

Every exported enhanced image carries

* a visible band appended below the picture: "AI ENHANCED · <model> · <scale>×"
  plus "details may be invented — not evidence" (the band adds rows, it never
  covers picture content; the on-screen preview may hide it, exports never do),
* a small "AI" tag inside the top-left corner of the picture itself, so a plain
  crop of the band still leaves a mark,
* PNG text chunks (Software, Description, model, scale, source camera, source
  timestamp, region ...), mirrored in the sidecar ``meta.json``.
"""
from __future__ import annotations

import struct
import zlib
from typing import Iterable

import cv2
import numpy as np

SOFTWARE = "SecurityCam Enhance"
DESCRIPTION = "AI-upscaled image — details may be invented, not evidence"
NOTICE = "AI upscaling invents plausible detail. Use for presentation and exports, not as evidence."
ORIGINAL_DESCRIPTION = "Original camera pixels (cropped, not enhanced)"

_BG = (39, 36, 34)            # #222427 (BGR)
_YELLOW = (31, 210, 255)      # #ffd21f
_WHITE = (242, 244, 244)      # #f4f4f2
_MUTED = (160, 160, 156)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def band_height(width: int) -> int:
    return int(min(160, max(30, round(width / 26))))


def band_text(model_label: str, scale: int, ai: bool = True) -> str:
    head = "AI ENHANCED" if ai else "UPSCALED (NO AI)"
    return f"{head} · {model_label} · {scale}x"   # PIL's bundled font has no U+00D7


def _font(px: int):
    """PIL's bundled scalable font, or None (-> OpenCV Hershey, ASCII only)."""
    try:
        from PIL import ImageFont

        return ImageFont.load_default(size=max(8, px))
    except Exception:
        return None


def _draw_texts(img: np.ndarray, items: Iterable[tuple[str, tuple[int, int], int, tuple[int, int, int], str]]) -> np.ndarray:
    """Draw (text, (x, y_center), px, bgr, anchor 'l'|'r') items. Returns the image (maybe a new array)."""
    items = list(items)
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, (x, y), px, bgr, anchor in items:
            font = _font(px)
            if font is None:
                raise RuntimeError("no font")
            draw.text((x, y), text, font=font, fill=(bgr[2], bgr[1], bgr[0]), anchor="lm" if anchor == "l" else "rm")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        for text, (x, y), px, bgr, anchor in items:
            ascii_text = text.replace("·", "|").replace("×", "x").replace("—", "-")
            ascii_text = ascii_text.encode("ascii", "replace").decode()
            fs = px / 30.0
            (tw, th), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ox = x if anchor == "l" else x - tw
            cv2.putText(img, ascii_text, (int(ox), int(y + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, fs, bgr, 1, cv2.LINE_AA)
        return img


def _text_width(text: str, px: int) -> int:
    font = _font(px)
    if font is not None:
        try:
            return int(font.getlength(text))
        except Exception:
            pass
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, 1)
    return tw


def add_corner_tag(img: np.ndarray, ai: bool = True) -> np.ndarray:
    """Small semi-transparent "AI" / "UPSCALED" tag inside the top-left corner (in place, returned)."""
    h, w = img.shape[:2]
    px = int(min(28, max(10, round(min(w, h) / 30))))
    label = "AI" if ai else "UPSCALED"
    pad = max(3, px // 3)
    tw = _text_width(label, px)
    x1, y1 = pad, pad
    x2, y2 = min(w - 1, x1 + tw + 2 * pad), min(h - 1, y1 + px + pad)
    if x2 <= x1 or y2 <= y1:
        return img
    roi = img[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = _YELLOW
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.25, tint, 0.75, 0)
    return _draw_texts(img, [(label, (x1 + pad, (y1 + y2) // 2), px, _BG, "l")])


def add_band(img: np.ndarray, model_label: str, scale: int, ai: bool = True) -> tuple[np.ndarray, int]:
    """Return (image with the band appended below, band height in px)."""
    h, w = img.shape[:2]
    bh = band_height(w)
    band = np.empty((bh, w, 3), np.uint8)
    band[:] = _BG
    band[:max(1, bh // 16), :] = _YELLOW            # thin yellow rule on top of the band
    chip_w = int(bh * 0.42)
    band[bh // 16:, :chip_w] = _YELLOW               # yellow chip on the left
    out = np.vstack([img, band])
    main_px = int(bh * 0.42)
    small_px = int(bh * 0.3)
    main = band_text(model_label, scale, ai)
    tail = "details may be invented - not evidence"
    y = h + bh // 2 + bh // 32
    items = [(main, (chip_w + int(bh * 0.35), y), main_px, _WHITE, "l")]
    # the disclaimer only when it fits next to the main label
    if chip_w + int(bh * 0.35) + _text_width(main, main_px) + _text_width(tail, small_px) + bh < w:
        items.append((tail, (w - int(bh * 0.35), y), small_px, _MUTED, "r"))
    return _draw_texts(out, items), bh


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _latin1(text: str) -> bytes:
    text = text.replace("—", "-").replace("–", "-").replace("·", "|")
    return text.encode("latin-1", "replace")


def png_with_text(png: bytes, text: dict[str, str]) -> bytes:
    """Insert tEXt chunks right after IHDR (keywords: 1-79 Latin-1 chars)."""
    if not png.startswith(_PNG_SIG):
        raise ValueError("not a PNG")
    ihdr_len = struct.unpack(">I", png[8:12])[0]
    cut = 8 + 12 + ihdr_len
    chunks = b"".join(
        _chunk(b"tEXt", _latin1(k)[:79] + b"\x00" + _latin1(str(v))) for k, v in text.items() if k
    )
    return png[:cut] + chunks + png[cut:]


def read_png_text(png: bytes) -> dict[str, str]:
    """tEXt chunks of a PNG (for tests and the history view)."""
    out: dict[str, str] = {}
    if not png.startswith(_PNG_SIG):
        return out
    pos = 8
    while pos + 12 <= len(png):
        length = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + length]
        if kind == b"tEXt" and b"\x00" in data:
            k, v = data.split(b"\x00", 1)
            out[k.decode("latin-1")] = v.decode("latin-1")
        if kind == b"IEND":
            break
        pos += 12 + length
    return out


def encode_png(img: np.ndarray, text: dict[str, str]) -> bytes:
    ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise ValueError("png_encode_failed")
    return png_with_text(buf.tobytes(), text)


def mark_export(enhanced: np.ndarray, model_label: str, scale: int, text: dict[str, str], ai: bool = True) -> tuple[bytes, int]:
    """Corner tag + band + PNG text. Returns (png bytes, band height)."""
    tagged = add_corner_tag(np.ascontiguousarray(enhanced).copy(), ai)
    banded, bh = add_band(tagged, model_label, scale, ai)
    return encode_png(banded, text), bh
