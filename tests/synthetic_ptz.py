"""Synthetic PTZ camera for tests: renders exact pinhole views of a random panorama.

The panorama is equirectangular (azimuth -180..180 deg on x, elevation 90..-90
deg on y). ``SyntheticPtz.render(pose)`` returns what an ideal rotating
pinhole camera with the given lens would see, so the ground-truth pose of
every frame is known.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from securitycam import geo


def make_panorama(width: int = 3600, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = width // 2
    pano = np.zeros((h, width, 3), np.float32)
    # multi-scale smooth noise
    for scale, amp in ((8, 60.0), (32, 50.0), (128, 40.0)):
        small = rng.random((max(2, h // scale), max(2, width // scale), 3)).astype(np.float32)
        pano += cv2.resize(small, (width, h), interpolation=cv2.INTER_CUBIC) * amp
    # crisp shapes (corners for tracking, landmarks for the eye)
    img = np.clip(pano + 40, 0, 255).astype(np.uint8)
    for _ in range(width // 6):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            s = int(rng.integers(6, 40))
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.5, 1.5))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(4, 22)), c, -1)
    for _ in range(width // 30):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, h)))
        p2 = (p1[0] + int(rng.integers(-200, 200)), p1[1] + int(rng.integers(-120, 120)))
        cv2.line(img, p1, p2, tuple(int(v) for v in rng.integers(0, 256, 3)), int(rng.integers(1, 4)))
    return img


class SyntheticPtz:
    def __init__(self, pano: np.ndarray, size=(640, 360), lens: geo.Lens = geo.Lens(100.0, 56.0)):
        self.pano = pano
        self.ph, self.pw = pano.shape[:2]
        self.size = size
        self.lens = lens
        self._grid_cache = {}
        # wrap columns so remap interpolates across the +-180 seam
        self._pano_wrapped = np.concatenate([pano[:, -2:], pano, pano[:, :2]], axis=1)

    def _cam_rays(self, zoom: float) -> np.ndarray:
        key = round(zoom, 6)
        rays = self._grid_cache.get(key)
        if rays is None:
            w, h = self.size
            fx, fy = self.lens.focal(zoom)
            xs = (np.arange(w) + 0.5) / w
            ys = (np.arange(h) + 0.5) / h
            u, v = np.meshgrid(xs, ys)
            rays = np.stack([(u - 0.5) / fx, (0.5 - v) / fy, np.ones_like(u)], axis=-1).reshape(-1, 3)
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            self._grid_cache[key] = rays
        return rays

    def render(self, pose: geo.Pose) -> np.ndarray:
        w, h = self.size
        world = self._cam_rays(pose.zoom) @ geo.rotation(pose).T
        az = np.arctan2(world[:, 0], world[:, 2])
        el = np.arcsin(np.clip(world[:, 1], -1, 1))
        mx = ((az + math.pi) / (2 * math.pi) * self.pw - 0.5 + 2).astype(np.float32).reshape(h, w)
        my = ((math.pi / 2 - el) / math.pi * self.ph - 0.5).astype(np.float32).reshape(h, w)
        return cv2.remap(self._pano_wrapped, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def project_px(self, rays, pose: geo.Pose):
        uv, front = geo.rays_to_image(rays, pose, self.lens)
        return uv * np.array(self.size, dtype=np.float64), front


def draw_person(frame: np.ndarray, x: int, y: int, scale: float = 1.0, color=(40, 40, 200)) -> tuple:
    """Paint a moving foreground blob (a "person") with texture; returns its normalized box."""
    h, w = frame.shape[:2]
    bw, bh = int(40 * scale), int(110 * scale)
    x1, y1 = max(0, x - bw // 2), max(0, y - bh)
    x2, y2 = min(w, x + bw // 2), min(h, y)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = color
        for k in range(y1, y2, 9):
            cv2.line(frame, (x1, k), (x2, k + 4), (230, 230, 230), 2)
        cv2.circle(frame, (x, max(0, y1 - int(14 * scale))), int(14 * scale), (180, 200, 230), -1)
    return (x1 / w, y1 / h, x2 / w, y2 / h)
