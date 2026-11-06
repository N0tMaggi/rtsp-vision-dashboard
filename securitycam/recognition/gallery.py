"""Local known-faces gallery: ``data/faces/<id>/`` with meta.json, embeddings and thumbnails.

Nothing here talks to the network. ``match()`` works on an in-memory matrix
that is rebuilt only when the gallery changes.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .common import DATA_DIR

FACES_DIR = DATA_DIR / "faces"
NAME_MAX = 40
MAX_SAMPLES = 24          # per person; oldest samples are dropped beyond this

_lock = threading.RLock()
_cache: Optional[Tuple[np.ndarray, List[str], List[str]]] = None   # (matrix, person_ids, names)
_version = 0


class GalleryError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def clean_name(raw: Any) -> str:
    if not isinstance(raw, str):
        raise GalleryError("invalid_name")
    name = " ".join(raw.split())
    if not name or len(name) > NAME_MAX:
        raise GalleryError("invalid_name")
    return name


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug[:40] or "person"


def _valid_id(person_id: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", person_id or ""))


def _person_dir(person_id: str) -> Path:
    if not _valid_id(person_id):
        raise GalleryError("unknown_person")
    return FACES_DIR / person_id


def _read_meta(folder: Path) -> Optional[Dict[str, Any]]:
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and isinstance(meta.get("name"), str) else None


def _write_meta(folder: Path, meta: Dict[str, Any]) -> None:
    tmp = folder / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(folder / "meta.json")


def _invalidate() -> None:
    global _cache, _version
    _cache = None
    _version += 1


def version() -> int:
    return _version


def _samples(folder: Path) -> List[int]:
    out = []
    for p in folder.glob("emb_*.npy"):
        try:
            out.append(int(p.stem.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(out)


def list_people() -> List[Dict[str, Any]]:
    with _lock:
        if not FACES_DIR.is_dir():
            return []
        people = []
        for folder in sorted(FACES_DIR.iterdir()):
            if not folder.is_dir() or not _valid_id(folder.name):
                continue
            meta = _read_meta(folder)
            if meta is None:
                continue
            samples = _samples(folder)
            people.append({
                "id": folder.name,
                "name": meta["name"],
                "samples": len(samples),
                "created": meta.get("created", 0),
                "updated": meta.get("updated", 0),
                "thumb": f"/api/recognition/people/{folder.name}/thumb?v={samples[-1] if samples else 0}",
            })
        people.sort(key=lambda p: p["name"].lower())
        return people


def find_by_name(name: str) -> Optional[str]:
    target = name.casefold()
    for p in list_people():
        if p["name"].casefold() == target:
            return p["id"]
    return None


def add_sample(name: str, embedding: np.ndarray, thumb: Optional[np.ndarray]) -> Dict[str, Any]:
    """Add one face sample; creates the person if ``name`` is new (case-insensitive)."""
    name = clean_name(name)
    emb = np.asarray(embedding, np.float32).reshape(-1)
    if emb.size < 16 or not np.all(np.isfinite(emb)):
        raise GalleryError("invalid_embedding")
    with _lock:
        person_id = find_by_name(name)
        now = time.time()
        if person_id is None:
            base = _slug(name)
            person_id, n = base, 2
            while (FACES_DIR / person_id).exists():
                person_id = f"{base}-{n}"
                n += 1
            folder = FACES_DIR / person_id
            folder.mkdir(parents=True, exist_ok=True)
            meta = {"name": name, "created": now, "updated": now}
        else:
            folder = FACES_DIR / person_id
            meta = _read_meta(folder) or {"name": name, "created": now}
            meta["updated"] = now
        samples = _samples(folder)
        idx = (samples[-1] + 1) if samples else 1
        np.save(folder / f"emb_{idx}.npy", emb / max(float(np.linalg.norm(emb)), 1e-9))
        if thumb is not None and thumb.size:
            cv2.imwrite(str(folder / f"thumb_{idx}.jpg"), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        for old in samples[: max(0, len(samples) + 1 - MAX_SAMPLES)]:
            (folder / f"emb_{old}.npy").unlink(missing_ok=True)
            (folder / f"thumb_{old}.jpg").unlink(missing_ok=True)
        _write_meta(folder, meta)
        _invalidate()
    return next(p for p in list_people() if p["id"] == person_id)


def rename(person_id: str, new_name: str) -> Dict[str, Any]:
    new_name = clean_name(new_name)
    with _lock:
        folder = _person_dir(person_id)
        meta = _read_meta(folder) if folder.is_dir() else None
        if meta is None:
            raise GalleryError("unknown_person")
        other = find_by_name(new_name)
        if other is not None and other != person_id:
            raise GalleryError("name_taken")
        meta["name"] = new_name
        meta["updated"] = time.time()
        _write_meta(folder, meta)
        _invalidate()
    return next(p for p in list_people() if p["id"] == person_id)


def delete(person_id: str) -> bool:
    with _lock:
        folder = _person_dir(person_id)
        if not folder.is_dir():
            return False
        shutil.rmtree(folder, ignore_errors=True)
        _invalidate()
        return True


def thumb_path(person_id: str) -> Optional[Path]:
    folder = _person_dir(person_id)
    thumbs = sorted(folder.glob("thumb_*.jpg"), key=lambda p: p.stat().st_mtime)
    return thumbs[-1] if thumbs else None


def _matrix() -> Tuple[np.ndarray, List[str], List[str]]:
    global _cache
    cache = _cache
    if cache is not None:
        return cache
    with _lock:
        if _cache is not None:
            return _cache
        rows, ids, names = [], [], []
        if FACES_DIR.is_dir():
            for folder in FACES_DIR.iterdir():
                if not folder.is_dir() or not _valid_id(folder.name):
                    continue
                meta = _read_meta(folder)
                if meta is None:
                    continue
                for idx in _samples(folder):
                    try:
                        v = np.load(folder / f"emb_{idx}.npy").astype(np.float32).reshape(-1)
                    except (OSError, ValueError):
                        continue
                    if rows and v.shape != rows[0].shape:
                        continue
                    rows.append(v)
                    ids.append(folder.name)
                    names.append(meta["name"])
        mat = np.stack(rows) if rows else np.zeros((0, 128), np.float32)
        _cache = (mat, ids, names)
        return _cache


def match(embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], Optional[str], float]:
    """(name, person_id, cosine) of the best gallery match, name/id None below ``threshold``."""
    mat, ids, names = _matrix()
    if embedding is None or mat.shape[0] == 0 or mat.shape[1] != embedding.size:
        return None, None, 0.0
    sims = mat @ embedding.reshape(-1)
    i = int(np.argmax(sims))
    score = float(sims[i])
    if score >= threshold:
        return names[i], ids[i], score
    return None, None, score


def size() -> int:
    return int(_matrix()[0].shape[0])
