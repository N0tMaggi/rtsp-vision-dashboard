"""Automatic face ids: unknown faces clustered online into persistent "Face #n" identities.

When Known Faces is on, every face the gallery does not recognize already has
an SFace embedding (``worker.recognize_frame``). This module clusters those
embeddings into *unknown identities* that keep their number across restarts:

* **Match**: cosine similarity against each identity's centroid and its few
  stored samples (best of both). ``face_cluster_threshold`` (default 0.45) is
  stricter than the name threshold, so two strangers rarely share an id. A
  face that belongs to a tracked person who already has an id gets a little
  slack (temporal consistency).
* **New identities** start as *pending* and only get a number after
  ``CONFIRM_HITS`` consistent sightings within ``PENDING_TTL_S`` (one blurry
  frame never creates "Face #93").
* **Learning**: only faces that are large enough, confidently detected and
  sharp (Laplacian variance) update a centroid (running mean, capped weight)
  or add a sample (kept diverse). Smaller faces can still *match*.
* **Merge**: when two centroids converge (similarity >= threshold + 0.15) the
  younger identity is folded into the older one.
* **Retention**: identities seen in only one sighting are pruned after
  ``RETENTION_DAYS``; at most ``MAX_IDENTITIES`` are kept.

Storage (local only) under ``data/faces/unknown/``::

    index.json             numbering + metadata (first / last seen, sightings)
    <uid>/samples.npy      up to MAX_SAMPLES L2-normalized embeddings
    <uid>/centroid.npy     running centroid
    <uid>/thumb.jpg        best face crop so far

All work happens in the recognition worker (``label_hits``) and the HTTP
handlers, never in the render path.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

CONFIRM_HITS = 3            # consistent sightings before a pending face gets a number
PENDING_TTL_S = 20.0
MAX_PENDING = 32
MAX_SAMPLES = 6
SAMPLE_DIVERSITY = 0.92     # a new sample must be less similar than this to every stored one
CENTROID_CAP = 30           # running-mean weight floor 1/30: identities keep adapting slowly
MERGE_MARGIN = 0.15         # centroids closer than threshold + this are merged
HINT_SLACK = 0.08           # same track -> slightly lower threshold
SIGHTING_GAP_S = 60.0       # unseen this long -> the next match counts as a new sighting
MAX_IDENTITIES = 300
RETENTION_DAYS = 3.0
SAVE_INTERVAL_S = 5.0
MIN_LEARN_PX = 48           # face width needed to learn (match works from MIN_RECOGNIZABLE_PX)
MIN_LEARN_SCORE = 0.8       # YuNet score needed to learn
MIN_SHARPNESS = 20.0        # Laplacian variance of the 112 px crop needed to learn
UID_MAX = 1_000_000


class UnknownFaceError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _norm(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return None
    return v / n


def sharpness(img: Optional[np.ndarray]) -> float:
    """Variance of the Laplacian (higher = sharper); 0 for empty images."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class Identity:
    uid: int
    centroid: np.ndarray
    samples: List[np.ndarray] = field(default_factory=list)
    n: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    sightings: int = 1
    hits: int = 1
    thumb_q: float = -1.0
    dirty_emb: bool = True
    thumb: Optional[np.ndarray] = None       # pending write

    def similarity(self, e: np.ndarray) -> float:
        best = float(self.centroid @ e)
        for s in self.samples:
            best = max(best, float(s @ e))
        return best

    def meta(self) -> Dict[str, Any]:
        return {"uid": self.uid, "n": self.n, "first_seen": round(self.first_seen, 3),
                "last_seen": round(self.last_seen, 3), "sightings": self.sightings, "hits": self.hits,
                "thumb_q": round(self.thumb_q, 4)}


@dataclass
class _Pending:
    centroid: np.ndarray
    samples: List[np.ndarray]
    hits: int
    first: float
    last: float
    thumb: Optional[np.ndarray]
    thumb_q: float


class UnknownFaces:
    def __init__(self, root: Optional[Path] = None, clock: Callable[[], float] = time.time):
        self._root = Path(root) if root is not None else None
        self.clock = clock
        self._lock = threading.RLock()
        self._ids: Dict[int, Identity] = {}
        self._pending: List[_Pending] = []
        self._next_uid = 1
        self._loaded = False
        self._index_dirty = False
        self._last_save = 0.0
        self._last_prune = 0.0
        self._deleted: List[int] = []
        self.version = 0
        self.merges: List[Tuple[int, int]] = []    # (from, into) since the last drain (tags follow merges)

    # ------------------------------------------------------------------ storage
    @property
    def root(self) -> Path:
        if self._root is None:
            from . import gallery          # resolved lazily so tests can point FACES_DIR elsewhere
            return gallery.FACES_DIR / "unknown"
        return self._root

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                doc = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                return
            except (OSError, ValueError) as exc:
                print(f"[RECOG] Unknown faces index unreadable ({exc.__class__.__name__}); starting empty.")
                return
            if not isinstance(doc, dict):
                return
            self._next_uid = max(1, int(doc.get("next_uid") or 1))
            for m in doc.get("faces") or []:
                try:
                    uid = int(m["uid"])
                    if not 0 < uid < UID_MAX:
                        continue
                    d = self.root / str(uid)
                    cen = _norm(np.load(d / "centroid.npy"))
                    samples = np.load(d / "samples.npy").astype(np.float32)
                    if cen is None:
                        continue
                    samples = [s for s in (_norm(r) for r in samples.reshape(-1, cen.size)) if s is not None]
                    self._ids[uid] = Identity(
                        uid=uid, centroid=cen, samples=samples[:MAX_SAMPLES], n=int(m.get("n", 1)),
                        first_seen=float(m.get("first_seen", 0)), last_seen=float(m.get("last_seen", 0)),
                        sightings=int(m.get("sightings", 1)), hits=int(m.get("hits", 1)),
                        thumb_q=float(m.get("thumb_q", -1)), dirty_emb=False)
                    self._next_uid = max(self._next_uid, uid + 1)
                except (KeyError, TypeError, ValueError, OSError):
                    continue

    def save(self, force: bool = False) -> None:
        """Write changed identities (index, embeddings, thumbnails). Throttled unless ``force``."""
        with self._lock:
            now = time.monotonic()
            if not force and (not self._index_dirty or now - self._last_save < SAVE_INTERVAL_S):
                return
            self._last_save = now
            jobs = []
            for ident in self._ids.values():
                if ident.dirty_emb or ident.thumb is not None:
                    jobs.append((ident.uid, ident.centroid.copy(), np.stack(ident.samples) if ident.samples
                                 else ident.centroid[None].copy(), ident.thumb))
                    ident.dirty_emb = False
                    ident.thumb = None
            index = {"version": 1, "next_uid": self._next_uid,
                     "faces": [i.meta() for i in sorted(self._ids.values(), key=lambda i: i.uid)]}
            deleted, self._deleted = self._deleted, []
            self._index_dirty = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for uid, cen, samples, thumb in jobs:
                d = self.root / str(uid)
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "centroid.npy", cen)
                np.save(d / "samples.npy", samples)
                if thumb is not None:
                    ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        tmp = d / "thumb.jpg.tmp"
                        tmp.write_bytes(buf.tobytes())
                        os.replace(tmp, d / "thumb.jpg")
            for uid in deleted:
                with self._lock:
                    if uid in self._ids:     # re-created meanwhile (never happens: uids are not reused)
                        continue
                shutil.rmtree(self.root / str(uid), ignore_errors=True)
            tmp = self.root / "index.json.tmp"
            tmp.write_text(json.dumps(index, indent=1), encoding="utf-8")
            os.replace(tmp, self.root / "index.json")
        except OSError as exc:
            print(f"[RECOG] Could not save unknown faces: {exc.__class__.__name__}: {exc}")
            with self._lock:
                self._index_dirty = True

    # ------------------------------------------------------------------ matching
    def _best(self, e: np.ndarray) -> Tuple[Optional[Identity], float]:
        best, best_sim = None, -1.0
        for ident in self._ids.values():
            sim = ident.similarity(e)
            if sim > best_sim:
                best, best_sim = ident, sim
        return best, best_sim

    def match(self, embedding: np.ndarray, threshold: float) -> Tuple[Optional[int], float]:
        e = _norm(embedding)
        if e is None:
            return None, 0.0
        self._ensure_loaded()
        with self._lock:
            ident, sim = self._best(e)
            return (ident.uid if ident is not None and sim >= threshold else None), max(0.0, sim)

    def observe(self, embedding: np.ndarray, threshold: float, thumb: Optional[np.ndarray] = None,
                quality: float = 1.0, learn: bool = True, hint: Optional[int] = None,
                now: Optional[float] = None) -> Optional[int]:
        """Assign one unknown face to an identity (creating / updating it). Returns the uid or None."""
        e = _norm(embedding)
        if e is None or e.size < 16:
            return None
        self._ensure_loaded()
        now = self.clock() if now is None else now
        with self._lock:
            chosen = None
            if hint is not None and hint in self._ids and self._ids[hint].similarity(e) >= threshold - HINT_SLACK:
                chosen = self._ids[hint]
            else:
                ident, sim = self._best(e)
                if ident is not None and sim >= threshold:
                    chosen = ident
            if chosen is not None:
                self._update(chosen, e, thumb, quality, learn, now, threshold)
                return chosen.uid if chosen.uid in self._ids else self._merged_into(chosen.uid)
            if not learn:
                return None
            return self._pending_observe(e, thumb, quality, now, threshold)

    def _merged_into(self, uid: int) -> Optional[int]:
        for src, dst in reversed(self.merges):
            if src == uid:
                return dst if dst in self._ids else None
        return None

    def _update(self, ident: Identity, e: np.ndarray, thumb, quality: float, learn: bool, now: float,
                threshold: float) -> None:
        if now - ident.last_seen > SIGHTING_GAP_S:
            ident.sightings += 1
        ident.last_seen = max(ident.last_seen, now)
        ident.hits += 1
        self._index_dirty = True
        if not learn:
            return
        w = 1.0 / min(ident.n + 1, CENTROID_CAP)
        cen = _norm((1.0 - w) * ident.centroid + w * e)
        if cen is not None:
            ident.centroid = cen
        ident.n += 1
        sims = [float(s @ e) for s in ident.samples]
        if not sims or max(sims) < SAMPLE_DIVERSITY:
            if len(ident.samples) < MAX_SAMPLES:
                ident.samples.append(e)
            else:
                # replace the most redundant sample (highest similarity to another sample)
                mat = np.stack(ident.samples)
                gram = mat @ mat.T
                np.fill_diagonal(gram, -1.0)
                ident.samples[int(np.argmax(gram.max(axis=1)))] = e
        ident.dirty_emb = True
        if thumb is not None and quality > ident.thumb_q * 1.1:
            ident.thumb_q = float(quality)
            ident.thumb = thumb.copy()
        self._maybe_merge(ident, threshold)

    def _maybe_merge(self, ident: Identity, threshold: float) -> None:
        limit = min(0.95, threshold + MERGE_MARGIN)
        for other in list(self._ids.values()):
            if other.uid == ident.uid:
                continue
            if float(other.centroid @ ident.centroid) >= limit:
                keep, drop = (other, ident) if other.uid < ident.uid else (ident, other)
                self._merge_locked(drop, keep)
                return

    def _merge_locked(self, src: Identity, dst: Identity) -> None:
        total = src.n + dst.n
        cen = _norm((dst.centroid * dst.n + src.centroid * src.n) / max(1, total))
        if cen is not None:
            dst.centroid = cen
        dst.n = min(total, 10_000)
        pool = dst.samples + src.samples
        kept: List[np.ndarray] = []
        for s in sorted(pool, key=lambda s: -float(s @ dst.centroid)):
            if len(kept) >= MAX_SAMPLES:
                break
            if all(float(s @ k) < SAMPLE_DIVERSITY for k in kept):
                kept.append(s)
        dst.samples = kept or [dst.centroid]
        dst.first_seen = min(dst.first_seen, src.first_seen)
        dst.last_seen = max(dst.last_seen, src.last_seen)
        dst.sightings += src.sightings
        dst.hits += src.hits
        if src.thumb_q > dst.thumb_q:
            src_thumb = src.thumb if src.thumb is not None else self._read_thumb(src.uid)
            if src_thumb is not None:
                dst.thumb, dst.thumb_q = src_thumb, src.thumb_q
        dst.dirty_emb = True
        del self._ids[src.uid]
        self._deleted.append(src.uid)
        self.merges.append((src.uid, dst.uid))
        del self.merges[:-64]
        self._index_dirty = True
        self.version += 1

    def _read_thumb(self, uid: int) -> Optional[np.ndarray]:
        p = self.root / str(uid) / "thumb.jpg"
        if not p.is_file():
            return None
        img = cv2.imread(str(p))
        return img if img is not None and img.size else None

    def _pending_observe(self, e: np.ndarray, thumb, quality: float, now: float, threshold: float) -> Optional[int]:
        self._pending = [p for p in self._pending if now - p.last <= PENDING_TTL_S]
        best, best_sim = None, -1.0
        for p in self._pending:
            sim = max([float(p.centroid @ e)] + [float(s @ e) for s in p.samples])
            if sim > best_sim:
                best, best_sim = p, sim
        if best is None or best_sim < threshold:
            if len(self._pending) >= MAX_PENDING:
                self._pending.sort(key=lambda p: p.last)
                self._pending.pop(0)
            self._pending.append(_Pending(centroid=e, samples=[e], hits=1, first=now, last=now,
                                          thumb=thumb.copy() if thumb is not None else None, thumb_q=quality))
            return None
        p = best
        p.hits += 1
        p.last = now
        cen = _norm(p.centroid * (p.hits - 1) + e)
        if cen is not None:
            p.centroid = cen
        if len(p.samples) < MAX_SAMPLES and all(float(s @ e) < SAMPLE_DIVERSITY for s in p.samples):
            p.samples.append(e)
        if thumb is not None and quality > p.thumb_q:
            p.thumb, p.thumb_q = thumb.copy(), quality
        if p.hits < CONFIRM_HITS:
            return None
        if len(self._ids) >= MAX_IDENTITIES and not self._evict_one(now):
            return None
        self._pending.remove(p)
        uid = self._next_uid
        self._next_uid += 1
        ident = Identity(uid=uid, centroid=p.centroid, samples=list(p.samples), n=p.hits, first_seen=p.first,
                         last_seen=now, sightings=1, hits=p.hits, thumb_q=p.thumb_q, thumb=p.thumb)
        self._ids[uid] = ident
        self._index_dirty = True
        self.version += 1
        self._maybe_merge(ident, threshold)
        return uid if uid in self._ids else self._merged_into(uid)

    def _evict_one(self, now: float) -> bool:
        """Make room at the cap: drop the stalest single-sighting identity (never established ones)."""
        singles = [i for i in self._ids.values() if i.sightings <= 1]
        if not singles:
            return False
        self._remove_locked(min(singles, key=lambda i: i.last_seen).uid)
        return True

    def prune(self, now: Optional[float] = None, retention_days: float = RETENTION_DAYS) -> List[int]:
        """Forget identities seen in only one sighting that are older than the retention period."""
        self._ensure_loaded()
        now = self.clock() if now is None else now
        cutoff = now - retention_days * 86400.0
        with self._lock:
            doomed = [i.uid for i in self._ids.values() if i.sightings <= 1 and i.last_seen < cutoff]
            for uid in doomed:
                self._remove_locked(uid)
            self._last_prune = time.monotonic()
        return doomed

    def maintenance(self) -> None:
        """Periodic upkeep from the worker: prune hourly, save throttled."""
        if time.monotonic() - self._last_prune > 3600:
            self.prune()
        self.save()

    # ------------------------------------------------------------------ management (HTTP)
    def _remove_locked(self, uid: int) -> None:
        if self._ids.pop(uid, None) is not None:
            self._deleted.append(uid)
            self._index_dirty = True
            self.version += 1

    def list(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            items = [(i.uid, i.first_seen, i.last_seen, i.sightings, i.hits, len(i.samples), i.thumb_q)
                     for i in self._ids.values()]
        out = [{"uid": uid, "label": f"Face #{uid}", "first_seen": fs, "last_seen": ls, "sightings": si,
                "hits": hi, "samples": ns,
                "thumb": f"/api/recognition/unknown/{uid}/thumb?v={int(ls)}" if q >= 0 else None}
               for uid, fs, ls, si, hi, ns, q in items]
        out.sort(key=lambda f: -f["last_seen"])
        return out

    def get(self, uid: int) -> Optional[Identity]:
        self._ensure_loaded()
        with self._lock:
            return self._ids.get(uid)

    def thumb_path(self, uid: int) -> Optional[Path]:
        if not isinstance(uid, int) or not 0 < uid < UID_MAX:
            return None
        self.save(force=self._index_dirty)            # a fresh identity's thumb may not be on disk yet
        p = self.root / str(uid) / "thumb.jpg"
        return p if p.is_file() else None

    def delete(self, uid: int) -> bool:
        self._ensure_loaded()
        with self._lock:
            if uid not in self._ids:
                return False
            self._remove_locked(uid)
        self.save(force=True)
        return True

    def merge(self, src: int, dst: int) -> Identity:
        self._ensure_loaded()
        if src == dst:
            raise UnknownFaceError("same_face")
        with self._lock:
            a, b = self._ids.get(src), self._ids.get(dst)
            if a is None or b is None:
                raise UnknownFaceError("unknown_face")
            self._merge_locked(a, b)
        self.save(force=True)
        return b

    def take(self, uid: int) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Samples + best thumbnail of an identity (for promotion into the gallery)."""
        self._ensure_loaded()
        with self._lock:
            ident = self._ids.get(uid)
            if ident is None:
                raise UnknownFaceError("unknown_face")
            samples = list(ident.samples) or [ident.centroid]
            thumb = ident.thumb if ident.thumb is not None else self._read_thumb(uid)
        return samples, thumb

    def drain_merges(self) -> List[Tuple[int, int]]:
        with self._lock:
            out, self.merges = self.merges, []
        return out

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._ids)


# ----------------------------------------------------------------------------- singleton
_store: Optional[UnknownFaces] = None
_store_lock = threading.Lock()


def get_store() -> UnknownFaces:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = UnknownFaces()
    return _store


def set_store(store: Optional[UnknownFaces]) -> None:
    """Swap the singleton (tests)."""
    global _store
    _store = store


def promote(uid: int, name: str) -> Dict[str, Any]:
    """Name an unknown identity: its samples move into the known gallery (new or existing person)."""
    from . import gallery

    store = get_store()
    samples, thumb = store.take(uid)
    name = gallery.clean_name(name)          # raises GalleryError before anything changes
    person = None
    for i, emb in enumerate(samples):
        person = gallery.add_sample(name, emb, thumb if i == 0 else None)
    store.delete(uid)
    return person or {}


def apply_merges() -> None:
    """Identities folded into others (automatically or by hand): tags and live tracks follow."""
    from .. import tags

    for src, dst in get_store().drain_merges():
        tags.STORE.move(f"f:{src}", f"f:{dst}")
        relabel_tracks(src, face_uid=dst)


def relabel_tracks(uid: int, face_uid: Optional[int] = None, name: Optional[str] = None,
                   person_id: Optional[str] = None) -> None:
    """Update every camera's per-track cache that carries unknown identity ``uid``."""
    from .common import _runtimes

    for rt in list(_runtimes.values()):
        with rt.lock:
            for ti in rt.track_info.values():
                if ti.face_uid == uid:
                    ti.face_uid = face_uid
                    if name:
                        ti.name, ti.person_id, ti.name_score = name, person_id, 1.0


# ----------------------------------------------------------------------------- worker hook
def learnable(row: np.ndarray, thumb: Optional[np.ndarray]) -> Tuple[bool, float]:
    """(may learn, quality) for one YuNet row: size, detector score and sharpness."""
    w = float(row[2])
    score = float(row[14]) if row.size > 14 else 1.0
    sharp = sharpness(thumb)
    quality = min(1.0, w / 160.0) * min(1.0, sharp / 200.0) * score
    return (w >= MIN_LEARN_PX and score >= MIN_LEARN_SCORE and sharp >= MIN_SHARPNESS), quality


def label_hits(found, frame: np.ndarray, threshold: float, hint_for: Callable[[Any], Optional[int]]) -> None:
    """Give every unrecognized face in ``found`` [(row, FaceHit)] its unknown identity (sets hit.face_uid)."""
    from . import faces as F

    store = get_store()
    for row, hit in found:
        if hit.name or not hit.recognizable or hit.embedding is None:
            continue
        thumb = F.aligned_thumb(frame, row)
        learn, quality = learnable(row, thumb)
        hit.face_uid = store.observe(hit.embedding, threshold, thumb=thumb, quality=quality, learn=learn,
                                     hint=hint_for(hit))
    store.maintenance()
    if store.merges:
        apply_merges()
