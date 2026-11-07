"""Persistent license-plate log: every confirmed plate read, with images, on disk.

Layout under ``data/plates/`` (``SECURITYCAM_DATA_DIR`` aware)::

    platelog.sqlite3                 index (one row per entry, JSON payload)
    watchlist.json                   plates that raise an alarm event
    <camera>/<entry id>/plate.jpg    upscaled + sharpened plate crop
    <camera>/<entry id>/vehicle.jpg  vehicle crop
    <camera>/<entry id>/frame.jpg    full frame (downscaled)

One entry per vehicle track and plate: the recognition worker calls
``observe()`` after every vote; the entry is created once the vote is
confirmed and then updated while the track lives (text, confidence, last
seen). Images are replaced when a read of clearly better quality comes in.

``observe()`` only does cheap bookkeeping and enqueues; JPEG encoding, file
and SQLite writes happen on a single background writer thread. Reads are
served from an in-memory index (loaded once from SQLite), so search and
filters never touch the disk.
"""
from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from ..config import _env_bool, _env_float, _env_int

# ----------------------------------------------------------------------------- settings
ENABLED = _env_bool("SECURITYCAM_PLATES_LOG", True)
MAX_COUNT = _env_int("SECURITYCAM_PLATES_MAX_COUNT", 2000, minimum=10, maximum=200000)
MAX_AGE_DAYS = _env_float("SECURITYCAM_PLATES_MAX_AGE_DAYS", 30.0, minimum=0.0, maximum=3650.0)

MIN_CHARS = 4
CONFIRM_READS = 2            # tracked vehicle: votes needed before an entry is logged
CONFIRM_SHARE = 0.45         # ... and the share of the winning text
UNTRACKED_MIN_CONF = 0.80    # without tracking a single, confident read is logged
UNTRACKED_DEDUPE_S = 120.0   # same plate without track id within this window -> same entry
TRACK_GAP_S = 90.0           # a track id unseen this long starts a new entry (tracker restarts ids)
IMAGE_MIN_INTERVAL_S = 1.0   # per entry, images are re-encoded at most this often
META_MIN_INTERVAL_S = 1.0
IMAGE_GAIN = 1.08            # new crop must be this much better to replace the stored one

PLATE_CROP_W = 400
VEHICLE_MAX_SIDE = 640
FRAME_MAX_W = 1600

CAMERA_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
WATCH_ID_RE = re.compile(r"^w[0-9a-f]{8}$")
IMAGE_NAMES = {"plate": "plate.jpg", "vehicle": "vehicle.jpg", "frame": "frame.jpg"}
NOTE_MAX = 80
WATCH_MAX = 500

# Characters OCR models confuse; both sides map to the same canonical symbol.
_FOLD = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"})


# ----------------------------------------------------------------------------- text helpers
def normalize(text: Any) -> str:
    return "".join(c for c in str(text or "").upper() if c.isascii() and c.isalnum())


def fold(text: Any) -> str:
    """Canonical form that is tolerant to 0/O/D/Q, 1/I/L, 2/Z, 5/S, 6/G, 8/B."""
    return normalize(text).translate(_FOLD)


def _substring_distance(q: str, s: str) -> int:
    """Minimum edit distance between ``q`` and any substring of ``s`` (Sellers)."""
    if not q:
        return 0
    prev = [0] * (len(s) + 1)
    for i, qc in enumerate(q, 1):
        cur = [i] + [0] * len(s)
        for j, sc in enumerate(s, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (qc != sc))
        prev = cur
    return min(prev)


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ac in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bc in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ac != bc))
        prev = cur
    return prev[-1]


def similar(a: str, b: str) -> bool:
    """Same plate, allowing OCR confusions and one more differing character."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return False
    return fa == fb or (min(len(fa), len(fb)) >= MIN_CHARS and _distance(fa, fb) <= 1)


def match_score(query: str, plate: str) -> int:
    """3 exact substring, 2 confusion-tolerant substring, 1 one/two typos away, 0 no match."""
    q = normalize(query)
    if not q:
        return 3
    p = normalize(plate)
    if q in p:
        return 3
    fq, fp = fold(q), fold(p)
    if fq in fp:
        return 2
    allowed = 2 if len(fq) >= 8 else 1 if len(fq) >= 4 else 0
    if allowed and _substring_distance(fq, fp) <= allowed:
        return 1
    return 0


# ----------------------------------------------------------------------------- image helpers
def _clip_box(box, w: int, h: int, mx: float = 0.0, my: float = 0.0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in box)
    px, py = int((x2 - x1) * mx), int((y2 - y1) * my)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def plate_crop(frame: np.ndarray, box) -> Optional[np.ndarray]:
    """Plate crop with a small margin, upscaled (bicubic) and unsharp-masked."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h, 0.12, 0.25)
    if x2 - x1 < 4 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    cw, ch = crop.shape[1], crop.shape[0]
    scale = min(6.0, PLATE_CROP_W / max(1, cw))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(cw * scale), max(1, int(ch * scale))), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0 + 0.25 * scale)
        crop = cv2.addWeighted(crop, 1.7, blur, -0.7, 0)
    elif cw > PLATE_CROP_W:
        crop = cv2.resize(crop, (PLATE_CROP_W, max(1, int(ch * PLATE_CROP_W / cw))), interpolation=cv2.INTER_AREA)
    return crop


def vehicle_crop(frame: np.ndarray, box) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h, 0.06, 0.06)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, (x1, y1, x2, y2)
    crop = frame[y1:y2, x1:x2]
    side = max(crop.shape[:2])
    if side > VEHICLE_MAX_SIDE:
        k = VEHICLE_MAX_SIDE / side
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))),
                          interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def frame_image(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= FRAME_MAX_W:
        return frame
    return cv2.resize(frame, (FRAME_MAX_W, max(1, int(h * FRAME_MAX_W / w))), interpolation=cv2.INTER_AREA)


def _write_jpeg(path: Path, img: Optional[np.ndarray], quality: int) -> bool:
    if img is None or img.size == 0:
        return False
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
    return True


def _norm_box(box, w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [round(min(1.0, max(0.0, v)), 4) for v in (x1 / w, y1 / h, x2 / w, y2 / h)]


# ----------------------------------------------------------------------------- the log
class PlateLog:
    def __init__(self, root: Path, max_count: int = MAX_COUNT, max_age_days: float = MAX_AGE_DAYS,
                 enabled: bool = ENABLED):
        self.root = Path(root)
        self.max_count = max_count
        self.max_age_days = max_age_days
        self.enabled = enabled
        self._lock = threading.RLock()
        self._entries: Dict[str, dict] = {}
        self._tracks: Dict[tuple, dict] = {}     # (camera, track id | "p"+key) -> bookkeeping
        self._watch: List[dict] = []
        self._loaded = False
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=64)
        self._writer: Optional[threading.Thread] = None
        self.version = 0                          # bumps on every change (cheap "anything new?" check)
        self.on_watch_hit: Optional[Callable[[dict, dict], None]] = None
        self._last_prune = 0.0

    # ------------------------------------------------------------------ storage
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                db = sqlite3.connect(str(self.root / "platelog.sqlite3"), check_same_thread=False)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, camera TEXT NOT NULL, "
                           "plate_key TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, "
                           "data TEXT NOT NULL)")
                db.execute("CREATE INDEX IF NOT EXISTS entries_last_seen ON entries(last_seen)")
                db.commit()
                self._db = db
                for (raw,) in db.execute("SELECT data FROM entries"):
                    try:
                        e = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(e, dict) and ID_RE.match(str(e.get("id"))) and CAMERA_RE.match(str(e.get("camera"))):
                        self._entries[e["id"]] = e
            except (OSError, sqlite3.Error) as exc:
                print(f"[PLATES] Plate log storage unavailable: {exc.__class__.__name__}: {exc}")
            self._load_watch()
            self._retention_locked()

    def _db_exec(self, sql: str, params: Iterable[Any] = ()) -> None:
        if self._db is None:
            return
        try:
            with self._db_lock:
                self._db.execute(sql, tuple(params))
                self._db.commit()
        except sqlite3.Error as exc:
            print(f"[PLATES] SQLite write failed: {exc}")

    def _db_upsert(self, e: dict) -> None:
        self._db_exec("INSERT OR REPLACE INTO entries (id, camera, plate_key, first_seen, last_seen, data) "
                      "VALUES (?, ?, ?, ?, ?, ?)",
                      (e["id"], e["camera"], e["plate_key"], e["first_seen"], e["last_seen"], json.dumps(e)))

    def entry_dir(self, camera_id: str, entry_id: str) -> Optional[Path]:
        if not CAMERA_RE.match(camera_id or "") or not ID_RE.match(entry_id or ""):
            return None
        return self.root / camera_id / entry_id

    def image_path(self, entry_id: str, name: str) -> Optional[Path]:
        """Path of an existing entry image, or None (ids and names are validated)."""
        if name not in IMAGE_NAMES or not ID_RE.match(entry_id or ""):
            return None
        e = self.get(entry_id)
        if e is None or not e.get("images", {}).get(name):
            return None
        d = self.entry_dir(e["camera"], entry_id)
        p = d / IMAGE_NAMES[name] if d else None
        return p if p is not None and p.is_file() else None

    def _remove_files(self, camera_id: str, entry_id: str) -> None:
        d = self.entry_dir(camera_id, entry_id)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def _retention_locked(self) -> List[str]:
        doomed: List[str] = []
        if self.max_age_days > 0:
            cutoff = time.time() - self.max_age_days * 86400.0
            doomed = [i for i, e in self._entries.items() if e.get("last_seen", 0) < cutoff]
        keep = sorted((i for i in self._entries if i not in set(doomed)),
                      key=lambda i: self._entries[i].get("last_seen", 0))
        excess = len(keep) - self.max_count
        if excess > 0:
            doomed += keep[:excess]
        for i in doomed:
            e = self._entries.pop(i, None)
            if e is None:
                continue
            self._forget_tracks_locked({i})
            self._remove_files(e["camera"], i)
            self._db_exec("DELETE FROM entries WHERE id = ?", (i,))
        if doomed:
            self.version += 1
        self._last_prune = time.monotonic()
        return doomed

    def _forget_tracks_locked(self, ids: set) -> None:
        for k in [k for k, st in self._tracks.items() if st["id"] in ids]:
            del self._tracks[k]

    # ------------------------------------------------------------------ watchlist
    def _watch_path(self) -> Path:
        return self.root / "watchlist.json"

    def _load_watch(self) -> None:
        try:
            raw = json.loads(self._watch_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print(f"[PLATES] Could not read watchlist.json: {exc.__class__.__name__}")
            return
        for w in raw.get("plates", []) if isinstance(raw, dict) else []:
            if isinstance(w, dict) and WATCH_ID_RE.match(str(w.get("id"))) and len(normalize(w.get("plate"))) >= 2:
                self._watch.append({"id": w["id"], "plate": normalize(w["plate"]),
                                    "note": str(w.get("note") or "")[:NOTE_MAX], "created": float(w.get("created") or 0)})

    def _save_watch_locked(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self._watch_path().with_name("watchlist.json.tmp")
            tmp.write_text(json.dumps({"version": 1, "plates": self._watch}, indent=2), encoding="utf-8")
            os.replace(tmp, self._watch_path())
        except OSError as exc:
            print(f"[PLATES] Could not save watchlist.json: {exc.__class__.__name__}")

    def watchlist(self) -> List[dict]:
        self._ensure_loaded()
        with self._lock:
            return [dict(w) for w in self._watch]

    def watch_add(self, plate: Any, note: Any = "") -> dict:
        p = normalize(plate)
        if not 2 <= len(p) <= 16:
            raise ValueError("invalid_plate")
        if note is not None and not isinstance(note, str):
            raise ValueError("invalid_note")
        note = " ".join((note or "").split())[:NOTE_MAX]
        self._ensure_loaded()
        with self._lock:
            for w in self._watch:
                if w["plate"] == p:
                    w["note"] = note or w["note"]
                    self._save_watch_locked()
                    return dict(w)
            if len(self._watch) >= WATCH_MAX:
                raise ValueError("watchlist_full")
            w = {"id": "w" + secrets.token_hex(4), "plate": p, "note": note, "created": time.time()}
            self._watch.append(w)
            self._save_watch_locked()
            self.version += 1
            return dict(w)

    def watch_remove(self, watch_id: str) -> bool:
        if not WATCH_ID_RE.match(watch_id or ""):
            return False
        self._ensure_loaded()
        with self._lock:
            before = len(self._watch)
            self._watch = [w for w in self._watch if w["id"] != watch_id]
            if len(self._watch) == before:
                return False
            self._save_watch_locked()
            self.version += 1
            return True

    def watch_match(self, plate: str) -> Optional[dict]:
        with self._lock:
            for w in self._watch:
                if similar(w["plate"], plate) if len(w["plate"]) >= MIN_CHARS else fold(w["plate"]) == fold(plate):
                    return dict(w)
        return None

    # ------------------------------------------------------------------ writes (recognition worker)
    def observe(self, camera_id: str, track_id: Optional[int], vehicle: str, text: str, share: float,
                reads: int, read, frame: np.ndarray, vehicle_box, now: Optional[float] = None) -> Optional[str]:
        """Record one voted read. Returns the entry id once the read is confirmed.

        ``read`` is the raw ``PlateRead`` of this attempt (box in frame pixels,
        OCR confidence); ``text``/``share``/``reads`` are the track's vote.
        """
        if not self.enabled or not CAMERA_RE.match(camera_id or "") or frame is None or frame.ndim != 3:
            return None
        plate = normalize(text)
        if len(plate) < MIN_CHARS:
            return None
        tracked = track_id is not None and track_id != -1
        if tracked and (reads < CONFIRM_READS or share < CONFIRM_SHARE):
            return None
        if not tracked and float(read.conf) < UNTRACKED_MIN_CONF:
            return None
        self._ensure_loaded()
        wall = time.time() if now is None else now
        mono = time.monotonic()
        h, w = frame.shape[:2]
        pw = max(1, read.box[2] - read.box[0])
        quality = float(read.conf) * min(1.0, pw / 140.0)
        key = (camera_id, int(track_id)) if tracked else (camera_id, "p" + fold(plate))
        with self._lock:
            st = self._tracks.get(key)
            gap = TRACK_GAP_S if tracked else UNTRACKED_DEDUPE_S
            if st is not None and (mono - st["mono"] > gap or not similar(st["plate"], plate)
                                   or st["id"] not in self._entries and not st["pending"]):
                st = None
            new = st is None
            if new:
                st = {"id": time.strftime("%Y%m%d-%H%M%S", time.localtime(wall)) + "-" + secrets.token_hex(3),
                      "plate": plate, "mono": mono, "quality": -1.0, "img_mono": -1e9, "meta_mono": -1e9,
                      "first_seen": wall, "pending": True, "watch": None, "ocr_conf": 0.0}
                self._tracks[key] = st
            changed = st["plate"] != plate
            st["plate"], st["mono"] = plate, mono
            st["ocr_conf"] = max(st["ocr_conf"], float(read.conf))
            meta = {
                "id": st["id"], "camera": camera_id, "plate": plate, "plate_key": fold(plate),
                "conf": round(float(share), 3), "ocr_conf": round(st["ocr_conf"], 3),
                "reads": int(reads) if tracked else int(st.get("reads", 0)) + 1,
                "first_seen": st["first_seen"], "last_seen": wall,
                "track_id": int(track_id) if tracked else None, "vehicle": str(vehicle or "vehicle"),
            }
            st["reads"] = meta["reads"]
            images = new or (quality > st["quality"] * IMAGE_GAIN and mono - st["img_mono"] >= IMAGE_MIN_INTERVAL_S)
            if not images and not changed and mono - st["meta_mono"] < META_MIN_INTERVAL_S:
                st["dirty"] = meta           # written by the writer's idle flush
                return st["id"]
            st["dirty"] = None
            watch = None
            if new or changed:
                hit = self.watch_match(plate)
                if hit and st["watch"] != hit["id"]:      # one alarm per entry and watchlist plate
                    watch = hit
                    st["watch"] = hit["id"]
            st["meta_mono"] = mono
            if images:
                st["quality"], st["img_mono"] = quality, mono
                job = ("images", meta, frame.copy(), tuple(read.box), tuple(int(v) for v in vehicle_box), quality, watch)
            else:
                job = ("meta", meta, None, None, None, None, watch)
            entry_id = st["id"]
        self._start_writer()
        try:
            self._q.put_nowait(job)
        except queue.Full:
            if new:                      # retry with the next read
                with self._lock:
                    self._tracks.pop(key, None)
                return None
        return entry_id

    def _start_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                return
            self._writer = threading.Thread(target=self._run, name="PlateLogWriter", daemon=True)
            self._writer.start()

    def _run(self) -> None:
        while True:
            try:
                job = self._q.get(timeout=META_MIN_INTERVAL_S)
            except queue.Empty:
                self._flush_dirty()
                continue
            try:
                self._apply(job)
            except Exception as exc:     # never let the writer die
                print(f"[PLATES] Plate log write failed: {exc.__class__.__name__}: {exc}")
            finally:
                self._q.task_done()

    def _flush_dirty(self) -> None:
        """Write the latest throttled metadata updates (last seen, reads, confidence)."""
        with self._lock:
            metas = []
            for st in self._tracks.values():
                if st.get("dirty") and not st["pending"]:
                    metas.append(st["dirty"])
                    st["dirty"] = None
                    st["meta_mono"] = time.monotonic()
        for meta in metas:
            try:
                self._apply(("meta", meta, None, None, None, None, None))
            except Exception as exc:
                print(f"[PLATES] Plate log write failed: {exc.__class__.__name__}: {exc}")

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued writes are done, including throttled updates (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        self._flush_dirty()
        while self._q.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def _apply(self, job: tuple) -> None:
        kind, meta, frame, pbox, vbox, quality, watch = job
        entry_id, cam = meta["id"], meta["camera"]
        img_meta: Dict[str, Any] = {}
        if kind == "images":
            d = self.entry_dir(cam, entry_id)
            if d is None:
                return
            d.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            vcrop, cbox = vehicle_crop(frame, vbox)
            images = {
                "plate": _write_jpeg(d / "plate.jpg", plate_crop(frame, pbox), 94),
                "vehicle": _write_jpeg(d / "vehicle.jpg", vcrop, 88),
                "frame": _write_jpeg(d / "frame.jpg", frame_image(frame), 85),
            }
            img_meta = {
                "images": images, "frame_w": w, "frame_h": h, "quality": round(float(quality), 3),
                "plate_box": _norm_box(pbox, w, h), "vehicle_box": _norm_box(vbox, w, h),
                "crop_box": _norm_box(cbox, w, h), "image_ts": meta["last_seen"],
            }
        with self._lock:
            e = self._entries.get(entry_id)
            if e is None:
                if kind != "images":
                    return                       # deleted meanwhile; a fresh entry starts with the next read
                st = next((s for s in self._tracks.values() if s["id"] == entry_id), None)
                if st is None:                   # deleted / cleared before the first write
                    self._remove_files(cam, entry_id)
                    return
                st["pending"] = False
                e = {"watch": None}
                self._entries[entry_id] = e
            e.update(meta)
            e.update(img_meta)
            if watch:
                e["watch"] = {"id": watch["id"], "note": watch["note"], "plate": watch["plate"]}
            snapshot = dict(e)
            self.version += 1
            if len(self._entries) > self.max_count or time.monotonic() - self._last_prune > 600:
                self._retention_locked()
        self._db_upsert(snapshot)
        if watch and self.on_watch_hit is not None and entry_id in self._entries:
            try:
                self.on_watch_hit(snapshot, watch)
            except Exception as exc:
                print(f"[PLATES] Watchlist alarm failed: {exc.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------ reads (HTTP)
    def get(self, entry_id: str) -> Optional[dict]:
        if not ID_RE.match(entry_id or ""):
            return None
        self._ensure_loaded()
        with self._lock:
            e = self._entries.get(entry_id)
            return dict(e) if e else None

    def query(self, camera: Optional[str] = None, q: str = "", since: Optional[float] = None,
              until: Optional[float] = None, plate_key: Optional[str] = None,
              watch_only: bool = False) -> List[dict]:
        """Matching entries, newest first (``match`` = search score, 3 best)."""
        self._ensure_loaded()
        with self._lock:
            items = [dict(e) for e in self._entries.values()]
        out = []
        for e in items:
            if camera and e["camera"] != camera:
                continue
            if since is not None and e.get("last_seen", 0) < since:
                continue
            if until is not None and e.get("first_seen", 0) > until:
                continue
            if plate_key is not None and e.get("plate_key") != plate_key:
                continue
            if watch_only and not e.get("watch"):
                continue
            score = match_score(q, e["plate"]) if q else 3
            if not score:
                continue
            e["match"] = score
            out.append(e)
        out.sort(key=lambda e: (-e["match"], -e.get("last_seen", 0)) if q else -e.get("last_seen", 0))
        return out

    def groups(self, **filters) -> List[dict]:
        """Entries grouped by plate (confusion-folded key): one row per distinct plate."""
        by_key: Dict[str, dict] = {}
        for e in self.query(**filters):
            g = by_key.get(e["plate_key"])
            if g is None:
                g = by_key[e["plate_key"]] = {
                    "plate_key": e["plate_key"], "plate": e["plate"], "count": 0, "cameras": [],
                    "first_seen": e["first_seen"], "last_seen": e["last_seen"], "best_id": e["id"],
                    "best_conf": -1.0, "latest_id": e["id"], "watch": e.get("watch"), "match": e["match"],
                    "texts": {},
                }
            g["count"] += 1
            if e["camera"] not in g["cameras"]:
                g["cameras"].append(e["camera"])
            g["first_seen"] = min(g["first_seen"], e["first_seen"])
            if e["last_seen"] > g["last_seen"]:
                g["last_seen"], g["latest_id"] = e["last_seen"], e["id"]
            score = e.get("conf", 0) * (1 + e.get("quality", 0))
            if score > g["best_conf"] and e.get("images", {}).get("plate"):
                g["best_conf"], g["best_id"] = score, e["id"]
            g["texts"][e["plate"]] = g["texts"].get(e["plate"], 0) + e.get("reads", 1)
            g["watch"] = g["watch"] or e.get("watch")
            g["match"] = max(g["match"], e["match"])
        out = []
        for g in by_key.values():
            g["plate"] = max(g["texts"], key=lambda t: (g["texts"][t], t))
            g["variants"] = sorted(g.pop("texts"), key=lambda t: t)
            g.pop("best_conf")
            out.append(g)
        out.sort(key=lambda g: (-g["match"], -g["last_seen"]))
        return out

    def summary(self, seen: float = 0.0, recent: int = 0) -> dict:
        self._ensure_loaded()
        with self._lock:
            entries = list(self._entries.values())
            version = self.version
        latest = max((e.get("first_seen", 0) for e in entries), default=0.0)
        out = {"total": len(entries), "new": sum(1 for e in entries if e.get("first_seen", 0) > seen),
               "latest": latest, "version": version}
        if recent > 0:
            times = sorted((float(e.get("first_seen", 0) or 0) for e in entries), reverse=True)
            out["recent"] = [round(t, 3) for t in times[:recent]]
        return out

    # ------------------------------------------------------------------ deletes
    def delete(self, entry_id: str) -> bool:
        if not ID_RE.match(entry_id or ""):
            return False
        self._ensure_loaded()
        with self._lock:
            e = self._entries.pop(entry_id, None)
            if e is None:
                return False
            self._forget_tracks_locked({entry_id})
            self.version += 1
        self._remove_files(e["camera"], entry_id)
        self._db_exec("DELETE FROM entries WHERE id = ?", (entry_id,))
        return True

    def clear(self, camera: Optional[str] = None) -> int:
        self._ensure_loaded()
        with self._lock:
            ids = [i for i, e in self._entries.items() if camera is None or e["camera"] == camera]
            doomed = [(i, self._entries.pop(i)["camera"]) for i in ids]
            self._forget_tracks_locked(set(ids))
            for k in [k for k in self._tracks if camera is None or k[0] == camera]:
                del self._tracks[k]
            self.version += 1
        for i, cam in doomed:
            self._remove_files(cam, i)
        if camera is None:
            self._db_exec("DELETE FROM entries")
        else:
            self._db_exec("DELETE FROM entries WHERE camera = ?", (camera,))
        return len(doomed)


# ----------------------------------------------------------------------------- module singleton
def _default_root() -> Path:
    from ..alerts import conf as alerts_conf     # SECURITYCAM_DATA_DIR aware, no heavy imports

    return alerts_conf.DATA_DIR / "plates"


_log: Optional[PlateLog] = None
_log_lock = threading.Lock()


def get_log() -> PlateLog:
    global _log
    if _log is None:
        with _log_lock:
            if _log is None:
                _log = PlateLog(_default_root())
                _log.on_watch_hit = raise_watch_alarm
    return _log


def set_log(log: Optional[PlateLog]) -> None:
    """Swap the singleton (tests)."""
    global _log
    _log = log


def observe(*args, **kwargs) -> Optional[str]:
    """Recognition-worker entry point; never raises."""
    try:
        return get_log().observe(*args, **kwargs)
    except Exception as exc:
        print(f"[PLATES] Plate log observe failed: {exc.__class__.__name__}: {exc}")
        return None


def raise_watch_alarm(entry: dict, watch: dict) -> None:
    """Watchlist hit -> alarm event through the alerts package (clip, snapshot, notifications)."""
    from .. import state
    from ..alerts import engine as alert_engine

    cam = state.get_camera_state(entry["camera"])
    if cam is None:
        return
    note = f" ({watch['note']})" if watch.get("note") else ""
    alert_engine.get(entry["camera"]).emit(
        cam, "plate", f"plate-{fold(watch['plate'])}", entry["plate"], entry.get("vehicle") or "vehicle",
        entry.get("track_id"), f"Watchlist plate {entry['plate']}{note}", True,
        extra={"plate": entry["plate"], "plate_log_id": entry["id"], "watch_note": watch.get("note", "")},
    )
