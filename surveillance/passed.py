"""Passed vehicles (WP17, P5): one record for every vehicle that crossed a camera, not only the ones that broke a rule.

What a record holds
    camera, run, the tracker's id, the detector's class, first / last seen (video clock and wall clock), a small
    thumbnail of the clearest view, a colour name, and the plate result. The global vehicle id is NOT copied: it is
    read from `tracklets` when a record is listed, because the cross-camera linker assigns and revises it minutes
    or days after the vehicle left, and this module must never hold a stale copy of someone else's decision.

Three costs, three rules
    * Recording (pipeline thread, per frame): a vehicle's few largest crops are kept as JPEG, at most one sample per
      `sample_sec`, and only when larger than the smallest one already held. Per finished vehicle: one decode, one
      thumbnail write, one colour name, one SQL insert. Measured numbers are in docs/VEHICLES.md.
    * Plate reading (background thread): the largest views of each vehicle go through the same plate and character
      models as the violators', in a worker with a BOUNDED queue. A full queue never blocks the detection loop: the
      vehicle's record is stored at once and marked "skipped: busy". Violators keep priority: the worker waits while
      the violators' reader has work (`busy_fn`), checked before every crop.
    * Colour (pipeline thread, ~0.1 ms): a name from an HSV histogram of the middle of the crop. No VLM here; the
      hook for an optional VLM look is `set_vlm_look` / the `look_vlm` column, filled by whoever runs a VLM later.

Privacy: these are records of people who broke no rule. They expire after PASSED_RETENTION_DAYS (default 30),
see `purge` and `PurgeLoop`; storage_mgr lists the same expiry in its dry-run plan.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from violations.base import NON_MOTORISED

from .anpr import MIN_PLATE_PX, read_crops

PASSED_DIR = "passed"              # under the data folder: data/passed/cam<id>/run<id>_t<track>.jpg
DEFAULT_RETENTION_DAYS = 30
DAY = 86400.0

# --- plate statuses, as stored and shown. "read" = confirmed (all parts), "partial" = the number only.
READ, PARTIAL, UNREADABLE = "read", "partial", "unreadable"
NO_PLATE = "no plate seen"
PENDING = "pending"
SKIP_BUSY = "skipped: busy"        # the reader's queue was full: the vehicle was not read, nothing was guessed
SKIP_SMALL = "skipped: small"      # the vehicle's best view is narrower than MIN_VEHICLE_PX
SKIP_PEDAL = "skipped: pedal"      # bicycles, rickshaws, vans, wheelbarrows carry no plate
NOT_READ = "not read"              # anpr_all is off for this camera, or the plate models are not installed
PLATE_STATUSES = (READ, PARTIAL, UNREADABLE, NO_PLATE, PENDING, SKIP_BUSY, SKIP_SMALL, SKIP_PEDAL, NOT_READ)

MIN_VEHICLE_PX = 100               # theirs: a vehicle narrower than this has no readable plate
READ_CROPS, GIVE_UP_AFTER = 10, 8  # theirs: the 10 largest views; stop at 8 when none of them shows a plate


def retention_days() -> float:
    """PASSED_RETENTION_DAYS from the environment (default 30); a bad value falls back to the default."""
    try:
        v = float(os.environ.get("PASSED_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return v if v > 0 else DEFAULT_RETENTION_DAYS


# ---------------------------------------------------------------- colour

COLOURS = ("white", "silver", "grey", "black", "red", "orange", "yellow", "green", "blue", "purple", "brown")
MIXED = "mixed"
CHROMA_SHARE = 0.30                # coloured pixels at least this share of the middle: the colour is the vehicle's


def name_colour(bgr, middle=0.5) -> tuple:
    """(colour name, share of the pixels with that name) from the middle of a vehicle crop.

    The edges of a box are road, shadow and neighbours, so only the central `middle` fraction in each direction
    is looked at, at most 48 px on a side. Each pixel is named from HSV: dark = black; low saturation = white /
    silver / grey by brightness; otherwise by hue (orange that is dark is brown). If at least CHROMA_SHARE of the
    pixels are coloured, the commonest colour among them wins - a red car with black windows and a shaded
    underside is red - otherwise the commonest of white / silver / grey / black. No clear winner (< 40% of the
    pixels of the chosen group) gives "mixed". Unreadable input gives ("", 0.0)."""
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 6:
        return "", 0.0
    h, w = bgr.shape[:2]
    my, mx = int(h * (1 - middle) / 2), int(w * (1 - middle) / 2)
    mid = bgr[my:h - my, mx:w - mx]
    if mid.size == 0:
        return "", 0.0
    scale = 48.0 / max(mid.shape[:2])
    if scale < 1.0:
        mid = cv2.resize(mid, (max(1, int(mid.shape[1] * scale)), max(1, int(mid.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.empty(len(hsv), dtype=np.int8)
    idx = {c: i for i, c in enumerate(COLOURS)}
    dark = val < 50
    grey = ~dark & (sat < 45)
    colour = ~dark & ~grey
    names[dark] = idx["black"]
    names[grey & (val >= 200)] = idx["white"]
    names[grey & (val >= 140) & (val < 200)] = idx["silver"]
    names[grey & (val < 140)] = idx["grey"]
    by_hue = np.full(len(hsv), idx["red"], np.int8)
    for lo, hi, name in ((8, 21, "orange"), (21, 36, "yellow"), (36, 86, "green"), (86, 131, "blue"),
                         (131, 172, "purple")):
        by_hue[(hue >= lo) & (hue < hi)] = idx[name]
    brown = (by_hue == idx["orange"]) & (val < 150) | (by_hue == idx["yellow"]) & (val < 120)
    by_hue[brown] = idx["brown"]
    names[colour] = by_hue[colour]
    group = colour if colour.mean() >= CHROMA_SHARE else ~colour
    if not group.any():
        return "", 0.0
    counts = np.bincount(names[group], minlength=len(COLOURS))
    top = int(counts.argmax())
    share = counts[top] / group.sum()
    if share < 0.4:
        return MIXED, round(float(share), 2)
    return COLOURS[top], round(float(share), 2)


# ---------------------------------------------------------------- plate text for searching

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_SEPARATORS = re.compile(r"[\s\-–—._/,]+")


def normalise_plate(text) -> str:
    """The form plate text is stored and searched in: Bangla digits turned into Latin ones, upper case, no spaces
    or dashes. "ঢাকা মেট্রো-গ ১২-৩৪৫৬" and a typed "dhaka metro-ga 12 3456" meet in the digits "123456"."""
    return _SEPARATORS.sub("", str(text or "").translate(_BN_DIGITS)).upper()


# ---------------------------------------------------------------- the table

SCHEMA = """
CREATE TABLE IF NOT EXISTS passed_vehicles (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    run_id INTEGER,                 -- no foreign key: a deleted run must not silently delete these
    track_id INTEGER NOT NULL,      -- the tracker's own id, unique only within one run of one camera
    cls TEXT DEFAULT '',
    colour TEXT DEFAULT '',
    first_video_t REAL, last_video_t REAL,      -- seconds on the video clock
    first_wall TEXT DEFAULT '', last_wall TEXT DEFAULT '',   -- camera wall clock, text, as the cases show it
    first_epoch REAL, last_epoch REAL,          -- the same, epoch seconds (NULL when the video has no clock)
    frames INTEGER DEFAULT 0,
    width_px INTEGER DEFAULT 0,     -- width of the vehicle's box in its best view
    thumb TEXT DEFAULT '',          -- path under data/passed/
    plate_status TEXT DEFAULT '',
    plate_bn TEXT DEFAULT '', plate_en TEXT DEFAULT '',
    plate_norm TEXT DEFAULT '',     -- normalise_plate of both scripts, for substring search
    plate_conf REAL, plate_px INTEGER,
    look_vlm TEXT DEFAULT '',       -- optional "grey SUV" from a VLM, filled later by set_vlm_look
    created_at REAL NOT NULL        -- when the record was made: retention counts from here
);
CREATE INDEX IF NOT EXISTS ix_passed_cam ON passed_vehicles(camera_id, run_id, first_epoch);
CREATE INDEX IF NOT EXISTS ix_passed_created ON passed_vehicles(created_at);
CREATE INDEX IF NOT EXISTS ix_passed_plate ON passed_vehicles(plate_norm);
"""


def ensure_schema(store) -> None:
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


def passed_dir(data_dir) -> Path:
    return Path(data_dir) / PASSED_DIR


def add_record(store, rec: dict) -> int:
    cols = ("camera_id", "run_id", "track_id", "cls", "colour", "first_video_t", "last_video_t", "first_wall",
            "last_wall", "first_epoch", "last_epoch", "frames", "width_px", "thumb", "plate_status", "created_at")
    return store.x(f"INSERT INTO passed_vehicles ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                   tuple(rec.get(c) for c in cols))


def set_plate(store, pid: int, status: str, text_bn="", text_en="", conf=None, plate_px=None) -> None:
    norm = (normalise_plate(text_en) + "|" + normalise_plate(text_bn)) if (text_bn or text_en) else ""
    store.x("UPDATE passed_vehicles SET plate_status=?, plate_bn=?, plate_en=?, plate_norm=?, plate_conf=?,"
            " plate_px=? WHERE id=?", (status, text_bn, text_en, norm, conf, plate_px, pid))


def set_vlm_look(store, pid: int, look: str) -> None:
    """HOOK for an optional VLM look ("grey SUV"). Nothing in this package calls a VLM for passing vehicles: on a
    CPU-only machine one call costs seconds, and there are thousands of vehicles an hour. Whoever runs one later
    (after a run, for the vehicles a user opens) stores its answer here; search and the page already show it."""
    store.x("UPDATE passed_vehicles SET look_vlm=? WHERE id=?", (str(look)[:120], pid))


def status_of(result) -> str:
    """The stored status for a PlateResult."""
    if result.status in (READ, PARTIAL):
        return result.status
    return UNREADABLE if result.plates_seen else NO_PLATE


# ---------------------------------------------------------------- plate reading, off the video loop

class PassedPlateWorker:
    """Reads the plates of passing vehicles on its own thread; submit() never blocks.

    The queue is bounded. When it is full `submit` returns False at once and counts a skip: the caller marks the
    vehicle "skipped: busy". Before each crop the worker waits while `busy_fn()` says the violators' reader has
    work, so a violator's plate is never queued behind a passing car's. `on_result(key, PlateResult)` is called from
    the worker thread. One worker per camera (each camera has its own reader)."""

    def __init__(self, reader, on_result: Callable, busy_fn: Optional[Callable] = None, maxsize: int = 16,
                 max_crops: int = READ_CROPS, give_up_after: int = GIVE_UP_AFTER, min_plate_px: int = MIN_PLATE_PX,
                 max_wait_s: float = 10.0):
        self.reader, self.on_result, self.busy_fn = reader, on_result, busy_fn
        self.max_crops, self.give_up_after, self.min_plate_px = max_crops, give_up_after, min_plate_px
        self.max_wait_s = max_wait_s
        self._q = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.queued = self.done = self.skipped = self.failed = 0
        self._ms = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="passed-anpr")
        self._thread.start()

    def submit(self, key, jpgs) -> bool:
        try:
            self._q.put_nowait((key, list(jpgs)))
        except queue.Full:
            with self._lock:
                self.skipped += 1
            return False
        with self._lock:
            self.queued += 1
        return True

    def _yield_to_violators(self):
        waited = 0.0
        while self.busy_fn is not None and self.busy_fn() and waited < self.max_wait_s and not self._stop.is_set():
            time.sleep(0.05)
            waited += 0.05

    def _run(self):
        while not self._stop.is_set():
            try:
                key, jpgs = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                result = read_crops(self.reader, jpgs, max_crops=self.max_crops, give_up_after=self.give_up_after,
                                    pause=self._yield_to_violators, min_plate_px=self.min_plate_px)
                self.on_result(key, result)
                with self._lock:
                    self.done += 1
                    self._ms += (time.perf_counter() - t0) * 1000.0
            except Exception as e:              # one vehicle must never end the worker
                print(f"passed-vehicle plate read failed: {e}")
                with self._lock:
                    self.failed += 1
            finally:
                self._q.task_done()

    def stats(self) -> dict:
        with self._lock:
            return {"queued": self.queued, "done": self.done, "skipped_busy": self.skipped, "failed": self.failed,
                    "waiting": self._q.qsize(), "mean_ms": round(self._ms / self.done, 1) if self.done else None}

    def close(self, timeout: float = 30.0) -> list:
        """Let the queue drain for up to `timeout` s, then stop. Returns the keys still unread (the caller marks
        them skipped)."""
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.05)
        self._stop.set()
        left = []
        while True:
            try:
                left.append(self._q.get_nowait()[0])
                self._q.task_done()
            except queue.Empty:
                break
        self._thread.join(timeout=5)
        return left


# ---------------------------------------------------------------- recording, on the pipeline thread

class PassedRecorder:
    """Turns the pipeline's per-frame boxes into one record per vehicle, when the vehicle has left.

    `observe` is called for every tracked vehicle on every processed frame and has to stay cheap; `tick` finishes
    the vehicles that have gone. Same shape as reid.TrackletCollector, with its own, larger crops (a plate needs
    the full-resolution view, a colour does not)."""

    def __init__(self, store, camera_id, data_dir, worker: Optional[PassedPlateWorker] = None, keep: int = READ_CROPS,
                 min_frames: int = 5, lost_sec: float = 1.5, sample_sec: float = 0.2, thumb_px: int = 160,
                 min_vehicle_px: int = MIN_VEHICLE_PX, max_crop_px: int = 1280, now_fn: Callable = time.time):
        self.store, self.camera_id, self.dir = store, camera_id, passed_dir(data_dir)
        self.worker, self.keep, self.min_frames = worker, keep, min_frames
        self.lost_sec, self.sample_sec, self.thumb_px = lost_sec, sample_sec, thumb_px
        self.min_vehicle_px, self.max_crop_px, self.now = min_vehicle_px, max_crop_px, now_fn
        self.run_id = None                    # set by the worker once the run exists
        self.state = {}                       # tid -> facts of a vehicle still in view
        self.rows = {}                        # tid -> record id (recent), so a late violator plate can reach it
        self.external = {}                    # tid -> PlateResult from the violators' reader
        self.recorded = self.skipped_busy = 0
        self.last_error = ""
        ensure_schema(store)
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- per frame
    def observe(self, tid, frame, box, cls, on_road, t, wall_t=None):
        st = self.state.get(tid)
        if st is None:
            st = self.state[tid] = {"first_t": t, "first_wall": wall_t, "frames": 0, "on_road": False,
                                    "crops": [], "last_sample": -1e9, "width": 0}
        st.update(last_t=t, last_wall=wall_t, cls=cls, seen=t)
        st["frames"] += 1
        st["on_road"] = st["on_road"] or bool(on_road)
        x1, y1, x2, y2 = box
        H, W = frame.shape[:2]
        if x1 < 2 or y1 < 2 or x2 > W - 2 or y2 > H - 2 or t - st["last_sample"] < self.sample_sec:
            return
        area = (x2 - x1) * (y2 - y1)
        crops = st["crops"]
        if area <= 0 or (len(crops) >= self.keep and area <= crops[-1][0] * 1.03):
            return                                      # not closer than the views already held
        st["last_sample"] = t
        pad = int(0.04 * (x2 - x1))
        crop = frame[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        if crop.size == 0:
            return
        if max(crop.shape[:2]) > self.max_crop_px:
            s = self.max_crop_px / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        crops.append((area, jpg, x2 - x1))
        crops.sort(key=lambda c: -c[0])
        del crops[self.keep:]
        st["width"] = crops[0][2]

    def tick(self, t, present):
        for tid, st in list(self.state.items()):
            if tid not in present and t - st["seen"] >= self.lost_sec:
                self._finish(tid)
        if len(self.rows) > 2000:                       # forget the oldest: a violator's plate comes within seconds
            for k in list(self.rows)[:1000]:
                self.rows.pop(k, None)
                self.external.pop(k, None)

    def finish_all(self):
        for tid in list(self.state):
            self._finish(tid)

    def _finish(self, tid):
        st = self.state.pop(tid)
        if not st["on_road"] or st["frames"] < self.min_frames or not st["crops"]:
            return                                      # flicker, or never on the road: not a passing vehicle
        try:
            self._record(tid, st)
        except Exception as e:                          # recording must never stop the analysis
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"passed vehicle not recorded: {self.last_error}")

    def _record(self, tid, st):
        crops = st["crops"]                             # largest first
        best = cv2.imdecode(np.frombuffer(crops[0][1], np.uint8), cv2.IMREAD_COLOR)
        colour = name_colour(best)[0]
        thumb = self._write_thumb(tid, best)
        ext = self.external.pop(tid, None)
        status = self._plate_plan(st)
        wall = lambda w: (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(w)) if w is not None else "")
        pid = add_record(self.store, {
            "camera_id": self.camera_id, "run_id": self.run_id, "track_id": tid, "cls": st["cls"], "colour": colour,
            "first_video_t": round(st["first_t"], 2), "last_video_t": round(st["last_t"], 2),
            "first_wall": wall(st["first_wall"]), "last_wall": wall(st["last_wall"]),
            "first_epoch": st["first_wall"], "last_epoch": st["last_wall"], "frames": st["frames"],
            "width_px": st["width"], "thumb": thumb, "plate_status": status, "created_at": self.now()})
        self.rows[tid] = pid
        self.recorded += 1
        if ext is not None:
            self._apply(pid, ext)
        if status == PENDING and not (ext is not None and status_of(ext) in (READ, PARTIAL)):
            if not self.worker.submit(pid, [c[1] for c in crops]):
                self.skipped_busy += 1
                set_plate(self.store, pid, SKIP_BUSY)

    def _plate_plan(self, st) -> str:
        """The status a record starts with: why its plate will not be read, or PENDING."""
        if st["cls"] in NON_MOTORISED:
            return SKIP_PEDAL
        if st["width"] < self.min_vehicle_px:
            return SKIP_SMALL
        return PENDING if self.worker is not None else NOT_READ

    def _write_thumb(self, tid, img) -> str:
        s = self.thumb_px / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        d = self.dir / f"cam{self.camera_id}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"run{self.run_id or 0}_t{tid}.jpg"
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return str(path.relative_to(self.dir))

    # -- plate results
    def plate_done(self, pid, result):
        """From the PassedPlateWorker thread."""
        self._apply(pid, result)

    def _apply(self, pid, result):
        set_plate(self.store, pid, status_of(result), result.text_bn, result.text_en, result.confidence,
                  result.plate_px or None)

    def violator_plate(self, tid, result):
        """The violators' reader finished this vehicle: reuse its result instead of reading twice. Only a result
        that found something is taken (a violator's "no plate" says nothing about the passing record's views)."""
        if result.status not in (READ, PARTIAL) and not result.plates_seen:
            return
        pid = self.rows.get(tid)
        if pid is None:
            self.external[tid] = result                 # the vehicle is still in view: used when it is recorded
            return
        row = self.store.q("SELECT plate_status FROM passed_vehicles WHERE id=?", (pid,))
        if row and row[0]["plate_status"] not in (READ, PARTIAL):
            self._apply(pid, result)

    def stats(self) -> dict:
        out = {"recorded": self.recorded, "in_view": len(self.state), "skipped_busy": self.skipped_busy,
               "error": self.last_error}
        if self.worker is not None:
            out["plates"] = self.worker.stats()
        return out

    def close(self, timeout: float = 30.0):
        """End of the run: record the vehicles still in view, let the plate queue drain (bounded), mark the rest."""
        self.finish_all()
        if self.worker is not None:
            for pid in self.worker.close(timeout):
                set_plate(self.store, pid, SKIP_BUSY)


# ---------------------------------------------------------------- retention

def purge(store, data_dir, days: Optional[float] = None, now: Optional[float] = None, dry_run=False) -> dict:
    """Delete passed-vehicle records older than `days` (PASSED_RETENTION_DAYS) and their thumbnails.

    Also removes thumbnails no record refers to once they are a day old (a camera that was deleted takes its
    records with it by cascade but not its files). dry_run counts only."""
    ensure_schema(store)
    now = time.time() if now is None else now
    days = retention_days() if days is None else days
    cutoff = now - days * DAY
    old = store.q("SELECT id, thumb FROM passed_vehicles WHERE created_at < ?", (cutoff,))
    root = passed_dir(data_dir)
    out = {"rows": len(old), "cutoff": cutoff, "days": days, "files": 0, "bytes": 0, "dry_run": dry_run}
    paths = [root / r["thumb"] for r in old if r["thumb"]]
    if not dry_run and old:
        with store.lock:
            store.db.executemany("DELETE FROM passed_vehicles WHERE id=?", [(r["id"],) for r in old])
            store.db.commit()
    keep = {r["thumb"] for r in store.q("SELECT thumb FROM passed_vehicles WHERE created_at >= ? AND thumb != ''",
                                        (cutoff,))}
    if root.is_dir():
        for p in root.rglob("*.jpg"):
            if str(p.relative_to(root)) not in keep and now - p.stat().st_mtime > DAY:
                paths.append(p)
    root_res = root.resolve()
    for p in dict.fromkeys(paths):
        try:
            if p.is_symlink() or not p.is_file() or root_res not in p.resolve().parents:
                continue
            size = p.stat().st_size
            if not dry_run:
                p.unlink()
            out["files"] += 1
            out["bytes"] += size
        except OSError:
            continue
    if not dry_run and root.is_dir():
        for d in sorted((x for x in root.rglob("*") if x.is_dir()), reverse=True):
            try:
                d.rmdir()                               # only if empty
            except OSError:
                pass
    return out


class PurgeLoop:
    """Runs `purge` once at start and then once a day, on a daemon thread. start() / stop() from the app's lifespan."""

    def __init__(self, store, data_dir, interval_s: float = DAY, log: Callable = print):
        self.store, self.data_dir, self.interval, self.log = store, data_dir, interval_s, log
        self._stop = threading.Event()
        self._thread = None
        self.last = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="passed-purge")
            self._thread.start()
        return self

    def run_once(self) -> dict:
        self.last = purge(self.store, self.data_dir)
        if self.last["rows"]:
            self.store.audit(None, "passed.purge", "passed_vehicles", "", f"{self.last['rows']} records older than "
                             f"{self.last['days']:g} days, {self.last['files']} thumbnails")
        return self.last

    def _run(self):
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:                      # a locked database tonight is tomorrow's problem
                self.log(f"passed-vehicle purge failed: {e}")
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
