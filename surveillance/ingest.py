"""Ingest queue (WP13, R-5): hundreds of videos a day must not all start analysing at once.

Uploads (or a watched folder, or a nightly import) call `enqueue(camera_id)`; a scheduler thread starts at most
`max_parallel` analyses, retries failures with exponential backoff, waits while the disk monitor says "pause", and
puts jobs that were running when the server stopped back in the queue. It never starts a camera that is already
running or already has an active job.

Everything about the analysis itself is injected, so the queue knows nothing about WorkerManager:
    start_fn(camera_id)        begin the analysis; raise NotReady to try again later without counting an attempt
    is_running_fn(camera_id)   True while the analysis (or the zone learning before it) is alive
    status_fn(camera_id)       optional, {"state": "done"/"error"/"stopped"/...}: how a finished run ended
    stop_fn(camera_id)         optional, used when a running job is cancelled
    paused_fn()                optional, True = start nothing new (StorageManager.ingest_paused)

Job states: queued -> running -> done | failed | cancelled. A failed attempt goes back to queued with
`next_try_at` pushed out until `max_attempts` is used up.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL,
    label TEXT DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'queued' CHECK (state IN ('queued','running','done','failed','cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    error TEXT DEFAULT '',
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    next_try_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ingest_state ON ingest_jobs(state, priority);
"""
ACTIVE = ("queued", "running")


class NotReady(Exception):
    """start_fn: the camera cannot start yet (e.g. still learning its zones). Not a failed attempt."""


def ensure_schema(store) -> None:
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


class IngestQueue:
    def __init__(self, store, start_fn: Callable, is_running_fn: Callable, max_parallel: int = 1, *,
                 status_fn: Optional[Callable] = None, stop_fn: Optional[Callable] = None,
                 paused_fn: Optional[Callable] = None, now_fn: Callable = time.time, backoff_base: float = 30.0,
                 backoff_cap: float = 3600.0, start_grace: float = 60.0, tick_sec: float = 2.0):
        self.store, self.start_fn, self.is_running_fn = store, start_fn, is_running_fn
        self.status_fn, self.stop_fn, self.paused_fn = status_fn, stop_fn, paused_fn
        self.max_parallel = max(1, int(max_parallel))
        self.now, self.backoff_base, self.backoff_cap = now_fn, backoff_base, backoff_cap
        self.start_grace, self.tick_sec = start_grace, tick_sec
        self.lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._halt = threading.Event()
        ensure_schema(store)
        # A job that was running when the server stopped did not finish: run it again.
        store.x("UPDATE ingest_jobs SET state='queued', error='interrupted by a restart', started_at=NULL,"
                " next_try_at=0 WHERE state='running'")

    # ------------------------------------------------------------ public
    def enqueue(self, camera_id: int, priority: int = 0, label: str = "", max_attempts: int = 3) -> dict:
        """Queue a camera's analysis. If it already has an active job, that job is returned (and moved up if the
        new priority is higher): the same camera is never queued twice."""
        with self.lock:
            cur = self.store.q("SELECT * FROM ingest_jobs WHERE camera_id=? AND state IN ('queued','running')",
                               (camera_id,))
            if cur:
                job = cur[0]
                if job["state"] == "queued" and priority > job["priority"]:
                    self.store.x("UPDATE ingest_jobs SET priority=? WHERE id=?", (priority, job["id"]))
                return self.job(job["id"])
            jid = self.store.x("INSERT INTO ingest_jobs (camera_id, label, priority, max_attempts, created_at)"
                               " VALUES (?,?,?,?,?)", (camera_id, label, priority, max_attempts, self.now()))
            return self.job(jid)

    def job(self, jid: int) -> Optional[dict]:
        r = self.store.q("SELECT * FROM ingest_jobs WHERE id=?", (jid,))
        return r[0] if r else None

    def jobs(self, state: Optional[str] = None, limit: int = 200) -> list:
        w, a = (" WHERE state=?", (state,)) if state else ("", ())
        return self.store.q("SELECT * FROM ingest_jobs" + w + " ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'queued' "
                            "THEN 1 ELSE 2 END, priority DESC, id DESC LIMIT ?", (*a, limit))

    def stats(self) -> dict:
        counts = {s: 0 for s in ("queued", "running", "done", "failed", "cancelled")}
        for r in self.store.q("SELECT state, COUNT(*) n FROM ingest_jobs GROUP BY state"):
            counts[r["state"]] = r["n"]
        return {"counts": counts, "max_parallel": self.max_parallel, "paused": self._paused()}

    def retry(self, jid: int) -> Optional[dict]:
        with self.lock:
            j = self.job(jid)
            if not j or j["state"] not in ("failed", "cancelled"):
                return None
            if self.store.q("SELECT 1 FROM ingest_jobs WHERE camera_id=? AND state IN ('queued','running')",
                            (j["camera_id"],)):
                return None
            self.store.x("UPDATE ingest_jobs SET state='queued', attempts=0, error='', next_try_at=0,"
                         " finished_at=NULL, started_at=NULL WHERE id=?", (jid,))
            return self.job(jid)

    def cancel(self, jid: int) -> Optional[dict]:
        with self.lock:
            j = self.job(jid)
            if not j or j["state"] not in ACTIVE:
                return None
            if j["state"] == "running" and self.stop_fn:
                try:
                    self.stop_fn(j["camera_id"])
                except Exception:
                    pass
            self.store.x("UPDATE ingest_jobs SET state='cancelled', finished_at=?, error=? WHERE id=?",
                         (self.now(), "cancelled by an administrator", jid))
            return self.job(jid)

    # ------------------------------------------------------------ scheduler
    def _paused(self) -> bool:
        try:
            return bool(self.paused_fn and self.paused_fn())
        except Exception:
            return False

    def _fail_attempt(self, j, why: str, count: bool = True):
        now = self.now()
        attempts = j["attempts"] + (1 if count else 0)
        if count and attempts >= j["max_attempts"]:
            self.store.x("UPDATE ingest_jobs SET state='failed', attempts=?, error=?, finished_at=? WHERE id=?",
                         (attempts, why[:500], now, j["id"]))
        else:
            wait = min(self.backoff_cap, self.backoff_base * (2 ** max(0, attempts - 1))) if count else self.tick_sec
            self.store.x("UPDATE ingest_jobs SET state='queued', attempts=?, error=?, started_at=NULL, next_try_at=?"
                         " WHERE id=?", (attempts, why[:500], now + wait, j["id"]))

    def _settle_running(self):
        now = self.now()
        for j in self.store.q("SELECT * FROM ingest_jobs WHERE state='running'"):
            cid = j["camera_id"]
            try:
                alive = bool(self.is_running_fn(cid))
            except Exception:
                alive = True
            if alive:
                continue
            state = ""
            if self.status_fn:
                try:
                    state = (self.status_fn(cid) or {}).get("state", "")
                except Exception:
                    state = ""
            if state == "error":
                try:
                    err = (self.status_fn(cid) or {}).get("error") or "analysis failed"
                except Exception:
                    err = "analysis failed"
                self._fail_attempt(j, err)
            elif state == "stopped":
                self.store.x("UPDATE ingest_jobs SET state='cancelled', finished_at=?, error='stopped' WHERE id=?",
                             (now, j["id"]))
            elif state == "done" or (not self.status_fn and now - (j["started_at"] or now) >= 0):
                self.store.x("UPDATE ingest_jobs SET state='done', finished_at=?, error='' WHERE id=?", (now, j["id"]))
            elif now - (j["started_at"] or now) >= self.start_grace:
                self._fail_attempt(j, "the analysis never started")

    def tick(self) -> list:
        """One scheduling pass. Returns the camera ids started in this pass."""
        started = []
        with self.lock:
            self._settle_running()
            if self._paused():
                return started
            now = self.now()
            running = self.store.q("SELECT COUNT(*) n FROM ingest_jobs WHERE state='running'")[0]["n"]
            busy = {r["camera_id"] for r in self.store.q("SELECT camera_id FROM ingest_jobs WHERE state='running'")}
            for j in self.store.q("SELECT * FROM ingest_jobs WHERE state='queued' AND next_try_at<=?"
                                  " ORDER BY priority DESC, id", (now,)):
                if running >= self.max_parallel:
                    break
                cid = j["camera_id"]
                if cid in busy:
                    continue
                try:
                    if self.is_running_fn(cid):
                        continue            # already running by hand: leave it, look again next pass
                except Exception:
                    continue
                try:
                    self.start_fn(cid)
                except NotReady:
                    self.store.x("UPDATE ingest_jobs SET next_try_at=? WHERE id=?", (now + 5, j["id"]))
                    continue
                except Exception as e:
                    self._fail_attempt(j, f"{type(e).__name__}: {e}")
                    continue
                self.store.x("UPDATE ingest_jobs SET state='running', started_at=?, error='' WHERE id=?",
                             (now, j["id"]))
                running += 1
                busy.add(cid)
                started.append(cid)
        return started

    def _loop(self):
        while not self._halt.wait(self.tick_sec):
            try:
                self.tick()
            except Exception as e:  # the scheduler must outlive any one bad pass
                print(f"[ingest] scheduler pass failed: {e}")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._halt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ingest-queue")
        self._thread.start()

    def stop(self):
        self._halt.set()
        if self._thread:
            self._thread.join(timeout=5)
