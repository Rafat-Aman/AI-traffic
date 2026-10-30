"""Storage management (WP13, requirement R-5): inventory, retention plans, compaction, disk monitor.

Nothing here deletes on its own. `StorageManager.plan()` is a DRY RUN: it walks the data folder, decides for every
file what the policy says and why, and saves the result as a plan. Only `apply(plan_id, user)` deletes, and it
deletes at most what the plan listed, after asking the same question again (a citation may have been issued
since the plan was made). Every deletion writes an audit row.

Policy (defaults; all editable by an admin, written to the `storage_settings` table)
  raw_video_days        7     Raw uploads are the bulk of the disk (a 10-minute 1080p file is ~0.3-1 GB) and every
                              result worth keeping is already extracted: tracklets, and for each case a 10 s clip
                              plus stills. A week covers re-analysis after a bug fix or a zone change. Held back
                              while the camera is running or queued, or has a pending case or a standing citation.
  dismissed_days        30    Media of rejected cases (false alarms) has no legal value; a month leaves room for an
                              appeal or a reviewer's second look.
  legal_retention_days  1095  Evidence of confirmed cases (3 years). THE AUTHORITY'S DECISION, not ours: until an
                              admin ticks `legal_retention_confirmed`, nothing in this class is auto-deleted.
  delivery_clip_days    30    The driver's 720p copy is regenerable from the sealed original. Never deleted while a
                              standing citation points at it (the driver's link would break).
  embed_days            30    Appearance vectors (MobileNet 576 floats, ReID 2048 floats, ~10 KB per row) are only
                              needed to link vehicles across cameras, which happens within minutes to days. After
                              that the row stays (segment index, journeys, section speed need class, time, speed,
                              colour, global_id) and only the two vector blobs are dropped.
  stale_part_minutes    60    A `.part.mp4` is a clip being written; older than an hour it is a crashed encode and
                              it blocks citing ("still being encoding") forever.
  warn_pct 80 / pause_pct 90  Disk monitor: warn, then stop new ingest (see ingest.py).

Hard rules, independent of the settings
  * A file belonging to a case with a standing citation (issued or paid; not withdrawn) is never deleted: the
    seal lists its hash and `verify` would report it "missing".
  * A pending case keeps all its files.
  * A file referenced by several rows is kept if any one of them says keep.
  * Only regular files inside data/videos and data/evidence are ever removed; data/autozone, data/zone_refs and the
    database are inventory only.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

DEFAULTS = {
    "raw_video_days": 7,
    "dismissed_days": 30,
    "legal_retention_days": 1095,
    "legal_retention_confirmed": False,
    "delivery_clip_days": 30,
    "embed_days": 30,
    "stale_part_minutes": 60,
    "orphan_days": 0,                  # unreferenced files in evidence/: 0 = report only, never delete
    "warn_pct": 80,
    "pause_pct": 90,
}
BOUNDS = {"raw_video_days": (1, 3650), "dismissed_days": (1, 3650), "legal_retention_days": (365, 36500),
          "delivery_clip_days": (1, 3650), "embed_days": (1, 3650), "stale_part_minutes": (10, 1440),
          "orphan_days": (0, 3650), "warn_pct": (10, 98), "pause_pct": (20, 99)}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts"}
DAY = 86400.0
PLAN_TTL = 24 * 3600.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS storage_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS storage_plans (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    created_by TEXT DEFAULT '',
    policy TEXT NOT NULL,
    items TEXT NOT NULL,
    kept TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned',      -- planned / applied
    applied_at REAL,
    applied_by TEXT DEFAULT '',
    result TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS storage_purged (
    path TEXT NOT NULL,
    purged_at REAL NOT NULL,
    plan_id INTEGER,
    rule TEXT DEFAULT '',
    bytes INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_purged_path ON storage_purged(path);
"""


def ensure_schema(store) -> None:
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


class PolicyError(ValueError):
    pass


def validate_policy(data: dict, base: Optional[dict] = None) -> dict:
    out = dict(base or DEFAULTS)
    for k, v in (data or {}).items():
        if k not in DEFAULTS:
            raise PolicyError(f"unknown setting: {k}")
        if isinstance(DEFAULTS[k], bool):
            if not isinstance(v, bool):
                raise PolicyError(f"{k} must be true or false")
            out[k] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            raise PolicyError(f"{k} must be a number")
        lo, hi = BOUNDS[k]
        if not lo <= v <= hi:
            raise PolicyError(f"{k} must be between {lo} and {hi}")
        out[k] = int(v)
    if out["warn_pct"] >= out["pause_pct"]:
        raise PolicyError("warn_pct must be lower than pause_pct")
    return out


def _ts(text) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(text)).timestamp() if text else None
    except ValueError:
        return None


def _rel(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def delivery_rel(clip_rel: str) -> str:
    p = PurePosixPath(clip_rel)
    return str(p.with_name(p.stem + "_delivery.mp4"))


def human(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


class StorageManager:
    def __init__(self, store, data_dir, *, usage_fn: Callable = shutil.disk_usage,
                 now_fn: Callable = time.time, busy_cameras_fn: Optional[Callable] = None):
        """busy_cameras_fn() -> set of camera ids whose video must not be removed right now (running or queued)."""
        self.store, self.data_dir = store, Path(data_dir)
        self.usage_fn, self.now, self.busy_cameras_fn = usage_fn, now_fn, busy_cameras_fn
        ensure_schema(store)
        self._inv_cache = (0.0, None)

    # ------------------------------------------------------------ policy
    def policy(self) -> dict:
        saved = {r["key"]: json.loads(r["value"]) for r in self.store.q("SELECT key, value FROM storage_settings")}
        try:
            return validate_policy(saved)
        except PolicyError:
            return dict(DEFAULTS)

    def set_policy(self, data: dict, user=None) -> dict:
        new = validate_policy(data, self.policy())
        for k, v in new.items():
            self.store.x("INSERT INTO storage_settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET"
                         " value=excluded.value", (k, json.dumps(v)))
        self.store.audit(user, "storage.policy", "settings", "storage", json.dumps(data, sort_keys=True),
                         (user or {}).get("address", ""))
        return new

    # ------------------------------------------------------------ disk monitor
    def disk(self, policy=None) -> dict:
        p = policy or self.policy()
        u = self.usage_fn(str(self.data_dir))
        pct = 100.0 * u.used / u.total if u.total else 0.0
        state = "pause" if pct >= p["pause_pct"] else "warn" if pct >= p["warn_pct"] else "ok"
        msg = {"ok": "", "warn": f"Disk {pct:.0f}% full (warning at {p['warn_pct']}%): run a retention plan.",
               "pause": f"Disk {pct:.0f}% full: new ingest is paused until it drops below {p['pause_pct']}%."}[state]
        return {"total": u.total, "used": u.used, "free": u.free, "pct": round(pct, 1), "state": state,
                "message": msg, "warn_pct": p["warn_pct"], "pause_pct": p["pause_pct"]}

    def ingest_paused(self) -> bool:
        return self.disk()["state"] == "pause"

    # ------------------------------------------------------------ references
    def _references(self):
        """({relative path: [ref dicts]}, {camera id: video relpath-or-abs source})."""
        refs: dict = {}
        cited = {r["violation_id"]: r for r in self.store.q(
            "SELECT violation_id, citation_no, status, delivery_clip FROM citations")}

        def add(path, **ref):
            if path:
                refs.setdefault(str(path), []).append(ref)

        for v in self.store.q("SELECT id, camera_id, status, snapshot, crop, clip, plate_img, evidence,"
                              " reviewed_at, created_at FROM violations"):
            ct = cited.get(v["id"])
            standing = bool(ct and ct["status"] != "withdrawn")
            base = {"vid": v["id"], "camera": v["camera_id"], "status": v["status"], "standing": standing,
                    "citation": ct["citation_no"] if ct else None,
                    "at": _ts(v["reviewed_at"]) or _ts(v["created_at"]) or 0.0}
            paths = [v["snapshot"], v["crop"], v["clip"], v["plate_img"]]
            try:
                paths += [e.get("path") for e in json.loads(v["evidence"] or "[]")]
            except (ValueError, AttributeError):
                pass
            for p in paths:
                add(p, kind="case", **base)
            if v["clip"]:
                add(delivery_rel(v["clip"]), kind="delivery", **base)
            if ct and ct["delivery_clip"]:
                add(ct["delivery_clip"], kind="delivery", **base)
        return refs

    def _camera_holds(self) -> dict:
        """camera id -> reasons its raw video must stay."""
        holds: dict = {}
        for r in self.store.q("SELECT camera_id, COUNT(*) n FROM violations WHERE status='pending' GROUP BY camera_id"):
            holds.setdefault(r["camera_id"], []).append(f"{r['n']} case(s) pending review")
        for r in self.store.q("SELECT v.camera_id, COUNT(*) n FROM citations ct JOIN violations v ON v.id=ct.violation_id"
                              " WHERE ct.status != 'withdrawn' GROUP BY v.camera_id"):
            holds.setdefault(r["camera_id"], []).append(f"{r['n']} standing citation(s)")
        for r in self.store.q("SELECT DISTINCT camera_id FROM runs WHERE status='running' AND ended_at IS NULL"):
            holds.setdefault(r["camera_id"], []).append("an analysis run is open")
        if self.busy_cameras_fn:
            for cid in self.busy_cameras_fn() or ():
                holds.setdefault(cid, []).append("running or queued for analysis")
        return holds

    # ------------------------------------------------------------ inventory
    def _walk(self, sub: str):
        root = self.data_dir / sub
        if not root.is_dir():
            return
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                it = list(os.scandir(d))
            except OSError:
                continue
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    yield Path(e.path), st.st_size, st.st_mtime

    def classify(self, refs, camera_sources, path: Path, rel: str) -> str:
        """Category of one file for the inventory."""
        top = rel.split("/", 1)[0]
        if top == "videos":
            return "raw_video"
        if top == "autozone":
            return "autozone"
        if top == "zone_refs":
            return "zone_refs"
        if rel.endswith(".part.mp4"):
            return "partial_clip"
        rs = refs.get(rel)
        if not rs:
            return "evidence_orphan"
        if any(r["kind"] == "delivery" for r in rs) and all(r["kind"] == "delivery" for r in rs):
            return "delivery_clip"
        if any(r["standing"] for r in rs):
            return "evidence_cited"
        if any(r["status"] == "pending" for r in rs):
            return "evidence_pending"
        if any(r["status"] == "confirmed" for r in rs):
            return "evidence_confirmed"
        return "evidence_dismissed"

    def inventory(self, fresh=False) -> dict:
        t, cached = self._inv_cache
        if cached and not fresh and self.now() - t < 30:
            return cached
        refs = self._references()
        cams = {c["id"]: c for c in self.store.cameras()}
        sources = {str(Path(c["source"]).resolve()): c["id"] for c in cams.values() if c["kind"] == "file"}
        cats: dict = {}
        per_day: dict = {}
        now = self.now()
        for sub in ("videos", "evidence", "autozone", "zone_refs"):
            for p, size, mtime in self._walk(sub):
                rel = _rel(p, self.data_dir)
                if p.name == ".gitkeep":
                    continue
                cat = self.classify(refs, sources, p, rel)
                c = cats.setdefault(cat, {"files": 0, "bytes": 0})
                c["files"] += 1
                c["bytes"] += size
                age_d = int((now - mtime) // DAY)
                if age_d < 30:
                    d = per_day.setdefault(datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"), {"bytes": 0, "files": 0})
                    d["bytes"] += size
                    d["files"] += 1
        tl = self.store.q("SELECT COUNT(*) n, COALESCE(SUM(LENGTH(embed)),0) e, COALESCE(SUM(LENGTH(embed_reid)),0) r,"
                          " COALESCE(SUM(LENGTH(hist)),0) h, COALESCE(SUM(LENGTH(lab)),0) l FROM tracklets")[0]
        dbf = self._db_bytes()
        inv = {"categories": cats, "tracklets": {"rows": tl["n"], "embed_bytes": tl["e"], "reid_bytes": tl["r"],
                                                  "hist_lab_bytes": tl["h"] + tl["l"]},
               "db_bytes": dbf, "growth_by_day": dict(sorted(per_day.items())),
               "files_total": sum(c["files"] for c in cats.values()),
               "bytes_total": sum(c["bytes"] for c in cats.values()), "at": now}
        self._inv_cache = (now, inv)
        return inv

    def _db_bytes(self) -> int:
        try:
            row = self.store.q("PRAGMA database_list")
            main = next((r["file"] for r in row if r["name"] == "main"), "")
        except Exception:
            main = ""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                total += os.path.getsize(main + suffix) if main else 0
            except OSError:
                pass
        return total

    def growth(self, inv=None, disk=None) -> dict:
        """Bytes per day added to media (from file modification times still on the disk, last 14 days) and the days
        until the disk is full at that rate. Files already deleted are not counted, so this is a lower bound."""
        inv = inv or self.inventory()
        disk = disk or self.disk()
        now = self.now()
        days = [datetime.fromtimestamp(now - i * DAY).strftime("%Y-%m-%d") for i in range(14)]
        series = [{"day": d, "bytes": inv["growth_by_day"].get(d, {}).get("bytes", 0)} for d in reversed(days)]
        active = [s["bytes"] for s in series]
        first = next((i for i, b in enumerate(active) if b), None)
        span = len(active) - first if first is not None else 0
        per_day = sum(active) / span if span else 0.0
        return {"series": series, "bytes_per_day": per_day,
                "days_to_full": round(disk["free"] / per_day, 1) if per_day > 0 else None,
                "note": "Estimated from files still on the disk, so deletions make it a lower bound."}

    def summary(self) -> dict:
        pol = self.policy()
        inv = self.inventory()
        disk = self.disk(pol)
        return {"disk": disk, "inventory": inv, "growth": self.growth(inv, disk), "policy": pol,
                "purged_files": self.store.q("SELECT COUNT(*) n FROM storage_purged")[0]["n"]}

    # ------------------------------------------------------------ the decision for one file
    def decide(self, rel: str, size: int, mtime: float, ctx: dict) -> tuple:
        """(action, rule, reason): action is "delete" or "keep"."""
        pol, now, refs, holds = ctx["policy"], ctx["now"], ctx["refs"], ctx["holds"]
        top = rel.split("/", 1)[0]
        if rel.endswith(".part.mp4"):
            age_min = (now - mtime) / 60
            if age_min >= pol["stale_part_minutes"]:
                return "delete", "stale_part", f"unfinished clip from a crashed encode, {age_min:.0f} min old; it blocks citing"
            return "keep", "stale_part", f"may still be encoding ({age_min:.0f} min old)"
        rs = refs.get(rel)
        if top == "videos":
            return self._decide_video(rel, mtime, ctx, size)
        if top != "evidence":
            return "keep", "inventory_only", "not managed"
        if not rs:
            if rel.endswith("_delivery.mp4"):
                age = (now - mtime) / DAY
                if age >= pol["delivery_clip_days"]:
                    return "delete", "delivery_clip", f"delivery copy not referenced by any case, {age:.0f} days old; regenerable"
                return "keep", "delivery_clip", "delivery copy, within its retention"
            if pol["orphan_days"] and (now - mtime) / DAY >= pol["orphan_days"]:
                return "delete", "orphan", f"referenced by no case, older than {pol['orphan_days']} days"
            return "keep", "orphan", "referenced by no case (reported only; orphan_days is 0)" if not pol["orphan_days"] \
                else "referenced by no case, within orphan_days"
        verdicts = []
        for r in rs:
            verdicts.append(self._decide_ref(r, pol, now, mtime))
        keeps = [v for v in verdicts if v[0] == "keep"]
        if keeps:
            return keeps[0]
        return verdicts[0]

    def _decide_ref(self, r, pol, now, mtime) -> tuple:
        if r["standing"]:
            return "keep", "sealed", f"sealed by citation {r['citation']}: deleting would make verification report it missing"
        if r["kind"] == "delivery":
            age = (now - mtime) / DAY
            if age >= pol["delivery_clip_days"]:
                return "delete", "delivery_clip", f"driver copy of case {r['vid']}, {age:.0f} days old; regenerable from the original"
            return "keep", "delivery_clip", "driver copy, within its retention"
        if r["status"] == "pending":
            return "keep", "pending", f"case {r['vid']} is pending review"
        age = (now - r["at"]) / DAY
        if r["status"] == "rejected":
            if age >= pol["dismissed_days"]:
                return "delete", "dismissed", f"case {r['vid']} dismissed {age:.0f} days ago (limit {pol['dismissed_days']})"
            return "keep", "dismissed", f"case {r['vid']} dismissed {age:.0f} days ago, within {pol['dismissed_days']}"
        # confirmed, no standing citation
        if not pol["legal_retention_confirmed"]:
            return "keep", "legal", (f"case {r['vid']} confirmed: legal retention period not yet confirmed by the "
                                     "authority, so nothing is auto-deleted")
        if age >= pol["legal_retention_days"]:
            return "delete", "legal", f"case {r['vid']} confirmed {age:.0f} days ago, past the {pol['legal_retention_days']}-day legal period"
        return "keep", "legal", f"case {r['vid']} confirmed {age:.0f} days ago, within the legal period"

    def _decide_video(self, rel, mtime, ctx, size) -> tuple:
        pol, now = ctx["policy"], ctx["now"]
        age = (now - mtime) / DAY
        path = str((self.data_dir / rel).resolve())
        cid = ctx["sources"].get(path)
        if age < pol["raw_video_days"]:
            return "keep", "raw_video", f"{age:.1f} days old, kept for {pol['raw_video_days']}"
        if cid is not None and ctx["holds"].get(cid):
            return "keep", "raw_video_held", f"camera {cid}: " + "; ".join(ctx["holds"][cid])
        who = f"camera {cid}" if cid is not None else "no camera"
        return "delete", "raw_video", (f"{age:.0f} days old (limit {pol['raw_video_days']}); {who} has no pending "
                                       "case or citation, and each case keeps its own clip and stills")

    # ------------------------------------------------------------ plan
    def _context(self, policy) -> dict:
        cams = self.store.cameras()
        sources = {}
        for c in cams:
            if c["kind"] == "file":
                try:
                    sources[str(Path(c["source"]).resolve())] = c["id"]
                except OSError:
                    pass
        return {"policy": policy, "now": self.now(), "refs": self._references(), "holds": self._camera_holds(),
                "sources": sources}

    def plan(self, user=None) -> dict:
        """DRY RUN. Saves and returns a plan; deletes nothing."""
        pol = self.policy()
        ctx = self._context(pol)
        items, kept = [], {}
        for sub in ("videos", "evidence"):
            for p, size, mtime in self._walk(sub):
                if p.name == ".gitkeep":
                    continue
                rel = _rel(p, self.data_dir)
                action, rule, reason = self.decide(rel, size, mtime, ctx)
                if action == "delete":
                    items.append({"action": "delete_file", "path": rel, "bytes": size, "rule": rule, "reason": reason})
                else:
                    k = kept.setdefault(rule, {"files": 0, "bytes": 0, "example": reason})
                    k["files"] += 1
                    k["bytes"] += size
        # tracklet compaction
        cutoff = datetime.fromtimestamp(ctx["now"] - pol["embed_days"] * DAY).isoformat(timespec="seconds")
        row = self.store.q("SELECT COUNT(*) n, COALESCE(SUM(COALESCE(LENGTH(embed),0)+COALESCE(LENGTH(embed_reid),0)),0) b"
                           " FROM tracklets WHERE created_at < ? AND (embed IS NOT NULL OR embed_reid IS NOT NULL)", (cutoff,))[0]
        if row["n"]:
            items.append({"action": "compact_tracklets", "path": "db:tracklets", "bytes": row["b"], "rule": "compaction",
                          "reason": f"{row['n']} tracklet rows older than {pol['embed_days']} days: drop appearance vectors, "
                                    "keep the rows (class, time, speed, colour, vehicle id)",
                          "rows": row["n"], "cutoff": cutoff})
        for i, it in enumerate(items):
            it["n"] = i
        totals = {"files": sum(1 for i in items if i["action"] == "delete_file"),
                  "bytes": sum(i["bytes"] for i in items)}
        by_rule: dict = {}
        for it in items:
            b = by_rule.setdefault(it["rule"], {"items": 0, "bytes": 0})
            b["items"] += 1
            b["bytes"] += it["bytes"]
        pid = self.store.x("INSERT INTO storage_plans (created_at, created_by, policy, items, kept) VALUES (?,?,?,?,?)",
                           (ctx["now"], (user or {}).get("username") or (user or {}).get("name") or "", json.dumps(pol),
                            json.dumps(items), json.dumps(kept)))
        return {"id": pid, "created_at": ctx["now"], "status": "planned", "items": items, "kept": kept,
                "totals": totals, "by_rule": by_rule, "policy": pol, "dry_run": True}

    def get_plan(self, pid: int) -> Optional[dict]:
        r = self.store.q("SELECT * FROM storage_plans WHERE id=?", (pid,))
        if not r:
            return None
        r = r[0]
        items = json.loads(r["items"])
        return {"id": r["id"], "created_at": r["created_at"], "created_by": r["created_by"], "status": r["status"],
                "applied_at": r["applied_at"], "applied_by": r["applied_by"], "items": items,
                "kept": json.loads(r["kept"]), "policy": json.loads(r["policy"]),
                "result": json.loads(r["result"]) if r["result"] else None,
                "totals": {"files": sum(1 for i in items if i["action"] == "delete_file"),
                           "bytes": sum(i["bytes"] for i in items)}}

    def plans(self, limit=10) -> list:
        return [{"id": r["id"], "created_at": r["created_at"], "status": r["status"], "applied_at": r["applied_at"],
                 "items": len(json.loads(r["items"]))} for r in
                self.store.q("SELECT id, created_at, status, applied_at, items FROM storage_plans ORDER BY id DESC LIMIT ?",
                             (limit,))]

    # ------------------------------------------------------------ apply
    def safe_path(self, rel: str) -> Optional[Path]:
        """The file for a plan item, or None unless it is a regular, non-symlink file under videos/ or evidence/."""
        if not rel or rel.startswith("/") or any(part in ("", ".", "..") for part in rel.split("/")):
            return None
        if rel.split("/", 1)[0] not in ("videos", "evidence"):
            return None
        p = self.data_dir / rel
        if p.is_symlink() or not p.is_file():
            return None
        if self.data_dir.resolve() not in p.resolve().parents:
            return None
        return p

    def apply(self, plan_id: int, user=None) -> dict:
        plan = self.get_plan(plan_id)
        if plan is None:
            raise KeyError(plan_id)
        if plan["status"] != "planned":
            raise PolicyError("this plan was already applied; make a new one")
        now = self.now()
        if now - plan["created_at"] > PLAN_TTL:
            raise PolicyError("this plan is more than 24 hours old; make a new one")
        pol = self.policy()
        ctx = self._context(pol)
        who = (user or {}).get("username") or (user or {}).get("name") or ""
        addr = (user or {}).get("address", "")
        freed, deleted, skipped, errors = 0, [], [], []
        for it in plan["items"]:
            if it["action"] == "delete_file":
                p = self.safe_path(it["path"])
                if p is None:
                    skipped.append({"path": it["path"], "why": "not a regular file under videos/ or evidence/ any more"})
                    continue
                st = p.stat()
                action, rule, reason = self.decide(it["path"], st.st_size, st.st_mtime, ctx)
                if action != "delete":
                    skipped.append({"path": it["path"], "why": f"no longer deletable: {reason}"})
                    continue
                try:
                    p.unlink()
                except OSError as e:
                    errors.append({"path": it["path"], "why": str(e)})
                    continue
                freed += st.st_size
                deleted.append(it["path"])
                self.store.x("INSERT INTO storage_purged (path, purged_at, plan_id, rule, bytes) VALUES (?,?,?,?,?)",
                             (it["path"], now, plan_id, rule, st.st_size))
                self.store.audit(user, "storage.delete", "file", it["path"],
                                 f"plan {plan_id}; rule {rule}; {st.st_size} bytes; {reason}", addr)
                self._prune_empty(p.parent)
            elif it["action"] == "compact_tracklets":
                n = self.store.q("SELECT COUNT(*) n FROM tracklets WHERE created_at < ? AND (embed IS NOT NULL OR"
                                 " embed_reid IS NOT NULL)", (it["cutoff"],))[0]["n"]
                b = self.store.q("SELECT COALESCE(SUM(COALESCE(LENGTH(embed),0)+COALESCE(LENGTH(embed_reid),0)),0) b"
                                 " FROM tracklets WHERE created_at < ?", (it["cutoff"],))[0]["b"]
                self.store.x("UPDATE tracklets SET embed=NULL, embed_reid=NULL, embed_model=NULL, embed_reid_model=NULL"
                             " WHERE created_at < ? AND (embed IS NOT NULL OR embed_reid IS NOT NULL)", (it["cutoff"],))
                freed += b
                deleted.append(f"db:tracklets ({n} rows)")
                self.store.audit(user, "storage.compact", "tracklets", "embeddings",
                                 f"plan {plan_id}; {n} rows older than {it['cutoff']}; {b} bytes of vectors dropped", addr)
        result = {"deleted": len(deleted), "freed_bytes": freed, "skipped": skipped, "errors": errors, "paths": deleted}
        self.store.x("UPDATE storage_plans SET status='applied', applied_at=?, applied_by=?, result=? WHERE id=?",
                     (now, who, json.dumps(result), plan_id))
        self.store.audit(user, "storage.apply", "plan", plan_id,
                         f"{len(deleted)} items, {freed} bytes freed, {len(skipped)} skipped, {len(errors)} errors", addr)
        self._inv_cache = (0.0, None)
        return result

    def _prune_empty(self, d: Path):
        root = (self.data_dir / "evidence").resolve()
        try:
            while d.resolve() != root and root in d.resolve().parents and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
        except OSError:
            pass
