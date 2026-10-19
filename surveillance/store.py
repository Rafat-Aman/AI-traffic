"""SQLite storage: cameras, versioned zone configs, analysis runs and violations (with review status)."""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT DEFAULT '',
    kind TEXT NOT NULL CHECK (kind IN ('file', 'live')),
    source TEXT NOT NULL,
    start_time TEXT,                -- recording start (files), local time ISO
    start_time_source TEXT DEFAULT '',
    settings TEXT DEFAULT '{}',
    learning TEXT DEFAULT '{}',     -- traffic direction statistics per road zone
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS zone_versions (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    config TEXT NOT NULL,
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    zone_version_id INTEGER,
    mode TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    frames INTEGER DEFAULT 0,
    video_seconds REAL DEFAULT 0,
    avg_fps REAL,
    status TEXT DEFAULT 'running',
    error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    zone_version_id INTEGER,
    code TEXT NOT NULL,
    title TEXT NOT NULL,
    track_id INTEGER,
    cls TEXT,
    class_conf REAL,
    video_time_s REAL,
    started_s REAL,
    wall_time TEXT,
    zone TEXT,
    areas TEXT,
    kph REAL,
    details TEXT,
    snapshot TEXT,
    crop TEXT,
    clip TEXT,
    plate_status TEXT DEFAULT '',   -- '' (not requested) / pending / read / partial / unreadable
    plate_text TEXT DEFAULT '',     -- Bangla, e.g. ঢাকা মেট্রো-গ ১২-৩৪৫৬
    plate_text_en TEXT DEFAULT '',  -- transliterated, e.g. DHAKA METRO-GA 12-3456
    plate_conf REAL,
    plate_img TEXT,
    evidence TEXT DEFAULT '[]',     -- extra photos: [{"label", "time_s", "wall_time", "path"}]
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'rejected')),
    note TEXT DEFAULT '',
    reviewed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_viol_cam ON violations(camera_id, created_at);
CREATE INDEX IF NOT EXISTS ix_viol_status ON violations(status);
"""

DEFAULT_SETTINGS = {
    "model": "bd_vehicles_best.pt",
    "imgsz": 640,
    "backend": "auto",            # auto = CoreML on Apple Silicon, else GPU / CPU
    "tracker": "bytetrack",
    "analysis_fps": 30,           # files: frames per second analysed (60 fps video -> every 2nd frame)
    "max_width": 0,               # analysis width; 0 = full resolution (needed to see helmets and plates)
    "realtime": False,            # files: play at real speed, like a live camera
    "auto_roi": True,
    "anpr": True,                 # read number plates of violating vehicles (needs the plate models)
    "helmet": True,               # V6 no-helmet detection (needs the helmet model)
    "clip_pre_sec": 5,
    "clip_post_sec": 5,
}


def now():
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript(SCHEMA)
            # Databases created by earlier versions: add the columns they are missing.
            have = {r[1] for r in self.db.execute("PRAGMA table_info(violations)")}
            for col, decl in (("plate_status", "TEXT DEFAULT ''"), ("plate_text", "TEXT DEFAULT ''"),
                              ("plate_text_en", "TEXT DEFAULT ''"), ("plate_conf", "REAL"), ("plate_img", "TEXT"),
                              ("evidence", "TEXT DEFAULT '[]'")):
                if col not in have:
                    self.db.execute(f"ALTER TABLE violations ADD COLUMN {col} {decl}")
            self.db.commit()

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------ cameras
    def cameras(self):
        return [self._camera(r) for r in self.q("SELECT * FROM cameras ORDER BY id")]

    def camera(self, cid):
        rows = self.q("SELECT * FROM cameras WHERE id=?", (cid,))
        return self._camera(rows[0]) if rows else None

    @staticmethod
    def _camera(r):
        r["settings"] = {**DEFAULT_SETTINGS, **json.loads(r["settings"] or "{}")}
        r["learning"] = json.loads(r["learning"] or "{}")
        return r

    def add_camera(self, name, kind, source, location="", start_time=None, start_time_source="", settings=None):
        return self.x("INSERT INTO cameras (name, location, kind, source, start_time, start_time_source, settings,"
                      " created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (name, location, kind, source, start_time, start_time_source, json.dumps(settings or {}),
                       now()))

    def update_camera(self, cid, **fields):
        allowed = {"name", "location", "source", "start_time", "start_time_source", "settings", "learning"}
        sets, args = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=?")
            args.append(json.dumps(v) if k in ("settings", "learning") else v)
        if sets:
            self.x(f"UPDATE cameras SET {', '.join(sets)} WHERE id=?", (*args, cid))

    def delete_camera(self, cid):
        self.x("DELETE FROM cameras WHERE id=?", (cid,))

    # ------------------------------------------------------------ zones
    def zones(self, cid):
        """Latest zone config of a camera: (version row with parsed config) or None."""
        rows = self.q("SELECT * FROM zone_versions WHERE camera_id=? ORDER BY version DESC LIMIT 1", (cid,))
        if not rows:
            return None
        rows[0]["config"] = json.loads(rows[0]["config"])
        return rows[0]

    def zone_history(self, cid):
        return self.q("SELECT id, version, note, created_at FROM zone_versions WHERE camera_id=? "
                      "ORDER BY version DESC", (cid,))

    def save_zones(self, cid, config, note=""):
        with self.lock:
            v = self.q("SELECT COALESCE(MAX(version), 0) AS v FROM zone_versions WHERE camera_id=?", (cid,))[0]["v"]
            return self.x("INSERT INTO zone_versions (camera_id, version, config, note, created_at) "
                          "VALUES (?,?,?,?,?)", (cid, v + 1, json.dumps(config), note, now())), v + 1

    # ------------------------------------------------------------ runs
    def start_run(self, cid, zone_version_id, mode):
        return self.x("INSERT INTO runs (camera_id, zone_version_id, mode, started_at) VALUES (?,?,?,?)",
                      (cid, zone_version_id, mode, now()))

    def end_run(self, run_id, status, frames, video_seconds, avg_fps, error=""):
        self.x("UPDATE runs SET ended_at=?, status=?, frames=?, video_seconds=?, avg_fps=?, error=? WHERE id=?",
               (now(), status, frames, video_seconds, avg_fps, error, run_id))

    def runs(self, cid, limit=20):
        return self.q("SELECT r.*, (SELECT COUNT(*) FROM violations v WHERE v.run_id=r.id) AS violations "
                      "FROM runs r WHERE camera_id=? ORDER BY id DESC LIMIT ?", (cid, limit))

    # ------------------------------------------------------------ violations
    def add_violation(self, cid, run_id, zone_version_id, rec, snapshot, crop, clip, plate_status="", evidence=()):
        return self.x(
            "INSERT INTO violations (camera_id, run_id, zone_version_id, code, title, track_id, cls, class_conf,"
            " video_time_s, started_s, wall_time, zone, areas, kph, details, snapshot, crop, clip, plate_status,"
            " evidence, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, run_id, zone_version_id, rec["code"], rec["violation"], rec["track_id"], rec["class"],
             rec["class_confidence"], rec["video_time_s"], rec["started_s"], rec["wall_time"], rec["zone"],
             rec["areas"], rec["kph"], json.dumps(rec["details"]), snapshot, crop, clip, plate_status,
             json.dumps(list(evidence)), now()))

    def set_plate(self, vid, status, text="", text_en="", conf=None, img=None):
        self.x("UPDATE violations SET plate_status=?, plate_text=?, plate_text_en=?, plate_conf=?, plate_img=? "
               "WHERE id=?", (status, text, text_en, conf, img, vid))

    def violation(self, vid):
        rows = self.violations(ids=[vid])
        return rows[0] if rows else None

    @staticmethod
    def _filters(camera_id=None, code=None, status=None, run_id=None, ids=None, plate=None):
        where, args = [], []
        if plate:  # matches Bangla or Latin text, ignoring spaces and dashes
            key = "%" + "".join(ch for ch in plate.upper() if ch not in " -") + "%"
            where.append("(REPLACE(REPLACE(UPPER(v.plate_text_en), ' ', ''), '-', '') LIKE ? OR "
                         "REPLACE(REPLACE(v.plate_text, ' ', ''), '-', '') LIKE ?)")
            args += [key, key]
        for col, val in (("v.camera_id", camera_id), ("v.code", code), ("v.status", status), ("v.run_id", run_id)):
            if val not in (None, ""):
                where.append(f"{col}=?")
                args.append(val)
        if ids:
            where.append(f"v.id IN ({','.join('?' * len(ids))})")
            args.extend(ids)
        return (" WHERE " + " AND ".join(where) if where else ""), args

    def count(self, **filters):
        w, args = self._filters(**filters)
        return self.q("SELECT COUNT(*) AS n FROM violations v" + w, args)[0]["n"]

    def violations(self, camera_id=None, code=None, status=None, run_id=None, ids=None, limit=100, offset=0,
                   order="DESC", plate=None):
        w, args = self._filters(camera_id, code, status, run_id, ids, plate)
        sql = ("SELECT v.*, c.name AS camera_name, c.location AS camera_location FROM violations v "
               "JOIN cameras c ON c.id=v.camera_id" + w +
               f" ORDER BY v.id {'ASC' if order == 'ASC' else 'DESC'} LIMIT ? OFFSET ?")
        rows = self.q(sql, (*args, limit, offset))
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
            r["evidence"] = json.loads(r.get("evidence") or "[]")
        return rows

    def count_violations(self, **filters):
        where, args = [], []
        for col, val in filters.items():
            if val not in (None, ""):
                where.append(f"{col}=?")
                args.append(val)
        return self.q("SELECT COUNT(*) AS n FROM violations" + (" WHERE " + " AND ".join(where) if where else ""),
                      args)[0]["n"]

    def review(self, vid, status, note=""):
        self.x("UPDATE violations SET status=?, note=?, reviewed_at=? WHERE id=?", (status, note, now(), vid))

    def clear_violations(self, cid):
        self.x("DELETE FROM violations WHERE camera_id=?", (cid,))
        self.x("DELETE FROM runs WHERE camera_id=?", (cid,))

    def stats(self, camera_id=None, include_rejected=False, days=30, basis="event"):
        # event: when it happened (wall time of the video, if known); detected: when the system found it
        when = "v.created_at" if basis == "detected" else "COALESCE(NULLIF(v.wall_time, ''), v.created_at)"
        where, args = [f"datetime({when}) >= datetime('now', 'localtime', ?)"], [f"-{int(days)} days"]
        if camera_id:
            where.append("v.camera_id=?")
            args.append(camera_id)
        if not include_rejected:
            where.append("v.status != 'rejected'")
        w = " WHERE " + " AND ".join(where)
        # Hour of day: the real (wall) time when known, otherwise when it was detected.
        hour = "CAST(strftime('%H', COALESCE(NULLIF(v.wall_time, ''), v.created_at)) AS INTEGER)"
        day = "date(COALESCE(NULLIF(v.wall_time, ''), v.created_at))"
        return {
            "by_code": self.q(f"SELECT code, title, COUNT(*) AS n FROM violations v{w} GROUP BY code ORDER BY code",
                              args),
            "by_class": self.q(f"SELECT cls, COUNT(*) AS n FROM violations v{w} GROUP BY cls ORDER BY n DESC", args),
            "by_hour": self.q(f"SELECT {hour} AS hour, COUNT(*) AS n FROM violations v{w} GROUP BY hour", args),
            "by_day": self.q(f"SELECT {day} AS day, code, COUNT(*) AS n FROM violations v{w} "
                             f"GROUP BY day, code ORDER BY day", args),
            "by_camera": self.q(f"SELECT c.name AS camera, COUNT(*) AS n FROM violations v JOIN cameras c "
                                f"ON c.id=v.camera_id{w} GROUP BY v.camera_id ORDER BY n DESC", args),
            "by_status": self.q("SELECT status, COUNT(*) AS n FROM violations v" +
                                (" WHERE v.camera_id=?" if camera_id else "") + " GROUP BY status",
                                [camera_id] if camera_id else []),
        }
