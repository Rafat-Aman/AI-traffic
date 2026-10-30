"""SQLite storage: cameras, versioned zone configs, analysis runs and violations (with review status)."""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .citation import ADVISORY_CODES
from .disputes import ensure_schema as ensure_disputes_schema
from .embed import MOBILENET_TAG
from .impact import ensure_schema as ensure_impact_schema

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
    vlm_status TEXT DEFAULT '',     -- '' (never asked) / pending / done / failed
    vlm TEXT DEFAULT '',            -- surveillance.vlm.Explanation as JSON: the reasoning over this case
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'rejected')),
    note TEXT DEFAULT '',
    reviewed_at TEXT,
    reviewed_by INTEGER,            -- deliberately NOT a foreign key: who decided a case has to survive
    reviewed_by_name TEXT DEFAULT '',  -- the account being closed, so the name is the record and the id
                                    -- is only a convenience for joining while the account still exists
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('driver', 'controller', 'admin')),
    name TEXT DEFAULT '',
    username TEXT UNIQUE,           -- staff sign-in name; NULL for drivers
    password_hash TEXT,             -- staff only
    licence_no TEXT UNIQUE,         -- drivers: normalised, the identifier they are looked up by
    phone TEXT DEFAULT '',          -- drivers: where the one-time code goes
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY,
    plate_key TEXT NOT NULL UNIQUE, -- normalised: upper case, no spaces or dashes
    plate_en TEXT DEFAULT '',       -- as shown to people
    plate_bn TEXT DEFAULT '',
    owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    source TEXT DEFAULT 'claimed',  -- 'admin' (loaded from a registry) / 'claimed' (the driver said so)
    verified INTEGER NOT NULL DEFAULT 0,
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_vehicle_owner ON vehicles(owner_id);
CREATE TABLE IF NOT EXISTS login_codes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,        -- the code itself is never stored
    expires_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_code_user ON login_codes(user_id, used);
CREATE TABLE IF NOT EXISTS sessions (
    fingerprint TEXT PRIMARY KEY,   -- sha256 of the cookie value, so the database holds no usable token
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    expires_at REAL NOT NULL,
    address TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_session_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    user_id INTEGER,                -- kept even if the account is later removed
    who TEXT DEFAULT '',            -- name / username as it was at the time
    role TEXT DEFAULT '',
    action TEXT NOT NULL,           -- 'violation.confirm', 'vehicle.verify', 'user.create', ...
    object_type TEXT DEFAULT '',
    object_id TEXT DEFAULT '',
    detail TEXT DEFAULT '',
    address TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_audit_at ON audit_log(at);
CREATE INDEX IF NOT EXISTS ix_audit_object ON audit_log(object_type, object_id);
CREATE TABLE IF NOT EXISTS tracklets (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,      -- the tracker's own id: only unique within one run of one camera
    cls TEXT,
    class_conf REAL,
    zone TEXT DEFAULT '',
    t_cross REAL,                   -- wall clock (epoch seconds) at the zone's reference line
    video_t REAL,                   -- the same moment as an offset into the video
    lateral REAL,                   -- metres across the carriageway
    speed REAL,                     -- m/s
    width REAL,                     -- physical width in metres
    direction TEXT DEFAULT 'unknown',
    extrap_m REAL DEFAULT 0,
    hist BLOB,                      -- HSV colour histogram  (float32)
    lab BLOB,                       -- CIELAB mean/spread    (float32)
    embed BLOB,                     -- MobileNet appearance vector (float32, surveillance.embed)
    embed_model TEXT,               -- Embedder.tag that made `embed` (NULL on rows older than this column: MobileNet)
    embed_reid BLOB,                -- ReID vector (float32), written later by the background worker
    embed_reid_model TEXT,          -- Embedder.tag that made `embed_reid`
    global_id INTEGER,              -- the vehicle this is an observation of, across all cameras
    link_source TEXT DEFAULT '',    -- how it got that id: online / batch
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_track_cam ON tracklets(camera_id, t_cross);
CREATE INDEX IF NOT EXISTS ix_track_global ON tracklets(global_id);
CREATE INDEX IF NOT EXISTS ix_track_run ON tracklets(run_id);
CREATE INDEX IF NOT EXISTS ix_viol_cam ON violations(camera_id, created_at);
CREATE INDEX IF NOT EXISTS ix_viol_status ON violations(status);
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY,
    citation_no TEXT NOT NULL UNIQUE,   -- DMP-2026-000123, the number printed on the notice
    year INTEGER NOT NULL,              -- numbering restarts each year, so (year, seq) is the real key
    seq INTEGER NOT NULL,
    violation_id INTEGER NOT NULL UNIQUE,  -- one citation per case. Deliberately NOT a foreign key: a citation
                                        -- is a legal record and has to outlive the operational data it was
                                        -- made from (the manifest below keeps the facts of the case)
    issued_at TEXT NOT NULL,
    issued_by INTEGER,                  -- not a foreign key either, for the reason given on violations.reviewed_by
    issued_by_name TEXT DEFAULT '',
    fine_bdt INTEGER,                   -- NULL = not set: the amounts are the authority's to supply
    status TEXT NOT NULL DEFAULT 'issued' CHECK (status IN ('issued', 'withdrawn', 'paid')),
    bundle_sha256 TEXT NOT NULL,        -- SHA-256 of `manifest`: the figure printed on the notice
    manifest TEXT NOT NULL,             -- the canonical JSON that was hashed, byte for byte
    delivery_clip TEXT,                 -- the driver's shorter copy of the clip; the original is what is sealed
    withdrawn_at TEXT,
    withdrawn_by INTEGER,
    withdrawn_by_name TEXT DEFAULT '',
    withdraw_reason TEXT DEFAULT '',
    UNIQUE (year, seq)
);
CREATE INDEX IF NOT EXISTS ix_citation_status ON citations(status);
"""

# What a case row carries about its citation, joined in so that no caller has to ask a second time (all NULL
# when the case has none). `delivery_clip` is the driver's shorter copy of the clip.
CITATION_COLUMNS = (" ct.citation_no, ct.status AS citation_status, ct.issued_at AS citation_issued_at,"
                    " ct.fine_bdt AS citation_fine_bdt, ct.delivery_clip AS delivery_clip")
CITATION_JOIN = " LEFT JOIN citations ct ON ct.violation_id=v.id"

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
    "reid": True,                 # summarise each vehicle's pass so it can be linked across cameras
    "impact": True,               # measure queue / speed / occupancy per road zone and score each violation's effect
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
                              ("evidence", "TEXT DEFAULT '[]'"), ("vlm_status", "TEXT DEFAULT ''"),
                              ("vlm", "TEXT DEFAULT ''"), ("reviewed_by", "INTEGER"),
                              ("reviewed_by_name", "TEXT DEFAULT ''")):
                if col not in have:
                    self.db.execute(f"ALTER TABLE violations ADD COLUMN {col} {decl}")
            have_t = {r[1] for r in self.db.execute("PRAGMA table_info(tracklets)")}
            if have_t and "embed" not in have_t:
                self.db.execute("ALTER TABLE tracklets ADD COLUMN embed BLOB")
            for col, decl in (("embed_model", "TEXT"), ("embed_reid", "BLOB"), ("embed_reid_model", "TEXT")):
                if have_t and col not in have_t:
                    self.db.execute(f"ALTER TABLE tracklets ADD COLUMN {col} {decl}")
            if have_t and "embed_model" not in have_t:
                # every vector stored before the column existed was made by MobileNet (576 float32 numbers)
                self.db.execute("UPDATE tracklets SET embed_model=? WHERE embed IS NOT NULL",
                                (MOBILENET_TAG,))
            self.db.commit()
        ensure_disputes_schema(self)  # drivers' disputes of citations
        ensure_impact_schema(self)    # time-of-day baseline, event impacts, fitted weights: no-op once they exist

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------ accounts
    def add_user(self, role, name="", username=None, password_hash=None, licence_no=None, phone=""):
        return self.x("INSERT INTO users (role, name, username, password_hash, licence_no, phone,"
                      " created_at) VALUES (?,?,?,?,?,?,?)",
                      (role, name, username or None, password_hash, licence_no or None, phone, now()))

    def user(self, uid):
        r = self.q("SELECT * FROM users WHERE id=?", (uid,))
        return r[0] if r else None

    def user_by_username(self, username):
        r = self.q("SELECT * FROM users WHERE username=? AND active=1", (username,))
        return r[0] if r else None

    def user_by_licence(self, licence_no):
        r = self.q("SELECT * FROM users WHERE licence_no=? AND active=1", (licence_no,))
        return r[0] if r else None

    def users(self, role=None):
        w, a = (" WHERE role=?", (role,)) if role else ("", ())
        return self.q("SELECT id, role, name, username, licence_no, phone, active, created_at, last_login"
                      " FROM users" + w + " ORDER BY role, COALESCE(name, username, licence_no)", a)

    def update_user(self, uid, **fields):
        allowed = {"name", "phone", "active", "role", "password_hash", "username", "licence_no"}
        sets, args = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                args.append(v)
        if sets:
            self.x(f"UPDATE users SET {', '.join(sets)} WHERE id=?", (*args, uid))

    def touch_login(self, uid):
        self.x("UPDATE users SET last_login=? WHERE id=?", (now(), uid))

    def count_users(self, role=None):
        w, a = (" WHERE role=?", (role,)) if role else ("", ())
        return self.q("SELECT COUNT(*) AS n FROM users" + w, a)[0]["n"]

    # ------------------------------------------------------------ vehicles owned by a driver
    def add_vehicle(self, plate_key, owner_id, plate_en="", plate_bn="", source="claimed", verified=0,
                    note=""):
        return self.x("INSERT INTO vehicles (plate_key, plate_en, plate_bn, owner_id, source, verified,"
                      " note, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (plate_key, plate_en, plate_bn, owner_id, source, int(verified), note, now()))

    def vehicle_by_plate(self, plate_key):
        r = self.q("SELECT * FROM vehicles WHERE plate_key=?", (plate_key,))
        return r[0] if r else None

    def vehicles_of(self, owner_id, verified_only=False):
        w = " AND verified=1" if verified_only else ""
        return self.q(f"SELECT * FROM vehicles WHERE owner_id=?{w} ORDER BY created_at DESC", (owner_id,))

    def all_vehicles(self, pending_only=False):
        w = " WHERE v.verified=0" if pending_only else ""
        return self.q("SELECT v.*, u.name AS owner_name, u.licence_no AS owner_licence FROM vehicles v"
                      " LEFT JOIN users u ON u.id=v.owner_id" + w + " ORDER BY v.created_at DESC")

    def set_vehicle(self, vid, **fields):
        allowed = {"verified", "note", "owner_id", "plate_en", "plate_bn"}
        sets, args = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                args.append(v)
        if sets:
            self.x(f"UPDATE vehicles SET {', '.join(sets)} WHERE id=?", (*args, vid))

    def delete_vehicle(self, vid):
        self.x("DELETE FROM vehicles WHERE id=?", (vid,))

    def violations_for_plates(self, plate_keys, limit=200, offset=0, exclude_codes=None):
        """Cases against any of these plates FOR WHICH A CITATION HAS BEEN ISSUED (standing or withdrawn; a
        withdrawn one stays so the driver sees the outcome). Drivers only ever see this, never the whole table:
        a case that no citation was issued for is not theirs to see (team decision 2026-10-30).

        Matched on the normalised plate, the same way the search box does, because the stored text carries
        spaces and dashes that nobody types consistently.

        Advisory codes (speed) are left out in the query itself, not by whoever formats the answer: a rule
        that may not support a citation must not reach a driver through a code path that forgot to filter.
        Leave `exclude_codes` unset for the configured policy; pass a collection only to change it on purpose."""
        if not plate_keys:
            return [], 0
        skip = sorted(ADVISORY_CODES if exclude_codes is None else exclude_codes)
        norm = "REPLACE(REPLACE(UPPER(v.plate_text_en), ' ', ''), '-', '')"
        marks = ",".join("?" * len(plate_keys))
        where = (f" WHERE ct.citation_no IS NOT NULL AND {norm} IN ({marks})")
        args = list(plate_keys)
        if skip:
            where += f" AND v.code NOT IN ({','.join('?' * len(skip))})"
            args += skip
        rows = self.q("SELECT v.*, c.name AS camera_name, c.location AS camera_location," + CITATION_COLUMNS +
                      " FROM violations v JOIN cameras c ON c.id=v.camera_id" + CITATION_JOIN + where +
                      " ORDER BY v.id DESC LIMIT ? OFFSET ?", (*args, limit, offset))
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
            r["evidence"] = json.loads(r.get("evidence") or "[]")
            r["vlm"] = json.loads(r["vlm"]) if r.get("vlm") else None
        total = self.q(f"SELECT COUNT(*) AS n FROM violations v{CITATION_JOIN}{where}", tuple(args))[0]["n"]
        return rows, total

    # ------------------------------------------------------------ sessions
    def add_session(self, fingerprint, user_id, role, expires_at, address=""):
        self.x("INSERT OR REPLACE INTO sessions (fingerprint, user_id, role, expires_at, address,"
               " created_at) VALUES (?,?,?,?,?,?)",
               (fingerprint, user_id, role, expires_at, address, now()))

    def session(self, fingerprint):
        r = self.q("SELECT s.*, u.name, u.username, u.licence_no, u.active FROM sessions s"
                   " JOIN users u ON u.id=s.user_id WHERE s.fingerprint=?", (fingerprint,))
        return r[0] if r else None

    def drop_session(self, fingerprint):
        self.x("DELETE FROM sessions WHERE fingerprint=?", (fingerprint,))

    def drop_sessions_of(self, user_id):
        self.x("DELETE FROM sessions WHERE user_id=?", (user_id,))

    def purge_sessions(self, before):
        self.x("DELETE FROM sessions WHERE expires_at < ?", (before,))

    # ------------------------------------------------------------ one-time login codes
    def add_login_code(self, user_id, code_hash, expires_at):
        self.x("UPDATE login_codes SET used=1 WHERE user_id=? AND used=0", (user_id,))
        return self.x("INSERT INTO login_codes (user_id, code_hash, expires_at, created_at)"
                      " VALUES (?,?,?,?)", (user_id, code_hash, expires_at, now()))

    def live_login_code(self, user_id, now_ts):
        r = self.q("SELECT * FROM login_codes WHERE user_id=? AND used=0 AND expires_at > ?"
                   " ORDER BY id DESC LIMIT 1", (user_id, now_ts))
        return r[0] if r else None

    def bump_code_attempts(self, code_id):
        self.x("UPDATE login_codes SET attempts=attempts+1 WHERE id=?", (code_id,))

    def spend_login_code(self, code_id):
        self.x("UPDATE login_codes SET used=1 WHERE id=?", (code_id,))

    # ------------------------------------------------------------ audit
    def audit(self, user, action, object_type="", object_id="", detail="", address=""):
        """Record that somebody did something. `user` is a session dict or None for the system itself."""
        u = user or {}
        self.x("INSERT INTO audit_log (at, user_id, who, role, action, object_type, object_id, detail,"
               " address) VALUES (?,?,?,?,?,?,?,?,?)",
               (now(), u.get("user_id"), u.get("name") or u.get("username") or u.get("licence_no") or
                "system", u.get("role", ""), action, object_type, str(object_id), detail, address))

    def audit_trail(self, object_type=None, object_id=None, user_id=None, limit=200):
        where, args = [], []
        for col, val in (("object_type", object_type), ("object_id", object_id), ("user_id", user_id)):
            if val not in (None, ""):
                where.append(f"{col}=?")
                args.append(str(val))
        sql = "SELECT * FROM audit_log" + (" WHERE " + " AND ".join(where) if where else "")
        return self.q(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))

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

    # ------------------------------------------------------------ tracklets (cross-camera Re-ID)
    def add_tracklet(self, camera_id, run_id, t):
        """Store one vehicle's pass through one camera. `t` is a surveillance.reid.Tracklet."""
        return self.x(
            "INSERT INTO tracklets (camera_id, run_id, track_id, cls, class_conf, zone, t_cross, video_t,"
            " lateral, speed, width, direction, extrap_m, hist, lab, embed, embed_model,"
            " embed_reid, embed_reid_model, global_id, link_source, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (camera_id, run_id, t.track_id, t.cls, getattr(t, "class_conf", None), t.zone, t.t_cross,
             getattr(t, "video_t", None), t.lateral, t.speed, t.width, t.direction, t.extrap_m,
             None if t.hist is None else t.hist.astype("float32").tobytes(),
             None if t.lab is None else t.lab.astype("float32").tobytes(),
             None if getattr(t, "embed", None) is None else t.embed.astype("float32").tobytes(),
             getattr(t, "embed_model", None) if getattr(t, "embed", None) is not None else None,
             None if getattr(t, "embed_reid", None) is None else t.embed_reid.astype("float32").tobytes(),
             getattr(t, "embed_reid_model", None) if getattr(t, "embed_reid", None) is not None else None,
             t.global_id, getattr(t, "link_source", ""), now()))

    def set_tracklet_reid(self, tracklet_id, vec, model):
        """Attach the ReID vector (made after the row was written) to a stored tracklet."""
        self.x("UPDATE tracklets SET embed_reid=?, embed_reid_model=? WHERE id=?",
               (vec.astype("float32").tobytes(), model, tracklet_id))

    def tracklets(self, camera_id=None, run_id=None, global_id=None, since=None, until=None, limit=100000):
        where, args = [], []
        for col, val in (("camera_id", camera_id), ("run_id", run_id), ("global_id", global_id)):
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        if since is not None:
            where.append("t_cross >= ?")
            args.append(since)
        if until is not None:
            where.append("t_cross <= ?")
            args.append(until)
        sql = "SELECT * FROM tracklets" + (" WHERE " + " AND ".join(where) if where else "")
        return self.q(sql + " ORDER BY t_cross LIMIT ?", (*args, limit))

    def tracklet_for_violation(self, v):
        """The tracklet of the vehicle a violation is about, if its pass was recorded."""
        rows = self.q("SELECT * FROM tracklets WHERE camera_id=? AND track_id=? AND (run_id=? OR ? IS NULL)"
                      " ORDER BY ABS(COALESCE(video_t, 0) - ?) LIMIT 1",
                      (v["camera_id"], v["track_id"], v.get("run_id"), v.get("run_id"),
                       v.get("video_time_s") or 0))
        return rows[0] if rows else None

    def journey_for_violation(self, v):
        """Where else this vehicle was seen, across every camera, ordered in time. [] when not linked."""
        t = self.tracklet_for_violation(v)
        if not t or not t["global_id"]:
            return []
        return self.journey(t["global_id"])

    def set_global_ids(self, assignments, source="batch"):
        """assignments: [(tracklet id, global id)], written in one transaction."""
        with self.lock:
            self.db.executemany("UPDATE tracklets SET global_id=?, link_source=? WHERE id=?",
                                [(g, source, i) for i, g in assignments])
            self.db.commit()

    def next_global_id(self):
        return (self.q("SELECT COALESCE(MAX(global_id), 0) AS g FROM tracklets")[0]["g"] or 0) + 1

    def vehicles(self, limit=50, min_cameras=2):
        """Vehicles seen by at least `min_cameras` cameras, most recent first."""
        return self.q(
            "SELECT global_id, COUNT(*) AS passes, COUNT(DISTINCT camera_id) AS cameras,"
            " MIN(t_cross) AS first_seen, MAX(t_cross) AS last_seen,"
            " GROUP_CONCAT(DISTINCT cls) AS classes FROM tracklets WHERE global_id IS NOT NULL"
            " GROUP BY global_id HAVING cameras >= ? ORDER BY last_seen DESC LIMIT ?", (min_cameras, limit))

    def journey(self, global_id):
        """Every camera this vehicle passed, in order."""
        return self.q(
            "SELECT t.*, c.name AS camera_name, c.location AS camera_location FROM tracklets t"
            " JOIN cameras c ON c.id=t.camera_id WHERE t.global_id=? ORDER BY t.t_cross", (global_id,))

    def clear_tracklets(self, camera_id=None):
        if camera_id is None:
            self.x("DELETE FROM tracklets")
        else:
            self.x("DELETE FROM tracklets WHERE camera_id=?", (camera_id,))

    def set_vlm(self, vid, status, explanation=None):
        """status: pending (queued) / done / failed. `explanation` is a surveillance.vlm.Explanation dict."""
        self.x("UPDATE violations SET vlm_status=?, vlm=? WHERE id=?",
               (status, json.dumps(explanation) if explanation else "", vid))

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
        sql = ("SELECT v.*, c.name AS camera_name, c.location AS camera_location," + CITATION_COLUMNS +
               " FROM violations v JOIN cameras c ON c.id=v.camera_id" + CITATION_JOIN + w +
               f" ORDER BY v.id {'ASC' if order == 'ASC' else 'DESC'} LIMIT ? OFFSET ?")
        rows = self.q(sql, (*args, limit, offset))
        for r in rows:
            r["details"] = json.loads(r["details"] or "{}")
            r["evidence"] = json.loads(r.get("evidence") or "[]")
            r["vlm"] = json.loads(r["vlm"]) if r.get("vlm") else None
        return rows

    def count_violations(self, **filters):
        where, args = [], []
        for col, val in filters.items():
            if val not in (None, ""):
                where.append(f"{col}=?")
                args.append(val)
        return self.q("SELECT COUNT(*) AS n FROM violations" + (" WHERE " + " AND ".join(where) if where else ""),
                      args)[0]["n"]

    def review(self, vid, status, note="", by=None):
        """`by` is the session of whoever decided: a case with no name on it is not evidence of anything."""
        u = by or {}
        self.x("UPDATE violations SET status=?, note=?, reviewed_at=?, reviewed_by=?, reviewed_by_name=?"
               " WHERE id=?", (status, note, now(), u.get("user_id"),
                               u.get("name") or u.get("username") or "", vid))

    # ------------------------------------------------------------ citations
    def citation(self, no):
        r = self.q("SELECT * FROM citations WHERE citation_no=?", (no,))
        return r[0] if r else None

    def citation_by_violation(self, vid):
        r = self.q("SELECT * FROM citations WHERE violation_id=?", (vid,))
        return r[0] if r else None

    def citations(self, status=None, plate=None, limit=100, offset=0):
        """(rows, total), newest first. A row carries the case and camera as they are now (blank if the case
        has since been deleted - the citation outlives it) and no manifest: that is for the single view."""
        where, args = [], []
        if status:
            where.append("ct.status=?")
            args.append(status)
        if plate:       # the same matching as the review search: ignores case, spaces and dashes
            where.append("REPLACE(REPLACE(UPPER(v.plate_text_en), ' ', ''), '-', '') LIKE ?")
            args.append("%" + "".join(ch for ch in plate.upper() if ch not in " -") + "%")
        base = (" FROM citations ct LEFT JOIN violations v ON v.id=ct.violation_id"
                " LEFT JOIN cameras c ON c.id=v.camera_id" + (" WHERE " + " AND ".join(where) if where else ""))
        rows = self.q("SELECT ct.id, ct.citation_no, ct.violation_id, ct.issued_at, ct.issued_by_name,"
                      " ct.fine_bdt, ct.status, ct.bundle_sha256, ct.delivery_clip, ct.withdrawn_at,"
                      " ct.withdraw_reason, v.code, v.title, v.plate_text_en AS plate, v.wall_time,"
                      " c.name AS camera_name" + base + " ORDER BY ct.id DESC LIMIT ? OFFSET ?",
                      (*args, limit, offset))
        return rows, self.q("SELECT COUNT(*) AS n" + base, args)[0]["n"]

    def create_citation(self, violation_id, by, fine_bdt, bundle_sha256, manifest, delivery_clip=None,
                        issued_at=None, prefix="DMP"):
        """Number a citation and store it as one step. Returns (row, created).

        The number is the next in this year's sequence. The look-up of the highest number and the insert
        happen inside one write transaction (taken up front, so the look-up cannot go stale between two
        requests, or between two server processes); the unique keys on the number and on the case are the
        backstop. If the case already has a citation, that one is returned and nothing is written."""
        u = by or {}
        at = issued_at or now()
        year = int(at[:4])
        with self.lock:
            if self.db.in_transaction:      # a statement that failed earlier can leave one open; it holds nothing
                self.db.commit()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                have = self.db.execute("SELECT * FROM citations WHERE violation_id=?", (violation_id,)).fetchone()
                if have:
                    self.db.rollback()
                    return dict(have), False
                seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM citations WHERE year=?",
                                      (year,)).fetchone()[0]
                no = f"{prefix}-{year:04d}-{seq:06d}"
                self.db.execute(
                    "INSERT INTO citations (citation_no, year, seq, violation_id, issued_at, issued_by,"
                    " issued_by_name, fine_bdt, bundle_sha256, manifest, delivery_clip)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (no, year, seq, violation_id, at, u.get("user_id"),
                     u.get("name") or u.get("username") or "", fine_bdt, bundle_sha256, manifest, delivery_clip))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
        return self.citation(no), True

    def set_delivery_clip(self, no, path):
        self.x("UPDATE citations SET delivery_clip=? WHERE citation_no=?", (path, no))

    def withdraw_citation(self, no, by, reason):
        u = by or {}
        self.x("UPDATE citations SET status='withdrawn', withdrawn_at=?, withdrawn_by=?, withdrawn_by_name=?,"
               " withdraw_reason=? WHERE citation_no=? AND status='issued'",
               (now(), u.get("user_id"), u.get("name") or u.get("username") or "", reason, no))

    def live_citations_for_camera(self, cid):
        """Numbers of the citations still standing (not withdrawn) for cases from this camera. Their evidence
        has to stay on the disk, so deleting the camera or its cases is refused while there are any."""
        return [r["citation_no"] for r in self.q(
            "SELECT ct.citation_no FROM citations ct JOIN violations v ON v.id=ct.violation_id"
            " WHERE v.camera_id=? AND ct.status != 'withdrawn' ORDER BY ct.id", (cid,))]

    def clear_violations(self, cid):
        self.x("DELETE FROM violations WHERE camera_id=?", (cid,))
        self.x("DELETE FROM tracklets WHERE camera_id=?", (cid,))
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
