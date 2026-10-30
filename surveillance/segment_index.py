"""Persistent segment index for the natural-language query panel (WP3, requirements R-3 and R-4).

Why it exists. The 499A slides handle long video the VideoRAG way: never feed hours of video to a VLM; keep a
persistent index, retrieve a few short candidate segments, and reason only over those. Spatialyze adds the other
half: analytics are computed ONCE, at ingest, and stored, so a question never re-runs detection. This module is
that index. It is built from rows the pipeline already wrote (tracklets, violations, event impacts, runs,
cameras); it runs no detector and reads no pixels.

Window length: WINDOW_S = 10 s, fixed per index (the value is stored in seg_meta; changing it needs a rebuild).
  * the evidence clips are 5 s before + 5 s after an event, so a window is the same size as the thing a reviewer
    already watches, and one window is one cheap thing to show a VLM (3-4 frames cover it);
  * a vehicle crosses a 20-40 m camera view in roughly 3-10 s at Dhaka speeds, so a pass almost always falls in
    one window, and "stopped 14 s" spans at most two;
  * 10 s windows are 360 rows per camera-hour: a month of 8 cameras is ~2 million rows at the very worst, and
    an hour of footage is hundreds of rows, not thousands. Shorter windows buy nothing the tracker's own timing
    resolves; longer ones make "which moment" vague and the VLM's frame sampling coarser.
  Windows are aligned to multiples of WINDOW_S on the camera's wall clock (or the video clock when the file has
  no start time, flagged wall = 0) so a window id is reproducible.

Tables (all created here, none altered elsewhere)
  seg_meta(key, value)                       window_s, fts flag, per-camera build cursors
  seg_windows                                one row per (camera, run, window): counts per class, crossings
                                             toward/away from the camera, mean/min speed, stopped vehicles,
                                             violation codes, traffic-state numbers around the worst event,
                                             source-video offsets so frames can be pulled on demand
  seg_vehicle                                one row per vehicle per window: a SENTENCE describing its motion
                                             ("white CNG crossed Mirpur Road southbound heading northbound at 23
                                             km/h ..."), its global id, colour, class, codes, plate, clip
  seg_fts                                    SQLite FTS5 over those sentences (when the SQLite build has it;
                                             search falls back to LIKE when it does not)
  seg_runs                                   what time span each analysed run covers, so "no data for that
                                             hour" can be told apart from "zero buses"

What a tracklet is, honestly. The pipeline stores ONE row per vehicle per camera: the moment it crossed the
zone's reference line, its mean speed over the whole track, and its direction. It does not store enter and exit
times. So "entries / exits" here are crossings TOWARD / AWAY from the camera, and "stopped" means a mean speed
under 1 m/s over the track or a stationary-rule violation (V1, V3, V7, with its measured dwell). Colour is
estimated from the stored CIELAB body colour and is approximate (lighting, bikes' colour is the rider's).

Entry points
    ensure_schema(store)
    build_for_run(store, run_id)               full, idempotent rebuild of one run's windows
    build_for_camera(store, cid, since=None)   incremental (new rows by id, plus the last LOOKBACK_S so late
                                               global-id linking and impact scores are picked up), or everything
                                               from `since` (epoch seconds) when given
    build_all(store, since=None)
    on_run_finished(store, run_id)             the worker hook: never raises
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from datetime import datetime
from typing import Iterable, Optional

import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:                       # pragma: no cover  (py < 3.9)
    ZoneInfo = None

WINDOW_S = 10
LOOKBACK_S = 20 * 60
STOPPED_MS = 1.0                          # mean track speed below this (m/s) = stopped or crawling
STATIONARY_CODES = ("V1", "V3", "V7")     # rules that are about a vehicle standing still
WALL_MIN = 946684800.0                    # anything above is a wall-clock epoch (year 2000)
DHAKA = "Asia/Dhaka"

_BUILD_LOCK = threading.Lock()

# ---------------------------------------------------------------- vocabulary (shared with query.py)

# detector class (bd_vehicles_best.pt, 16 classes) -> words a person might use; the first is the display name
CLASS_WORDS = {
    "Bicycle": ["bicycle", "bicycles", "cycle", "cycles", "pushbike"],
    "Bus": ["bus", "buses", "minibus"],
    "Bhotbhoti": ["bhotbhoti", "vhotvhoti", "bhatbhati", "shallow-engine van"],
    "Car": ["car", "cars", "sedan", "saloon"],
    "CNG": ["cng", "cngs", "auto", "autos", "auto-rickshaw", "auto-rickshaws", "auto rickshaw", "auto rickshaws",
            "autorickshaw", "autorickshaws", "baby taxi", "baby-taxi"],
    "Easybike": ["easybike", "easy-bike", "easy bike", "easybikes", "battery rickshaw", "e-rickshaw", "toto"],
    "Leguna": ["leguna", "legunas", "human hauler", "human-hauler", "hauler"],
    "Motorbike": ["motorbike", "motorbikes", "motorcycle", "motorcycles", "bike rider", "scooter", "scooters"],
    "MPV": ["mpv", "minivan", "microbus", "micro-bus", "noah", "hiace"],
    "Pedestrian": ["pedestrian", "pedestrians", "person", "people", "walker"],
    "Pickup": ["pickup", "pickups", "pick-up", "pick up"],
    "PowerTiller": ["powertiller", "power tiller", "tiller", "nosimon", "korimon"],
    "Rickshaw": ["rickshaw", "rickshaws", "rikshaw", "pedal rickshaw"],
    "Van": ["van", "vans", "rickshaw van", "cart"],
    "Truck": ["truck", "trucks", "lorry", "lorries"],
    "Wheelbarrow": ["wheelbarrow", "wheelbarrows", "handcart"],
}
CLASS_LABEL = {"Easybike": "easy-bike", "Bhotbhoti": "bhotbhoti", "PowerTiller": "power tiller",
               "Leguna": "leguna", "CNG": "CNG", "MPV": "MPV"}

# words that name more than one class: the parser expands them and says so
AMBIGUOUS_CLASS_WORDS = {"bike": ["Motorbike", "Bicycle"], "bikes": ["Motorbike", "Bicycle"],
                         "two-wheeler": ["Motorbike", "Bicycle"], "two wheeler": ["Motorbike", "Bicycle"],
                         "two-wheelers": ["Motorbike", "Bicycle"], "heavy vehicle": ["Bus", "Truck"],
                         "heavy vehicles": ["Bus", "Truck"]}

COLOUR_WORDS = {
    "white": ["white"], "black": ["black"], "grey": ["grey", "gray", "silver"], "red": ["red", "maroon"],
    "orange": ["orange"], "yellow": ["yellow"], "green": ["green"], "blue": ["blue"], "purple": ["purple"],
}

# rule code -> (title keywords for search/FTS, plain words a person would say)
CODE_WORDS = {
    "V1": "bus stopped outside bus stop stopping blocking",
    "V2": "speeding overspeed over speed limit fast",
    "V3": "parked parking waiting no-parking kerb",
    "V4": "wrong way wrong-way wrong side contraflow opposite direction",
    "V5": "lane line crossing solid line lane change",
    "V6": "helmet no helmet without helmet",
    "V7": "blocking lane road block blocked running lane stopped in lane",
    "V8": "banned restricted not allowed prohibited vehicle",
    "V9": "bus lane",
    "V10": "erratic unsafe swerving cutting manoeuvre maneuver",
    "V11": "near collision near-collision crash accident collision",
}

DIRECTION_WORDS = {"incoming": "toward the camera approaching", "outgoing": "away from the camera leaving",
                   "unknown": ""}
POINTS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")

# ---------------------------------------------------------------- time (Dhaka local, same clock as cameras)

def local_zone():
    return ZoneInfo(DHAKA) if ZoneInfo else None


def now_local() -> datetime:
    """Dhaka wall clock as a naive datetime: the same convention as cameras.start_time and violations.wall_time."""
    z = local_zone()
    return datetime.now(z).replace(tzinfo=None) if z else datetime.now()


def to_epoch(dt: datetime) -> float:
    """Naive Dhaka datetime -> the epoch value the pipeline stores in tracklets.t_cross (it uses the same
    naive-datetime .timestamp(), so whatever the machine zone is, queries and index agree)."""
    return dt.timestamp()


def from_epoch(t: float) -> datetime:
    return datetime.fromtimestamp(t)


def clock(t: float) -> str:
    return from_epoch(t).strftime("%H:%M:%S") if t >= WALL_MIN else f"video {t:.0f}s"


def day_clock(t: float) -> str:
    return from_epoch(t).strftime("%Y-%m-%d %H:%M:%S") if t >= WALL_MIN else f"video {t:.0f}s"


# ---------------------------------------------------------------- schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS seg_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS seg_windows (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL DEFAULT 0,
    w_start REAL NOT NULL,
    w_end REAL NOT NULL,
    wall INTEGER NOT NULL DEFAULT 1,       -- 1: w_start is a wall-clock epoch; 0: seconds into the video
    video_t0 REAL,                         -- offset into the camera's source video (NULL: cannot be located)
    video_t1 REAL,
    n_pass INTEGER NOT NULL DEFAULT 0,     -- tracklets that crossed in this window
    n_in INTEGER NOT NULL DEFAULT 0,       -- ... toward the camera
    n_out INTEGER NOT NULL DEFAULT 0,      -- ... away from it
    n_unknown INTEGER NOT NULL DEFAULT 0,
    counts TEXT NOT NULL DEFAULT '{}',     -- {class: crossings}
    mean_kph REAL,
    min_kph REAL,
    n_stopped INTEGER NOT NULL DEFAULT 0,  -- distinct vehicles stopped or crawling (see module doc)
    max_dwell_s REAL,
    events TEXT NOT NULL DEFAULT '{}',     -- {violation code: n}
    n_events INTEGER NOT NULL DEFAULT 0,
    impact_max REAL,                       -- highest event-impact score of an event in the window
    state TEXT NOT NULL DEFAULT '{}',      -- traffic state during the worst event: queue_m, speed_kph, ...
    zones TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    built_at REAL,
    UNIQUE (camera_id, run_id, w_start)
);
CREATE INDEX IF NOT EXISTS ix_segw_time ON seg_windows(camera_id, w_start);
CREATE INDEX IF NOT EXISTS ix_segw_run ON seg_windows(run_id);
CREATE TABLE IF NOT EXISTS seg_vehicle (
    id INTEGER PRIMARY KEY,
    window_id INTEGER NOT NULL REFERENCES seg_windows(id) ON DELETE CASCADE,
    camera_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL,                    -- pass (a tracklet) / event (a violation with no tracklet in the window)
    pkey TEXT NOT NULL,                    -- camera:run:track, the vehicle pass this row is about
    tracklet_id INTEGER,
    violation_id INTEGER,
    global_id INTEGER,
    track_id INTEGER,
    cls TEXT,
    colour TEXT DEFAULT '',
    direction TEXT DEFAULT 'unknown',
    zone TEXT DEFAULT '',
    kph REAL,
    t REAL,
    video_t REAL,
    stopped INTEGER NOT NULL DEFAULT 0,
    dwell_s REAL,
    codes TEXT DEFAULT '',                 -- " V1 V4 " (space-padded so LIKE '% V1 %' is exact)
    plate TEXT DEFAULT '',
    impact REAL,
    text TEXT NOT NULL,                    -- the motion verbalisation
    terms TEXT DEFAULT '',                 -- extra search words (synonyms, behaviours), not shown
    clip TEXT DEFAULT '',
    snapshot TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_segv_window ON seg_vehicle(window_id);
CREATE INDEX IF NOT EXISTS ix_segv_cam_t ON seg_vehicle(camera_id, t);
CREATE INDEX IF NOT EXISTS ix_segv_global ON seg_vehicle(global_id);
CREATE INDEX IF NOT EXISTS ix_segv_cls ON seg_vehicle(cls);
CREATE TABLE IF NOT EXISTS seg_runs (
    camera_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    t0 REAL,
    t1 REAL,
    wall INTEGER NOT NULL DEFAULT 1,
    indexed_at REAL,
    PRIMARY KEY (camera_id, run_id)
);
"""


def ensure_schema(store) -> bool:
    """Create the tables (and the FTS5 table when the SQLite build has it). Idempotent. Returns has_fts."""
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.execute("INSERT OR IGNORE INTO seg_meta(key, value) VALUES ('window_s', ?)", (str(WINDOW_S),))
        have = store.db.execute("SELECT value FROM seg_meta WHERE key='fts'").fetchone()
        if have is None:
            try:
                store.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS seg_fts USING fts5(body, tokenize='unicode61')")
                ok = "1"
            except Exception:
                ok = "0"
            store.db.execute("INSERT OR REPLACE INTO seg_meta(key, value) VALUES ('fts', ?)", (ok,))
        store.db.commit()
    return has_fts(store)


def has_fts(store) -> bool:
    r = store.q("SELECT value FROM seg_meta WHERE key='fts'")
    return bool(r) and r[0]["value"] == "1"


def window_s(store) -> int:
    r = store.q("SELECT value FROM seg_meta WHERE key='window_s'")
    try:
        return int(r[0]["value"])
    except Exception:
        return WINDOW_S


def _meta_get(store, key, default=None):
    r = store.q("SELECT value FROM seg_meta WHERE key=?", (key,))
    return r[0]["value"] if r else default


def _meta_set(store, key, value):
    store.x("INSERT OR REPLACE INTO seg_meta(key, value) VALUES (?,?)", (key, str(value)))


# ---------------------------------------------------------------- small pure helpers

def finite(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def class_label(cls) -> str:
    return CLASS_LABEL.get(cls, (cls or "vehicle").lower())


def colour_of_lab(blob) -> str:
    """A colour word from the stored CIELAB body colour (6 floats: mean L, a, b, then spread, each divided by
    LAB_SCALE in reid.py), or "" when it is missing or unclear. OpenCV's 8-bit Lab: L 0..255, a and b offset by 128.
    A guess: dusk, shade, glare and a rider's clothes all move it."""
    if blob is None:
        return ""
    try:
        v = np.frombuffer(blob, np.float32) if isinstance(blob, (bytes, bytearray, memoryview)) else np.asarray(blob, np.float32)
    except Exception:
        return ""
    if v.size < 3 or not np.all(np.isfinite(v[:3])):
        return ""
    L = float(v[0]) * 20.0 * 100.0 / 255.0
    a, b = float(v[1]) * 8.0 - 128.0, float(v[2]) * 8.0 - 128.0
    chroma = math.hypot(a, b)
    if L < 22:
        return "black"
    if chroma < 11:
        if L > 72:
            return "white"
        return "black" if L < 32 else "grey"
    if L < 28 and chroma < 22:
        return "black"
    h = math.degrees(math.atan2(b, a)) % 360
    if h >= 330 or h < 38:
        return "red"
    if h < 68:
        return "orange"
    if h < 105:
        return "yellow"
    if h < 200:
        return "green"
    if h < 290:
        return "blue"
    return "purple"


def compass(deg: float) -> str:
    return POINTS[int(((deg % 360) + 22.5) // 45) % 8]


def travel_word(direction: str, heading: Optional[float]) -> str:
    """"northbound" when the camera's compass heading is known, else "toward the camera". Heading is where the
    camera points: a vehicle coming toward it travels the opposite way."""
    if direction not in ("incoming", "outgoing"):
        return ""
    if heading is None:
        return "toward the camera" if direction == "incoming" else "away from the camera"
    d = heading + 180.0 if direction == "incoming" else heading
    return compass(d) + "bound"


def parse_wall(text) -> Optional[float]:
    if not text:
        return None
    try:
        return to_epoch(datetime.fromisoformat(str(text).replace("T", " ")[:19]))
    except ValueError:
        return None


def _loads(s, default):
    try:
        v = json.loads(s) if isinstance(s, str) and s else default
    except ValueError:
        return default
    return v if isinstance(v, type(default)) else default


def w_floor(t: float, w: int) -> float:
    return math.floor(t / w) * w


# ---------------------------------------------------------------- loading what the pipeline already stored

def _tracklet_rows(store, cid, run_id, t_lo, t_hi):
    sql = ("SELECT id, camera_id, run_id, track_id, cls, zone, t_cross, video_t, speed, direction, lab, global_id"
           " FROM tracklets WHERE camera_id=? AND t_cross IS NOT NULL")
    args = [cid]
    if run_id is not None:
        sql += " AND run_id=?"
        args.append(run_id)
    if t_lo is not None:
        sql += " AND t_cross>=?"
        args.append(t_lo)
    if t_hi is not None:
        sql += " AND t_cross<?"
        args.append(t_hi)
    return store.q(sql + " ORDER BY t_cross", args)


def _violation_rows(store, cid, run_id):
    sql = ("SELECT v.id, v.camera_id, v.run_id, v.code, v.title, v.track_id, v.cls, v.video_time_s, v.started_s,"
           " v.wall_time, v.zone, v.kph, v.details, v.clip, v.snapshot, v.plate_text_en, v.status,"
           " e.score AS impact_score, e.impact AS impact_json"
           " FROM violations v LEFT JOIN event_impact e ON e.violation_id=v.id WHERE v.camera_id=?")
    args = [cid]
    if run_id is not None:
        sql += " AND v.run_id=?"
        args.append(run_id)
    try:
        return store.q(sql, args)
    except Exception:       # event_impact missing in some very old database
        return store.q(sql.replace("e.score AS impact_score, e.impact AS impact_json", "NULL AS impact_score, NULL AS impact_json")
                       .replace("LEFT JOIN event_impact e ON e.violation_id=v.id", ""), args)


def _run_bases(store, cid) -> dict:
    """run_id -> (wall epoch of video second 0), from tracklets that carry both clocks."""
    out = {}
    for r in store.q("SELECT run_id, t_cross, video_t FROM tracklets WHERE camera_id=? AND video_t IS NOT NULL"
                     " AND t_cross>=?", (cid, WALL_MIN)):
        out.setdefault(r["run_id"] or 0, []).append(r["t_cross"] - r["video_t"])
    return {k: statistics.median(v) for k, v in out.items()}


def _violation_time(v, bases):
    """(epoch seconds, is_wall, video offset) of a violation."""
    vt = finite(v.get("video_time_s"))
    t = parse_wall(v.get("wall_time"))
    if t is not None:
        return t, True, vt
    base = bases.get(v.get("run_id") or 0)
    if base is not None and vt is not None:
        return base + vt, True, vt
    return (vt if vt is not None else 0.0), False, vt


def _geo_headings(store) -> dict:
    try:
        return {r["camera_id"]: r["heading_deg"] for r in store.q("SELECT camera_id, heading_deg FROM camera_geo")}
    except Exception:
        return {}


# ---------------------------------------------------------------- one window

def _terms(cls, colour, codes, direction, extra=""):
    words = list(CLASS_WORDS.get(cls, []))
    if colour:
        words += COLOUR_WORDS.get(colour, [colour])
    for c in codes:
        words.append(c.lower())
        words.append(CODE_WORDS.get(c, ""))
    words.append(DIRECTION_WORDS.get(direction, ""))
    words.append(extra)
    return " ".join(w for w in words if w)


def _pass_text(it, cam_name, heading, codes_info, others):
    bits = [f"{(it['colour'] + ' ') if it['colour'] else ''}{class_label(it['cls'])}",
            f"crossed {it['zone'] or cam_name}"]
    tw = travel_word(it["direction"], heading)
    if tw:
        bits.append(f"heading {tw}")
    s = " ".join(bits)
    if it["kph"] is not None:
        s += f" at {it['kph']:.0f} km/h"
    if it["slow"]:
        s += ", nearly stationary"
    s += f" at {clock(it['t'])}"
    for c in codes_info:
        d = f", stopped {c['dwell']:.0f} s" if c.get("dwell") else ""
        s += f"; flagged {c['title'].lower()} ({c['code']}{d})"
    if it.get("plate"):
        s += f"; plate {it['plate']}"
    if it.get("gid"):
        s += f"; vehicle #{it['gid']}"
        if others:
            s += f", also seen at {', '.join(others)}"
    return s


def _event_text(ev, cam_name):
    s = (f"{(ev['colour'] + ' ') if ev['colour'] else ''}{class_label(ev['cls'])} {ev['title'].lower()} ({ev['code']})"
         f" at {ev['zone'] or cam_name}")
    if ev.get("dwell"):
        s += f", stopped {ev['dwell']:.0f} s"
    s += f" at {clock(ev['t'])}"
    if ev.get("plate"):
        s += f"; plate {ev['plate']}"
    if ev.get("gid"):
        s += f"; vehicle #{ev['gid']}"
    return s


def _impact_state(row):
    imp = _loads(row.get("impact_json"), {})
    during = imp.get("during") if isinstance(imp.get("during"), dict) else {}
    state = {k: finite(during.get(k)) for k in ("queue_m", "speed_kph", "occupancy", "density", "wait_s")
             if finite(during.get(k)) is not None}
    return state


def _window_rows(cid, cam_name, heading, run, ws, w, tracklets, violations, bases, cam_vid, other_cams, lookup):
    """Compute the seg_windows row and its seg_vehicle rows for one (run, window) from raw rows."""
    passes = []
    for r in tracklets:
        k = finite(r["speed"])
        kph = None if k is None else k * 3.6
        passes.append({"row": r, "t": r["t_cross"], "vt": finite(r["video_t"]), "kph": kph,
                       "slow": k is not None and k < STOPPED_MS, "colour": colour_of_lab(r["lab"]),
                       "direction": r["direction"] or "unknown", "zone": r["zone"] or "", "cls": r["cls"],
                       "gid": r["global_id"], "stationary": False, "plate": ""})
    by_track = {(p["row"]["track_id"]): p for p in passes}
    merged, events = {}, []
    for v in violations:
        details = _loads(v["details"], {})
        dwell = finite(details.get("dwell_s")) or finite(details.get("dwell_sec"))
        info = {"code": v["code"], "title": v["title"] or v["code"], "dwell": dwell, "id": v["id"],
                "status": v["status"], "clip": v["clip"] or "", "snapshot": v["snapshot"] or "",
                "impact": finite(v["impact_score"]), "state": _impact_state(v), "plate": v["plate_text_en"] or "",
                "t": v["_t"], "vt": v["_vt"], "track": v["track_id"], "cls": v["cls"], "zone": v["zone"] or "",
                "kph": finite(v["kph"])}
        p = by_track.get(v["track_id"]) if v["track_id"] is not None else None
        if p is not None and (v["run_id"] or 0) == (p["row"]["run_id"] or 0):
            merged.setdefault(id(p), []).append(info)
            if v["code"] in STATIONARY_CODES:
                p["stationary"] = True
            if info["plate"]:
                p["plate"] = info["plate"]
        else:
            events.append(info)
    vrows, counts = [], {}
    codes_all, zones = {}, set()
    kphs, in_, out_, unk = [], 0, 0, 0
    stopped, dwell_max = set(), 0.0
    impact_best, state_best = None, {}
    for p in sorted(passes, key=lambda x: x["t"]):
        r = p["row"]
        infos = merged.get(id(p), [])
        codes = [i["code"] for i in infos]
        others = []
        if p["gid"]:
            others = sorted({other_cams[c] for c in lookup.get(p["gid"], ()) if c != cid and c in other_cams})
        stop = p["slow"] or p["stationary"]
        dw = max([i["dwell"] for i in infos if i["dwell"]] or [0]) or None
        imp = max([i["impact"] for i in infos if i["impact"] is not None] or [None], key=lambda z: -1 if z is None else z)
        pkey = f"{cid}:{run}:{r['track_id']}"
        text = _pass_text(p, cam_name, heading, infos, others)
        vrows.append(dict(kind="pass", pkey=pkey, tracklet_id=r["id"], violation_id=infos[0]["id"] if infos else None,
                          global_id=p["gid"], track_id=r["track_id"], cls=p["cls"], colour=p["colour"],
                          direction=p["direction"], zone=p["zone"], kph=p["kph"], t=p["t"], video_t=p["vt"],
                          stopped=int(stop), dwell_s=dw, codes=" " + " ".join(codes) + " " if codes else "",
                          plate=p["plate"], impact=imp, text=text,
                          terms=_terms(p["cls"], p["colour"], codes, p["direction"], "stopped stationary" if stop else ""),
                          clip=next((i["clip"] for i in infos if i["clip"]), ""),
                          snapshot=next((i["snapshot"] for i in infos if i["snapshot"]), "")))
        counts[p["cls"]] = counts.get(p["cls"], 0) + 1
        if p["kph"] is not None:
            kphs.append(p["kph"])
        in_ += p["direction"] == "incoming"
        out_ += p["direction"] == "outgoing"
        unk += p["direction"] not in ("incoming", "outgoing")
        if p["zone"]:
            zones.add(p["zone"])
        if stop:
            stopped.add(pkey)
        for i in infos:
            codes_all[i["code"]] = codes_all.get(i["code"], 0) + 1
            dwell_max = max(dwell_max, i["dwell"] or 0)
            if i["impact"] is not None and (impact_best is None or i["impact"] > impact_best):
                impact_best, state_best = i["impact"], i["state"]
    for ev in events:
        look = lookup_event_tracklet(ev, cid, run, tracklets_by=None, store=lookup.get("_store"))
        colour, gid = look
        ev["colour"], ev["gid"] = colour, gid
        pkey = f"{cid}:{run}:{ev['track']}" if ev["track"] is not None else f"{cid}:{run}:v{ev['id']}"
        stop = ev["code"] in STATIONARY_CODES
        text = _event_text(ev, cam_name)
        vrows.append(dict(kind="event", pkey=pkey, tracklet_id=None, violation_id=ev["id"], global_id=gid,
                          track_id=ev["track"], cls=ev["cls"], colour=colour, direction="unknown", zone=ev["zone"],
                          kph=ev["kph"], t=ev["t"], video_t=ev["vt"], stopped=int(stop), dwell_s=ev["dwell"],
                          codes=f" {ev['code']} ", plate=ev["plate"], impact=ev["impact"], text=text,
                          terms=_terms(ev["cls"], colour, [ev["code"]], "unknown", "stopped stationary" if stop else ""),
                          clip=ev["clip"], snapshot=ev["snapshot"]))
        codes_all[ev["code"]] = codes_all.get(ev["code"], 0) + 1
        if ev["zone"]:
            zones.add(ev["zone"])
        if stop:
            stopped.add(pkey)
        dwell_max = max(dwell_max, ev["dwell"] or 0)
        if ev["impact"] is not None and (impact_best is None or ev["impact"] > impact_best):
            impact_best, state_best = ev["impact"], ev["state"]
    # where in the source video this window is: pair every row's wall time with its video offset
    offs = [r["video_t"] - (r["t"] - ws) for r in vrows if r["video_t"] is not None]
    if offs:
        v0 = statistics.median(offs)
    elif run in bases and ws >= WALL_MIN:
        v0 = ws - bases[run]
    else:
        v0 = cam_vid
    wall = int(ws >= WALL_MIN)
    parts = []
    if counts:
        parts.append(f"{sum(counts.values())} crossing{'s' if sum(counts.values()) != 1 else ''} ("
                     + ", ".join(f"{n} {class_label(c)}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])) + ")")
    if kphs:
        parts.append(f"mean {statistics.fmean(kphs):.0f} km/h")
    if stopped:
        parts.append(f"{len(stopped)} stopped")
    if codes_all:
        parts.append("events " + ", ".join(f"{c} x{n}" for c, n in sorted(codes_all.items())))
    win = dict(camera_id=cid, run_id=run, w_start=ws, w_end=ws + w, wall=wall,
               video_t0=None if v0 is None else max(0.0, v0), video_t1=None if v0 is None else max(0.0, v0) + w,
               n_pass=len(passes), n_in=in_, n_out=out_, n_unknown=unk, counts=json.dumps(counts),
               mean_kph=statistics.fmean(kphs) if kphs else None, min_kph=min(kphs) if kphs else None,
               n_stopped=len(stopped), max_dwell_s=dwell_max or None, events=json.dumps(codes_all),
               n_events=sum(codes_all.values()), impact_max=impact_best, state=json.dumps(state_best),
               zones=json.dumps(sorted(zones)), summary="; ".join(parts), built_at=time.time())
    return win, vrows


def lookup_event_tracklet(ev, cid, run, tracklets_by, store):
    """(colour, global id) of the vehicle a stand-alone event is about, when its tracklet exists elsewhere."""
    if store is None or ev["track"] is None:
        return "", None
    rows = store.q("SELECT lab, global_id FROM tracklets WHERE camera_id=? AND track_id=? AND COALESCE(run_id,0)=?"
                   " ORDER BY ABS(COALESCE(video_t,0)-?) LIMIT 1", (cid, ev["track"], run, ev["vt"] or 0))
    if not rows:
        return "", None
    return colour_of_lab(rows[0]["lab"]), rows[0]["global_id"]


# ---------------------------------------------------------------- building

def _write_windows(store, cid, scope_runs, new, keep_scope):
    """Replace the windows in `scope`: delete old rows (and their FTS entries), insert `new`.
    new: [(win dict, [vehicle dicts])]. keep_scope(run, w_start) says which existing windows this build owns."""
    fts = has_fts(store)
    with store.lock:
        db = store.db
        existing = db.execute("SELECT id, run_id, w_start FROM seg_windows WHERE camera_id=?", (cid,)).fetchall()
        drop = [r["id"] for r in existing if keep_scope(r["run_id"], r["w_start"])]
        for chunk in range(0, len(drop), 500):
            ids = drop[chunk:chunk + 500]
            ph = ",".join("?" * len(ids))
            if fts:
                db.execute(f"DELETE FROM seg_fts WHERE rowid IN (SELECT id FROM seg_vehicle WHERE window_id IN ({ph}))", ids)
            db.execute(f"DELETE FROM seg_vehicle WHERE window_id IN ({ph})", ids)
            db.execute(f"DELETE FROM seg_windows WHERE id IN ({ph})", ids)
        n_veh = 0
        wcols = list(new[0][0].keys()) if new else []
        for win, vrows in new:
            cur = db.execute(f"INSERT INTO seg_windows ({','.join(wcols)}) VALUES ({','.join('?' * len(wcols))})",
                             [win[c] for c in wcols])
            wid = cur.lastrowid
            for v in vrows:
                cols = ["window_id", "camera_id", "run_id"] + list(v.keys())
                vals = [wid, cid, win["run_id"]] + list(v.values())
                c2 = db.execute(f"INSERT INTO seg_vehicle ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
                if fts:
                    db.execute("INSERT INTO seg_fts(rowid, body) VALUES (?,?)", (c2.lastrowid, v["text"] + " " + v["terms"]))
                n_veh += 1
        db.commit()
    return len(new), n_veh


def _rebuild(store, cid, run_id=None, keys=None, since=None):
    """Recompute windows of one camera. `keys`: {(run, w_start)} to recompute (others untouched); otherwise every
    window of `run_id` (or all runs) from `since`. Returns {"windows", "vehicles"}."""
    cam = store.camera(cid)
    if cam is None:
        return {"windows": 0, "vehicles": 0}
    w = window_s(store)
    bases = _run_bases(store, cid)
    headings = _geo_headings(store)
    cams = {c["id"]: c["name"] for c in store.cameras()}
    if keys is not None:
        t_lo, t_hi = (min(k[1] for k in keys), max(k[1] for k in keys) + w) if keys else (None, None)
    else:
        t_lo, t_hi = (w_floor(since, w) if since is not None else None), None
    trs = _tracklet_rows(store, cid, run_id, t_lo, t_hi) if (keys is None or keys) else []
    vios = _violation_rows(store, cid, run_id)
    for v in vios:
        v["_t"], v["_wall"], v["_vt"] = _violation_time(v, bases)
    vios = [v for v in vios if (t_lo is None or v["_t"] >= t_lo) and (t_hi is None or v["_t"] < t_hi)]
    gids = {r["global_id"] for r in trs if r["global_id"]}
    lookup = {"_store": store}
    if gids:
        ids = sorted(gids)
        for i in range(0, len(ids), 500):
            ch = ids[i:i + 500]
            for r in store.q(f"SELECT global_id, camera_id FROM tracklets WHERE global_id IN ({','.join('?' * len(ch))})", ch):
                lookup.setdefault(r["global_id"], set()).add(r["camera_id"])
    cam_vid = None
    grouped_t, grouped_v = {}, {}
    for r in trs:
        grouped_t.setdefault((r["run_id"] or 0, w_floor(r["t_cross"], w)), []).append(r)
    for v in vios:
        grouped_v.setdefault((v["run_id"] or 0, w_floor(v["_t"], w)), []).append(v)
    want = set(grouped_t) | set(grouped_v)
    if keys is not None:
        want &= set(keys)
    new = []
    for (run, ws) in sorted(want):
        win, vrows = _window_rows(cid, cam["name"], headings.get(cid), run, ws, w, grouped_t.get((run, ws), []),
                                  grouped_v.get((run, ws), []), bases, cam_vid, cams, lookup)
        new.append((win, vrows))
    if keys is not None:
        def scope(run, ws):
            return (run, ws) in keys
    else:
        def scope(run, ws):
            return ((run_id is None or run == run_id) and (t_lo is None or ws >= t_lo))
    n_w, n_v = _write_windows(store, cid, None, new, scope)
    _update_coverage(store, cid, {run for run, _ in want} | ({run_id} if run_id else set()), bases)
    return {"windows": n_w, "vehicles": n_v}


def _update_coverage(store, cid, runs, bases):
    for run in runs:
        rows = store.q("SELECT MIN(w_start) AS a, MAX(w_end) AS b, MAX(wall) AS wall FROM seg_windows"
                       " WHERE camera_id=? AND run_id=?", (cid, run))
        if not rows or rows[0]["a"] is None:
            store.x("DELETE FROM seg_runs WHERE camera_id=? AND run_id=?", (cid, run))
            continue
        a, b, wall = rows[0]["a"], rows[0]["b"], rows[0]["wall"]
        info = store.q("SELECT video_seconds FROM runs WHERE id=?", (run,)) if run else []
        base = bases.get(run)
        if wall and base is not None and info and info[0]["video_seconds"]:
            a, b = min(a, base), max(b, base + info[0]["video_seconds"])
        store.x("INSERT OR REPLACE INTO seg_runs (camera_id, run_id, t0, t1, wall, indexed_at) VALUES (?,?,?,?,?,?)",
                (cid, run, a, b, wall, time.time()))


def build_for_run(store, run_id) -> dict:
    """Full, idempotent rebuild of every window of one run. Calling it twice leaves the index byte-for-byte equal."""
    ensure_schema(store)
    rows = store.q("SELECT camera_id FROM runs WHERE id=?", (run_id,))
    if not rows:
        return {"run_id": run_id, "windows": 0, "vehicles": 0, "error": "unknown run"}
    with _BUILD_LOCK:
        out = _rebuild(store, rows[0]["camera_id"], run_id=run_id)
    return {"run_id": run_id, "camera_id": rows[0]["camera_id"], **out}


def build_for_camera(store, cid, since: Optional[float] = None) -> dict:
    """Incremental: only the windows touched by rows newer than the last build (by row id), plus the last
    LOOKBACK_S of data so late global-id linking and event-impact scores reach the index. With `since`
    (epoch seconds) every window from then on is recomputed."""
    ensure_schema(store)
    w = window_s(store)
    with _BUILD_LOCK:
        if since is not None:
            out = _rebuild(store, cid, since=since)
            _set_cursors(store, cid)
            return {"camera_id": cid, **out}
        ct = int(_meta_get(store, f"cur:{cid}:t", 0))
        cv = int(_meta_get(store, f"cur:{cid}:v", 0))
        keys = set()
        for r in store.q("SELECT run_id, t_cross FROM tracklets WHERE camera_id=? AND id>? AND t_cross IS NOT NULL", (cid, ct)):
            keys.add((r["run_id"] or 0, w_floor(r["t_cross"], w)))
        bases = None
        newv = store.q("SELECT id FROM violations WHERE camera_id=? AND id>?", (cid, cv))
        if newv:
            bases = _run_bases(store, cid)
            for v in _violation_rows(store, cid, None):
                if v["id"] > cv:
                    t, _, _ = _violation_time(v, bases)
                    keys.add((v["run_id"] or 0, w_floor(t, w)))
        top = store.q("SELECT MAX(t_cross) AS m FROM tracklets WHERE camera_id=? AND t_cross>=?", (cid, WALL_MIN))
        if top and top[0]["m"]:
            for r in store.q("SELECT run_id, t_cross FROM tracklets WHERE camera_id=? AND t_cross>=?",
                             (cid, top[0]["m"] - LOOKBACK_S)):
                keys.add((r["run_id"] or 0, w_floor(r["t_cross"], w)))
        out = _rebuild(store, cid, keys=keys) if keys else {"windows": 0, "vehicles": 0}
        _set_cursors(store, cid)
        return {"camera_id": cid, **out}


def _set_cursors(store, cid):
    t = store.q("SELECT COALESCE(MAX(id),0) AS m FROM tracklets WHERE camera_id=?", (cid,))[0]["m"]
    v = store.q("SELECT COALESCE(MAX(id),0) AS m FROM violations WHERE camera_id=?", (cid,))[0]["m"]
    _meta_set(store, f"cur:{cid}:t", t)
    _meta_set(store, f"cur:{cid}:v", v)


def build_all(store, since: Optional[float] = None, cameras: Optional[Iterable[int]] = None) -> dict:
    ensure_schema(store)
    total = {"windows": 0, "vehicles": 0, "cameras": 0}
    for c in store.cameras():
        if cameras is not None and c["id"] not in set(cameras):
            continue
        r = build_for_camera(store, c["id"], since)
        total["windows"] += r["windows"]
        total["vehicles"] += r["vehicles"]
        total["cameras"] += 1
    return total


def on_run_finished(store, run_id) -> dict:
    """The worker hook: index a run that has just ended. Never raises, so a failure here cannot fail the run."""
    try:
        return build_for_run(store, run_id)
    except Exception as e:                      # noqa: BLE001
        return {"run_id": run_id, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- reading

def status(store) -> dict:
    ensure_schema(store)
    one = store.q("SELECT COUNT(*) AS windows, COALESCE(SUM(n_pass),0) AS crossings, COALESCE(SUM(n_events),0) AS events,"
                  " MIN(w_start) AS first, MAX(w_end) AS last, MAX(built_at) AS built_at FROM seg_windows")[0]
    per = store.q("SELECT camera_id, COUNT(*) AS windows, MIN(w_start) AS first, MAX(w_end) AS last FROM seg_windows"
                  " GROUP BY camera_id ORDER BY camera_id")
    return {**one, "vehicle_rows": store.q("SELECT COUNT(*) AS n FROM seg_vehicle")[0]["n"], "window_s": window_s(store),
            "fts": has_fts(store), "cameras": per,
            "unindexed_tracklets": store.q(
                "SELECT COUNT(*) AS n FROM tracklets t WHERE NOT EXISTS (SELECT 1 FROM seg_vehicle s WHERE s.tracklet_id=t.id)")[0]["n"]}


def window(store, wid) -> Optional[dict]:
    r = store.q("SELECT * FROM seg_windows WHERE id=?", (wid,))
    if not r:
        return None
    w = r[0]
    w["counts"], w["events"] = _loads(w["counts"], {}), _loads(w["events"], {})
    w["state"], w["zones"] = _loads(w["state"], {}), _loads(w["zones"], [])
    return w


def vehicles_in(store, wid) -> list:
    return store.q("SELECT * FROM seg_vehicle WHERE window_id=? ORDER BY t", (wid,))


def coverage(store, camera_ids, t0, t1) -> list:
    """Analysed spans that overlap [t0, t1] on those cameras (wall-clock runs only)."""
    sql = "SELECT camera_id, run_id, t0, t1 FROM seg_runs WHERE wall=1 AND t1>=? AND t0<=?"
    args = [t0 if t0 is not None else -1e18, t1 if t1 is not None else 1e18]
    if camera_ids:
        sql += f" AND camera_id IN ({','.join('?' * len(camera_ids))})"
        args += list(camera_ids)
    return store.q(sql, args)
