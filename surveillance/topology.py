"""Camera-network topology for cross-camera vehicle matching (WP19).

Why it exists. Matching a vehicle that has just left one camera against everything every other camera has
seen costs more as the archive grows and, worse, invites look-alikes. Liu et al. (AI City 2021, "City-Scale
Multi-Camera Vehicle Tracking Guided by Crossroad Zones") answer with a map of the road network: a vehicle
leaving camera A through a particular part of A's picture can only appear next at the camera that road leads
to, inside the time the trip takes. Only those few vehicles are worth comparing. The idea, and the two-link
fallback and out-of-order handling below, are ported in spirit from a teammate's system (see
docs/COMPARISON_SHIHAB.md) and fitted to ours.

Vocabulary
    gate    a named polygon on one camera's frame (coordinates are fractions of the frame, 0..1, so they do
            not depend on the snapshot size) where vehicles come into or go out of view. kind: in / out / both.
    link    camera A's out-gate -> camera B's in-gate with a travel-time window [min_s, max_s]: the time from
            the vehicle leaving A's view to entering B's.
    route   a link, or two links through a camera in between that missed the vehicle (hops = 2). Two-link
            routes are precomputed whenever the topology changes and stored in `camera_routes`.

How it combines with the existing scoring (surveillance.reid). Topology is a candidate FILTER (gate and
window) and its timing is a CUE: the pair cost is `reid.pair_terms`, i.e. the same speed / size / colour /
embedding LLR terms as always, with the time term centred on the middle of the link window and as wide as
the window, so it can only add a bounded penalty inside it. Class is still a hard rule. Where a camera has
no links the old path (`link.OnlineLinker` flows, `reid.link_all`) is untouched; where BOTH cameras have
topology, only the topology path may link them.

One decision procedure serves the online and the batch path (`choose`). Batch replays the tracklets in the
order they were stored over an in-memory index; online runs the same function over the database. That is
why the two give the same answer on in-order data.

Tables (all created by `ensure_schema`, idempotent): camera_gates, camera_links, camera_routes,
vehicle_merges, topology_meta; and new tracklet columns entry_x, entry_y, exit_x, exit_y, t_enter, t_exit,
entry_gate, exit_gate. Existing columns are neither renamed nor retyped.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import os
import time
from collections import defaultdict
from typing import Optional

from . import geo as geo_mod
from . import reid

# ---------------------------------------------------------------- settings
GATE_TOLERANCE = 0.05       # a point just outside a gate (fraction of the frame) still belongs to the nearest
                            # one: a vehicle is first and last seen at the edge of what was drawn
SKIP_DWELL_S = 60.0         # two links: the most a vehicle may spend inside the view of the camera that missed it


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# Acceptance bars, overridable from the environment. Lower accept / higher margin = fewer, surer links.
ACCEPT = _env("TOPOLOGY_ACCEPT", 3.0)          # one link: cost must be under this ...
MARGIN = _env("TOPOLOGY_MARGIN", 3.0)           # ... and the runner-up this much worse
SKIP_ACCEPT = _env("TOPOLOGY_SKIP_ACCEPT", 1.5)             # two links: stricter again (wider window, more look-alikes)
SKIP_MARGIN = _env("TOPOLOGY_SKIP_MARGIN", 3.0)
MAX_GATES = 12
MAX_POINTS = 40
KINDS = ("in", "out", "both")
DISPLAY_PREFIX = "DV-"

SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_gates (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- in / out / both
    points TEXT NOT NULL,               -- JSON [[x, y], ...] as fractions of the frame
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT '',
    UNIQUE (camera_id, name)
);
CREATE TABLE IF NOT EXISTS camera_links (
    id INTEGER PRIMARY KEY,
    from_camera INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    from_gate INTEGER NOT NULL REFERENCES camera_gates(id) ON DELETE CASCADE,
    to_camera INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    to_gate INTEGER NOT NULL REFERENCES camera_gates(id) ON DELETE CASCADE,
    min_s REAL NOT NULL,                -- leaving the first camera's view -> entering the second's
    max_s REAL NOT NULL,
    distance_m REAL,                    -- straight line between the placed cameras, when both are placed
    window_source TEXT DEFAULT 'manual',   -- suggested (from the map) / manual
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT '',
    UNIQUE (from_gate, to_gate)
);
CREATE TABLE IF NOT EXISTS camera_routes (
    from_camera INTEGER NOT NULL,
    from_gate INTEGER NOT NULL,
    to_camera INTEGER NOT NULL,
    to_gate INTEGER NOT NULL,
    min_s REAL NOT NULL,
    max_s REAL NOT NULL,
    hops INTEGER NOT NULL,              -- 1 = a link, 2 = through a camera that missed the vehicle
    via_camera INTEGER
);
CREATE TABLE IF NOT EXISTS vehicle_merges (
    old_gid INTEGER PRIMARY KEY,        -- an id that no longer labels any tracklet
    new_gid INTEGER NOT NULL,           -- the id that took its tracklets (the first camera's)
    merged_at REAL NOT NULL,
    reason TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_merge_new ON vehicle_merges(new_gid);
CREATE TABLE IF NOT EXISTS topology_meta (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""
TRACKLET_COLUMNS = (("entry_x", "REAL"), ("entry_y", "REAL"), ("exit_x", "REAL"), ("exit_y", "REAL"),
                    ("t_enter", "REAL"), ("t_exit", "REAL"), ("entry_gate", "INTEGER"), ("exit_gate", "INTEGER"))


class TopologyError(ValueError):
    """A gate or link that cannot be stored. The message is safe to show to the person who sent it."""


def ensure_schema(store) -> None:
    """Create the tables and add the tracklet columns. Idempotent and cheap after the first call."""
    if getattr(store, "_topology_ready", False):
        return
    geo_mod.ensure_schema(store)      # clock offsets and distances are read from the placed cameras
    with store.lock:
        store.db.executescript(SCHEMA)
        have = {r[1] for r in store.db.execute("PRAGMA table_info(tracklets)")}
        for col, decl in TRACKLET_COLUMNS:
            if col not in have:
                store.db.execute(f"ALTER TABLE tracklets ADD COLUMN {col} {decl}")
        # candidates for a sighting are read by (camera, exit gate, exit time) and (camera, entry gate, entry time)
        store.db.execute("CREATE INDEX IF NOT EXISTS ix_track_exit ON tracklets(camera_id, exit_gate, t_exit)")
        store.db.execute("CREATE INDEX IF NOT EXISTS ix_track_enter ON tracklets(camera_id, entry_gate, t_enter)")
        store.db.commit()
    store._topology_ready = True


# ---------------------------------------------------------------- display ids

def display_id(gid) -> str:
    """123 -> "DV-000123". Display only: the stored id stays the integer."""
    return f"{DISPLAY_PREFIX}{int(gid):06d}"


def parse_display_id(text) -> Optional[int]:
    """"DV-000123", "dv123" or "123" -> 123; None when the text is not a vehicle id."""
    import re
    m = re.match(r"^(?:DV)?[\s-]*0*(\d{1,12})$", str(text).strip().upper())
    return int(m.group(1)) if m else None


def resolve(store, gid) -> Optional[int]:
    """The id an old (merged away) id now stands for; the id itself when it was never merged."""
    seen = set()
    while gid is not None and gid not in seen:
        seen.add(gid)
        rows = store.q("SELECT new_gid FROM vehicle_merges WHERE old_gid=?", (int(gid),))
        if not rows:
            break
        gid = rows[0]["new_gid"]
    return gid


def merged_from(store, gid) -> list:
    """Every id that was merged into `gid` (directly or through others), so a printed old id stays findable."""
    out, todo = [], [int(gid)]
    while todo:
        rows = store.q("SELECT old_gid FROM vehicle_merges WHERE new_gid=?", (todo.pop(),))
        for r in rows:
            out.append(r["old_gid"])
            todo.append(r["old_gid"])
    return sorted(out)


def merge_vehicles(store, keep, old, reason="") -> None:
    """Fold vehicle `old` into `keep`: its tracklets take the id, and the old id is recorded as merged."""
    keep, old = int(keep), int(old)
    if keep == old:
        return
    with store.lock:
        store.db.execute("UPDATE tracklets SET global_id=?, link_source='merge' WHERE global_id=?", (keep, old))
        store.db.execute("UPDATE vehicle_merges SET new_gid=? WHERE new_gid=?", (keep, old))   # keep chains flat
        store.db.execute("INSERT OR REPLACE INTO vehicle_merges (old_gid, new_gid, merged_at, reason)"
                         " VALUES (?,?,?,?)", (old, keep, time.time(), reason))
        store.db.commit()


# ---------------------------------------------------------------- gates

def _inside(p, poly) -> bool:
    hit = False
    j = len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            hit = not hit
        j = i
    return hit


def _segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / ln)) if ln else 0.0
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def gate_at(pt, gates, kinds=KINDS) -> Optional[int]:
    """The id of the gate (of one of `kinds`) a point lies in; else the nearest within GATE_TOLERANCE; else None.

    `gates` are rows with id, kind and points (a list, or its JSON text)."""
    if pt is None or not gates:
        return None
    p = (float(pt[0]), float(pt[1]))
    best, best_d = None, GATE_TOLERANCE
    for g in gates:
        if g["kind"] not in kinds:
            continue
        poly = g["points"] if isinstance(g["points"], list) else json.loads(g["points"])
        if len(poly) < 3:
            continue
        if _inside(p, poly):
            return g["id"]
        for i in range(len(poly)):
            d = _segment_distance(p, poly[i], poly[(i + 1) % len(poly)])
            if d < best_d:
                best, best_d = g["id"], d
    return best


def gates_of(store, camera_id) -> list:
    rows = store.q("SELECT * FROM camera_gates WHERE camera_id=? ORDER BY id", (int(camera_id),))
    for r in rows:
        r["points"] = json.loads(r["points"])
    return rows


def assign_gates(store, camera_id, t) -> None:
    """Set `t.entry_gate` / `t.exit_gate` from where it came into and left the picture. No gates drawn: nothing
    changes. A gate only counts for the right direction (an entry point can only be in an in / both gate)."""
    gates = gates_of(store, camera_id)
    if not gates:
        return
    if getattr(t, "entry_xy", None) is not None:
        t.entry_gate = gate_at(t.entry_xy, gates, ("in", "both"))
    if getattr(t, "exit_xy", None) is not None:
        t.exit_gate = gate_at(t.exit_xy, gates, ("out", "both"))


def reassign_gates(store, camera_id) -> int:
    """Recompute the gates of every stored tracklet of a camera (after its gates were edited). Returns the count."""
    gates = gates_of(store, camera_id)
    rows = store.q("SELECT id, entry_x, entry_y, exit_x, exit_y FROM tracklets WHERE camera_id=?", (int(camera_id),))
    upd = []
    for r in rows:
        gin = gate_at((r["entry_x"], r["entry_y"]), gates, ("in", "both")) if r["entry_x"] is not None else None
        gout = gate_at((r["exit_x"], r["exit_y"]), gates, ("out", "both")) if r["exit_x"] is not None else None
        upd.append((gin, gout, r["id"]))
    with store.lock:
        store.db.executemany("UPDATE tracklets SET entry_gate=?, exit_gate=? WHERE id=?", upd)
        store.db.commit()
    return len(upd)


def _clean_gate(g, i) -> dict:
    if not isinstance(g, dict):
        raise TopologyError(f"gate {i + 1}: send an object with name, kind and points")
    name = str(g.get("name") or "").strip()
    if not name or len(name) > 40:
        raise TopologyError(f"gate {i + 1}: give it a name of 1-40 characters")
    kind = g.get("kind", "both")
    if kind not in KINDS:
        raise TopologyError(f"gate '{name}': kind must be one of {', '.join(KINDS)}")
    pts = g.get("points")
    if not isinstance(pts, list) or not 3 <= len(pts) <= MAX_POINTS:
        raise TopologyError(f"gate '{name}': draw a polygon of 3 to {MAX_POINTS} points")
    clean = []
    for p in pts:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise TopologyError(f"gate '{name}': every point is [x, y]") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise TopologyError(f"gate '{name}': points are fractions of the frame, between 0 and 1")
        clean.append([round(x, 5), round(y, 5)])
    gid = g.get("id")
    return {"id": int(gid) if gid not in (None, "") else None, "name": name, "kind": kind, "points": clean}


def save_gates(store, camera_id, gates, who="") -> dict:
    """Replace one camera's gate set. A gate that is left out is deleted, and with it every link that used it.
    Returns {"gates": [...], "removed_links": n, "retagged": tracklets re-assigned}."""
    ensure_schema(store)
    if not isinstance(gates, list) or len(gates) > MAX_GATES:
        raise TopologyError(f"send a list of at most {MAX_GATES} gates")
    clean = [_clean_gate(g, i) for i, g in enumerate(gates)]
    names = [g["name"].lower() for g in clean]
    if len(set(names)) != len(names):
        raise TopologyError("two gates share a name")
    have = {r["id"] for r in store.q("SELECT id FROM camera_gates WHERE camera_id=?", (int(camera_id),))}
    for g in clean:
        if g["id"] is not None and g["id"] not in have:
            raise TopologyError(f"gate {g['id']} does not belong to this camera")
    keep = {g["id"] for g in clean if g["id"] is not None}
    gone = have - keep
    removed = 0
    now = time.time()
    with store.lock:
        db = store.db
        if gone:
            marks = ",".join("?" * len(gone))
            removed = db.execute(f"SELECT COUNT(*) FROM camera_links WHERE from_gate IN ({marks}) OR to_gate IN ({marks})",
                                 (*gone, *gone)).fetchone()[0]
            db.execute(f"DELETE FROM camera_links WHERE from_gate IN ({marks}) OR to_gate IN ({marks})", (*gone, *gone))
            db.execute(f"DELETE FROM camera_gates WHERE id IN ({marks})", tuple(gone))
        # names are unique per camera: park the kept ones first so two gates can swap names in one save
        for g in clean:
            if g["id"] is not None:
                db.execute("UPDATE camera_gates SET name=? WHERE id=?", (f"~{g['id']}", g["id"]))
        for g in clean:
            if g["id"] is None:
                try:
                    cur = db.execute("INSERT INTO camera_gates (camera_id, name, kind, points, updated_at, updated_by)"
                                     " VALUES (?,?,?,?,?,?)",
                                     (int(camera_id), g["name"], g["kind"], json.dumps(g["points"]), now, who))
                except Exception:
                    db.rollback()
                    raise TopologyError(f"a gate named '{g['name']}' already exists on this camera") from None
                g["id"] = cur.lastrowid
            else:
                db.execute("UPDATE camera_gates SET name=?, kind=?, points=?, updated_at=?, updated_by=? WHERE id=?",
                           (g["name"], g["kind"], json.dumps(g["points"]), now, who, g["id"]))
        db.commit()
    changed = _bump_and_rebuild(store)
    retagged = reassign_gates(store, camera_id)
    return {"gates": gates_of(store, camera_id), "removed_links": removed, "retagged": retagged, "routes": changed}


# ---------------------------------------------------------------- links and routes

def suggest_window(store, from_camera, to_camera) -> Optional[dict]:
    """The map's suggestion for a link window (see geo.link_window), or None when a camera is not placed."""
    a, b = geo_mod.get(store, from_camera), geo_mod.get(store, to_camera)
    return geo_mod.link_window(a, b) if a and b else None


def list_links(store) -> list:
    return store.q("SELECT l.*, fg.name AS from_gate_name, tg.name AS to_gate_name FROM camera_links l"
                   " JOIN camera_gates fg ON fg.id=l.from_gate JOIN camera_gates tg ON tg.id=l.to_gate ORDER BY l.id")


def save_link(store, data, who="", link_id=None) -> dict:
    """Create (or, with `link_id`, replace) a link. Without min_s / max_s the window comes from the map."""
    ensure_schema(store)
    if not isinstance(data, dict):
        raise TopologyError("send a JSON object")
    try:
        fc, tc = int(data["from_camera"]), int(data["to_camera"])
        fg, tg = int(data["from_gate"]), int(data["to_gate"])
    except (KeyError, TypeError, ValueError):
        raise TopologyError("from_camera, from_gate, to_camera and to_gate are required numbers") from None
    if fc == tc:
        raise TopologyError("a link joins two different cameras")
    g_from = store.q("SELECT * FROM camera_gates WHERE id=? AND camera_id=?", (fg, fc))
    g_to = store.q("SELECT * FROM camera_gates WHERE id=? AND camera_id=?", (tg, tc))
    if not g_from or not g_to:
        raise TopologyError("each gate must belong to its camera")
    if g_from[0]["kind"] not in ("out", "both"):
        raise TopologyError(f"gate '{g_from[0]['name']}' is an entry gate: a link leaves through an out or both gate")
    if g_to[0]["kind"] not in ("in", "both"):
        raise TopologyError(f"gate '{g_to[0]['name']}' is an exit gate: a link arrives through an in or both gate")
    sug = suggest_window(store, fc, tc)
    if data.get("min_s") in (None, "") and data.get("max_s") in (None, ""):
        if sug is None:
            raise TopologyError("place both cameras on the map to get a suggested window, or type min and max seconds")
        lo, hi, source = sug["min_s"], sug["max_s"], "suggested"
    else:
        try:
            lo, hi = float(data["min_s"]), float(data["max_s"])
        except (KeyError, TypeError, ValueError):
            raise TopologyError("min_s and max_s must be numbers") from None
        source = "manual"
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or hi <= lo or hi > geo_mod.WINDOW_MAX_S:
        raise TopologyError(f"the window needs 0 <= min < max <= {geo_mod.WINDOW_MAX_S:g} seconds")
    args = (fc, fg, tc, tg, lo, hi, sug["distance_m"] if sug else None, source, time.time(), who)
    try:
        if link_id is None:
            lid = store.x("INSERT INTO camera_links (from_camera, from_gate, to_camera, to_gate, min_s, max_s,"
                          " distance_m, window_source, updated_at, updated_by) VALUES (?,?,?,?,?,?,?,?,?,?)", args)
        else:
            if not store.q("SELECT 1 FROM camera_links WHERE id=?", (int(link_id),)):
                raise TopologyError("no such link")
            store.x("UPDATE camera_links SET from_camera=?, from_gate=?, to_camera=?, to_gate=?, min_s=?, max_s=?,"
                    " distance_m=?, window_source=?, updated_at=?, updated_by=? WHERE id=?", (*args, int(link_id)))
            lid = int(link_id)
    except TopologyError:
        raise
    except Exception:
        raise TopologyError("those two gates are already linked") from None
    _bump_and_rebuild(store)
    return next(l for l in list_links(store) if l["id"] == lid)


def delete_link(store, link_id) -> bool:
    ensure_schema(store)
    had = bool(store.q("SELECT 1 FROM camera_links WHERE id=?", (int(link_id),)))
    store.x("DELETE FROM camera_links WHERE id=?", (int(link_id),))
    if had:
        _bump_and_rebuild(store)
    return had


def build_routes(links, dwell_s=SKIP_DWELL_S) -> list:
    """Every route of one or two links, as dicts. A two-link route X -> Y -> Z takes [min1 + min2,
    max1 + max2 + dwell_s] (the extra is time inside Y's view); it never returns to X, and is dropped where a
    direct link X -> Z already exists. Which of Y's exits the vehicle took is unknown (no turn table), so any
    link leaving Y counts. Built once per topology change; read per sighting."""
    direct, by_cam = {}, defaultdict(list)
    for l in links:
        if l["from_camera"] == l["to_camera"]:
            continue
        r = dict(from_camera=l["from_camera"], from_gate=l["from_gate"], to_camera=l["to_camera"],
                 to_gate=l["to_gate"], min_s=float(l["min_s"]), max_s=float(l["max_s"]), hops=1, via_camera=None)
        direct[(r["from_camera"], r["from_gate"], r["to_camera"], r["to_gate"])] = r
        by_cam[l["from_camera"]].append(r)
    twice = {}
    for a in direct.values():
        for l in by_cam.get(a["to_camera"], ()):
            if l["to_camera"] in (a["from_camera"], a["to_camera"]):
                continue
            key = (a["from_camera"], a["from_gate"], l["to_camera"], l["to_gate"])
            if key in direct:
                continue
            r = dict(from_camera=a["from_camera"], from_gate=a["from_gate"], to_camera=l["to_camera"],
                     to_gate=l["to_gate"], min_s=a["min_s"] + l["min_s"],
                     max_s=a["max_s"] + l["max_s"] + dwell_s, hops=2, via_camera=a["to_camera"])
            had = twice.get(key)
            if had is None:
                twice[key] = r
            else:                                  # several middle cameras: the union of their windows
                had["min_s"], had["max_s"] = min(had["min_s"], r["min_s"]), max(had["max_s"], r["max_s"])
    return list(direct.values()) + list(twice.values())


def _bump_and_rebuild(store) -> int:
    """The topology changed: recompute the two-link routes, bump the revision readers cache against."""
    routes = build_routes(store.q("SELECT * FROM camera_links"))
    with store.lock:
        store.db.execute("DELETE FROM camera_routes")
        store.db.executemany(
            "INSERT INTO camera_routes (from_camera, from_gate, to_camera, to_gate, min_s, max_s, hops, via_camera)"
            " VALUES (:from_camera,:from_gate,:to_camera,:to_gate,:min_s,:max_s,:hops,:via_camera)", routes)
        store.db.execute("INSERT INTO topology_meta (key, value) VALUES ('revision', 1)"
                         " ON CONFLICT(key) DO UPDATE SET value=value+1")
        store.db.commit()
    return len(routes)


def revision(store) -> int:
    ensure_schema(store)
    rows = store.q("SELECT value FROM topology_meta WHERE key='revision'")
    return int(rows[0]["value"]) if rows else 0


class Topology:
    """A snapshot of gates, links and routes, indexed for the matcher."""

    def __init__(self, gates, links, routes, offsets=None, rev=0):
        self.gates, self.links, self.routes, self.rev = gates, links, routes, rev
        self.offsets = offsets or {}
        self.cameras = {l["from_camera"] for l in links} | {l["to_camera"] for l in links}
        self._into, self._out = defaultdict(list), defaultdict(list)
        for r in routes:
            self._into[(r["to_camera"], r["to_gate"])].append(r)
            self._out[(r["from_camera"], r["from_gate"])].append(r)

    def active(self) -> bool:
        return bool(self.links)

    def offset(self, camera_id) -> float:
        return self.offsets.get(camera_id, 0.0)

    def into(self, camera_id, gate, hops) -> list:
        return [r for r in self._into.get((camera_id, gate), ()) if r["hops"] == hops] if gate is not None else []

    def out_of(self, camera_id, gate, hops=1) -> list:
        return [r for r in self._out.get((camera_id, gate), ()) if r["hops"] == hops] if gate is not None else []


def load(store) -> Topology:
    ensure_schema(store)
    links = store.q("SELECT * FROM camera_links")
    return Topology(store.q("SELECT id, camera_id, name, kind FROM camera_gates"), links,
                    store.q("SELECT * FROM camera_routes"),
                    {cid: g.get("clock_offset_s", 0.0) or 0.0 for cid, g in _geos(store).items()}, revision(store))


def _geos(store) -> dict:
    return geo_mod.all_geo(store)


# ---------------------------------------------------------------- scoring

def t_in(t) -> float:
    return t.t_enter if t.t_enter is not None else t.t_cross


def t_out(t) -> float:
    return t.t_exit if t.t_exit is not None else t.t_cross


def pair_cost(a, b, route, topo, llr=None):
    """reid.pair_terms for `a` (upstream) -> `b`, or None when it cannot be the same vehicle.

    Times are the moment `a` left and `b` entered (plus each camera's clock offset), so the gap is exactly what
    the link window describes. The window must contain the gap. The time term is centred on the middle of the
    window with a sigma of a quarter of its width: at the window edge it costs 2 and nothing inside costs more,
    so timing cannot veto a pair the window admits. Lateral position means nothing across two cameras and is
    switched off (NaN)."""
    ta, tb = t_out(a) + topo.offset(a.camera_id), t_in(b) + topo.offset(b.camera_id)
    gap = tb - ta
    if not route["min_s"] <= gap <= route["max_s"]:
        return None
    mid, half = (route["min_s"] + route["max_s"]) / 2, max(1.0, (route["max_s"] - route["min_s"]) / 2)
    v = 0.5 * (a.speed + b.speed)
    if v <= 0:
        return None
    P = dict(reid.default_params(v * mid), t_sig0=max(0.4, half / 2), t_sig1=0.0)
    nan = float("nan")
    return reid.pair_terms(dataclasses.replace(a, t_cross=ta, lateral=nan), dataclasses.replace(b, t_cross=tb, lateral=nan),
                           P, llr)


def decide(scored, accept, margin):
    """The cheapest candidate if it is under `accept` and `margin` ahead of the next; else None.
    `scored` is [(cost, a, route, terms)]."""
    ok = sorted((s for s in scored if s[0] < accept), key=lambda s: (s[0], getattr(s[1], "id", 0) or 0))
    if not ok:
        return None
    if len(ok) > 1 and ok[1][0] - ok[0][0] < margin:     # like reid.match_online: only the acceptable ones count
        return None
    return ok[0][1], ok[0][2], ok[0][3]


def choose(b, index, topo, llr=None, record=None):
    """Which upstream tracklet is `b`? (a, route, terms) or None.

    One-link routes first. Only when nothing is accepted are the two-link routes tried, with a stricter bar.
    `index` supplies candidates and the one-to-one test; `record`, when given, collects every scored pair
    (a, b, terms) so the colour LLRs can be fitted from them."""
    for hops, accept, margin in ((1, ACCEPT, MARGIN), (2, SKIP_ACCEPT, SKIP_MARGIN)):
        scored = []
        for r in topo.into(b.camera_id, b.entry_gate, hops):
            for a in index.candidates(b, r):
                if a is b or (a.id is not None and a.id == b.id):
                    continue
                if index.taken(a, b, r):
                    continue
                terms = pair_cost(a, b, r, topo, llr)
                if terms is None:
                    continue
                scored.append((terms["cost"], a, r, terms))
                if record is not None:
                    record.append((a, b, terms, reid.appearance(a, b)[1]))
        best = decide(scored, accept, margin)
        if best is not None:
            return best
    return None


# ---------------------------------------------------------------- the two indexes

class MemoryIndex:
    """Candidates over tracklets held in memory (the batch path). Mirrors StoreIndex exactly."""

    def __init__(self, tracklets, topo):
        self.topo = topo
        self.exit, self.enter = defaultdict(list), defaultdict(list)
        for t in tracklets:
            if t.exit_gate is not None and t.t_exit is not None:
                self.exit[(t.camera_id, t.exit_gate)].append(t)
            if t.entry_gate is not None and t.t_enter is not None:
                self.enter[(t.camera_id, t.entry_gate)].append(t)
        self._exit_keys = {}
        for k, v in self.exit.items():
            v.sort(key=lambda t: (t.t_exit, t.id or 0))
            self._exit_keys[k] = [t.t_exit for t in v]
        self._enter_keys = {}
        for k, v in self.enter.items():
            v.sort(key=lambda t: (t.t_enter, t.id or 0))
            self._enter_keys[k] = [t.t_enter for t in v]
        self.parent, self.cams = {}, {}
        self.n_candidates = 0

    def candidates(self, b, route):
        lo, hi = _exit_bounds(b, route, self.topo)
        key = (route["from_camera"], route["from_gate"])
        keys = self._exit_keys.get(key, [])
        got = self.exit.get(key, [])[bisect.bisect_left(keys, lo):bisect.bisect_right(keys, hi)]
        self.n_candidates += len(got)
        return got

    def _find(self, t):
        k = id(t)
        self.parent.setdefault(k, k)
        self.cams.setdefault(k, {t.camera_id})
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def taken(self, a, b, route):
        """Is `a` already the predecessor of another vehicle at b's camera - or, over two links, did the
        camera in between in fact see it (so it did not miss it)?"""
        cams = self.cams.get(self._find(a), {a.camera_id})
        return b.camera_id in cams or (route["via_camera"] is not None and route["via_camera"] in cams)

    def link(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[rb] = ra
            self.cams[ra] = self.cams[ra] | self.cams[rb]

    def downstream(self, a, route):
        lo, hi = _enter_bounds(a, route, self.topo)
        key = (route["to_camera"], route["to_gate"])
        keys = self._enter_keys.get(key, [])
        return self.enter.get(key, [])[bisect.bisect_left(keys, lo):bisect.bisect_right(keys, hi)]


def _exit_bounds(b, route, topo):
    """The exit times at the upstream camera that put a gap inside the window, in that camera's own clock."""
    tb = t_in(b) + topo.offset(b.camera_id) - topo.offset(route["from_camera"])
    return tb - route["max_s"], tb - route["min_s"]


def _enter_bounds(a, route, topo):
    ta = t_out(a) + topo.offset(a.camera_id) - topo.offset(route["to_camera"])
    return ta + route["min_s"], ta + route["max_s"]


class StoreIndex:
    """Candidates read from the database (the online path): an indexed range query per route, so the work is
    the number of vehicles inside the link window, not the number stored."""

    def __init__(self, store, topo):
        self.store, self.topo = store, topo
        self.n_candidates = 0

    def candidates(self, b, route):
        from .link import to_tracklet
        lo, hi = _exit_bounds(b, route, self.topo)
        rows = self.store.q("SELECT * FROM tracklets WHERE camera_id=? AND exit_gate=? AND t_exit BETWEEN ? AND ?"
                            " AND t_cross IS NOT NULL AND speed > 0 ORDER BY t_exit, id",
                            (route["from_camera"], route["from_gate"], lo, hi))
        self.n_candidates += len(rows)
        return [to_tracklet(r) for r in rows]

    def taken(self, a, b, route):
        if a.global_id is None:
            return False
        cams = (b.camera_id,) if route["via_camera"] is None else (b.camera_id, route["via_camera"])
        marks = ",".join("?" * len(cams))
        return bool(self.store.q(f"SELECT 1 FROM tracklets WHERE global_id=? AND camera_id IN ({marks}) AND id<>? LIMIT 1",
                                 (a.global_id, *cams, b.id if b.id is not None else -1)))

    def downstream(self, a, route):
        from .link import to_tracklet
        lo, hi = _enter_bounds(a, route, self.topo)
        rows = self.store.q("SELECT * FROM tracklets WHERE camera_id=? AND entry_gate=? AND t_enter BETWEEN ? AND ?"
                            " AND t_cross IS NOT NULL AND speed > 0 ORDER BY t_enter, id",
                            (route["to_camera"], route["to_gate"], lo, hi))
        return [to_tracklet(r) for r in rows]


# ---------------------------------------------------------------- batch

def _order(t):
    return (t_out(t), t.id or 0)


def replay(tracklets, topo, llr=None, record=None):
    """The online algorithm replayed over stored tracklets in the order they were stored (by exit time).
    Returns [(a, b, route, terms)] - the links between tracklets of cameras that have topology."""
    mine = [t for t in tracklets if t.camera_id in topo.cameras]
    idx = MemoryIndex(mine, topo)
    links = []
    for b in sorted(mine, key=_order):
        got = choose(b, idx, topo, llr, record)
        if got is not None:
            a, route, terms = got
            idx.link(a, b)
            links.append((a, b, route, terms))
    return links


def fit_llr(tracklets, topo, iters=4):
    """The colour / embedding LLRs for topology candidates, fitted like reid.refit does for a flow.

    Every scored candidate pair is recorded; the positives are a greedy one-to-one assignment of the cheapest
    pairs under the accept bar (a tracklet is used once on each side), everything else is the background.
    Taking the confident-and-unambiguous pairs (what `choose` would accept) is not enough here: in dense
    traffic nothing is unambiguous until the colour cue exists, so the cue would never be learned. Like
    reid.refit, the first pass has no colour cue at all (timing, speed, size, class) and each later pass uses
    what the one before learned, so a mostly-right first guess is enough."""
    llr = defaultdict(reid.LLR)
    for _ in range(iters):
        rec = []
        replay(tracklets, topo, llr, rec)
        used_a, used_b, taken = set(), set(), set()
        for a, b, terms, tag in sorted(rec, key=lambda r: (r[2]["cost"], r[0].id or 0, r[1].id or 0)):
            if terms["cost"] >= reid.ACCEPT_COST:
                break
            if id(a) in used_a or id(b) in used_b:
                continue
            used_a.add(id(a))
            used_b.add(id(b))
            taken.add((id(a), id(b)))
        pos, neg = defaultdict(list), defaultdict(list)
        for a, b, terms, tag in rec:
            where = pos if (id(a), id(b)) in taken else neg
            where[("hist", a.group)].append(terms["hist"])
            where[("lab", a.group)].append(terms["lab"])
            if tag:
                where[("embed:" + tag, a.group)].append(terms["embed"])
        nxt = defaultdict(reid.LLR)
        for key in set(pos) | set(neg):
            nxt[key] = reid.LLR(pos.get(key), neg.get(key), weight=1.0)
        llr = nxt
    return llr


def combine(tracklets, flow_links, topo, llr=None, log=lambda m: None):
    """Batch result when topology exists. `flow_links` are reid.link_all's links; those between two cameras that
    both have topology are dropped (the topology decides those), the rest are kept, and the topology links are
    added. Returns (links, groups) shaped like reid.link_all's, with global ids assigned on the tracklets."""
    if llr is None:
        llr = fit_llr(tracklets, topo)
    topo_links = replay(tracklets, topo, llr)
    kept = [l for l in flow_links if not (l[0].camera_id in topo.cameras and l[1].camera_id in topo.cameras)]
    log(f"  topology: {len(topo_links)} link(s) over {len(topo.cameras)} camera(s); "
        f"{len(flow_links) - len(kept)} flow link(s) between them set aside")
    u = reid.Union()
    for t in tracklets:
        u.find(id(t))
    for a, b, *_ in kept + topo_links:
        u.union(id(a), id(b))
    by_id = {id(t): t for t in tracklets}
    groups = {}
    for gid, (_, members) in enumerate(sorted(u.groups().items(), key=lambda kv: -len(kv[1])), start=1):
        group = [by_id[m] for m in members]
        for t in group:
            t.global_id = gid
        groups[gid] = sorted(group, key=lambda t: t.t_cross)
    return kept + [(a, b, terms) for a, b, _, terms in topo_links], groups


# ---------------------------------------------------------------- online

def _first_time(store, gid) -> float:
    rows = store.q("SELECT MIN(COALESCE(t_enter, t_cross)) AS t FROM tracklets WHERE global_id=?", (gid,))
    return rows[0]["t"] if rows and rows[0]["t"] is not None else float("inf")


def _is_head(store, d) -> bool:
    """Is `d` the first sighting of its vehicle (so nothing upstream was found for it)?"""
    if d.global_id is None:
        return True
    rows = store.q("SELECT id FROM tracklets WHERE global_id=? ORDER BY COALESCE(t_enter, t_cross), id LIMIT 1",
                   (d.global_id,))
    return bool(rows) and rows[0]["id"] == d.id


def join(store, a, d) -> int:
    """`a` and `d` are one vehicle: give them one id. When both already have one, the vehicle whose first
    sighting is earlier keeps its id (the first camera's) and the other is merged into it and recorded."""
    ga, gd = a.global_id, d.global_id
    if ga is None and gd is None:
        gid = store.next_global_id()
        store.set_global_ids([(a.id, gid), (d.id, gid)], source="online")
    elif ga is None:
        gid = gd
        store.set_global_ids([(a.id, gid)], source="online")
    elif gd is None:
        gid = ga
        store.set_global_ids([(d.id, gid)], source="online")
    else:
        if ga == gd:
            return ga
        first_a, first_d = _first_time(store, ga), _first_time(store, gd)
        keep, old = (ga, gd) if (first_a, ga) <= (first_d, gd) else (gd, ga)
        merge_vehicles(store, keep, old, "out-of-order record joined two ids")
        gid = keep
    a.global_id = gid
    return gid


def link_online(store, topo, b, llr=None) -> Optional[int]:
    """Give the tracklet `b`, just stored, a global id by topology. Two steps:

    1. upstream - the usual one: which vehicle that left a linked camera is this?
    2. out-of-order - `b` may be the upstream record of a vehicle whose NEXT camera was processed first and
       already gave it its own id. Each such sighting is asked again, this time with `b` among its candidates;
       if it picks `b`, the two ids merge (the earlier camera's id survives).
    Returns the id `b` ended up with, or None."""
    idx = StoreIndex(store, topo)
    got = choose(b, idx, topo, llr)
    gid = None
    if got is not None:
        a, route, _ = got
        gid = a.global_id or store.next_global_id()
        pairs = [(b.id, gid)] + ([] if a.global_id else [(a.id, gid)])
        store.set_global_ids(pairs, source="online" if route["hops"] == 1 else "online-skip")
        b.global_id = gid
    if b.exit_gate is not None and b.t_exit is not None:
        for route in topo.out_of(b.camera_id, b.exit_gate, hops=1):
            for d in idx.downstream(b, route):
                if d.id == b.id or not _is_head(store, d):
                    continue
                again = choose(d, idx, topo, llr)
                if again is not None and again[0].id == b.id:
                    b.global_id = _reload_gid(store, b)
                    gid = join(store, b, d)
    return gid


def _reload_gid(store, t):
    rows = store.q("SELECT global_id FROM tracklets WHERE id=?", (t.id,))
    return rows[0]["global_id"] if rows else t.global_id
