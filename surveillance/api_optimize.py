"""HTTP surface of the 3D city view and the optimiser (WP10, requirement R-6 and the pipeline's OPTIMIZE stage).

A feature module with its own router, built on the real code in surveillance/optimize/. It adds a network EDITOR
(the admin draws intersections and roads, attaches each placed camera and its road zone to a road), and serves the
live view, the signal-timing recommendation, the simulator and the router. Nothing here is applied to a real signal.

Wiring (the only edits server.py needs):
    from . import api_optimize
    api_optimize.ensure_schema(store)
    app.include_router(api_optimize.make_router(
        store, current_user, require, CONTROLLER, ADMIN,
        snapshots_fn=lambda: api_optimize.snapshots_from_workers(manager.workers)))
and optionally `google_fn=` (a callable returning a GoogleTraffic or None). Left out, the router makes one from the
server key (api_map.server_key) and keeps it, so its cache and call budget survive between requests.

Endpoints (/api/ is behind the server's fail-shut guard; `staff` = controller or admin, a driver is refused)
    GET    /city                              page (redirects to /login, drivers to /portal)
    GET    /api/optimize/network              staff   the drawn network (or the labelled demo), cameras, zones
    PUT    /api/optimize/network              admin   save the network (validated, audited)
    DELETE /api/optimize/network              admin   delete the drawn network
    POST   /api/optimize/network/validate     admin   check a network without saving it
    POST   /api/optimize/network/demo         admin   a SYNTHETIC "demo network around my cameras" (not saved)
    GET    /api/optimize/live                 staff   links and cameras with the camera-measured state + freshness
    GET    /api/optimize/recommendation       staff   suggested green seconds per phase (a suggestion only)
    POST   /api/optimize/simulate             staff   fixed-time vs max pressure (+ emergency corridor), bounded
    POST   /api/optimize/route                staff   k alternatives, incidents from recorded violations' impact
    GET    /api/optimize/google               admin   Google traffic status (never the key)
    PUT    /api/optimize/google               admin   enable / disable Google traffic as a planner input

GOOGLE RULE (R-8). Google Maps Platform terms forbid showing Google Maps content on a non-Google map. So:
  * the 3D view (MapLibre) never asks for Google: /live and /route are called with use_google=false there, and
    with it false Google is not even called, so no Google-derived number can be in the response;
  * Google is used only when the admin has enabled it AND a server key exists (`google_allowed`), only through
    GoogleTraffic (cache + per-minute call budget), and only in two places: /live and /route with
    use_google=true. Their responses say `contains_google: true`; they are for text/tables outside any map and
    for the Google-provider view of /map, never for MapLibre;
  * recommendation and simulation never use Google (queues come from cameras; the simulator is synthetic).

Honesty. The simulator is a mesoscopic queue model written for this project, not SUMO and not calibrated to
Dhaka; max pressure is an established adaptive rule (Varaiya 2013), not a trained model; 1200 vehicles/h/lane
saturation flow is an assumption; default demand is synthetic. `NOTES` carries these into every response.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse

from . import api_map
from . import geo as geo_mod
from .optimize import network as N
from .optimize import routing, signals as S, sim as M
from .optimize.google_traffic import GoogleTraffic, apply_google

WEB = Path(__file__).resolve().parents[1] / "web"
SITE_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
NODE_ID_RE = re.compile(r"^[A-Za-z0-9_.]{1,24}$")          # no '-' or '>': link ids are "<from>-<to>"
MAX_NODES, MAX_ROADS, MAX_FLOWS = 120, 250, 60
LANES = (1, 8)
SPEED_KPH = (5.0, 120.0)
LENGTH_M = (5.0, 50000.0)
SIM_HORIZON_S = (60.0, 900.0)
SIM_DRAIN_S = 300.0
SIM_FRAMES_MAX = 300
SIM_TIMEOUT_S = 60.0
INCIDENT_WINDOW_MIN = (1, 720)

SCHEMA = """
CREATE TABLE IF NOT EXISTS optimize_networks (
    site TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS optimize_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT ''
);
"""

NOTES = [
    "Simulator results are synthetic: a mesoscopic queue model written for this project (not SUMO, not calibrated "
    "to Dhaka). They say how the controllers behave under these rules, not what a street would do.",
    "Max pressure is an established adaptive control rule (Varaiya 2013), not a trained model. Nothing here has "
    "controlled a real signal and nothing is applied to one.",
    f"Saturation flow of {N.DHAKA_SAT_FLOW_VPH_PER_LANE:.0f} vehicles/h/lane is an assumption, not a measurement.",
    "Default simulated demand is synthetic, not measured.",
]
GOOGLE_RULE = ("Google traffic is a planner input shown only as text or on a Google map. It is never drawn on the "
               "MapLibre 3D view (Google Maps Platform terms).")


class NetworkError(ValueError):
    """A network that cannot be stored. `problems` is every reason, so one pass fixes the lot."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def ensure_schema(store) -> None:
    """Create this module's tables (optimize_networks, optimize_settings). Idempotent; touches nothing else."""
    api_map.ensure_schema(store)           # camera_geo + map_settings (the server key lives there)
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


# ---------------------------------------------------------------- the editor's network document

def _num(v, what, problems, lo=None, hi=None):
    if isinstance(v, bool) or v is None:
        problems.append(f"{what} must be a number")
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        problems.append(f"{what} must be a number")
        return None
    if not math.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        rng = "" if lo is None else f" between {lo:g} and {hi:g}"
        problems.append(f"{what} must be a finite number{rng}")
        return None
    return x


def validate_doc(doc, known_cameras: Optional[dict] = None, placed: Optional[set] = None) -> tuple:
    """(clean document, warnings). Raises NetworkError listing every problem.

    Document: {"name", "synthetic", "nodes": [{"id", "kind", "lat", "lon", "name"}], "roads": [{"from", "to",
    "name", "lanes", "speed_kph", "length_m"?, "two_way", "cameras": [{"camera_id", "zone", "dir": "fwd"|"back"}]}]}.
    A road is drawn once and is two directed links ("a-b" and, if two_way, "b-a"); a camera watches one direction:
    "fwd" is the a->b link, "back" is b->a. `known_cameras` {id: name} and `placed` {ids} are checked when given."""
    problems, warnings = [], []
    if not isinstance(doc, dict):
        raise NetworkError(["send a JSON object"])
    nodes_in, roads_in = doc.get("nodes"), doc.get("roads")
    if not isinstance(nodes_in, list) or not isinstance(roads_in, list):
        raise NetworkError(["nodes and roads must be lists"])
    if len(nodes_in) > MAX_NODES or len(roads_in) > MAX_ROADS:
        raise NetworkError([f"at most {MAX_NODES} intersections and {MAX_ROADS} roads"])
    nodes, ids = [], set()
    (la0, la1), (lo0, lo1) = geo_mod.DHAKA_BOX["lat"], geo_mod.DHAKA_BOX["lon"]
    for i, n in enumerate(nodes_in):
        if not isinstance(n, dict):
            problems.append(f"intersection {i + 1}: must be an object")
            continue
        nid = str(n.get("id", "")).strip()
        tag = f"intersection {nid or i + 1}"
        if not NODE_ID_RE.match(nid):
            problems.append(f"{tag}: id must be 1-24 letters, digits, '_' or '.'")
            continue
        if nid in ids:
            problems.append(f"{tag}: id used twice")
            continue
        ids.add(nid)
        kind = n.get("kind") or N.SIGNAL
        if kind not in N.NODE_KINDS:
            problems.append(f"{tag}: kind must be one of {', '.join(N.NODE_KINDS)}")
            continue
        lat = _num(n.get("lat"), f"{tag}: lat", problems, -90, 90)
        lon = _num(n.get("lon"), f"{tag}: lon", problems, -180, 180)
        if lat is not None and lon is not None and not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            warnings.append(f"{tag} is outside greater Dhaka")
        nodes.append({"id": nid, "kind": kind, "lat": None if lat is None else round(lat, 7),
                      "lon": None if lon is None else round(lon, 7), "name": str(n.get("name", ""))[:80]})
    by_id = {n["id"]: n for n in nodes}
    roads, pairs, zones_seen = [], set(), {}
    for i, r in enumerate(roads_in):
        if not isinstance(r, dict):
            problems.append(f"road {i + 1}: must be an object")
            continue
        a, b = str(r.get("from", "")).strip(), str(r.get("to", "")).strip()
        tag = f"road {a}-{b}"
        if a not in by_id or b not in by_id:
            problems.append(f"{tag}: both ends must be intersections of the network")
            continue
        if a == b:
            problems.append(f"{tag}: a road cannot start and end at the same intersection")
            continue
        if (a, b) in pairs or (b, a) in pairs:
            problems.append(f"{tag}: there is already a road between these two intersections")
            continue
        pairs.add((a, b))
        lanes = _num(r.get("lanes", 2), f"{tag}: lanes", problems, *LANES)
        speed = _num(r.get("speed_kph", N.DEFAULT_FREE_FLOW_KPH), f"{tag}: speed_kph", problems, *SPEED_KPH)
        length = r.get("length_m")
        straight = None
        pa, pb = by_id[a], by_id[b]
        if None not in (pa["lat"], pa["lon"], pb["lat"], pb["lon"]):
            straight = N.haversine_m(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
            if straight < LENGTH_M[0]:
                problems.append(f"{tag}: the two intersections are under {LENGTH_M[0]:g} m apart")
        if length in (None, ""):
            length = None
        else:
            length = _num(length, f"{tag}: length_m", problems, *LENGTH_M)
            if length is not None and straight and length < 0.9 * straight:
                warnings.append(f"{tag}: length {length:.0f} m is shorter than the straight line ({straight:.0f} m)")
        two_way = r.get("two_way", True)
        if not isinstance(two_way, bool):
            problems.append(f"{tag}: two_way must be true or false")
            two_way = True
        cams, dirs = [], set()
        for c in r.get("cameras") or []:
            if not isinstance(c, dict):
                problems.append(f"{tag}: a camera attachment must be an object")
                continue
            cid, zone, d = c.get("camera_id"), str(c.get("zone", "")).strip(), c.get("dir", "fwd")
            if isinstance(cid, bool) or not isinstance(cid, int):
                problems.append(f"{tag}: camera_id must be a whole number")
                continue
            if known_cameras is not None and cid not in known_cameras:
                problems.append(f"{tag}: camera {cid} does not exist")
                continue
            if not zone or len(zone) > 80:
                problems.append(f"{tag}: camera {cid} needs the name of its road zone (1-80 characters)")
                continue
            if d not in ("fwd", "back") or (d == "back" and not two_way):
                problems.append(f"{tag}: camera {cid} dir must be 'fwd', or 'back' on a two-way road")
                continue
            if d in dirs:
                problems.append(f"{tag}: only one camera can watch each direction")
                continue
            if (cid, zone) in zones_seen:
                problems.append(f"{tag}: camera {cid} zone {zone!r} is already attached to {zones_seen[(cid, zone)]}")
                continue
            dirs.add(d)
            zones_seen[(cid, zone)] = tag
            if placed is not None and cid not in placed:
                warnings.append(f"{tag}: camera {cid} is not placed on the map yet")
            cams.append({"camera_id": cid, "zone": zone, "dir": d})
        roads.append({"from": a, "to": b, "name": str(r.get("name", ""))[:80], "lanes": None if lanes is None else int(lanes),
                      "speed_kph": speed, "length_m": None if length is None else round(length, 1),
                      "two_way": two_way, "cameras": cams})
    if len(nodes) < 2:
        problems.append("a network needs at least two intersections")
    if not roads and len(nodes) >= 2:
        problems.append("a network needs at least one road")
    if problems:
        raise NetworkError(problems)
    degree = {n["id"]: 0 for n in nodes}
    for r in roads:
        degree[r["from"]] += 1
        degree[r["to"]] += 1
    for n in nodes:
        d = degree[n["id"]]
        if d == 0:
            warnings.append(f"intersection {n['id']} has no road")
        elif n["kind"] == N.SIGNAL and d < 2:
            warnings.append(f"signal {n['id']} has only one road: a signal needs at least two arms to do anything")
        elif n["kind"] == N.BOUNDARY and d > 1:
            warnings.append(f"boundary {n['id']} has {d} roads: a boundary is where traffic enters or leaves")
    comp = {n["id"]: n["id"] for n in nodes}

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x
    for r in roads:
        comp[find(r["from"])] = find(r["to"])
    if len({find(n) for n in comp}) > 1:
        warnings.append("the roads form separate pieces that are not connected to each other")
    clean = {"name": str(doc.get("name", ""))[:120], "synthetic": bool(doc.get("synthetic", False)),
             "nodes": nodes, "roads": roads}
    try:
        build_network(clean)
    except ValueError as e:
        raise NetworkError([str(e)]) from None
    return clean, warnings


def build_network(doc: dict, circuity: float = 1.0) -> N.Network:
    """The optimiser Network for a validated document, through network.from_points."""
    pts = [{"id": n["id"], "lat": n["lat"], "lon": n["lon"], "name": n["name"], "kind": n["kind"]} for n in doc["nodes"]]
    roads = []
    for r in doc["roads"]:
        a, b = r["from"], r["to"]
        road = {"from": a, "to": b, "lanes": r["lanes"] if r.get("lanes") is not None else 2,
                "free_flow_kph": r.get("speed_kph") or N.DEFAULT_FREE_FLOW_KPH, "name": r.get("name", ""),
                "two_way": r.get("two_way", True)}
        if r.get("length_m"):
            road["length_m"] = r["length_m"]
        for c in r.get("cameras") or []:
            key = "" if c["dir"] == "fwd" else "back_"
            road[key + "camera_zone"], road[key + "camera_id"] = c["zone"], str(c["camera_id"])
        roads.append(road)
    return N.from_points(pts, roads, circuity=circuity, name=doc.get("name") or "network")


# ---------------------------------------------------------------- the demo network (synthetic)

DEMO_NAME = "Demo network around my cameras (synthetic)"
STUB_M = 250.0


def _ang(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def demo_doc(geos: dict, names: Optional[dict] = None, zones: Optional[dict] = None) -> tuple:
    """(document, notes): a SYNTHETIC layout for when nothing has been drawn.

    One signalised junction at each placed camera (cameras under 40 m apart share one), the junctions joined
    by a minimum spanning tree of straight roads, and boundary stubs of 250 m so every junction has at least
    three arms. Each camera is attached to the arm it looks along, using the first road zone it has. The streets
    are NOT the real street plan; only the camera positions are real. `zones` {camera id: [zone names]}."""
    names, zones = names or {}, zones or {}
    notes = ["Synthetic layout: only the camera positions are real. The roads between them are straight lines and "
             "the stubs are invented. Draw the real network in the editor."]
    cams = sorted(geos)
    if not cams:
        return None, notes + ["No camera is placed on the map yet."]
    clusters: list = []
    for cid in cams:
        g = geos[cid]
        for cl in clusters:
            if N.haversine_m(g["lat"], g["lon"], cl["lat"], cl["lon"]) < 40.0:
                cl["cams"].append(cid)
                break
        else:
            clusters.append({"lat": g["lat"], "lon": g["lon"], "cams": [cid]})
    nodes, roads = [], []
    for i, cl in enumerate(clusters, 1):
        cl["id"] = f"J{i}"
        label = " / ".join(names.get(c) or f"camera {c}" for c in cl["cams"])
        nodes.append({"id": cl["id"], "kind": N.SIGNAL, "lat": cl["lat"], "lon": cl["lon"], "name": label[:80]})
    arms = {cl["id"]: [] for cl in clusters}          # junction -> [{"bearing", "road index", "inbound dir"}]
    done, left = {0}, set(range(1, len(clusters)))
    while left:                                          # Prim: join the nearest unconnected junction
        d, i, j = min((N.haversine_m(clusters[i]["lat"], clusters[i]["lon"], clusters[j]["lat"], clusters[j]["lon"]), i, j)
                      for i in done for j in left)
        a, b = clusters[i], clusters[j]
        roads.append({"from": a["id"], "to": b["id"], "lanes": 2, "speed_kph": N.DEFAULT_FREE_FLOW_KPH, "two_way": True,
                      "cameras": [], "name": ""})
        k = len(roads) - 1
        arms[a["id"]].append({"bearing": geo_mod.bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"]), "road": k, "inbound": "back"})
        arms[b["id"]].append({"bearing": geo_mod.bearing_deg(b["lat"], b["lon"], a["lat"], a["lon"]), "road": k, "inbound": "fwd"})
        done.add(j)
        left.discard(j)
    nstub = 0
    for cl in clusters:
        heading = geos[cl["cams"][0]]["heading_deg"]
        for turn in (0, 180, 90, 270, 45, 225, 135, 315):
            if len(arms[cl["id"]]) >= 3:
                break
            bearing = (heading + turn) % 360.0
            if any(_ang(bearing, x["bearing"]) < 35.0 for x in arms[cl["id"]]):
                continue
            nstub += 1
            la, lo = geo_mod.destination(cl["lat"], cl["lon"], bearing, STUB_M)
            sid = f"B{nstub}"
            nodes.append({"id": sid, "kind": N.BOUNDARY, "lat": round(la, 7), "lon": round(lo, 7), "name": f"edge {nstub} (invented)"})
            roads.append({"from": sid, "to": cl["id"], "lanes": 2, "speed_kph": N.DEFAULT_FREE_FLOW_KPH, "two_way": True,
                          "cameras": [], "name": ""})
            arms[cl["id"]].append({"bearing": bearing, "road": len(roads) - 1, "inbound": "fwd"})
    for cl in clusters:
        taken: set = set()
        for cid in cl["cams"]:
            zs = zones.get(cid) or []
            if not zs:
                notes.append(f"camera {cid} ({names.get(cid, '')}) has no road zone yet, so it is not attached to a road")
                continue
            heading = geos[cid]["heading_deg"]
            for arm in sorted(arms[cl["id"]], key=lambda x: _ang(heading, x["bearing"])):
                if (arm["road"], arm["inbound"]) not in taken:
                    taken.add((arm["road"], arm["inbound"]))
                    roads[arm["road"]]["cameras"].append({"camera_id": cid, "zone": zs[0], "dir": arm["inbound"]})
                    break
            else:
                notes.append(f"camera {cid} found no free road at its junction")
    return {"name": DEMO_NAME, "synthetic": True, "nodes": nodes, "roads": roads}, notes


# ---------------------------------------------------------------- live traffic state

_ADVANCE: dict = {}
_ADVANCE_LOCK = threading.Lock()


def snapshots_from_workers(workers: dict, now: Optional[float] = None) -> dict:
    """{camera id: {zone: state + "wall"}} from WorkerManager.workers, for the `snapshots_fn` of make_router.

    `wall` is the clock time of the zone's latest sample, so a camera that stalled is not "fresh": the
    collector's own wall_at(t) where the video carries a clock, otherwise the last moment this function saw the
    zone's sample time ADVANCE (a recorded file has no clock). A worker with no collector is left out."""
    now = time.time() if now is None else now
    out: dict = {}
    for cid, w in list(workers.items()):
        col = getattr(w, "traffic", None)
        if col is None:
            continue
        try:
            snap = col.snapshot()
        except Exception:
            continue
        zones = {}
        for zone, st in snap.items():
            st = dict(st)
            t = st.get("t")
            if t is not None:
                wall = col.wall_at(t)
                if wall is None:
                    with _ADVANCE_LOCK:
                        prev = _ADVANCE.get((cid, zone))
                        if prev is None or prev[0] != t:
                            _ADVANCE[(cid, zone)] = prev = (t, now)
                        wall = prev[1]
                st["wall"] = wall
            zones[zone] = st
        out[cid] = zones
    return out


def _fresh_only(net: N.Network, now: float) -> N.Network:
    """A copy whose links keep a measurement only if a camera took it recently: stale queues are no basis for a
    timing recommendation."""
    out = net.copy()
    for l in out.links.values():
        if l.state and not l.fresh_camera(now):
            l.state = None
    return out


def _r(x, n=2):
    return None if x is None or not isinstance(x, (int, float)) or not math.isfinite(x) else round(float(x), n)


def link_view(l: N.Link, now: float) -> dict:
    """One link as the API shows it. `status`: live (fresh camera), stale (camera older than CAMERA_FRESH_S),
    google (only when Google was asked for), none (no measurement: free flow assumed)."""
    st = l.state or {}
    age = None if l.updated_at is None else max(0.0, now - l.updated_at)
    src = l.source
    if src == N.CAMERA:
        status = "live" if age is not None and age <= N.CAMERA_FRESH_S else "stale"
    elif src == N.GOOGLE:
        status = "google"
    else:
        status = "none"
    tt = l.travel_time_s
    out = {"id": l.id, "from": l.from_node, "to": l.to_node, "name": l.name, "length_m": _r(l.length_m, 1),
           "lanes": l.lanes, "free_flow_kph": l.free_flow_kph, "free_flow_s": _r(l.free_flow_s, 1),
           "camera_id": None if l.camera_id is None else int(l.camera_id), "camera_zone": l.camera_zone,
           "source": "free_flow" if src == N.FREE_FLOW else src, "status": status, "age_s": _r(age, 1),
           "travel_time_s": _r(tt, 1), "delay_s": _r(max(0.0, tt - l.free_flow_s), 1),
           "ratio": _r(tt / l.free_flow_s, 3) if l.free_flow_s > 0 else None}
    if src == N.CAMERA:
        out["measured"] = {k: _r(st.get(k), 3) for k in ("queue_m", "queue_veh", "speed_kph", "occupancy", "density",
                                                       "wait_s")}
        out["measured"]["speed_source"] = st.get("speed_source")
    elif src == N.GOOGLE:
        out["google"] = {"ratio": _r(st.get("ratio"), 3), "jam_fraction": _r(st.get("jam_fraction"), 3)}
    return out


def camera_views(store, snaps: dict, net: N.Network, now: float, state_fn=None) -> list:
    """Every camera: placement + wedge, and what OUR collector measures in each of its zones, with age."""
    geos = geo_mod.all_geo(store)
    attached: dict = {}
    for l in net.links.values():
        if l.camera_id is not None:
            attached.setdefault(int(l.camera_id), []).append(l.id)
    out = []
    for c in store.cameras():
        cid = c["id"]
        g = geos.get(cid)
        zones = []
        for zone, st in (snaps.get(cid) or {}).items():
            if not N.usable_state(st):
                zones.append({"zone": zone, "usable": False})
                continue
            wall = st.get("wall")
            age = None if wall is None else max(0.0, now - float(wall))
            zones.append({"zone": zone, "usable": True, "density": _r(st.get("density"), 2), "occupancy": _r(st.get("occupancy"), 3),
                          "queue_m": _r(st.get("queue_m"), 1), "speed_kph": _r(st.get("speed_kph"), 1),
                          "vehicles": st.get("vehicles"), "lanes": st.get("lanes"), "age_s": _r(age, 1),
                          "fresh": age is not None and age <= N.CAMERA_FRESH_S})
        used = [z for z in zones if z["usable"] and z.get("density") is not None]
        dens = sum(z["density"] for z in used) / len(used) if used else None
        out.append({"id": cid, "name": c["name"], "placed": g is not None, "geo": geo_mod.view(g) if g else None,
                    "state": (state_fn(cid) if state_fn else "unknown"), "zones": zones, "links": attached.get(cid, []),
                    "density": _r(dens, 2), "fresh": bool(used) and any(z["fresh"] for z in used),
                    "stale": bool(used) and not any(z["fresh"] for z in used)})
    return out


# ---------------------------------------------------------------- incidents from recorded violations

def incident_delays(store, net: N.Network, now: float, window_s: float) -> list:
    """[{"link", "extra_delay_s", "violation_id", "code", "age_s"}] - for each link a camera watches, the delay a
    vehicle should expect from the worst recent recorded violation there, from its scored impact
    (routing.extra_delay_from_impact). Rejected cases are not incidents."""
    try:
        rows = store.q("SELECT e.violation_id, e.camera_id, e.zone, e.code, e.impact, e.created_at FROM event_impact e"
                       " JOIN violations v ON v.id=e.violation_id WHERE v.status!='rejected'"
                       " ORDER BY e.violation_id DESC LIMIT 500")
    except Exception:
        return []
    best: dict = {}
    for r in rows:
        try:
            when = datetime.fromisoformat(r["created_at"]).timestamp()
            impact = json.loads(r["impact"])
        except (TypeError, ValueError):
            continue
        age = now - when
        if age > window_s or age < -60:
            continue
        delay = routing.extra_delay_from_impact(impact)
        if delay <= 0:
            continue
        for l in net.links.values():
            if l.camera_zone == r["zone"] and (l.camera_id is None or l.camera_id == str(r["camera_id"])):
                if l.id not in best or delay > best[l.id]["extra_delay_s"]:
                    best[l.id] = {"link": l.id, "extra_delay_s": round(delay, 1), "violation_id": r["violation_id"],
                                  "code": r["code"], "age_s": round(max(0.0, age), 1)}
    return sorted(best.values(), key=lambda x: -x["extra_delay_s"])


# ---------------------------------------------------------------- simulation helpers

def default_flows(net: N.Network, load: float, horizon_s: float) -> list:
    """SYNTHETIC demand: from each boundary point to the two boundary points farthest from it, with a mid-run
    peak. Not measured; the response says so."""
    bounds = sorted(n.id for n in net.nodes.values() if n.kind == N.BOUNDARY)
    if len(bounds) < 2:
        raise HTTPException(400, "The simulator needs at least two boundary points (where traffic enters and leaves)")
    ff = {lid: l.free_flow_s for lid, l in net.links.items()}
    flows = []
    for o in bounds:
        ranked = []
        for d in bounds:
            if d != o:
                r = routing.shortest_path(net, o, d, weights=ff)
                if r is not None:
                    ranked.append((r.time_s, d))
        for (_, d), base in zip(sorted(ranked, reverse=True)[:2], (220.0, 110.0)):
            v = base * load
            flows.append(M.Flow(o, d, [(0.0, 0.6 * v), (0.25 * horizon_s, 1.2 * v), (0.65 * horizon_s, 0.7 * v)]))
    if not flows:
        raise HTTPException(400, "No boundary point can reach another: the network has no through route")
    return flows


def _round_result(r: dict) -> dict:
    keys = ("generated", "finished", "unfinished", "in_network", "mean_delay_s", "p95_delay_s", "mean_trip_s", "throughput_vph",
            "max_queue_veh", "mean_total_queue_veh", "forced_releases", "ended_s")
    return {k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in keys}


# ---------------------------------------------------------------- Google holder

class _GoogleHolder:
    """One GoogleTraffic per server key, so its cache and call budget outlive a request. The key is read from
    the settings at use time (api_map.server_key) and is never stored here except inside the provider."""

    def __init__(self, store, env=None):
        self.store, self.env, self._p, self._key = store, env, None, None

    def __call__(self) -> Optional[GoogleTraffic]:
        key = api_map.server_key(self.store, self.env)
        if not key:
            self._p = self._key = None
            return None
        if key != self._key:
            self._p, self._key = GoogleTraffic(api_key=key), key
        return self._p


# ---------------------------------------------------------------- the router

NO_NETWORK = "There is no network yet: place cameras on the map or draw one in the editor"


def make_router(store, current_user, require, controller: str = "controller", admin: str = "admin", snapshots_fn: Optional[Callable] = None,
                google_fn: Optional[Callable] = None, *, status_fn: Optional[Callable] = None, web_dir: Optional[Path] = None,
                env=None, clock: Callable = time.time, sim_timeout_s: float = SIM_TIMEOUT_S) -> APIRouter:
    """`snapshots_fn()` -> {camera id: {zone: traffic-state dict (+ "wall")}} (see snapshots_from_workers);
    `google_fn()` -> a GoogleTraffic or None; `status_fn(camera id)` -> {"state": ...}."""
    ensure_schema(store)
    router = APIRouter()
    staff, admin_only = require(controller), require(admin)
    page = (web_dir or WEB) / "city.html"
    google_fn = google_fn or _GoogleHolder(store, env)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="optimize-sim")
    slots = threading.BoundedSemaphore(2)

    # ---- helpers
    def site_of(site: str) -> str:
        if not SITE_RE.match(site or ""):
            raise HTTPException(400, "site must be 1-32 lower-case letters, digits, - or _")
        return site

    def who(user):
        return user.get("username") or user.get("name") or ""

    def snaps() -> dict:
        if snapshots_fn is None:
            return {}
        try:
            return snapshots_fn() or {}
        except Exception:
            return {}

    def state_of(cid):
        if status_fn is None:
            return "unknown"
        try:
            return (status_fn(cid) or {}).get("state", "unknown")
        except Exception:
            return "unknown"

    def camera_zones(snapshot: dict) -> dict:
        """{camera id: [road zone names]} from the stored zone config, plus any zone the collector reports."""
        out = {}
        for c in store.cameras():
            names = []
            try:
                zv = store.zones(c["id"])
                names = [z["name"] for z in ((zv or {}).get("config") or {}).get("zones", []) if z.get("type") == "road" and z.get("name")]
            except Exception:
                pass
            for z in (snapshot.get(c["id"]) or {}):
                if z not in names:
                    names.append(z)
            out[c["id"]] = names
        return out

    def stored(site):
        rows = store.q("SELECT * FROM optimize_networks WHERE site=?", (site,))
        return rows[0] if rows else None

    def effective(site, snapshot=None):
        """(document, source) - the saved network, else the synthetic demo, else (None, "none")."""
        row = stored(site)
        if row:
            return json.loads(row["data"]), "saved"
        geos = geo_mod.all_geo(store)
        if geos:
            snapshot = snaps() if snapshot is None else snapshot
            doc, _ = demo_doc(geos, {c["id"]: c["name"] for c in store.cameras()}, camera_zones(snapshot))
            if doc:
                try:
                    return validate_doc(doc)[0], "demo"
                except NetworkError:
                    pass
        return None, "none"

    def need_network(site):
        doc, source = effective(site)
        if doc is None:
            raise HTTPException(404, NO_NETWORK)
        return doc, source

    def google_allowed() -> bool:
        rows = store.q("SELECT value FROM optimize_settings WHERE key='google_enabled'")
        if not (rows and rows[0]["value"] == "1"):
            return False
        try:
            p = google_fn()
        except Exception:
            return False
        return p is not None and bool(getattr(p, "configured", False))

    def live_net(doc, now, use_google=False, snapshot=None):
        net = N.from_traffic_state(build_network(doc), snaps() if snapshot is None else snapshot, now=now)
        used = False
        if use_google and google_allowed():
            net = apply_google(net, google_fn(), now=now)
            used = any(l.source == N.GOOGLE for l in net.links.values())
        return net, used

    def bool_q(v) -> bool:
        return str(v).lower() in ("1", "true", "yes")

    # ---- the page
    @router.get("/city")
    def city_page(user=Depends(current_user)):
        if user is None:
            return RedirectResponse("/login", status_code=302)
        if user["role"] not in (controller, admin):
            return RedirectResponse("/portal", status_code=302)
        return FileResponse(page, headers={"Cache-Control": "no-store"})

    # ---- the network (editor)
    @router.get("/api/optimize/network")
    def get_network(site: str = "default", user=Depends(staff)):
        site = site_of(site)
        snapshot = snaps()
        doc, source = effective(site, snapshot)
        cams = []
        geos = geo_mod.all_geo(store)
        zones = camera_zones(snapshot)
        for c in store.cameras():
            g = geos.get(c["id"])
            cams.append({"id": c["id"], "name": c["name"], "placed": g is not None, "geo": geo_mod.view(g) if g else None,
                         "zones": zones.get(c["id"], [])})
        row = stored(site)
        warnings = []
        if doc:
            try:
                warnings = validate_doc(doc, {c["id"]: c["name"] for c in cams}, set(geos))[1]
            except NetworkError as e:
                warnings = e.problems
        return {"site": site, "source": source, "doc": doc, "synthetic": bool(doc and doc.get("synthetic")),
                "updated_at": row["updated_at"] if row else None, "updated_by": row["updated_by"] if row else None,
                "cameras": cams, "warnings": warnings, "can_edit": user["role"] == admin,
                "limits": {"nodes": MAX_NODES, "roads": MAX_ROADS, "lanes": list(LANES), "speed_kph": list(SPEED_KPH)},
                "notes": NOTES}

    async def read_doc(request: Request):
        try:
            return await request.json()
        except ValueError:
            raise HTTPException(400, "send a JSON object") from None

    def check_doc(doc):
        cams = {c["id"]: c["name"] for c in store.cameras()}
        try:
            return validate_doc(doc, cams, set(geo_mod.all_geo(store)))
        except NetworkError as e:
            raise HTTPException(400, {"message": "The network has problems", "problems": e.problems}) from None

    @router.put("/api/optimize/network")
    async def put_network(request: Request, site: str = "default", user=Depends(admin_only)):
        site = site_of(site)
        clean, warnings = check_doc(await read_doc(request))
        store.x("INSERT INTO optimize_networks (site, name, data, updated_at, updated_by) VALUES (?,?,?,?,?)"
                " ON CONFLICT(site) DO UPDATE SET name=excluded.name, data=excluded.data,"
                " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (site, clean["name"], json.dumps(clean), clock(), who(user)))
        store.audit(user, "optimize.network", "network", site,
                    f"{len(clean['nodes'])} intersections, {len(clean['roads'])} roads, "
                    f"{sum(len(r['cameras']) for r in clean['roads'])} camera attachments"
                    f"{', synthetic' if clean['synthetic'] else ''}", user.get("address", ""))
        return {"site": site, "doc": clean, "warnings": warnings}

    @router.post("/api/optimize/network/validate")
    async def validate_network(request: Request, user=Depends(admin_only)):
        clean, warnings = check_doc(await read_doc(request))
        return {"ok": True, "doc": clean, "warnings": warnings}

    @router.delete("/api/optimize/network")
    def delete_network(site: str = "default", user=Depends(admin_only)):
        site = site_of(site)
        if not stored(site):
            raise HTTPException(404, "No drawn network to delete")
        store.x("DELETE FROM optimize_networks WHERE site=?", (site,))
        store.audit(user, "optimize.network.delete", "network", site, "deleted", user.get("address", ""))
        return {"ok": True}

    @router.post("/api/optimize/network/demo")
    def make_demo(user=Depends(admin_only)):
        geos = geo_mod.all_geo(store)
        doc, notes = demo_doc(geos, {c["id"]: c["name"] for c in store.cameras()}, camera_zones(snaps()))
        if doc is None:
            raise HTTPException(400, notes[-1])
        clean, warnings = check_doc(doc)
        return {"doc": clean, "warnings": warnings, "notes": notes, "synthetic": True}

    # ---- the live view
    @router.get("/api/optimize/live")
    def live(site: str = "default", use_google: str = "false", user=Depends(staff)):
        """Camera-measured state per link and per camera. `use_google` is for text and the Google map only; the
        MapLibre 3D view never sends it (see the module docstring)."""
        site, want_google, now = site_of(site), bool_q(use_google), clock()
        doc, source = effective(site)
        if doc is None:     # an expected state, not an error: 200 keeps the browser console and the logs clean
            return {"network": False, "site": site, "reason": NO_NETWORK, "links": [], "nodes": [], "cameras": []}
        snapshot = snaps()
        net, used = live_net(doc, now, want_google, snapshot)
        links = [link_view(l, now) for l in net.links.values()]
        nodes = []
        for n in net.nodes.values():
            nodes.append({"id": n.id, "kind": n.kind, "lat": n.lat, "lon": n.lon, "name": n.name})
        return {"site": site, "name": doc.get("name", ""), "source": source, "synthetic": bool(doc.get("synthetic")),
                "network": True, "generated_at": now, "nodes": nodes, "links": links,
                "cameras": camera_views(store, snapshot, net, now, state_fn=state_of),
                "fresh_s": N.CAMERA_FRESH_S, "notes": list(net.notes),
                "contains_google": used, "google_requested": want_google, "google_used": used,
                "assumptions": NOTES, "google_rule": GOOGLE_RULE}

    # ---- the recommendation
    @router.get("/api/optimize/recommendation")
    def recommendation(site: str = "default", cycle_s: float = Query(90.0, ge=40.0, le=180.0), user=Depends(staff)):
        site, now = site_of(site), clock()
        doc, source = need_network(site)
        net, _ = live_net(doc, now, False)
        fresh = _fresh_only(net, now)
        rec = S.recommend_timings(fresh, cycle_s=cycle_s)
        out = []
        for nid, r in rec.items():
            ins = fresh.in_links(nid)
            measured = [l for l in ins if fresh.links[l].state]
            phases = fresh.phases[nid]
            rows = []
            for ph, p in zip(phases, r["phases"]):
                serves = sorted({fresh.movements[m].from_link for m in ph.movements})
                rows.append({**p, "links": serves})
            favoured = max(range(len(rows)), key=lambda i: rows[i]["green_s"]) if rows else None
            out.append({"node": nid, "name": fresh.nodes[nid].name, "cycle_s": r["cycle_s"], "lost_s": r["lost_s"],
                        "demand": r["demand"], "phases": rows, "favoured_phase": favoured if measured else None,
                        "approaches": len(ins), "approaches_measured": len(measured),
                        "basis": "camera" if measured else "none",
                        "message": ("Based on queues measured by cameras in the last "
                                    f"{N.CAMERA_FRESH_S:.0f} s on {len(measured)} of {len(ins)} approaches; the others are "
                                    "counted as empty." if measured else
                                    "No fresh camera data on any approach: equal greens, no basis to change anything.")})
        return {"site": site, "source": source, "synthetic": bool(doc.get("synthetic")), "generated_at": now,
                "cycle_s": cycle_s, "signals": out,
                "disclaimer": "A suggestion. Nothing is applied to any real signal.",
                "method": "Greens proportional to the max-pressure (Varaiya 2013) pressure of each phase from the camera-measured "
                          "queues, at least 7 s each. Not a trained model; Webster's cycle formula needs arrival rates, which "
                          "queues alone do not give.",
                "assumptions": NOTES}

    # ---- the simulator
    def clean_flows(raw, net):
        if not isinstance(raw, list) or not raw or len(raw) > MAX_FLOWS:
            raise HTTPException(400, f"flows must be a list of 1 to {MAX_FLOWS} {{origin, dest, vph}}")
        out = []
        for f in raw:
            try:
                o, d, vph = str(f["origin"]), str(f["dest"]), float(f["vph"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "each flow needs origin, dest and vph") from None
            if o not in net.nodes or d not in net.nodes or o == d:
                raise HTTPException(400, f"flow {o}->{d}: both must be different intersections of the network")
            if not (math.isfinite(vph) and 0 <= vph <= 3000):
                raise HTTPException(400, "vph must be between 0 and 3000")
            out.append(M.Flow.constant(o, d, vph))
        return out

    def run_one(net, flows, make, kw, evs, cancel):
        simulator = M.Simulator(net, flows, make(), evs=evs, **kw)
        res = simulator.run(stop=cancel.is_set)
        return res

    @router.post("/api/optimize/simulate")
    async def simulate(request: Request, site: str = "default", user=Depends(staff)):
        site = site_of(site)
        try:
            body = await request.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(400, "send a JSON object")
        doc, source = need_network(site)
        net = build_network(doc)           # structure only: no camera or Google state reaches the simulator
        try:
            horizon = float(body.get("horizon_s", 300.0))
            load = float(body.get("load", 0.8))
            seed = int(body.get("seed", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "horizon_s, load and seed must be numbers") from None
        if not (SIM_HORIZON_S[0] <= horizon <= SIM_HORIZON_S[1]):
            raise HTTPException(400, f"horizon_s must be between {SIM_HORIZON_S[0]:g} and {SIM_HORIZON_S[1]:g}")
        if not (0.1 <= load <= 2.0) or not (0 <= seed <= 10**6):
            raise HTTPException(400, "load must be 0.1-2.0 and seed 0-1000000")
        wanted = body.get("controllers") or ["fixed", "max_pressure"]
        if not isinstance(wanted, list) or not set(wanted) <= {"fixed", "max_pressure"}:
            raise HTTPException(400, "controllers may contain 'fixed' and 'max_pressure'")
        flows = clean_flows(body["flows"], net) if body.get("flows") else default_flows(net, load, horizon)
        evs, ev_in = [], body.get("ev")
        if ev_in:
            try:
                o, d, dep = str(ev_in["origin"]), str(ev_in["dest"]), float(ev_in.get("depart_s", horizon * 0.3))
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "ev needs origin, dest and depart_s") from None
            if o not in net.nodes or d not in net.nodes or o == d or not (0 <= dep < horizon):
                raise HTTPException(400, "ev origin and dest must be different intersections and depart_s inside the horizon")
            r = routing.shortest_path(net, o, d, weights={k: l.free_flow_s for k, l in net.links.items()})
            if r is None:
                raise HTTPException(400, "The emergency vehicle has no route between those intersections")
            evs = [M.EV(dep, o, d, r.links)]
        want_frames = bool(body.get("frames", False))
        end = horizon + SIM_DRAIN_S
        every = max(1, math.ceil(end / SIM_FRAMES_MAX))
        kw = {"horizon_s": horizon, "drain_s": SIM_DRAIN_S, "seed": seed, "record_frames": want_frames, "frame_every": every}
        avg = M.expected_link_flows(net, flows, horizon)
        makers = {"fixed": ("Fixed-time (splits tuned to the average demand)", lambda: S.FixedTimeController.from_flows(net, avg)),
                  "max_pressure": ("Max pressure (adaptive; min green 15 s)", lambda: S.MaxPressureController(net, min_green=15, max_green=60))}
        runs = [(k, makers[k][0], makers[k][1], False) for k in ("fixed", "max_pressure") if k in wanted]
        if evs:
            mp = makers["max_pressure"][1]
            runs.append(("max_pressure_corridor", "Max pressure + emergency corridor",
                         lambda: S.EmergencyCorridor(mp(), net), True))
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Simulations are already running; try again in a moment")
        cancel = threading.Event()

        def job():
            try:
                t0 = time.time()
                results = [(k, label, run_one(net, flows, make, kw, evs, cancel), corridor) for k, label, make, corridor in runs]
                return results, time.time() - t0
            finally:
                slots.release()

        fut = pool.submit(job)
        loop_wait = sim_timeout_s
        try:
            import asyncio
            results, elapsed = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=loop_wait)
        except (asyncio.TimeoutError, FutureTimeout):
            cancel.set()
            raise HTTPException(504, f"The simulation took longer than {sim_timeout_s:g} s and was stopped. "
                                     "Shorten the horizon or lower the load.") from None
        out_runs = []
        for k, label, res, corridor in results:
            item = {"id": k, "label": label, "summary": _round_result(res),
                    "ev": [{kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in e.items()} for e in res["ev"]],
                    "queue_series": [[round(t, 1), round(q, 1)] for t, q in res["queue_series"]]}
            if want_frames:
                item["frames"] = res["frames"]
                item["frame_every"] = every
            out_runs.append(item)
        by = {r["id"]: r for r in out_runs}
        cmp = {}
        if "fixed" in by and "max_pressure" in by and by["fixed"]["summary"]["mean_delay_s"] > 0:
            f, m = by["fixed"]["summary"]["mean_delay_s"], by["max_pressure"]["summary"]["mean_delay_s"]
            cmp["max_pressure_vs_fixed_mean_delay_pct"] = round(100.0 * (m / f - 1.0), 1)
        if "max_pressure" in by and "max_pressure_corridor" in by:
            e0 = [e["travel_time_s"] for e in by["max_pressure"]["ev"]]
            e1 = [e["travel_time_s"] for e in by["max_pressure_corridor"]["ev"]]
            if e0 and e1 and e0[0] > 0:
                cmp["corridor_ev_time_pct"] = round(100.0 * (e1[0] / e0[0] - 1.0), 1)
            d0, d1 = by["max_pressure"]["summary"]["mean_delay_s"], by["max_pressure_corridor"]["summary"]["mean_delay_s"]
            if d0 > 0:
                cmp["corridor_others_delay_pct"] = round(100.0 * (d1 / d0 - 1.0), 1)
        return {"site": site, "source": source, "synthetic_network": bool(doc.get("synthetic")),
                "params": {"horizon_s": horizon, "load": load, "seed": seed, "drain_s": SIM_DRAIN_S,
                           "custom_flows": bool(body.get("flows")), "ev": bool(evs)},
                "demand": "synthetic" if not body.get("flows") else "as given",
                "flows": [{"origin": f.origin, "dest": f.dest, "rates": f.rates} for f in flows],
                "runs": out_runs, "comparison": cmp, "elapsed_s": round(elapsed, 2),
                "notes": NOTES + ["Both controllers saw exactly the same arrivals (same seed)."]}

    # ---- routing
    def snap(net, spec, what):
        """A node id, or {"lat", "lon"} snapped to the nearest intersection (with the distance reported)."""
        if isinstance(spec, str):
            if spec not in net.nodes:
                raise HTTPException(400, f"{what}: unknown intersection {spec}")
            return spec, 0.0
        if isinstance(spec, dict):
            try:
                la, lo = float(spec["lat"]), float(spec["lon"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, f"{what}: give an intersection id or lat and lon") from None
            geo = [n for n in net.nodes.values() if n.geolocated]
            if not geo:
                raise HTTPException(400, "The network has no positions")
            d, n = min((N.haversine_m(la, lo, n.lat, n.lon), n.id) for n in geo)
            return n, d
        raise HTTPException(400, f"{what}: give an intersection id or lat and lon")

    @router.post("/api/optimize/route")
    async def route(request: Request, site: str = "default", user=Depends(staff)):
        site, now = site_of(site), clock()
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or "origin" not in body or "dest" not in body:
            raise HTTPException(400, "send {origin, dest} (intersection ids or {lat, lon})")
        doc, source = need_network(site)
        try:
            k = int(body.get("k", 3))
            window = float(body.get("incident_window_min", 30))
        except (TypeError, ValueError):
            raise HTTPException(400, "k and incident_window_min must be numbers") from None
        if not 1 <= k <= 4 or not (INCIDENT_WINDOW_MIN[0] <= window <= INCIDENT_WINDOW_MIN[1]):
            raise HTTPException(400, "k must be 1-4 and incident_window_min 1-720")
        use_inc, want_google = body.get("use_incidents", True) is not False, bool(body.get("use_google", False))
        net, used = live_net(doc, now, want_google)
        o, od = snap(net, body["origin"], "origin")
        d, dd = snap(net, body["dest"], "dest")
        incidents = incident_delays(store, net, now, window * 60.0) if use_inc else []
        extra = {i["link"]: i["extra_delay_s"] for i in incidents}
        routes = routing.k_alternatives(net, o, d, k, extra_delay=extra)
        if not routes:
            raise HTTPException(404, "There is no route between those intersections")
        out = []
        for i, r in enumerate(routes):
            item = r.to_dict()
            for leg in item["breakdown"]:
                ua = leg.get("updated_at")
                leg["source"] = leg["source"].replace("-", "_")
                leg["age_s"] = None if ua is None else round(max(0.0, now - ua), 1)
                leg["status"] = {"camera": "live" if leg["age_s"] is not None and leg["age_s"] <= N.CAMERA_FRESH_S else "stale",
                                 "google": "google"}.get(leg["source"], "none")
                leg["data_source"] = {"camera": "camera measurement", "google": "Google traffic (text only)",
                                      "free_flow": "no measurement: free flow assumed"}.get(leg["source"], leg["source"])
            item["label"] = "Fastest" if i == 0 else f"Alternative {i}"
            item["stale_legs"] = sum(1 for x in item["breakdown"] if x["status"] == "stale")
            out.append(item)
        comparison = None
        if extra and o != d:
            rr = routing.reroute_around(net, o, d, extra)
            if rr:
                comparison = {"saving_s": None if rr["saving_s"] is None else round(rr["saving_s"], 1),
                              "usual_route_time_s": round(rr["unaware"].time_s, 1) if math.isfinite(rr["unaware"].time_s) else None,
                              "usual_route_closed": rr["unaware_closed"], "avoided": rr["avoided"]}
        return {"site": site, "source": source, "origin": {"node": o, "snapped_m": round(od, 1)},
                "dest": {"node": d, "snapped_m": round(dd, 1)}, "routes": out, "incidents": incidents,
                "incident_window_min": window, "incident_comparison": comparison, "generated_at": now,
                "contains_google": used, "google_requested": want_google,
                "notes": list(net.notes) + ["Times are camera-measured where a fresh measurement exists and free flow elsewhere. "
                                            "Incident delay comes from the scored impact of recent recorded violations."]}

    # ---- Google status
    def google_state():
        rows = store.q("SELECT value, updated_by, updated_at FROM optimize_settings WHERE key='google_enabled'")
        enabled = bool(rows and rows[0]["value"] == "1")
        key, key_source = api_map.get_key(store, "server_key", env)
        prov = None
        if key:
            try:
                p = google_fn()
                prov = p.status() if p is not None else None
            except Exception:
                prov = None
        return {"enabled": enabled, "server_key_set": key is not None, "key_source": key_source,
                "active": enabled and key is not None, "provider": prov, "rule": GOOGLE_RULE,
                "updated_by": rows[0]["updated_by"] if rows else None}

    @router.get("/api/optimize/google")
    def google_status(user=Depends(admin_only)):
        return google_state()

    @router.put("/api/optimize/google")
    async def google_set(request: Request, user=Depends(admin_only)):
        body = await read_doc(request)
        if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
            raise HTTPException(400, "send {\"enabled\": true|false}")
        store.x("INSERT INTO optimize_settings (key, value, updated_at, updated_by) VALUES ('google_enabled',?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at,"
                " updated_by=excluded.updated_by", ("1" if body["enabled"] else "0", clock(), who(user)))
        store.audit(user, "optimize.google", "settings", "optimize", "google traffic " + ("enabled" if body["enabled"] else "disabled"),
                    user.get("address", ""))
        return google_state()

    return router
