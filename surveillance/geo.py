"""Camera geolocation: where each camera stands, which way it points, and what it can see.

A feature module (WP9, requirement R-2). It owns one table, `camera_geo`, and never alters `cameras`: a
camera with no row has simply not been placed on the map yet, and deleting a camera deletes its row (foreign
key). Call `ensure_schema(store)` once at start-up; it is idempotent.

Conventions
    lat, lon       WGS84 degrees. Validation rejects what is not a place on Earth; a position outside greater
                   Dhaka is accepted with a WARNING (a swapped lat/lon pair is the usual cause), because the
                   system is not meant to be locked to one city.
    heading_deg    the direction the camera POINTS, compass degrees, 0 = north, 90 = east, clockwise. Stored
                   normalised to [0, 360). It is the direction of view, not the direction traffic flows.
    fov_deg        horizontal field of view, 5..180. range_m  how far along the heading the camera usefully
                   sees, 5..1000 m. Both only shape the drawn wedge: nothing measured depends on them.
    clock_offset_s seconds to ADD to this camera's wall times before comparing with other cameras (default 0).
                   See section_speed.py: the speed between two cameras is only as good as their clocks.

`network_inputs` / `build_network` turn the placed cameras into the input of
`surveillance.optimize.network.from_points`, so the optimiser can be built on the map the admin drew.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Optional

from .optimize.network import from_points, haversine_m, project, unproject

SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_geo (
    camera_id INTEGER PRIMARY KEY REFERENCES cameras(id) ON DELETE CASCADE,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    heading_deg REAL NOT NULL DEFAULT 0,
    fov_deg REAL NOT NULL DEFAULT 60,
    range_m REAL NOT NULL DEFAULT 80,
    clock_offset_s REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    updated_by TEXT DEFAULT ''
);
"""

# Greater Dhaka, generously: a position outside is probably a typo or a swapped pair, not a crime.
DHAKA_BOX = {"lat": (23.60, 23.95), "lon": (90.25, 90.55)}
DEFAULTS = {"heading_deg": 0.0, "fov_deg": 60.0, "range_m": 80.0, "clock_offset_s": 0.0}
FOV_RANGE = (5.0, 180.0)
RANGE_M_RANGE = (5.0, 1000.0)
CLOCK_OFFSET_MAX_S = 86400.0


class GeoError(ValueError):
    """A placement that cannot be stored. The message is safe to show to the person who sent it."""


def ensure_schema(store) -> None:
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


# ---------------------------------------------------------------- validation

def _number(value, what: str) -> float:
    if isinstance(value, bool) or value is None:
        raise GeoError(f"{what} must be a number")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise GeoError(f"{what} must be a number") from None
    if not math.isfinite(x):
        raise GeoError(f"{what} must be a finite number")
    return x


def normalise_heading(deg) -> float:
    """Compass degrees in [0, 360), to 0.1 degree."""
    h = round(_number(deg, "heading_deg") % 360.0, 1)
    return 0.0 if h >= 360.0 else h


def validate(data: dict, base: Optional[dict] = None) -> tuple:
    """(clean placement, warnings). `data` may be partial when `base` (the stored row) is given.

    Raises GeoError for anything that is not storable; out-of-Dhaka is a warning, not an error."""
    if not isinstance(data, dict):
        raise GeoError("send a JSON object")
    merged = {**DEFAULTS, **{k: v for k, v in (base or {}).items() if k in
                             ("lat", "lon", *DEFAULTS)}, **{k: v for k, v in data.items() if v is not None
                                                           or k in ("lat", "lon")}}
    if merged.get("lat") is None or merged.get("lon") is None:
        raise GeoError("lat and lon are required")
    lat, lon = _number(merged["lat"], "lat"), _number(merged["lon"], "lon")
    if not -90.0 <= lat <= 90.0:
        swapped = DHAKA_BOX["lat"][0] <= lon <= DHAKA_BOX["lat"][1] and DHAKA_BOX["lon"][0] <= lat <= DHAKA_BOX["lon"][1]
        raise GeoError("lat must be between -90 and 90" + (" (lat and lon look swapped)" if swapped else ""))
    if not -180.0 <= lon <= 180.0:
        raise GeoError("lon must be between -180 and 180")
    fov = _number(merged["fov_deg"], "fov_deg")
    if not FOV_RANGE[0] <= fov <= FOV_RANGE[1]:
        raise GeoError(f"fov_deg must be between {FOV_RANGE[0]:g} and {FOV_RANGE[1]:g}")
    rng = _number(merged["range_m"], "range_m")
    if not RANGE_M_RANGE[0] <= rng <= RANGE_M_RANGE[1]:
        raise GeoError(f"range_m must be between {RANGE_M_RANGE[0]:g} and {RANGE_M_RANGE[1]:g}")
    off = _number(merged["clock_offset_s"], "clock_offset_s")
    if abs(off) > CLOCK_OFFSET_MAX_S:
        raise GeoError(f"clock_offset_s must be within +/-{CLOCK_OFFSET_MAX_S:g} s")
    clean = {"lat": round(lat, 7), "lon": round(lon, 7), "heading_deg": normalise_heading(merged["heading_deg"]),
             "fov_deg": round(fov, 1), "range_m": round(rng, 1), "clock_offset_s": round(off, 3)}
    warnings = []
    (la0, la1), (lo0, lo1) = DHAKA_BOX["lat"], DHAKA_BOX["lon"]
    if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
        warnings.append(f"({lat:.5f}, {lon:.5f}) is outside greater Dhaka; saved as given")
    return clean, warnings


# ---------------------------------------------------------------- storage

def get(store, camera_id: int) -> Optional[dict]:
    rows = store.q("SELECT * FROM camera_geo WHERE camera_id=?", (int(camera_id),))
    return rows[0] if rows else None


def all_geo(store) -> dict:
    """{camera id: row} for every placed camera."""
    return {r["camera_id"]: r for r in store.q("SELECT * FROM camera_geo ORDER BY camera_id")}


def save(store, camera_id: int, data: dict, who: str = "") -> tuple:
    """Validate and upsert one placement. Returns (row, warnings). Partial updates keep the stored values."""
    clean, warnings = validate(data, get(store, camera_id))
    store.x("INSERT INTO camera_geo (camera_id, lat, lon, heading_deg, fov_deg, range_m, clock_offset_s,"
            " updated_at, updated_by) VALUES (?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(camera_id) DO UPDATE SET lat=excluded.lat, lon=excluded.lon,"
            " heading_deg=excluded.heading_deg, fov_deg=excluded.fov_deg, range_m=excluded.range_m,"
            " clock_offset_s=excluded.clock_offset_s, updated_at=excluded.updated_at,"
            " updated_by=excluded.updated_by",
            (int(camera_id), clean["lat"], clean["lon"], clean["heading_deg"], clean["fov_deg"],
             clean["range_m"], clean["clock_offset_s"], time.time(), who))
    return get(store, camera_id), warnings


def remove(store, camera_id: int) -> bool:
    had = get(store, camera_id) is not None
    store.x("DELETE FROM camera_geo WHERE camera_id=?", (int(camera_id),))
    return had


# ---------------------------------------------------------------- geometry

def destination(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple:
    """The (lat, lon) `dist_m` metres from a point along a compass bearing (local plane, fine at camera scale)."""
    b = math.radians(bearing_deg)
    return unproject(dist_m * math.sin(b), dist_m * math.cos(b), (lat, lon))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from point 1 to point 2, [0, 360)."""
    x, y = project(lat2, lon2, (lat1, lon1))
    return math.degrees(math.atan2(x, y)) % 360.0


def fov_wedge(lat: float, lon: float, heading_deg: float, fov_deg: float, range_m: float,
              arc_points: int = 12) -> list:
    """The camera's field of view as a polygon [[lat, lon], ...]: the apex, then an arc across the view."""
    n = max(2, int(arc_points))
    pts = [[lat, lon]]
    for i in range(n + 1):
        a = heading_deg - fov_deg / 2.0 + fov_deg * i / n
        la, lo = destination(lat, lon, a, range_m)
        pts.append([round(la, 7), round(lo, 7)])
    return pts


def view(row: dict) -> dict:
    """A stored row as the API shows it: the numbers plus the wedge to draw."""
    return {"lat": row["lat"], "lon": row["lon"], "heading_deg": row["heading_deg"], "fov_deg": row["fov_deg"],
            "range_m": row["range_m"], "clock_offset_s": row["clock_offset_s"],
            "updated_at": row["updated_at"], "updated_by": row["updated_by"],
            "wedge": fov_wedge(row["lat"], row["lon"], row["heading_deg"], row["fov_deg"], row["range_m"])}


def distance_m(a: dict, b: dict) -> float:
    return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])


# ---------------------------------------------------------------- travel-time window for a camera link (WP19)

WINDOW_FAST_KPH = 60.0      # quickest plausible free-flow speed on an urban road: sets the earliest arrival
WINDOW_SLOW_KPH = 5.0       # a crawl in a Dhaka jam: sets the latest arrival
WINDOW_CIRCUITY = 1.4       # road length / straight line; the same ASSUMPTION section_speed.py makes
WINDOW_SLACK_S = 10.0       # clock and exit-point jitter, added to the late end
WINDOW_MAX_S = 1800.0       # the same ceiling section_speed.py puts on a believable camera-to-camera trip


def link_window(a: dict, b: dict, fast_kph: float = WINDOW_FAST_KPH, slow_kph: float = WINDOW_SLOW_KPH,
                circuity: float = WINDOW_CIRCUITY) -> dict:
    """Suggested [min_s, max_s] for a vehicle leaving camera `a` and appearing at camera `b`, from the map alone.

    Why these numbers. The time that matters is from the vehicle LEAVING a's view to ENTERING b's view, so
    the distance is the camera-to-camera distance less what each camera already covers (each one's view
    range, an upper bound on how much of the road it spans). The earliest arrival is that shortened straight
    line at 60 km/h (a road is never shorter than a straight line); the latest is the full straight line times
    1.4 (the assumed road factor) at 5 km/h, plus 10 s of slack. 5-60 km/h is deliberately wide: a link
    window is a FILTER, and a window that is too narrow silently drops the real vehicle while one that is too
    wide only costs a few more candidates for the appearance score to rule out. The admin can edit it, and
    should once the real travel times are known. Capped at 30 minutes."""
    d = distance_m(a, b)
    covered = float(a.get("range_m") or 0) + float(b.get("range_m") or 0)
    near = max(0.0, d - covered)
    lo = math.floor(near / (fast_kph / 3.6))
    hi = math.ceil((d * circuity / (slow_kph / 3.6) + WINDOW_SLACK_S) / 5.0) * 5.0
    capped = hi > WINDOW_MAX_S
    hi = min(hi, WINDOW_MAX_S)
    return {"min_s": float(min(lo, hi - 1)), "max_s": float(hi), "distance_m": round(d, 1), "capped": capped,
            "basis": (f"{round(d)} m apart; earliest {fast_kph:g} km/h over the {round(near)} m not already in view, "
                      f"latest {slow_kph:g} km/h over {circuity:g} x the straight line + {WINDOW_SLACK_S:g} s")}


# ---------------------------------------------------------------- the optimiser's input

def node_id(camera_id) -> str:
    """The network node id a placed camera gets. section_speed looks cameras up by this name."""
    return f"cam{int(camera_id)}"


def network_inputs(geos: dict, links: Iterable = (), names: Optional[dict] = None) -> tuple:
    """(intersections, roads) in the shape `surveillance.optimize.network.from_points` takes.

    geos   {camera id: row} (all_geo). Every placed camera becomes a point "cam<id>".
    links  [(camera id a, camera id b)] or [(a, b, length_m)] - the road segments between cameras. A road with
           no length is as long as the straight line between its cameras (from_points applies its circuity).
           Pairs naming an unplaced camera are skipped and reported in the third element.
    The admin map says WHERE cameras are; which cameras are joined by a road is a separate fact the caller
    supplies (a camera is not an intersection, and two cameras on one avenue are one link, not a junction)."""
    points = [{"id": node_id(cid), "lat": g["lat"], "lon": g["lon"], "name": (names or {}).get(cid, "")}
              for cid, g in sorted(geos.items())]
    roads, skipped = [], []
    for link in links:
        a, b = int(link[0]), int(link[1])
        if a not in geos or b not in geos or a == b:
            skipped.append((a, b))
            continue
        road = {"from": node_id(a), "to": node_id(b)}
        if len(link) > 2 and link[2] is not None:
            road["length_m"] = float(link[2])
        roads.append(road)
    return points, roads, skipped


def build_network(geos: dict, links: Iterable = (), *, circuity: float = 1.0, **kw):
    """A Network from the placed cameras and the given road links (see network_inputs)."""
    points, roads, _ = network_inputs(geos, links)
    linked = {r["from"] for r in roads} | {r["to"] for r in roads}
    points = [p for p in points if p["id"] in linked]       # a camera on no road is not a node of the network
    return from_points(points, roads, circuity=circuity, **kw)
