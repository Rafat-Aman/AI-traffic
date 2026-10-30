"""Point-to-point (section) average speed: how fast the SAME vehicle covered the road between two cameras.

A feature module (WP9, requirement R-2). It reads the tracklets the cross-camera matcher has already linked
(`tracklets.global_id`, surveillance/reid.py + link.py) and the camera positions (surveillance/geo.py); it
writes nothing.

Per vehicle segment
    Sightings of one global id are ordered by wall time. Between two CONSECUTIVE sightings at different placed
    cameras A -> B:
        distance   road distance when `road_fn(a, b)` knows a route (the optimiser network built from the map,
                   geo.build_network + routing.shortest_path), otherwise the straight line between the two
                   cameras times CIRCUITY. Which one was used is reported as `distance_basis` ("road" or
                   "straight_line"), never hidden.
        time       t_B - t_A in seconds, each tracklet's `t_cross` (the moment it crossed its zone's reference
                   line, wall clock) plus that camera's `clock_offset_s`.
        speed      distance / time.
    Consecutive sightings at the SAME camera collapse to the last one (the one nearest the departure), so a
    vehicle that was tracked twice at A is timed from its last sighting there.

Per camera pair (directed A -> B), over a time window
    n (segments), n_vehicles (distinct global ids), median, mean, p15, p85 (linear-interpolated percentiles),
    and `space_mean_kph` = distance / mean(time), the harmonic-type mean that is the right average for speed
    over a fixed distance (the plain mean of speeds overstates it when speeds vary). A pair with fewer than
    MIN_VEHICLES distinct vehicles gets NO speed numbers - only the count and the reason - because a median of
    two cars says nothing about the road.

Plausibility bounds (a segment outside them is rejected AND counted, per reason; it is never clipped)
    speed < MIN_SPEED_KPH = 2   below walking pace for a vehicle that was moving at both cameras: it stopped
                                in between (parked, broke down) or is a different vehicle - not a section speed.
    speed > MAX_SPEED_KPH = 120 no Dhaka road permits it; far above the ~80 km/h of the fastest expressway, so
                                the margin absorbs timing noise and what remains is a wrong match or a clock error.
    travel time == 0            `clock_inconsistent`: both sightings at the same instant. Sightings are ordered
                                by time, so a clock error larger than the travel time does not show up as a
                                negative time but as a vehicle driving B -> A; it betrays itself as a pair whose
                                rejected counts (too_fast/too_slow) are high in BOTH directions.
    travel time > MAX_TRAVEL_S = 1800 s   `gap_too_long`: half an hour between two cameras is a second trip,
                                not the transit of one section.
    distance < MIN_DISTANCE_M = 50   `cameras_too_close`: a one-second clock error would already be a large
                                fraction of the travel time; no honest speed comes from it.

Clock assumptions (read this before trusting a number)
    Both times come from the tracklets' wall clocks. For a LIVE camera that is the server's clock - the same
    clock for every live camera, so they agree. For an UPLOADED video it is the operator-entered start time
    plus the offset into the video: if two videos were not entered against one clock, every speed between
    them is wrong by distance/(true time +/- the error), and nothing here can detect it except the
    implausibility counters (a lot of `clock_inconsistent`/`too_fast` on one pair is the signature). Each
    camera therefore has `clock_offset_s` in camera_geo, added to its times; an admin who knows camera B's
    clock runs 40 s slow sets +40 there. No synchronisation is estimated automatically.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

from . import geo as geo_mod

CIRCUITY = 1.4              # road length / straight line, when no road route is known. An ASSUMPTION for dense
                            # urban grids (planning literature uses ~1.2-1.5), not a Dhaka measurement.
MIN_VEHICLES = 5
MIN_SPEED_KPH = 2.0
MAX_SPEED_KPH = 120.0
MAX_TRAVEL_S = 1800.0
MIN_DISTANCE_M = 50.0
REASONS = ("clock_inconsistent", "gap_too_long", "cameras_too_close", "too_slow", "too_fast")

BOUNDS = {"min_vehicles": MIN_VEHICLES, "min_speed_kph": MIN_SPEED_KPH, "max_speed_kph": MAX_SPEED_KPH,
          "max_travel_s": MAX_TRAVEL_S, "min_distance_m": MIN_DISTANCE_M, "circuity": CIRCUITY}
CLOCK_NOTE = ("Times are each tracklet's wall-clock crossing time plus the camera's clock_offset_s. Uploaded "
              "videos carry an operator-entered start time; if two videos were not entered against the same "
              "clock the speed between them is wrong. Clock agreement is assumed, not checked.")


def percentile(sorted_vals: list, p: float) -> float:
    """Linear-interpolated percentile (0..100) of an ascending list."""
    if not sorted_vals:
        raise ValueError("no values")
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def network_road_fn(network) -> Callable:
    """road_fn(camera a, camera b) -> metres along the network's fastest route, or None if there is none.

    Nodes are named by geo.node_id. The route is the one routing.shortest_path picks on the current travel
    times, which is the road a driver would take, so its length is the distance to divide by."""
    from .optimize.routing import shortest_path

    def road(a, b):
        na, nb = geo_mod.node_id(a), geo_mod.node_id(b)
        if na not in network.nodes or nb not in network.nodes:
            return None
        try:
            r = shortest_path(network, na, nb)
        except ValueError:
            return None
        return r.length_m if r is not None and r.length_m > 0 else None
    return road


def pair_distance(a: int, b: int, geos: dict, road_fn: Optional[Callable] = None,
                  circuity: float = CIRCUITY) -> tuple:
    """(metres, basis) between two placed cameras. basis: "road" or "straight_line"."""
    if road_fn is not None:
        d = road_fn(a, b)
        if d is not None and math.isfinite(d) and d > 0:
            return float(d), "road"
    return geo_mod.distance_m(geos[a], geos[b]) * circuity, "straight_line"


def _adjusted_sightings(rows: Iterable, geos: dict) -> list:
    """[(time, camera id, row)] ascending, using only placed cameras and rows with a time; consecutive
    sightings at one camera collapse to the last."""
    s = []
    for r in rows:
        cam, t = r.get("camera_id"), r.get("t_cross")
        if t is None or cam not in geos:
            continue
        s.append((float(t) + float(geos[cam].get("clock_offset_s") or 0.0), cam, r))
    s.sort(key=lambda x: x[0])
    out = []
    for item in s:
        if out and out[-1][1] == item[1]:
            out[-1] = item
        else:
            out.append(item)
    return out


def judge(dt: float, dist: float) -> tuple:
    """(speed kph or None, reason or None) for one segment."""
    if dt <= 0:
        return None, "clock_inconsistent"
    if dt > MAX_TRAVEL_S:
        return None, "gap_too_long"
    if dist < MIN_DISTANCE_M:
        return None, "cameras_too_close"
    kph = dist / dt * 3.6
    if kph < MIN_SPEED_KPH:
        return None, "too_slow"
    if kph > MAX_SPEED_KPH:
        return None, "too_fast"
    return kph, None


def segments_of_vehicle(rows: Iterable, geos: dict, road_fn: Optional[Callable] = None,
                        circuity: float = CIRCUITY) -> list:
    """Every consecutive camera-to-camera segment of one vehicle (rows = its tracklets), plausible or not."""
    sights = _adjusted_sightings(rows, geos)
    out = []
    for (t0, a, r0), (t1, b, r1) in zip(sights, sights[1:]):
        dist, basis = pair_distance(a, b, geos, road_fn, circuity)
        dt = t1 - t0
        kph, reason = judge(dt, dist)
        raw = round(dist / dt * 3.6, 1) if dt > 0 else None
        out.append({"global_id": r0.get("global_id"), "from": a, "to": b, "t_from": round(t0, 2),
                    "t_to": round(t1, 2), "travel_s": round(dt, 2), "distance_m": round(dist, 1),
                    "distance_basis": basis, "speed_kph": None if kph is None else round(kph, 1),
                    "raw_kph": raw, "rejected": reason, "cls": r0.get("cls") or r1.get("cls")})
    return out


def compute(rows: Iterable, geos: dict, *, road_fn: Optional[Callable] = None, circuity: float = CIRCUITY,
            since: Optional[float] = None, until: Optional[float] = None, cam_from: Optional[int] = None,
            cam_to: Optional[int] = None, min_vehicles: int = MIN_VEHICLES, with_speeds: bool = True) -> dict:
    """Per-pair section speeds from tracklet rows (dicts, any order, any mix of vehicles).

    A segment belongs to the window when its ARRIVAL time (at B) is within [since, until]. `rows` should reach
    back at least MAX_TRAVEL_S before `since` so a vehicle that left A just before the window is not lost."""
    by_vehicle: dict = {}
    for r in rows:
        if r.get("global_id") is not None:
            by_vehicle.setdefault(r["global_id"], []).append(r)
    pairs: dict = {}
    rejected_total = {k: 0 for k in REASONS}
    for gid, vrows in by_vehicle.items():
        for seg in segments_of_vehicle(vrows, geos, road_fn, circuity):
            if (since is not None and seg["t_to"] < since) or (until is not None and seg["t_to"] > until):
                continue
            if (cam_from is not None and seg["from"] != cam_from) or (cam_to is not None and seg["to"] != cam_to):
                continue
            p = pairs.setdefault((seg["from"], seg["to"]), {
                "from": seg["from"], "to": seg["to"], "distance_m": seg["distance_m"],
                "distance_basis": seg["distance_basis"], "_ok": [], "_gids": set(), "_times": [],
                "rejected": {k: 0 for k in REASONS}})
            if seg["rejected"]:
                p["rejected"][seg["rejected"]] += 1
                rejected_total[seg["rejected"]] += 1
            else:
                p["_ok"].append(seg["speed_kph"])
                p["_times"].append(seg["travel_s"])
                p["_gids"].add(gid)
    out = []
    for key in sorted(pairs):
        p = pairs[key]
        speeds = sorted(p.pop("_ok"))
        times = p.pop("_times")
        gids = p.pop("_gids")
        p["n"] = len(speeds)
        p["n_vehicles"] = len(gids)
        p["n_rejected"] = sum(p["rejected"].values())
        p["min_vehicles"] = min_vehicles
        if p["n_vehicles"] >= min_vehicles:
            p["enough"] = True
            p["speed"] = {
                "median_kph": round(percentile(speeds, 50), 1), "mean_kph": round(sum(speeds) / len(speeds), 1),
                "p15_kph": round(percentile(speeds, 15), 1), "p85_kph": round(percentile(speeds, 85), 1),
                "space_mean_kph": round(p["distance_m"] / (sum(times) / len(times)) * 3.6, 1)}
            p["reason"] = None
            if with_speeds:
                p["speeds_kph"] = [round(s, 1) for s in speeds[:1000]]
        else:
            p["enough"] = False
            p["speed"] = None
            p["reason"] = (f"{p['n_vehicles']} vehicle{'s' if p['n_vehicles'] != 1 else ''} matched; "
                           f"at least {min_vehicles} are needed before a speed is shown")
        out.append(p)
    return {"pairs": out, "rejected": rejected_total, "bounds": {**BOUNDS, "min_vehicles": min_vehicles},
            "clock_note": CLOCK_NOTE, "window": {"since": since, "until": until}}


def fetch_rows(store, since: Optional[float] = None, until: Optional[float] = None) -> list:
    """Linked tracklets that can matter for a window: those up to MAX_TRAVEL_S before `since` through `until`."""
    lo = None if since is None else since - MAX_TRAVEL_S - 86400.0   # slack for a camera's clock_offset_s
    hi = None if until is None else until + 86400.0
    return [r for r in store.tracklets(since=lo, until=hi) if r.get("global_id") is not None]
