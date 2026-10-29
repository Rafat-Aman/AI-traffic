"""Google Routes API congestion as a numeric fill-in where no camera looks (requirement R-8).

Cameras cover a handful of approaches; the planner needs a travel time for every link. Where a link has no
camera, Google's live traffic is the only measurement available at all, so this module asks the Routes API how
long it takes to drive between a link's two end points with traffic and without, and the ratio of the two - the
congestion ratio - scales the link's own free-flow time (`apply_google`). Cameras win wherever they are fresh;
Google only fills the gaps.

Google's terms - read before enabling
    The Google Maps Platform terms restrict how Google content may be used: it may only be displayed on a
    Google map, and it may not be stored or cached beyond what the terms and the Routes API's service-specific
    terms allow. This module is written to stay inside both:
      * Google data is used here ONLY as a numeric input to the planner, one congestion ratio per link. It is
        never drawn, so it never appears on a non-Google map. The free MapLibre 3D view shows the system's
        own camera-measured densities; Google's traffic belongs on Google's own TrafficLayer in the browser.
      * Nothing Google returns is ever written to disk. The cache is a dict in process memory, entries expire
        after `ttl_s` (ten minutes by default) and exist only so the same element is not paid for twice.
        `Network.save` drops all live link state, so a saved network holds no Google-derived number.
      * The key is a SERVER key, read from the environment (GOOGLE_MAPS_SERVER_KEY) or passed in; it is sent in
        a header, never put in a URL, never logged, never returned by `status()`. It must not be in the
        repository. The browser map needs a different key, restricted by referrer.
    This is not legal advice: the current Google Maps Platform Terms of Service and the Routes API terms are
    the authority, and they change.

What it asks, and why a ratio
    computeRoutes between a link's two end points with routingPreference TRAFFIC_AWARE and the
    TRAFFIC_ON_POLYLINE extra computation returns `duration` (with live traffic), `staticDuration` (without),
    the route's length, and speedReadingIntervals that label stretches of the route NORMAL, SLOW or
    TRAFFIC_JAM. computeRouteMatrix returns the durations only, for many pairs in one request.
    Google's route between two points need not be the link we are modelling (it may take a parallel road), and
    its static duration rests on its own speed assumptions, not ours. So the planner does not adopt Google's
    absolute duration: it takes the RATIO of traffic to static duration and applies it to the link's own
    free-flow time, and it ignores a route whose length is nothing like the link's (`apply_google`).
    `jam_fraction` is the share of the route's polyline covered by TRAFFIC_JAM intervals, counted in polyline
    points, which are denser in bends than on straights: a rough share, not a share of the distance.

Cost control
    Traffic-aware requests are billed per element (one origin-destination pair) at a higher rate than a plain
    route, and the rate is Google's to state; no price is assumed here. So: results are cached for `ttl_s`;
    `apply_google` asks only about links that have no fresh camera data; a rolling one-minute budget caps both
    requests and elements (past it the caller gets what the cache holds and a log line, never an exception);
    and `method="matrix"` groups the links that leave one junction into a single 1-by-k request, which costs
    k elements, the same as k separate routes but one round trip - at the price of the jam fraction, which the
    matrix does not carry.

Failure is quiet by design
    No key, an HTTP error, a timeout, an error object inside a 200 response, an exhausted budget: every one of
    them returns what is available (usually {}) and logs the reason, because the optimiser must keep working
    on camera data and free flow when Google is not there. The first failure of a pass stops that pass, so a
    revoked key is not retried once per link.

Not verified live
    The request and response shapes follow Google's published Routes API reference, and are exercised in
    tests/test_optimize.py ONLY against canned responses through an injected transport. This code has never
    been called against the live service. The first live call is where a mistake in a field name, a quota, or
    billing not being enabled would show up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from typing import Callable, Optional

from .network import CAMERA_FRESH_S, GOOGLE, Link, Network

log = logging.getLogger("surveillance.optimize.google")

ENV_KEY = "GOOGLE_MAPS_SERVER_KEY"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MATRIX_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
ROUTES_FIELD_MASK = ("routes.duration,routes.staticDuration,routes.distanceMeters,"
                     "routes.travelAdvisory.speedReadingIntervals")
MATRIX_FIELD_MASK = "originIndex,destinationIndex,duration,staticDuration,distanceMeters,condition"

_DURATION = re.compile(r"^\s*(-?\d+(?:\.\d+)?)s\s*$")


# ---------------------------------------------------------------- request and response shapes

def parse_duration(value) -> Optional[float]:
    """Seconds from Google's duration string ("123s", "0.5s"); a plain number passes; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value == value and abs(value) != float("inf") else None
    m = _DURATION.match(value) if isinstance(value, str) else None
    return float(m.group(1)) if m else None


def _waypoint(lat_lon) -> dict:
    return {"location": {"latLng": {"latitude": float(lat_lon[0]), "longitude": float(lat_lon[1])}}}


def routes_body(origin, destination) -> dict:
    """The computeRoutes request body between two (lat, lon) points. No departureTime: Google means 'now'."""
    return {"origin": _waypoint(origin), "destination": _waypoint(destination), "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE", "extraComputations": ["TRAFFIC_ON_POLYLINE"]}


def matrix_body(origins, destinations) -> dict:
    return {"origins": [{"waypoint": _waypoint(o)} for o in origins],
            "destinations": [{"waypoint": _waypoint(d)} for d in destinations],
            "travelMode": "DRIVE", "routingPreference": "TRAFFIC_AWARE"}


def speed_shares(intervals) -> tuple:
    """(jam_fraction, slow_fraction) of the polyline from speedReadingIntervals, or (None, None).

    Spans are in polyline point indices. Proto3 JSON leaves out zero-valued fields, so a missing start index
    is 0; a missing speed is 'unspecified' and counts toward the total but toward neither share."""
    spans = []
    for it in intervals or []:
        try:
            a, b = int(it.get("startPolylinePointIndex", 0)), int(it.get("endPolylinePointIndex", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        if b > a:
            spans.append((b - a, it.get("speed")))
    total = sum(n for n, _ in spans)
    if total <= 0:
        return None, None
    return (sum(n for n, s in spans if s == "TRAFFIC_JAM") / total,
            sum(n for n, s in spans if s == "SLOW") / total)


def _summarise(traffic_s, free_s, distance_m, jam, slow, at) -> Optional[dict]:
    if traffic_s is None or free_s is None or free_s <= 0 or traffic_s < 0:
        return None
    return {"traffic_s": traffic_s, "free_s": free_s, "ratio": round(traffic_s / free_s, 4),
            "jam_fraction": None if jam is None else round(jam, 4),
            "slow_fraction": None if slow is None else round(slow, 4),
            "distance_m": distance_m, "fetched_at": at}


def parse_route(response, at: float) -> Optional[dict]:
    """The summary of the first route in a computeRoutes response, or None when there is no usable route."""
    routes = response.get("routes") if isinstance(response, dict) else None
    if not routes or not isinstance(routes[0], dict):
        return None
    r = routes[0]
    jam, slow = speed_shares((r.get("travelAdvisory") or {}).get("speedReadingIntervals"))
    dist = r.get("distanceMeters")
    return _summarise(parse_duration(r.get("duration")), parse_duration(r.get("staticDuration")),
                      float(dist) if isinstance(dist, (int, float)) else None, jam, slow, at)


def parse_matrix(response, at: float) -> list:
    """[{"origin", "destination", ...summary}] for the elements of a computeRouteMatrix response that have a
    route. The response is a JSON array of elements (a single element object is accepted too); indices that
    are 0 are omitted from the JSON and so default to 0."""
    elements = response if isinstance(response, list) else [response]
    out = []
    for el in elements:
        if not isinstance(el, dict) or (el.get("status") or {}).get("code"):
            continue
        if el.get("condition", "ROUTE_EXISTS") != "ROUTE_EXISTS":
            continue
        dist = el.get("distanceMeters")
        s = _summarise(parse_duration(el.get("duration")), parse_duration(el.get("staticDuration")),
                       float(dist) if isinstance(dist, (int, float)) else None, None, None, at)
        if s:
            out.append({"origin": int(el.get("originIndex", 0)), "destination": int(el.get("destinationIndex", 0)),
                        **s})
    return out


# ---------------------------------------------------------------- transport

def urllib_transport(timeout_s: float = 10.0) -> Callable:
    """The real transport: POST JSON, return the decoded JSON. Raises on any HTTP or network failure.

    Tests never use it - they inject a function that returns canned responses."""
    def post(url: str, headers: dict, body: dict):
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={**headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        try:
            return json.loads(raw)
        except ValueError:                      # newline-delimited elements
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return post


class _Budget(Exception):
    """The per-minute budget is spent."""


class _Failed(Exception):
    """A request failed; the message says why (and never contains the key)."""


# ---------------------------------------------------------------- the provider

class GoogleTraffic:
    """Server-side Routes API client with an in-memory TTL cache and a per-minute budget.

    transport  (url, headers, body) -> decoded JSON; injectable so tests use canned responses and nothing
               here touches the network. Default: `urllib_transport`.
    ttl_s      how long a result may be reused; also an upper bound on how stale a figure the planner sees.
    max_calls_per_min / max_elements_per_min
               a rolling 60-second budget on requests and on billable elements (origin-destination pairs).
    clock      time.time unless a test supplies one."""

    def __init__(self, api_key: Optional[str] = None, transport: Optional[Callable] = None,
                 ttl_s: float = 600.0, max_calls_per_min: int = 30, max_elements_per_min: int = 120,
                 clock: Callable = time.time, timeout_s: float = 10.0, max_cache: int = 2000):
        self.api_key = (api_key or os.environ.get(ENV_KEY) or "").strip() or None
        self._transport = transport or urllib_transport(timeout_s)
        self.ttl_s = float(ttl_s)
        self.max_calls_per_min = int(max_calls_per_min)
        self.max_elements_per_min = int(max_elements_per_min)
        self.clock = clock
        self.max_cache = int(max_cache)
        self._cache: OrderedDict = OrderedDict()
        self._spent: deque = deque()              # (time, elements) of recent requests
        self._lock = threading.Lock()
        self._warned: set = set()
        self.last_error: Optional[str] = None
        self.stats = {"calls": 0, "elements": 0, "cache_hits": 0, "errors": 0, "budget_skips": 0}

    def __repr__(self) -> str:                    # never show the key
        return f"GoogleTraffic(configured={self.configured}, ttl_s={self.ttl_s})"

    @property
    def configured(self) -> bool:
        return self.api_key is not None

    def status(self) -> dict:
        """For an admin page: whether a key is set, the limits, what has been used. Never the key."""
        with self._lock:
            now = self.clock()
            recent = [(t, n) for t, n in self._spent if now - t < 60.0]
            return {"configured": self.configured, "ttl_s": self.ttl_s,
                    "max_calls_per_min": self.max_calls_per_min,
                    "max_elements_per_min": self.max_elements_per_min,
                    "calls_last_minute": len(recent), "elements_last_minute": sum(n for _, n in recent),
                    "cached": len(self._cache), "last_error": self.last_error, **self.stats}

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # -------------------------------------------------------- plumbing
    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("google traffic: %s", message)

    def _scrub(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    def _describe(self, exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            detail = ""
            try:
                body = json.loads(exc.read().decode("utf-8", "replace"))
                detail = (body[0] if isinstance(body, list) and body else body).get("error", {}).get("message", "")
            except Exception:
                pass
            return f"HTTP {exc.code} {exc.reason}" + (f": {detail}" if detail else "")
        return f"{type(exc).__name__}: {exc}"

    def _cached(self, key):
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and self.clock() - hit[0] <= self.ttl_s:
                self.stats["cache_hits"] += 1
                return hit[1]
            if hit is not None:
                del self._cache[key]
        return None

    def _store(self, key, value) -> None:
        with self._lock:
            self._cache[key] = (self.clock(), value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)

    def _spend(self, elements: int) -> None:
        with self._lock:
            now = self.clock()
            while self._spent and now - self._spent[0][0] >= 60.0:
                self._spent.popleft()
            if (len(self._spent) + 1 > self.max_calls_per_min
                    or sum(n for _, n in self._spent) + elements > self.max_elements_per_min):
                self.stats["budget_skips"] += 1
                raise _Budget()
            self._spent.append((now, elements))
            self.stats["calls"] += 1
            self.stats["elements"] += elements

    def _post(self, url: str, field_mask: str, body: dict, elements: int):
        """One billed request. Raises _Budget when the minute's budget is spent, _Failed on any failure."""
        self._spend(elements)
        headers = {"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": field_mask}
        try:
            resp = self._transport(url, headers, body)
        except Exception as e:                      # HTTP error, timeout, DNS, bad JSON: all the same to us
            raise _Failed(self._scrub(self._describe(e))) from None
        err = resp.get("error") if isinstance(resp, dict) else None
        if err:
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise _Failed(self._scrub(f"API error: {msg}"))
        return resp

    def _fail(self, reason: str) -> None:
        with self._lock:
            self.stats["errors"] += 1
            self.last_error = reason
        log.warning("google traffic: request failed (%s); falling back to cameras and free flow", reason)

    @staticmethod
    def _rounded(p) -> tuple:
        return (round(float(p[0]), 5), round(float(p[1]), 5))      # about a metre: one cache entry per place

    # -------------------------------------------------------- the two endpoints
    def _route(self, origin, destination) -> Optional[dict]:
        key = ("route", self._rounded(origin), self._rounded(destination))
        hit = self._cached(key)
        if hit is not None:
            return hit or None                  # {} is a remembered 'no route', not a miss
        resp = self._post(ROUTES_URL, ROUTES_FIELD_MASK, routes_body(origin, destination), 1)
        out = parse_route(resp, self.clock())
        if out is None:
            log.info("google traffic: no usable route between %s and %s", origin, destination)
        self._store(key, out or {})             # a missing route is cached too: do not pay to ask again
        return out

    def route(self, origin, destination) -> Optional[dict]:
        """Traffic and static duration, ratio and jam share for one drive between two (lat, lon) points,
        or None (no key, no route, budget spent, or a failed request - the reason is logged)."""
        if not self.configured:
            self._warn_once("nokey", f"no key configured (set {ENV_KEY}); Google traffic is off")
            return None
        try:
            return self._route(origin, destination)
        except _Budget:
            self._warn_once("budget", "per-minute budget spent; serving the cache only")
            return None
        except _Failed as e:
            self._fail(str(e))
            return None

    def route_matrix(self, origins, destinations) -> list:
        """computeRouteMatrix: [{"origin", "destination", "traffic_s", "free_s", "ratio", "distance_m",
        "fetched_at"}] for each origin-destination pair that has a route; [] on any failure. Billed per
        element, origins x destinations of them."""
        if not self.configured:
            self._warn_once("nokey", f"no key configured (set {ENV_KEY}); Google traffic is off")
            return []
        key = ("matrix", tuple(self._rounded(o) for o in origins), tuple(self._rounded(d) for d in destinations))
        hit = self._cached(key)
        if hit is not None:
            return hit
        try:
            resp = self._post(MATRIX_URL, MATRIX_FIELD_MASK, matrix_body(origins, destinations),
                              len(origins) * len(destinations))
        except _Budget:
            self._warn_once("budget", "per-minute budget spent; serving the cache only")
            return []
        except _Failed as e:
            self._fail(str(e))
            return []
        out = parse_matrix(resp, self.clock())
        self._store(key, out)
        return out

    # -------------------------------------------------------- links
    def link_travel_times(self, network: Network, links=None, method: str = "routes") -> dict:
        """{link id: {"traffic_s", "free_s", "ratio", "jam_fraction", "fetched_at", ...}} for the links
        (ids or Link objects; default all) whose two end nodes both have a lat/lon.

        ratio = traffic duration / static duration, Google's own two figures. `method` "routes" asks per
        link and returns the jam share too; "matrix" asks once per origin junction and has no jam share.
        Returns {} without a key, and what it has so far if a request fails or the budget runs out."""
        if method not in ("routes", "matrix"):
            raise ValueError("method must be 'routes' or 'matrix'")
        if not self.configured:
            self._warn_once("nokey", f"no key configured (set {ENV_KEY}); Google traffic is off")
            return {}
        todo = []
        for l in (network.links.values() if links is None else links):
            link = l if isinstance(l, Link) else network.links.get(l)
            if link is None:
                continue
            a, b = network.nodes[link.from_node], network.nodes[link.to_node]
            if a.geolocated and b.geolocated:
                todo.append((link.id, (a.lat, a.lon), (b.lat, b.lon), link.from_node))
        out: dict = {}
        try:
            if method == "routes":
                for lid, o, d, _ in todo:
                    r = self._route(o, d)
                    if r is not None:
                        out[lid] = r
            else:
                groups: dict = {}
                for item in todo:
                    groups.setdefault(item[3], []).append(item)
                for items in groups.values():
                    origin = items[0][1]
                    key = ("matrix", (self._rounded(origin),), tuple(self._rounded(i[2]) for i in items))
                    els = self._cached(key)
                    if els is None:
                        resp = self._post(MATRIX_URL, MATRIX_FIELD_MASK, matrix_body([origin], [i[2] for i in items]),
                                          len(items))
                        els = parse_matrix(resp, self.clock())
                        self._store(key, els)
                    for el in els:
                        if 0 <= el["destination"] < len(items):
                            res = {k: v for k, v in el.items() if k not in ("origin", "destination")}
                            res.update(jam_fraction=None, slow_fraction=None)    # the matrix has no polyline
                            out[items[el["destination"]][0]] = res
        except _Budget:
            self._warn_once("budget", "per-minute budget spent; serving the cache only")
        except _Failed as e:
            self._fail(str(e))
        return out


# ---------------------------------------------------------------- merging with the cameras

def apply_google(network: Network, provider: Optional[GoogleTraffic], *, now: Optional[float] = None,
                 camera_fresh_s: float = CAMERA_FRESH_S, max_length_mismatch: float = 0.5,
                 max_ratio: float = 10.0, method: str = "routes", inplace: bool = False) -> Network:
    """Fill the travel times of links the cameras do not cover, from Google, and say so.

    A link with a camera measurement younger than `camera_fresh_s` is NEVER touched, and Google is not even
    asked about it (that would be billing for an answer that is thrown away). Every other link whose end
    nodes have coordinates gets travel time = its own free-flow time x Google's congestion ratio, marked
    source "google" with the time Google was asked. A ratio below 1 is read as 1 (never faster than free
    flow) and one above `max_ratio` is cut there. A route whose length differs from the link's by more than
    `max_length_mismatch` (50 % by default) is another road, and is skipped with a note. Stale camera data is
    replaced when Google has an answer and kept (with its age showing) when it does not.

    Returns a copy unless `inplace`. With no provider, no key, or any failure the network comes back as it
    was: cameras and free flow, which is exactly what the planner would have used without Google."""
    net = network if inplace else network.copy()
    if provider is None:
        return net
    now = provider.clock() if now is None else float(now)
    need = [l.id for l in net.links.values() if not l.fresh_camera(now, camera_fresh_s)]
    if not need:
        return net
    filled = 0
    for lid, r in provider.link_travel_times(net, need, method=method).items():
        link = net.links[lid]
        dist = r.get("distance_m")
        if dist and abs(dist - link.length_m) / link.length_m > max_length_mismatch:
            net.notes.append(f"link {lid}: Google's route is {dist:.0f} m against the link's "
                             f"{link.length_m:.0f} m, so it is another road; not used")
            continue
        ratio = min(max(1.0, r["ratio"]), max_ratio)
        net.set_link_state(lid, GOOGLE, link.free_flow_s * ratio, r["fetched_at"], ratio=r["ratio"],
                           jam_fraction=r.get("jam_fraction"), slow_fraction=r.get("slow_fraction"),
                           google_traffic_s=r["traffic_s"], google_free_s=r["free_s"], distance_m=dist)
        filled += 1
    if filled:
        net.notes.append(f"Google traffic filled {filled} link(s) with no fresh camera data")
    return net
