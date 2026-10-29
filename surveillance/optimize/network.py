"""The road network that routing and signal control act on (OPTIMIZE stage, work package WP4).

The surveillance side measures what happens at a handful of camera approaches: how far back the queue
reaches, how fast traffic moves, how long vehicles have been held. Acting on that - telling a controller
which approach to release, or a driver which way round - needs to know how those approaches connect. This
module is that map. It is deliberately the minimum a max-pressure controller and a shortest-path router need,
and nothing a transport planner would recognise as a calibrated traffic model.

What this package is, and is not
    A simulation-validated controller and router. Nothing here has ever controlled a real signal and nothing
    is connected to one: the closed loop the report lists as future work (traffic-state estimator -> signal
    control) is built as software and tested on a simulator. There is no SUMO bridge and no reinforcement
    learning: MPLight, CoLight, the fog-cloud MARL of Ha et al. and the federated RL of Bao et al. all need a
    GPU and a traffic simulator to train, and neither exists on this machine. Max-pressure control (Varaiya,
    "Max pressure control of a network of signalized intersections", Transportation Research Part C 36, 2013)
    is the training-free, throughput-optimal controller that MPLight's pressure reward is built on, so the
    idea being exercised is the same one; the learned policy is what is left out.

The model
    node      a junction - "signal" (controlled) or "uncontrolled" (every movement may always flow) - or a
              "boundary" point where traffic enters or leaves the modelled area. A node may carry lat/lon
              (a map placed it) and/or x/y metres (a hand-written spec); lat/lon is projected onto a local
              plane around the first geolocated node, which is all turn classification needs at city scale
    link      one directed approach between two nodes: length, lanes, free-flow speed, the rate its stop line
              discharges at while green (the saturation flow) and, optionally, the camera zone that watches it
    movement  a permitted turn from one link into the next at a node, with the share of the first link's
              traffic that takes it. Max pressure weighs the downstream queue by these turn ratios
              (Varaiya 2013), so they are part of the network, not of the controller
    phase     a set of movements that may be green together without conflicting

    Bangladesh drives on the left, so the turn that crosses opposing traffic is the RIGHT turn.

Length from coordinates
    A link given no length gets the great-circle (haversine) distance between its end nodes. That is the
    straight line; a road is longer than the straight line, so `from_points` takes a `circuity` factor and
    any road can be given an explicit length instead.

Saturation flow in Dhaka
    1800 vehicles per hour of green per lane (a 2-second headway) is the textbook planning figure for a queue
    of cars (the US Highway Capacity Manual's base value is 1900), and it does not describe a Dhaka stop
    line. A mixed queue leaves at the pace of its slowest members: a cycle rickshaw is human-powered and
    clears the junction at roughly jogging pace, the cars and CNGs behind it cannot pass in a lane they
    share, and there are no lane markings to keep the streams apart. The default here,
    DHAKA_SAT_FLOW_VPH_PER_LANE = 1200 (a 3-second headway, two thirds of the textbook figure), is a
    conservative planning ASSUMPTION, NOT a measurement and not taken from any study. It is a per-link
    parameter so that a measured value replaces it: the camera pipeline can time the gaps between vehicles
    crossing the stop line while a queue is discharging, and that headway is the saturation headway. Mixed-
    traffic saturation flow in Dhaka is the subject of a micro-simulation study calibrated on Dhaka data
    (Hossain, Transportation Research Part A 35(2), 2001), which also expresses it in vehicles per hour
    rather than passenger-car units because the PCU concept breaks down with non-motorised vehicles in the
    mix; that is the place to check the figure before it is quoted. What the choice affects: every queue-delay
    and discharge estimate scales with it. The RANKING of phases by pressure does not (a uniform factor on
    every movement cancels), so the controller's choices are far less sensitive to it than the delays are.

From camera measurements to travel times (`update_from_state`, `from_traffic_state`)
    `TrafficStateCollector.snapshot()` reports, per camera zone: queue_m, speed_kph, occupancy (0..1),
    density (vehicles per km per lane), wait_s, vehicles, lanes, flow, t. Per link this becomes

        queue_veh   queue_m / spacing per lane, times the lanes          ~7 m of road per queued vehicle
        in_queue_s  max(queue per lane / saturation flow per lane, wait_s)
                    the time to discharge the vehicles ahead, floored by how long the stopped ones have
                    already been held: queue / saturation flow assumes the queue is moving, and behind a
                    broken-down bus it is not
        running_s   (length - queue_m) / running speed
        travel time running_s + in_queue_s, and never below the free-flow time

    queue_m is read as the back-of-queue distance from the stop line - how far up the road the stopped chain
    reaches, which is what traffic_state measures and what a camera sees - so each lane holds queue_m /
    spacing vehicles and the time to discharge them does not depend on the lane count. If a source reports
    a length summed over lanes instead, pass queue_m_is_total=True and it is divided by the lanes first.

    The running speed needs care, because the camera zone's mean speed INCLUDES the stopped vehicles of its
    own queue. Charging a queue both through a low mean speed over the whole link and through its discharge
    time would count it twice, so with a queue present the unqueued stretch is taken at free-flow speed and
    the queue is charged once, through in_queue_s. With no queue the mean speed applies to the whole link
    (floored at SPEED_FLOOR_KPH, because a camera watching a standing vehicle measures zero and length / 0 is
    no travel time anyone can route on). With no speed reported - nothing was moving in view - Greenshields'
    linear speed-density relation turns the density into a speed, and with neither the link stays free-flow.

    The result never makes a link faster than its free-flow time: a measured mean above the free-flow speed
    is noise or speeding, and a route should be planned on neither.

Where a link's travel time came from
    Each link carries `state`: None (free-flow), or a dict whose "source" is "camera" or "google", whose "at"
    is the wall-clock second the underlying measurement was taken, and whose "travel_time_s" is what routing
    reads. Cameras and Google are merged by `google_traffic.apply_google`: a fresh camera measurement is
    never replaced, Google only fills links the cameras do not cover. The state is NOT saved by `save` or
    `to_dict` (default): Google-derived numbers must never be written to disk (see google_traffic), and a
    camera measurement is stale the moment it is written. The functions that apply state return a COPY, so a
    server can hold one structural network and derive a fresh view per request.

Limits
    A camera sees 50-100 m of an approach that may be 500 m long; its speed and queue are assumed to hold over
    the whole link, and a queue longer than the zone is cut off at the zone (queue_m cannot exceed what the
    camera sees), so the longest queues are under-estimated. Nothing here models lane choice, buses blocking
    a lane, or pedestrians: they appear only through the measured queue and speed and the saturation-flow
    assumption above. Routing is over nodes; turn bans and per-turn delays are not modelled.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

SIGNAL, UNCONTROLLED, BOUNDARY = "signal", "uncontrolled", "boundary"
NODE_KINDS = (SIGNAL, UNCONTROLLED, BOUNDARY)
TURNS = ("left", "through", "right", "u")
FREE_FLOW, CAMERA, GOOGLE = "free-flow", "camera", "google"
SOURCES = (FREE_FLOW, CAMERA, GOOGLE)

EARTH_RADIUS_M = 6_371_008.8
TEXTBOOK_SAT_FLOW_VPH_PER_LANE = 1800.0
DHAKA_SAT_FLOW_VPH_PER_LANE = 1200.0     # see the module docstring: an assumption, not a measurement
AVG_SPACING_M = 7.0          # road one queued vehicle occupies: a ~4.5 m car plus a ~2.5 m gap
SPEED_FLOOR_KPH = 5.0        # walking pace; a stopped queue must still give a finite travel time
DEFAULT_FREE_FLOW_KPH = 40.0
# Shares used only when nothing better is known; measured turning counts should replace them
# (Network.set_turn_ratios). Through traffic dominates on arterials; the two turns split the rest.
DEFAULT_TURN_SHARE = {"through": 0.6, "left": 0.2, "right": 0.2, "u": 0.0}
PHASINGS = ("split", "protected", "two")
COMPASS = ("N", "E", "S", "W")
CAMERA_FRESH_S = 120.0       # a camera measurement older than this no longer beats a fill-in source


# ---------------------------------------------------------------- geometry

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def project(lat: float, lon: float, origin: tuple) -> tuple:
    """(x metres east, y metres north) of a point on the local plane around `origin` (lat, lon).

    Equirectangular: exact enough across a city (the error is well under a metre per kilometre), and all
    it is used for is telling a left turn from a right one and drawing."""
    lat0, lon0 = origin
    return (math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_RADIUS_M)


def unproject(x: float, y: float, origin: tuple) -> tuple:
    """The (lat, lon) that `project` maps to (x, y)."""
    lat0, lon0 = origin
    return (lat0 + math.degrees(y / EARTH_RADIUS_M),
            lon0 + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))))


def _coord(lat, lon, what: str = "node"):
    """(lat, lon) as floats, (None, None) when neither is given; ValueError for half a pair or nonsense."""
    if lat is None and lon is None:
        return None, None
    if lat is None or lon is None:
        raise ValueError(f"{what}: give both lat and lon, or neither")
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"{what}: lat/lon ({lat}, {lon}) is not a position on Earth")
    return lat, lon


# ---------------------------------------------------------------- elements

@dataclass
class Node:
    id: str
    kind: str = SIGNAL
    lat: Optional[float] = None
    lon: Optional[float] = None
    x: Optional[float] = None       # metres east of the network origin (None: position unknown)
    y: Optional[float] = None       # metres north
    name: str = ""

    @property
    def geolocated(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def has_xy(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class Link:
    """One directed approach. `state` is what a camera or Google last said about it; None = free flow."""
    id: str
    from_node: str
    to_node: str
    length_m: float
    lanes: int = 1
    free_flow_kph: float = DEFAULT_FREE_FLOW_KPH
    sat_flow_vph_per_lane: float = DHAKA_SAT_FLOW_VPH_PER_LANE
    name: str = ""
    camera_zone: Optional[str] = None     # the traffic-state zone name that watches this approach
    camera_id: Optional[str] = None       # which camera; None: a zone of that name on any camera
    state: Optional[dict] = None

    @property
    def free_flow_s(self) -> float:
        return self.length_m / (self.free_flow_kph / 3.6)

    @property
    def sat_flow_vph(self) -> float:
        """What the whole stop line discharges per hour of green, all lanes together."""
        return self.lanes * self.sat_flow_vph_per_lane

    def storage(self, spacing_m: float = AVG_SPACING_M) -> float:
        """Vehicles the link holds when queued end to end: past this, traffic spills back upstream."""
        return max(1.0, self.lanes * self.length_m / spacing_m)

    @property
    def travel_time_s(self) -> float:
        return self.state["travel_time_s"] if self.state else self.free_flow_s

    @property
    def source(self) -> str:
        return self.state["source"] if self.state else FREE_FLOW

    @property
    def updated_at(self) -> Optional[float]:
        return self.state["at"] if self.state else None

    def fresh_camera(self, now: float, max_age_s: float = CAMERA_FRESH_S) -> bool:
        """A camera measurement recent enough that nothing else should replace it."""
        return bool(self.state and self.state["source"] == CAMERA and now - self.state["at"] <= max_age_s)


@dataclass
class Movement:
    """A permitted turn from `from_link` into `to_link` at the node between them."""
    from_link: str
    to_link: str
    turn: str = "through"
    ratio: float = 0.0          # share of from_link's traffic taking this movement
    # None: any lane of the from-link can serve it, so it may use the whole stop line's saturation flow.
    # That is the honest default for Dhaka, where lane markings do not sort traffic by direction.
    sat_flow_vph: Optional[float] = None

    @property
    def id(self) -> str:
        return movement_id(self.from_link, self.to_link)


@dataclass
class Phase:
    name: str
    movements: list = field(default_factory=list)       # movement ids


def movement_id(from_link: str, to_link: str) -> str:
    return f"{from_link}>{to_link}"


# ---------------------------------------------------------------- the network

class Network:
    """Nodes, directed links, movements and signal phases, with the lookups routing and control need."""

    def __init__(self, name: str = "", spacing_m: float = AVG_SPACING_M):
        self.name = name
        self.spacing_m = spacing_m
        self.origin: Optional[tuple] = None      # (lat, lon) the x/y plane is centred on
        self.nodes: dict[str, Node] = {}
        self.links: dict[str, Link] = {}
        self.movements: dict[str, Movement] = {}
        self.phases: dict[str, list[Phase]] = {}
        self.notes: list[str] = []               # what the last state update could not use, for the UI
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}
        self._from: dict[str, list[str]] = {}    # link id -> movement ids leaving it

    # -------------------------------------------------------- building
    def add_node(self, id: str, kind: str = SIGNAL, *, lat=None, lon=None, x=None, y=None,
                 name: str = "") -> Node:
        if kind not in NODE_KINDS:
            raise ValueError(f"node {id}: kind must be one of {', '.join(NODE_KINDS)}")
        if id in self.nodes:
            raise ValueError(f"duplicate node {id}")
        if (x is None) != (y is None):
            raise ValueError(f"node {id}: give both x and y, or neither")
        lat, lon = _coord(lat, lon, f"node {id}")
        if lat is not None:
            if self.origin is None:
                self.origin = (lat, lon)
            if x is None:
                x, y = project(lat, lon, self.origin)
        n = self.nodes[id] = Node(id, kind, lat, lon, None if x is None else float(x),
                                  None if y is None else float(y), name)
        self._out.setdefault(id, [])
        self._in.setdefault(id, [])
        return n

    def distance_m(self, a: str, b: str) -> float:
        """Straight-line distance between two nodes: haversine if both are geolocated, else from x/y."""
        na, nb = self.nodes[a], self.nodes[b]
        if na.geolocated and nb.geolocated:
            return haversine_m(na.lat, na.lon, nb.lat, nb.lon)
        if na.has_xy and nb.has_xy:
            return math.hypot(nb.x - na.x, nb.y - na.y)
        raise ValueError(f"no distance between {a} and {b}: give the link a length_m, or both nodes "
                         "a lat/lon (or x/y)")

    def add_link(self, id: str, from_node: str, to_node: str, length_m: Optional[float] = None, *,
                 lanes: int = 1, free_flow_kph: float = DEFAULT_FREE_FLOW_KPH,
                 sat_flow_vph_per_lane: float = DHAKA_SAT_FLOW_VPH_PER_LANE, name: str = "",
                 camera_zone: Optional[str] = None, camera_id=None, circuity: float = 1.0) -> Link:
        """Add a directed link. `length_m` None: haversine between the end nodes (x circuity)."""
        if id in self.links:
            raise ValueError(f"duplicate link {id}")
        for n in (from_node, to_node):
            if n not in self.nodes:
                raise ValueError(f"link {id}: unknown node {n}")
        if length_m is None:
            length_m = self.distance_m(from_node, to_node) * circuity
        link = self.links[id] = Link(id, from_node, to_node, float(length_m), int(lanes),
                                     float(free_flow_kph), float(sat_flow_vph_per_lane), name,
                                     None if camera_zone is None else str(camera_zone),
                                     None if camera_id is None else str(camera_id))
        self._out[from_node].append(id)
        self._in[to_node].append(id)
        self._from.setdefault(id, [])
        return link

    def add_movement(self, from_link: str, to_link: str, turn: Optional[str] = None, ratio: float = 0.0,
                     sat_flow_vph: Optional[float] = None) -> Movement:
        a, b = self.links.get(from_link), self.links.get(to_link)
        if a is None or b is None:
            raise ValueError(f"movement {from_link}>{to_link}: unknown link")
        if a.to_node != b.from_node:
            raise ValueError(f"movement {from_link}>{to_link}: the links do not meet at a node")
        m = Movement(from_link, to_link, turn or self.turn(from_link, to_link), float(ratio), sat_flow_vph)
        if m.id in self.movements:
            raise ValueError(f"duplicate movement {m.id}")
        self.movements[m.id] = m
        self._from[from_link].append(m.id)
        return m

    def auto_movements(self, nodes: Optional[Iterable[str]] = None, u_turns: bool = False,
                       shares: Optional[dict] = None) -> None:
        """Every turn from each approach into every exit of the same junction (no U-turns unless asked),
        with turn ratios from `shares` renormalised over the turns that exist - a T-junction has no
        through movement for one of its arms."""
        shares = shares or DEFAULT_TURN_SHARE
        for nid in (nodes if nodes is not None else self.nodes):
            if self.nodes[nid].kind == BOUNDARY:
                continue                       # traffic enters or leaves here; it does not pass through
            for a in self._in[nid]:
                for b in self._out[nid]:
                    if movement_id(a, b) in self.movements:
                        continue
                    turn = self.turn(a, b)
                    if turn == "u" and not u_turns:
                        continue
                    self.add_movement(a, b, turn)
                self._share_out(a, shares)

    def _share_out(self, link: str, shares: dict) -> None:
        mids = self._from.get(link, [])
        w = [max(0.0, shares.get(self.movements[m].turn, 0.0)) for m in mids]
        if mids and sum(w) <= 0:
            w = [1.0] * len(mids)
        for m, x in zip(mids, w):
            self.movements[m].ratio = x / sum(w)

    def set_phases(self, node: str, phases: list[Phase]) -> None:
        self.phases[node] = list(phases)

    def auto_phases(self, scheme: str = "split", nodes: Optional[Iterable[str]] = None) -> None:
        """Conflict-free phases for every signalised node.

        split      one approach at a time, all of its movements together. Conflict-free without assuming any
                   lane discipline, which is the safe assumption for mixed Dhaka traffic where right-turners,
                   rickshaws and through traffic share every lane. The default.
        protected  opposing approaches together for through + left (left does not cross traffic when
                   driving on the left), then a protected right-turn phase per axis. Four phases; assumes
                   right-turners wait in their own lane.
        two        north-south then east-west, every movement. Right turns filter through opposing traffic,
                   which this model does not represent, so it is optimistic.
        The last two need node coordinates (to tell north-south from east-west)."""
        if scheme not in PHASINGS:
            raise ValueError(f"phasing must be one of {', '.join(PHASINGS)}")
        for nid in (nodes if nodes is not None else self.nodes):
            if self.nodes[nid].kind != SIGNAL:
                continue
            arms = sorted((l for l in self._in[nid] if self._from[l]),
                          key=lambda l: (COMPASS.index(self.arm(l)) if self.arm(l) else 9, l))
            if scheme == "split":
                letters = [self.arm(l) or l for l in arms]
                phases = [Phase(a if letters.count(a) == 1 else l, list(self._from[l]))
                          for a, l in zip(letters, arms)]
            else:
                if any(self.arm(l) is None for l in arms):
                    raise ValueError(f"signal {nid}: phasing {scheme!r} needs node coordinates; use 'split'")
                axis = {"NS": [l for l in arms if self.arm(l) in "NS"],
                        "EW": [l for l in arms if self.arm(l) in "EW"]}
                phases = []
                for ax in ("NS", "EW"):
                    mids = [m for l in axis[ax] for m in self._from[l]]
                    if scheme == "two":
                        groups = [(ax, mids)]
                    else:
                        groups = [(f"{ax} through+left",
                                   [m for m in mids if self.movements[m].turn in ("through", "left")]),
                                  (f"{ax} right", [m for m in mids if self.movements[m].turn in ("right", "u")])]
                    phases += [Phase(name, ms) for name, ms in groups if ms]
            self.phases[nid] = phases

    # -------------------------------------------------------- lookups
    def out_links(self, node: str) -> list[str]:
        return self._out.get(node, [])

    def in_links(self, node: str) -> list[str]:
        return self._in.get(node, [])

    def movements_from(self, link: str) -> list[str]:
        return self._from.get(link, [])

    def node_of(self, movement: str) -> str:
        return self.links[self.movements[movement].from_link].to_node

    def phases_with(self, movement: str) -> list[int]:
        """Indices of the phases at the movement's node that serve it."""
        node = self.node_of(movement)
        return [i for i, p in enumerate(self.phases.get(node, [])) if movement in p.movements]

    def signals(self) -> list[str]:
        return sorted(n for n, v in self.nodes.items() if v.kind == SIGNAL)

    def link_between(self, a: str, b: str) -> Optional[str]:
        """Id of the (first) link from node a to node b, or None."""
        for l in self._out.get(a, []):
            if self.links[l].to_node == b:
                return l
        return None

    def links_of_path(self, nodes: list) -> list[str]:
        """The link ids along a node path; ValueError if two consecutive nodes are not linked."""
        out = []
        for a, b in zip(nodes, nodes[1:]):
            l = self.link_between(a, b)
            if l is None:
                raise ValueError(f"no link from {a} to {b}")
            out.append(l)
        return out

    def heading(self, link: str) -> Optional[float]:
        """Direction of travel in radians counter-clockwise from east; None without coordinates."""
        a, b = self.nodes[self.links[link].from_node], self.nodes[self.links[link].to_node]
        if not (a.has_xy and b.has_xy):
            return None
        return math.atan2(b.y - a.y, b.x - a.x)

    def turn(self, from_link: str, to_link: str) -> str:
        """left / through / right / u from the change of heading (counter-clockwise is a left turn).
        Without coordinates the heading is unknown and every movement is 'through'."""
        h1, h2 = self.heading(from_link), self.heading(to_link)
        if h1 is None or h2 is None:
            return "through"
        d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
        if abs(d) >= math.radians(135):
            return "u"
        if abs(d) <= math.radians(45):
            return "through"
        return "left" if d > 0 else "right"

    def arm(self, link: str) -> Optional[str]:
        """Compass side of its junction that an approach comes in from (N, E, S or W); None without coordinates."""
        a, b = self.nodes[self.links[link].from_node], self.nodes[self.links[link].to_node]
        if not (a.has_xy and b.has_xy):
            return None
        dx, dy = a.x - b.x, a.y - b.y
        if abs(dy) >= abs(dx):
            return "N" if dy > 0 else "S"
        return "E" if dx > 0 else "W"

    # -------------------------------------------------------- turn ratios
    def set_turn_ratios(self, counts: dict) -> None:
        """Replace turn ratios with observed shares: {movement id: count}. Links with no counts keep theirs.

        In deployment these are camera turning counts; in simulation, the routes vehicles actually took."""
        by_link: dict[str, float] = {}
        for mid, n in counts.items():
            if mid in self.movements:
                by_link[self.movements[mid].from_link] = by_link.get(self.movements[mid].from_link, 0.0) + n
        for link, total in by_link.items():
            if total > 0:
                for mid in self._from[link]:
                    self.movements[mid].ratio = counts.get(mid, 0.0) / total

    # -------------------------------------------------------- state
    def set_link_state(self, link: str, source: str, travel_time_s: float, at: float, **detail) -> None:
        """Record where a link's current travel time came from. Detail keys are kept for display."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}")
        self.links[link].state = {"source": source, "at": float(at), "travel_time_s": float(travel_time_s),
                                  **detail}

    def clear_state(self) -> None:
        for l in self.links.values():
            l.state = None
        self.notes = []

    def travel_times(self) -> dict:
        """{link id: current travel time in seconds}, the weights routing runs on."""
        return {lid: l.travel_time_s for lid, l in self.links.items()}

    def link_table(self, now: Optional[float] = None) -> list[dict]:
        """One row per link for a table or a map: time, delay over free flow, source, age."""
        now = time.time() if now is None else now
        rows = []
        for l in self.links.values():
            t = l.travel_time_s
            rows.append({"id": l.id, "from": l.from_node, "to": l.to_node, "name": l.name,
                         "length_m": round(l.length_m, 1), "lanes": l.lanes,
                         "free_flow_s": round(l.free_flow_s, 1), "travel_time_s": round(t, 1),
                         "delay_s": round(max(0.0, t - l.free_flow_s), 1),
                         "ratio": round(t / l.free_flow_s, 3) if l.free_flow_s > 0 else None,
                         "source": l.source,
                         "age_s": None if l.updated_at is None else round(max(0.0, now - l.updated_at), 1),
                         "camera_zone": l.camera_zone, "camera_id": l.camera_id})
        return rows

    # -------------------------------------------------------- integrity and persistence
    def validate(self) -> None:
        """Raise ValueError listing every problem, so a hand-written spec is fixed in one pass."""
        bad = []
        for l in self.links.values():
            if not (l.length_m > 0 and l.lanes >= 1 and l.free_flow_kph > 0 and l.sat_flow_vph_per_lane > 0):
                bad.append(f"link {l.id}: length, lanes, free-flow speed and saturation flow must be positive")
        for nid, phases in self.phases.items():
            if nid not in self.nodes:
                bad.append(f"phases given for unknown node {nid}")
                continue
            for p in phases:
                for m in p.movements:
                    if m not in self.movements:
                        bad.append(f"node {nid} phase {p.name}: unknown movement {m}")
                    elif self.node_of(m) != nid:
                        bad.append(f"node {nid} phase {p.name}: movement {m} belongs to another junction")
        for nid in self.signals():
            if not self.phases.get(nid):
                bad.append(f"signal {nid} has no phases")
                continue
            served = {m for p in self.phases[nid] for m in p.movements}
            for l in self._in[nid]:
                for m in self._from[l]:
                    if m not in served:
                        bad.append(f"signal {nid}: movement {m} is never green")
        seen: dict = {}
        for l in self.links.values():
            if l.camera_zone is not None:
                key = (l.camera_id, l.camera_zone)
                if key in seen:
                    bad.append(f"camera zone {l.camera_zone!r} is mapped to both {seen[key]} and {l.id} "
                               "(set camera_id to tell the cameras apart)")
                seen[key] = l.id
        if bad:
            raise ValueError("; ".join(bad))

    def to_dict(self, include_state: bool = False) -> dict:
        """JSON-ready. `include_state` adds each link's current state for a display; `save` never does."""
        nodes = []
        for n in self.nodes.values():
            d = {"id": n.id, "kind": n.kind}
            if n.geolocated:
                d["lat"], d["lon"] = round(n.lat, 7), round(n.lon, 7)
            if n.has_xy:
                d["x"], d["y"] = round(n.x, 2), round(n.y, 2)
            if n.name:
                d["name"] = n.name
            nodes.append(d)
        links = []
        for l in self.links.values():
            d = {"id": l.id, "from": l.from_node, "to": l.to_node, "length_m": round(l.length_m, 2),
                 "lanes": l.lanes, "free_flow_kph": l.free_flow_kph,
                 "sat_flow_vph_per_lane": l.sat_flow_vph_per_lane}
            for k, v in (("name", l.name), ("camera_zone", l.camera_zone), ("camera_id", l.camera_id)):
                if v:
                    d[k] = v
            if include_state:
                d["travel_time_s"] = round(l.travel_time_s, 2)
                d["source"] = l.source
                d["updated_at"] = l.updated_at
                if l.state:
                    d["state"] = dict(l.state)
            links.append(d)
        movements = []
        for m in self.movements.values():
            d = {"from": m.from_link, "to": m.to_link, "turn": m.turn, "ratio": round(m.ratio, 6)}
            if m.sat_flow_vph is not None:
                d["sat_flow_vph"] = m.sat_flow_vph
            movements.append(d)
        out = {"name": self.name, "spacing_m": self.spacing_m, "nodes": nodes, "links": links,
               "movements": movements,
               "phases": {nid: [{"name": p.name, "movements": list(p.movements)} for p in ps]
                          for nid, ps in self.phases.items()}}
        if self.origin is not None:
            out["origin"] = list(self.origin)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        """Build and validate. `movements` and `phases` may be left out of a hand-written spec: every
        non-U turn is then allowed with default ratios, and signals get split phasing. A link may leave out
        `length_m` if both of its nodes have lat/lon. A top-level `zones` {zone name: link id} attaches
        camera zones to links without editing them (cameras added later from a map)."""
        net = cls(d.get("name", ""), float(d.get("spacing_m", AVG_SPACING_M)))
        try:
            if d.get("origin"):
                net.origin = (float(d["origin"][0]), float(d["origin"][1]))
            for n in d["nodes"]:
                net.add_node(str(n["id"]), n.get("kind", SIGNAL), lat=n.get("lat"), lon=n.get("lon"),
                             x=n.get("x"), y=n.get("y"), name=n.get("name", ""))
            for l in d["links"]:
                net.add_link(str(l["id"]), str(l["from"]), str(l["to"]), l.get("length_m"),
                             lanes=int(l.get("lanes", 1)),
                             free_flow_kph=float(l.get("free_flow_kph", DEFAULT_FREE_FLOW_KPH)),
                             sat_flow_vph_per_lane=float(l.get("sat_flow_vph_per_lane",
                                                               DHAKA_SAT_FLOW_VPH_PER_LANE)),
                             name=l.get("name", ""), camera_zone=l.get("camera_zone"),
                             camera_id=l.get("camera_id"))
            if "movements" in d:
                for m in d["movements"]:
                    net.add_movement(str(m["from"]), str(m["to"]), m.get("turn"), m.get("ratio", 0.0),
                                     m.get("sat_flow_vph"))
                for link in net.links:      # a spec without ratios gets the defaults, renormalised
                    if net._from[link] and sum(net.movements[x].ratio for x in net._from[link]) <= 0:
                        net._share_out(link, DEFAULT_TURN_SHARE)
            else:
                net.auto_movements()
            if "phases" in d:
                for nid, ps in d["phases"].items():
                    net.set_phases(nid, [Phase(p.get("name", str(i)), list(p["movements"]))
                                         for i, p in enumerate(ps)])
            net.auto_phases("split", [n for n in net.signals() if n not in net.phases])
            for zone, link in (d.get("zones") or {}).items():
                if str(link) not in net.links:
                    raise ValueError(f"zone {zone!r} maps to unknown link {link}")
                net.links[str(link)].camera_zone = str(zone)
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"network spec is malformed: {e!r}") from None
        net.validate()
        return net

    def save(self, path) -> None:
        """Write the structure (never live state: see the module docstring) as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path) -> "Network":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> "Network":
        return copy.deepcopy(self)


# ---------------------------------------------------------------- builders

def grid(rows: int = 3, cols: int = 3, block_m: float = 300.0, edge_m: float = 200.0, lanes: int = 2,
         free_flow_kph: float = DEFAULT_FREE_FLOW_KPH,
         sat_flow_vph_per_lane: float = DHAKA_SAT_FLOW_VPH_PER_LANE, phasing: str = "split",
         name: str = "", lat0: Optional[float] = None, lon0: Optional[float] = None) -> Network:
    """A rows x cols grid of signalised junctions with an entry/exit point beyond every edge approach.

    Junctions are J<row><col>, row 0 at the north; boundary points are N<col>, S<col>, W<row>, E<row>.
    Links are named "<from>-<to>", so "S1-J21" is the northbound entry into the bottom-middle junction.
    With lat0/lon0 the grid is also placed on the map, junction J00 at that position, so a map or 3D view can
    draw it; the layout is then synthetic, not a real street plan."""
    net = Network(name or f"grid {rows}x{cols}")
    kw = dict(lanes=lanes, free_flow_kph=free_flow_kph, sat_flow_vph_per_lane=sat_flow_vph_per_lane)
    if lat0 is not None or lon0 is not None:
        net.origin = _coord(lat0, lon0, "grid origin")

    def node(nid, kind, x, y):
        if net.origin is not None:
            lat, lon = unproject(x, y, net.origin)
            net.add_node(nid, kind, lat=lat, lon=lon, x=x, y=y)
        else:
            net.add_node(nid, kind, x=x, y=y)

    J = lambda r, c: f"J{r}{c}"
    for r in range(rows):
        for c in range(cols):
            node(J(r, c), SIGNAL, c * block_m, -r * block_m)
    bottom, right = -(rows - 1) * block_m, (cols - 1) * block_m
    for c in range(cols):
        node(f"N{c}", BOUNDARY, c * block_m, edge_m)
        node(f"S{c}", BOUNDARY, c * block_m, bottom - edge_m)
    for r in range(rows):
        node(f"W{r}", BOUNDARY, -edge_m, -r * block_m)
        node(f"E{r}", BOUNDARY, right + edge_m, -r * block_m)

    def both(a, b, length):
        net.add_link(f"{a}-{b}", a, b, length, **kw)
        net.add_link(f"{b}-{a}", b, a, length, **kw)

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                both(J(r, c), J(r, c + 1), block_m)
            if r + 1 < rows:
                both(J(r, c), J(r + 1, c), block_m)
    for c in range(cols):
        both(f"N{c}", J(0, c), edge_m)
        both(f"S{c}", J(rows - 1, c), edge_m)
    for r in range(rows):
        both(f"W{r}", J(r, 0), edge_m)
        both(f"E{r}", J(r, cols - 1), edge_m)
    net.auto_movements()
    net.auto_phases(phasing)
    net.validate()
    return net


def from_points(intersections: Iterable, roads: Iterable, *, lanes: int = 2,
                free_flow_kph: float = DEFAULT_FREE_FLOW_KPH,
                sat_flow_vph_per_lane: float = DHAKA_SAT_FLOW_VPH_PER_LANE, phasing: str = "split",
                circuity: float = 1.0, name: str = "") -> Network:
    """A network from points on a map, so an admin map can drive it.

    intersections  [{"id", "lat", "lon", "name"?, "kind"?}] or [(id, lat, lon)]. A point with no `kind`
                   that only one road reaches is a "boundary" (traffic enters or leaves the modelled area
                   there); every other is a "signal".
    roads          [{"from", "to", "lanes"?, "free_flow_kph"?, "length_m"?, "name"?, "two_way"? (True),
                   "camera_zone"?, "camera_id"? (the from->to link), "back_camera_zone"?,
                   "back_camera_id"? (the to->from link), "id"?}] or [(from, to)].
                   Link ids are "<from>-<to>" (and "<to>-<from>" for the way back) unless `id` is given.
                   A road with no `length_m` is as long as the straight line between its ends times
                   `circuity` (1.0 = the straight line, an under-estimate of any real road).
    Movements get default turn ratios and signals the requested phasing; replace the ratios with
    `set_turn_ratios` once turning counts exist."""
    pts = [p if isinstance(p, dict) else {"id": p[0], "lat": p[1], "lon": p[2]} for p in intersections]
    rds = [r if isinstance(r, dict) else {"from": r[0], "to": r[1]} for r in roads]
    degree: dict[str, set] = {str(p["id"]): set() for p in pts}
    for r in rds:
        a, b = str(r["from"]), str(r["to"])
        for end, other in ((a, b), (b, a)):
            if end not in degree:
                raise ValueError(f"road {a}-{b}: unknown intersection {end}")
            degree[end].add(other)
    net = Network(name or "map network")
    for p in pts:
        pid = str(p["id"])
        kind = p.get("kind") or (BOUNDARY if len(degree[pid]) == 1 else SIGNAL)
        net.add_node(pid, kind, lat=p.get("lat"), lon=p.get("lon"), name=p.get("name", ""))
    for r in rds:
        a, b = str(r["from"]), str(r["to"])
        kw = dict(lanes=int(r.get("lanes", lanes)), free_flow_kph=float(r.get("free_flow_kph", free_flow_kph)),
                  sat_flow_vph_per_lane=float(r.get("sat_flow_vph_per_lane", sat_flow_vph_per_lane)),
                  name=r.get("name", ""), circuity=circuity)
        length = r.get("length_m")
        net.add_link(str(r.get("id") or f"{a}-{b}"), a, b, length, camera_zone=r.get("camera_zone"),
                     camera_id=r.get("camera_id"), **kw)
        if r.get("two_way", True):
            net.add_link(f"{b}-{a}", b, a, length, camera_zone=r.get("back_camera_zone"),
                         camera_id=r.get("back_camera_id"), **kw)
    net.auto_movements()
    net.auto_phases(phasing)
    net.validate()
    return net


# ---------------------------------------------------------------- measured state -> travel times

def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


_METRICS = ("queue_m", "speed_kph", "occupancy", "density", "wait_s")


def usable_state(state) -> bool:
    """Does a traffic-state dict carry a measurement? One with `samples` of 0 (nothing was observed in the
    window) or with no finite quantity does not: it says nothing, which is not the same as 'free flow'."""
    if not isinstance(state, dict):
        return False
    if "samples" in state and not (_num(state.get("samples")) or 0) > 0:
        return False
    return any(_num(state.get(k)) is not None for k in _METRICS)


def link_measurement(link: Link, state: dict, spacing_m: float = AVG_SPACING_M,
                     speed_floor_kph: float = SPEED_FLOOR_KPH, queue_m_is_total: bool = False) -> dict:
    """What one approach's measured state means for travel through it. See the module docstring."""
    queue_m = max(0.0, _num(state.get("queue_m")) or 0.0)
    if queue_m_is_total:
        queue_m /= link.lanes
    queue_m = min(queue_m, link.length_m)            # a queue longer than the link is spillback upstream
    vf = link.free_flow_kph
    speed, speed_source = _num(state.get("speed_kph")), "measured"
    density = _num(state.get("density"))             # vehicles per km per lane
    if speed is None or speed < 0:
        if density is not None and density >= 0:
            # Greenshields: speed falls linearly from free flow to zero at jam density, one vehicle per
            # `spacing_m` of lane.
            speed, speed_source = vf * max(0.0, 1.0 - density / (1000.0 / spacing_m)), "from density"
        else:
            speed, speed_source = vf, "free flow (no speed)"
    queued = queue_m > 0
    # With a queue the zone's mean speed is mostly the queue's own standstill, already charged below.
    running_kph = vf if queued else min(max(speed, speed_floor_kph), vf)
    per_lane = queue_m / spacing_m
    discharge_s = per_lane / (link.sat_flow_vph_per_lane / 3600.0)
    waiting = _num(state.get("wait_s"))
    in_queue_s = max(discharge_s, waiting or 0.0)
    free_s = link.free_flow_s
    run_s = (link.length_m - queue_m) / (running_kph / 3.6)
    total = max(free_s, run_s + in_queue_s)
    slow_s = max(0.0, min(total, run_s + queue_m / (vf / 3.6)) - free_s)
    return {"travel_time_s": total, "free_flow_s": free_s, "slow_s": slow_s, "queue_s": total - free_s - slow_s,
            "queue_veh": per_lane * link.lanes, "queue_m": queue_m, "speed_kph": running_kph,
            "speed_source": speed_source, "occupancy": _num(state.get("occupancy")), "density": density,
            "wait_s": waiting, "camera_lanes": _num(state.get("lanes"))}


def update_from_state(network: Network, state_by_link: dict, *, now: Optional[float] = None,
                      spacing_m: Optional[float] = None, speed_floor_kph: float = SPEED_FLOOR_KPH,
                      queue_m_is_total: bool = False, inplace: bool = False) -> Network:
    """The network with travel times from measured state: {link id: traffic-state dict}.

    Links with no (usable) entry keep what they had. A state may carry `wall` - the epoch second its sample
    was taken (TrafficStateCollector.wall_at) - which becomes the link's "at"; otherwise `now` does, so a
    caller replaying old video should pass it. Returns a copy unless `inplace`, so a server can hold one
    structural network and derive a fresh view per request."""
    net = network if inplace else network.copy()
    now = time.time() if now is None else float(now)
    spacing = spacing_m or net.spacing_m
    for lid, state in state_by_link.items():
        link = net.links.get(lid)
        if link is None:
            net.notes.append(f"state for unknown link {lid} ignored")
        elif not usable_state(state):
            net.notes.append(f"link {lid}: the camera reported nothing usable")
        else:
            m = link_measurement(link, state, spacing, speed_floor_kph, queue_m_is_total)
            net.set_link_state(lid, CAMERA, m.pop("travel_time_s"), _num(state.get("wall")) or now, **m)
    return net


def merge_states(states: list) -> dict:
    """Several cameras on one approach: the worst queue and wait, the mean speed / occupancy / density.

    The worst queue because a queue seen by any camera is real; the mean speed because each camera sees a
    different stretch and the link's travel time is over all of them."""
    states = [s for s in states if usable_state(s)]

    def vals(k):
        return [v for v in (_num(s.get(k)) for s in states) if v is not None]
    out = {}
    for k in ("queue_m", "wait_s"):
        if vals(k):
            out[k] = max(vals(k))
    for k in ("speed_kph", "occupancy", "density"):
        if vals(k):
            out[k] = sum(vals(k)) / len(vals(k))
    if vals("wall"):
        out["wall"] = min(vals("wall"))      # the oldest sample bounds how fresh the merge is
    if vals("lanes"):
        out["lanes"] = max(vals("lanes"))
    return out


def _zone_states(snapshots) -> list:
    """[(camera id or None, zone name, state)] from {camera: {zone: state}} or from a flat {zone: state}."""
    out = []
    for k, v in (snapshots or {}).items():
        nested = (isinstance(v, dict) and v and not (set(v) & set(_METRICS))
                  and all(isinstance(x, dict) for x in v.values()))
        if nested:
            out += [(k, zone, st) for zone, st in v.items()]
        else:
            out.append((None, k, v))
    return out


def from_traffic_state(network: Network, snapshots_by_camera: dict, *, now: Optional[float] = None,
                       **kw) -> Network:
    """The network updated from the traffic-state module, through each link's `camera_zone`.

    `snapshots_by_camera` is {camera id: TrafficStateCollector.snapshot()} - or one snapshot, {zone: state} -
    and each link takes the zones named by its `camera_zone` (and `camera_id`, when set). Zones that map to
    no link, and mapped links the cameras said nothing usable about, are listed in the result's `notes`
    rather than silently dropped, so the UI can say which cameras are not yet placed on the map. Other
    keywords go to `update_from_state`."""
    found = _zone_states(snapshots_by_camera)
    used: set = set()
    by_link: dict = {}
    notes = []
    for link in network.links.values():
        if link.camera_zone is None:
            continue
        hit = [(i, st) for i, (cam, zone, st) in enumerate(found)
               if zone == link.camera_zone
               and (link.camera_id is None or cam is None or str(cam) == link.camera_id)]
        used.update(i for i, _ in hit)
        merged = merge_states([st for _, st in hit])
        if merged:
            by_link[link.id] = merged
        else:
            notes.append(f"link {link.id}: no usable data from camera zone {link.camera_zone!r}")
    notes += [f"zone {zone!r} is not mapped to a link" for i, (cam, zone, _) in enumerate(found)
              if i not in used]
    net = update_from_state(network, by_link, now=now, **kw)
    net.notes = notes + net.notes
    return net
