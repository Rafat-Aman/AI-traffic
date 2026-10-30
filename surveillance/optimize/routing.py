"""Routing on the live travel times: shortest path, alternatives, and routes that go round an incident.

The network (network.py) holds one travel time per link, derived from the cameras where there is one, from Google
where there is not, and from free flow otherwise. Routing is shortest path on those times. Nothing cleverer is
needed to be useful: the times already carry the queues, and what a driver or a dispatcher wants to know is
"which way is fastest now, and what does each link of it cost".

What is here
    shortest_path     Dijkstra with a binary heap (heapq) on non-negative link times. Deterministic: equal-cost
                      ties are broken by node id, so the same network always gives the same route.
    k_alternatives    up to k different routes by the penalty method: find the shortest route, multiply the cost
                      of its links by `penalty`, search again, and keep each new route that is not too much
                      slower and not mostly the same road. The times reported are the real ones, without the
                      penalties; the penalties only steer the search.
    reroute_around    incident-aware routing: {link id: extra delay in seconds} is added to those links' costs
                      (math.inf closes a link). Returns the route that goes round the trouble AND the route a
                      driver unaware of it would have taken, timed with the incident, so the saving is visible.
    tree_to           the shortest-path tree INTO one destination (reverse Dijkstra): the cost-to-go and the
                      next link from every node, which is what the simulator needs to route each departing
                      vehicle without a search per vehicle.
    extra_delay_from_impact
                      turns the impact module's excess delay on an approach (vehicle-seconds) into the seconds
                      of delay the next vehicle there should expect.
    A Route carries a per-link breakdown: length, free-flow and current time, the extra delay, and WHERE that
    time came from ("camera", "google", "free-flow") and how old it is, so a dashboard can show how much of a
    route is measured and how much is assumed.

Choices worth knowing about
    Boundary nodes are terminals. A route may start or end at one but never passes through one: leaving the
    modelled area and coming back in is not a shortcut. Routing is over nodes: turn bans and the delay of a
    particular turn are not modelled (with the default "every non-U turn is allowed" movement set nothing is
    lost, because a U-turn can never be on a shortest simple path).

Limits - what a shortest path on current times does not know
    * Times are "now" and are held fixed over the whole trip. On a long route the far links will have changed
      by the time the vehicle gets there.
    * It is all-or-nothing: every vehicle asked is given the same best route. If a thousand drivers are told to
      take the alternative, the alternative becomes the jam. The simulator (sim.py) routes each departure on
      the queues at that moment, which damps this a little, but a proper answer is a traffic assignment
      (user equilibrium), which this is not.
    * The delay at a signal is in the link time only as far as the camera sees a queue; the time spent waiting
      for the red that has not yet formed a queue is not.
    * Incident delays are estimates from a window of measurements and stop being true when the incident clears;
      the caller must drop the entry then.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from .network import BOUNDARY, Network


# ---------------------------------------------------------------- routes

@dataclass
class Route:
    nodes: list
    links: list
    time_s: float               # travel time including any extra delay: what the driver would experience
    free_flow_s: float
    length_m: float
    extra_delay_s: float = 0.0
    breakdown: list = field(default_factory=list)

    @property
    def delay_s(self) -> float:
        """Time beyond free flow (queues, slow traffic, and any incident delay)."""
        return self.time_s - self.free_flow_s

    def source_mix(self) -> dict:
        """{source: number of links} - how much of this route is measured and how much is assumed."""
        mix: dict = {}
        for b in self.breakdown:
            mix[b["source"]] = mix.get(b["source"], 0) + 1
        return mix

    def to_dict(self) -> dict:
        return {"nodes": list(self.nodes), "links": list(self.links), "time_s": round(self.time_s, 1),
                "free_flow_s": round(self.free_flow_s, 1), "delay_s": round(self.delay_s, 1),
                "length_m": round(self.length_m, 1), "extra_delay_s": round(self.extra_delay_s, 1),
                "source_mix": self.source_mix(), "breakdown": self.breakdown}


def link_costs(network: Network, extra_delay: Optional[dict] = None, weights: Optional[dict] = None) -> dict:
    """{link id: cost in seconds}: the link's current travel time (or `weights[id]` where given) plus its
    extra delay. math.inf closes a link."""
    extra_delay = extra_delay or {}
    weights = weights or {}
    out = {}
    for lid, l in network.links.items():
        out[lid] = float(weights.get(lid, l.travel_time_s)) + float(extra_delay.get(lid, 0.0))
    return out


def route_from_links(network: Network, links: list, extra_delay: Optional[dict] = None,
                     weights: Optional[dict] = None) -> Route:
    """A Route (with its breakdown) for a given sequence of consecutive links."""
    extra_delay = extra_delay or {}
    weights = weights or {}
    nodes, rows, t, free, length, extra = [], [], 0.0, 0.0, 0.0, 0.0
    for i, lid in enumerate(links):
        l = network.links[lid]
        if i == 0:
            nodes.append(l.from_node)
        elif network.links[links[i - 1]].to_node != l.from_node:
            raise ValueError(f"links {links[i - 1]} and {lid} are not consecutive")
        nodes.append(l.to_node)
        base = float(weights.get(lid, l.travel_time_s))
        add = float(extra_delay.get(lid, 0.0))
        t += base + add
        free += l.free_flow_s
        length += l.length_m
        extra += add
        rows.append({"link": lid, "from": l.from_node, "to": l.to_node, "name": l.name,
                     "length_m": round(l.length_m, 1), "free_flow_s": round(l.free_flow_s, 1),
                     "travel_time_s": round(base, 1), "extra_delay_s": round(add, 1),
                     "source": "override" if lid in weights else l.source,
                     "updated_at": l.updated_at, "end_s": round(t, 1)})
    return Route(nodes, list(links), t, free, length, extra, rows)


# ---------------------------------------------------------------- shortest paths

def _search(network: Network, origin: str, dest: str, cost: dict, avoid) -> Optional[list]:
    """Dijkstra from origin to dest on `cost`; the link ids of the path, or None when dest cannot be reached."""
    dist = {origin: 0.0}
    via: dict = {}
    done: set = set()
    heap = [(0.0, origin)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == dest:
            break
        if u != origin and network.nodes[u].kind == BOUNDARY:
            continue                                     # an exit of the area, not a shortcut through it
        for lid in network.out_links(u):
            c = cost[lid]
            if lid in avoid or not math.isfinite(c):
                continue
            v = network.links[lid].to_node
            nd = d + c
            if nd < dist.get(v, math.inf) - 1e-12:
                dist[v] = nd
                via[v] = lid
                heapq.heappush(heap, (nd, v))
    if dest not in done:
        return None
    path, node = [], dest
    while node != origin:
        lid = via[node]
        path.append(lid)
        node = network.links[lid].from_node
    path.reverse()
    return path


def _check_nodes(network: Network, *nodes) -> None:
    for n in nodes:
        if n not in network.nodes:
            raise ValueError(f"unknown node {n}")


def shortest_path(network: Network, origin: str, dest: str, *, extra_delay: Optional[dict] = None,
                  weights: Optional[dict] = None, avoid=()) -> Optional[Route]:
    """The fastest route from origin to dest on the network's current travel times, or None if there is none.

    extra_delay  {link id: seconds} added to those links (an incident); math.inf closes a link
    weights      {link id: seconds} used INSTEAD of the network's times for those links
    avoid        link ids that may not be used"""
    _check_nodes(network, origin, dest)
    if origin == dest:
        return Route([origin], [], 0.0, 0.0, 0.0)
    path = _search(network, origin, dest, link_costs(network, extra_delay, weights), set(avoid))
    return None if path is None else route_from_links(network, path, extra_delay, weights)


def tree_to(network: Network, dest: str, *, extra_delay: Optional[dict] = None,
            weights: Optional[dict] = None) -> tuple:
    """The shortest-path tree into `dest`: (cost_to, next_link), each keyed by node.

    cost_to[n] is the time from n to dest; next_link[n] is the first link to take from n. A node missing from
    the dicts cannot reach dest. Reverse Dijkstra over the in-links; boundary nodes other than dest appear (as
    possible origins) but are never expanded through, matching `shortest_path`."""
    _check_nodes(network, dest)
    cost = link_costs(network, extra_delay, weights)
    dist = {dest: 0.0}
    nxt: dict = {}
    done: set = set()
    heap = [(0.0, dest)]
    while heap:
        d, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        if v != dest and network.nodes[v].kind == BOUNDARY:
            continue
        for lid in network.in_links(v):
            c = cost[lid]
            if not math.isfinite(c):
                continue
            u = network.links[lid].from_node
            nd = d + c
            if nd < dist.get(u, math.inf) - 1e-12:
                dist[u] = nd
                nxt[u] = lid
                heapq.heappush(heap, (nd, u))
    return dist, nxt


def follow_tree(network: Network, next_link: dict, origin: str, dest: str) -> Optional[list]:
    """The link ids from origin to dest by following a `tree_to` result; None if dest is unreachable."""
    path, node = [], origin
    while node != dest:
        lid = next_link.get(node)
        if lid is None or len(path) > len(network.links):      # unreachable, or a malformed tree
            return None
        path.append(lid)
        node = network.links[lid].to_node
    return path


# ---------------------------------------------------------------- alternatives

def _overlap(route: Route, others: list, network: Network) -> float:
    """Largest share of `route`'s length that it has in common with any one of `others`."""
    total = sum(network.links[l].length_m for l in route.links) or 1.0
    best = 0.0
    for o in others:
        shared = set(o.links)
        best = max(best, sum(network.links[l].length_m for l in route.links if l in shared) / total)
    return best


def k_alternatives(network: Network, origin: str, dest: str, k: int = 3, *, penalty: float = 1.5,
                   max_stretch: float = 1.6, max_overlap: float = 0.85, extra_delay: Optional[dict] = None,
                   weights: Optional[dict] = None, avoid=(), max_tries: Optional[int] = None) -> list:
    """Up to k different routes, fastest first, by the penalty method.

    The first is the true shortest path. Each further route is the shortest path after the links of the routes
    already found were made `penalty` times dearer, and is kept only if its real time is within `max_stretch`
    times the fastest and it shares at most `max_overlap` of its length with any one route already kept. May
    return fewer than k: a corridor with no genuine alternative has none to offer. Times are real, unpenalised."""
    _check_nodes(network, origin, dest)
    if origin == dest:
        return [Route([origin], [], 0.0, 0.0, 0.0)]
    base = link_costs(network, extra_delay, weights)
    work = dict(base)
    avoid = set(avoid)
    found: list = []
    seen: set = set()
    for _ in range(max_tries or 6 * max(1, k)):
        path = _search(network, origin, dest, work, avoid)
        if path is None:
            break
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            route = route_from_links(network, path, extra_delay, weights)
            if not found:
                found.append(route)
            elif (route.time_s <= found[0].time_s * max_stretch
                  and _overlap(route, found, network) <= max_overlap):
                found.append(route)
        if len(found) >= k:
            break
        for lid in path:
            work[lid] = work[lid] * penalty
    found.sort(key=lambda r: r.time_s)
    return found


# ---------------------------------------------------------------- incidents

def reroute_around(network: Network, origin: str, dest: str, incidents: dict, *,
                   avoid=()) -> Optional[dict]:
    """Incident-aware routing: the route that goes round `incidents` {link id: extra delay s}, and the saving.

    Returns {"route": the best route given the incidents, "unaware": the route found WITHOUT knowing of them,
    timed with the incidents, "saving_s": unaware - route, "avoided": links of `unaware` that `route` does
    not use, "unaware_closed": the usual route runs over a closed link}, or None if dest cannot be reached.
    With no saving the two routes are the same. `saving_s` is None when the usual route is closed outright (it
    has no finite time to compare). The entry for a link is the delay a vehicle should expect there (see
    `extra_delay_from_impact`); math.inf closes it."""
    unknown = [l for l in incidents if l not in network.links]
    if unknown:
        raise ValueError(f"incident on unknown link(s) {', '.join(unknown)}")
    route = shortest_path(network, origin, dest, extra_delay=incidents, avoid=avoid)
    if route is None:
        return None
    plain = shortest_path(network, origin, dest, avoid=avoid)
    closed = any(not math.isfinite(incidents.get(l, 0.0)) for l in plain.links)
    unaware = plain if closed else route_from_links(network, plain.links, incidents)
    return {"route": route, "unaware": unaware, "unaware_closed": closed,
            "saving_s": None if closed else unaware.time_s - route.time_s,
            "avoided": [l for l in plain.links if l not in route.links]}


def extra_delay_from_impact(impact: dict, cap_s: float = 900.0) -> float:
    """Seconds of delay the next vehicle on an approach should expect, from the impact module's excess delay.

    Accepts both shapes in the code base - TrafficStateCollector.excess_delay ({"veh_s", "vehicles",
    "affected"}) and impact.compute_event_impact ({"excess_delay_veh_s", "vehicles_seen",
    "vehicles_affected"}). The total excess delay (vehicle-seconds) is divided by the vehicles that crossed
    the approach in the window: the mean delay per vehicle, which is what one more vehicle should expect while
    the incident lasts. If the window saw nobody, the vehicles actually held up are used. Capped at `cap_s`:
    a window of a few vehicles must not close a road by arithmetic."""
    total = impact.get("veh_s", impact.get("excess_delay_veh_s"))
    seen = impact.get("vehicles", impact.get("vehicles_seen"))
    held = impact.get("affected", impact.get("vehicles_affected"))
    try:
        total = float(total)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(total) or total <= 0:
        return 0.0
    n = seen if seen else held
    if not n:
        return min(total, cap_s)
    return min(total / float(n), cap_s)
