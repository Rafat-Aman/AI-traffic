"""A deterministic mesoscopic queue simulator: the bench the controllers are validated on.

SUMO is not available here and a controller that has only ever met its own assumptions proves little, so this is
the smallest simulator that still has the things signal control is about: demand that varies in time and space,
stop-line queues that discharge only while green, finite storage that spills back, and routing that reacts to
queues. It is NOT a calibrated model of Dhaka; results from it say how the controllers behave under THESE rules.

The rules (one-second steps)
    Vehicles   arrive as a Poisson process per origin-destination flow, rate piecewise-constant in time, drawn
               up front from a seeded generator (one independent stream per flow), so a run is reproducible and
               two controllers face exactly the same arrivals. Each takes the shortest path at departure, on
               free-flow time plus the queue delay of each link at that moment (tables refreshed every
               `route_refresh_s`); the route is then fixed.
    Link       free-flow travel to the stop line (the link's free-flow time, no car-following), then a FIFO
               queue per movement. A vehicle on the last link of its route leaves at the end of it.
    Discharge  a link's stop line serves its green movements at the link's saturation flow with fractional
               accumulation (a credit of saturation_flow/3600 per second; one vehicle leaves per whole credit),
               the earliest-queued head vehicle first. The first `startup_lost_s` of every green discharge
               nothing (start-up loss), and the controllers' clearance intervals discharge nothing: that is
               what makes switching cost something. A movement at an uncontrolled node is always green.
    Spillback  a vehicle enters the next link only if it has room (lanes x length / spacing vehicles on it);
               otherwise it stays in its queue and blocks the movement. A vehicle blocked for
               `gridlock_release_s` is let through anyway and counted in `forced_releases`, which breaks the
               deadlock a closed loop of full links would otherwise be and is reported so it is not hidden.
               Vehicles that cannot enter the network wait outside it, FIFO, and their wait counts as delay.
    Emergency vehicles
               an EV is an ordinary vehicle in the queues (it cannot overtake, the conservative choice): a
               controller helps it only by having the queue ahead discharged and the green waiting.

Measures
    delay      trip time minus the free-flow time of the quickest route between its origin and destination, so a
               detour counts as delay. A vehicle still in the network when the run ends is counted with the delay
               it has so far (a lower bound) and reported as `unfinished`: leaving the stuck ones out would flatter
               exactly the controller that strands them.
    Also throughput (vehicles finished inside the demand horizon, per hour), queues over time, the emergency
    vehicle's travel time, and `frames`: per step, each link's [moving, queued] count and each signal's phase
    (-1 = clearance), for a 3D view to animate.
    Conservation: generated = finished + in the network (queued, travelling, or waiting to enter), at every step.

Limits
    No car-following, lane changing, buses blocking, pedestrians, or mixed speeds; saturation flow is one number
    per link (default 1200 vehicles/h/lane for Dhaka, an assumption - see network.py); turn conflicts are not
    simulated (the phases are taken to be conflict-free). A rule that wins here may not win on a street.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import routing
from .network import SIGNAL, Network


@dataclass
class Flow:
    """Demand from origin to dest. rates: [(start_s, vehicles_per_hour), ...], each holding until the next."""
    origin: str
    dest: str
    rates: list

    @classmethod
    def constant(cls, origin: str, dest: str, vph: float) -> "Flow":
        return cls(origin, dest, [(0.0, float(vph))])

    def rate_at(self, t: float) -> float:
        r = 0.0
        for start, vph in sorted(self.rates):
            if start <= t:
                r = vph
        return r


@dataclass
class EV:
    depart_s: float
    origin: str
    dest: str
    route: Optional[list] = None            # link ids; default: the free-flow shortest path


def generate_arrivals(flows: list, horizon_s: float, seed: int = 0) -> list:
    """[(time, flow index)] sorted: a Poisson process per flow with piecewise-constant rate, one independent
    stream per flow (so adding a flow does not change the others)."""
    streams = np.random.SeedSequence(seed).spawn(max(1, len(flows)))
    out = []
    for k, (flow, ss) in enumerate(zip(flows, streams)):
        rng = np.random.default_rng(ss)
        rates = sorted(flow.rates)
        for i, (start, vph) in enumerate(rates):
            end = rates[i + 1][0] if i + 1 < len(rates) else horizon_s
            end = min(end, horizon_s)
            if vph <= 0 or start >= end:
                continue
            t = start
            while True:
                t += rng.exponential(3600.0 / vph)
                if t >= end:
                    break
                out.append((t, k))
    out.sort()
    return out


def expected_link_flows(network: Network, flows: list, horizon_s: float) -> dict:
    """Mean vehicles/hour on each link over the horizon if everyone takes the free-flow shortest path."""
    ff = {lid: l.free_flow_s for lid, l in network.links.items()}
    out: dict = {}
    for f in flows:
        r = routing.shortest_path(network, f.origin, f.dest, weights=ff)
        if r is None:
            continue
        veh = 0.0
        for i, (s, vph) in enumerate(sorted(f.rates)):
            e = sorted(f.rates)[i + 1][0] if i + 1 < len(f.rates) else horizon_s
            veh += vph * max(0.0, min(e, horizon_s) - s) / 3600.0
        for l in r.links:
            out[l] = out.get(l, 0.0) + veh * 3600.0 / horizon_s
    return out


class Vehicle:
    __slots__ = ("id", "t_gen", "route", "idx", "ready", "qt", "ev", "blocked", "fft", "t_done")

    def __init__(self, vid, t_gen, route, fft, ev=False):
        self.id, self.t_gen, self.route, self.idx, self.ev = vid, t_gen, route, 0, ev
        self.ready = self.qt = 0.0
        self.blocked = None
        self.fft = fft
        self.t_done = None


class Simulator:
    """One run. `controller.decide(node, self, t)` is asked once per second per signal; the simulator itself is
    the state the controllers read (`queue`, `movement_queue`, `evs`, `network`)."""

    def __init__(self, network: Network, flows: list, controller, *, horizon_s: float = 1800.0,
                 drain_s: float = 600.0, seed: int = 0, startup_lost_s: float = 2.0,
                 route_refresh_s: float = 30.0, gridlock_release_s: float = 60.0, evs=(),
                 record_frames: bool = False, frame_every: int = 1):
        self.network, self.controller = network, controller
        self.flows, self.horizon_s, self.drain_s = list(flows), float(horizon_s), float(drain_s)
        self.startup_lost_s, self.route_refresh_s = float(startup_lost_s), float(route_refresh_s)
        self.gridlock_release_s = float(gridlock_release_s)
        self.record_frames, self.frame_every = record_frames, max(1, int(frame_every))
        self.t = 0.0
        self._arrivals = generate_arrivals(self.flows, self.horizon_s, seed)
        self._ai = 0
        self._evs = sorted(evs, key=lambda e: e.depart_s)
        self._ei = 0
        L = network.links
        self._transit = {l: deque() for l in L}
        self._qm = {m: deque() for m in network.movements}
        self._qcount = {l: 0 for l in L}
        self._count = {l: 0 for l in L}                      # vehicles on the link: travelling + queued
        self._credit = {l: 0.0 for l in L}
        self._store = {l: L[l].storage(network.spacing_m) for l in L}
        self._backlog = {l: deque() for l in L}              # waiting outside to enter the first link
        self._phase = {n: None for n in network.signals()}
        self._phase_start = {n: 0.0 for n in network.signals()}
        self._green: dict = {n: frozenset() for n in network.signals()}
        self._next: dict = {}
        self._next_at = -math.inf
        self._fft_od: dict = {}
        self._ff = {lid: l.free_flow_s for lid, l in L.items()}
        self.generated = self.finished = 0
        self.forced_releases = 0
        self.vehicles: list = []
        self._ev_vehicles: list = []
        self._qsum = 0.0
        self._qmax = 0.0
        self._steps = 0
        self.queue_series: list = []
        self.frames: list = []

    # -- the state controllers read
    def queue(self, link: str) -> float:
        return float(self._qcount[link])

    def movement_queue(self, movement: str) -> float:
        return float(len(self._qm[movement]))

    def evs(self) -> list:
        out = []
        for v in self._ev_vehicles:
            if v.t_done is None:
                lid = v.route[v.idx]
                link = self.network.links[lid]
                if v.qt > 0 and v.ready <= self.t:           # queued at the stop line
                    pos, speed = link.length_m, 0.0
                else:
                    left = max(0.0, v.ready - self.t)
                    pos = link.length_m * (1.0 - left / max(1e-9, link.free_flow_s))
                    speed = link.length_m / link.free_flow_s
                out.append({"route": v.route, "idx": v.idx, "pos_m": pos, "speed_mps": speed})
        return out

    # -- routing
    def _refresh_routes(self) -> None:
        w = {lid: self._ff[lid] + self._qcount[lid] / (self.network.links[lid].sat_flow_vph / 3600.0)
             for lid in self.network.links}
        self._next = {d: routing.tree_to(self.network, d, weights=w)[1] for d in self._dests}
        self._next_at = self.t

    def _route(self, origin: str, dest: str) -> Optional[list]:
        if self.t - self._next_at >= self.route_refresh_s or dest not in self._next:
            if dest not in self._dests:
                self._dests.add(dest)
            self._refresh_routes()
        return routing.follow_tree(self.network, self._next[dest], origin, dest)

    def _free_flow_time(self, origin: str, dest: str) -> float:
        k = (origin, dest)
        if k not in self._fft_od:
            r = routing.shortest_path(self.network, origin, dest, weights=self._ff)
            if r is None:
                raise ValueError(f"no route from {origin} to {dest}")
            self._fft_od[k] = r.time_s
        return self._fft_od[k]

    @property
    def _dests(self) -> set:
        if not hasattr(self, "_dest_set"):
            self._dest_set = {f.dest for f in self.flows} | {e.dest for e in self._evs}
        return self._dest_set

    # -- movement of vehicles
    def _enter(self, v: Vehicle, link: str) -> None:
        self._count[link] += 1
        v.ready = self.t + self._ff[link]
        v.qt = 0.0
        self._transit[link].append(v)

    def _depart(self, v: Vehicle) -> None:
        first = v.route[0]
        if self._count[first] < self._store[first] and not self._backlog[first]:
            self._enter(v, first)
        else:
            self._backlog[first].append(v)

    def _spawn(self, origin, dest, route=None, ev=False) -> None:
        route = route or self._route(origin, dest)
        if not route:
            raise ValueError(f"no route from {origin} to {dest}")
        v = Vehicle(len(self.vehicles), self.t, route, self._free_flow_time(origin, dest), ev)
        self.vehicles.append(v)
        if ev:
            self._ev_vehicles.append(v)
        self.generated += 1
        self._depart(v)

    def in_network(self) -> int:
        return sum(self._count.values()) + sum(len(b) for b in self._backlog.values())

    def conserved(self) -> bool:
        return self.generated == self.finished + self.in_network()

    def step(self) -> None:
        t = self.t
        net = self.network
        while self._ai < len(self._arrivals) and self._arrivals[self._ai][0] <= t:
            f = self.flows[self._arrivals[self._ai][1]]
            self._spawn(f.origin, f.dest)
            self._ai += 1
        while self._ei < len(self._evs) and self._evs[self._ei].depart_s <= t:
            e = self._evs[self._ei]
            self._spawn(e.origin, e.dest, e.route or self._route(e.origin, e.dest), ev=True)
            self._ei += 1
        for lid, b in self._backlog.items():                  # let waiting vehicles in as room appears
            while b and self._count[lid] < self._store[lid]:
                self._enter(b.popleft(), lid)
        # signals
        for n in self._phase:
            ph = self.controller.decide(n, self, t)
            if ph != self._phase[n]:
                if ph is not None:
                    self._phase_start[n] = t
                self._phase[n] = ph
                self._green[n] = frozenset(net.phases[n][ph].movements) if ph is not None else frozenset()
        # vehicles reaching the end of their link
        for lid, tr in self._transit.items():
            while tr and tr[0].ready <= t:
                v = tr.popleft()
                if v.idx == len(v.route) - 1:
                    v.t_done = t
                    self._count[lid] -= 1
                    self.finished += 1
                else:
                    v.qt = t if t > 0 else 1e-9
                    self._qm[f"{lid}>{v.route[v.idx + 1]}"].append(v)
                    self._qcount[lid] += 1
        # discharge
        for lid, link in net.links.items():
            if not self._qcount[lid]:
                self._credit[lid] = min(self._credit[lid], 1.0)
                continue
            node = link.to_node
            if net.nodes[node].kind == SIGNAL:
                if t - self._phase_start[node] < self.startup_lost_s and self._phase[node] is not None:
                    continue
                green = self._green[node]
                if not green:
                    continue
                mids = [m for m in net.movements_from(lid) if m in green]
            else:
                mids = net.movements_from(lid)
            self._credit[lid] += link.sat_flow_vph / 3600.0
            while self._credit[lid] >= 1.0:
                best = None
                for m in mids:
                    q = self._qm[m]
                    if q and (best is None or q[0].qt < self._qm[best][0].qt):
                        nl = net.movements[m].to_link
                        v = q[0]
                        if self._count[nl] < self._store[nl]:
                            best = m
                        else:
                            if v.blocked is None:
                                v.blocked = t
                            if t - v.blocked >= self.gridlock_release_s:
                                best = m
                                self.forced_releases += 1
                if best is None:
                    self._credit[lid] = min(self._credit[lid], 1.0)
                    break
                v = self._qm[best].popleft()
                v.blocked = None
                self._qcount[lid] -= 1
                self._count[lid] -= 1
                v.idx += 1
                self._enter(v, net.movements[best].to_link)
                self._credit[lid] -= 1.0
        # records
        total_q = sum(self._qcount.values())
        self._qsum += total_q
        self._qmax = max(self._qmax, max(self._qcount.values()))
        if self._steps % 10 == 0:
            self.queue_series.append((t, total_q))
        if self.record_frames and self._steps % self.frame_every == 0:
            self.frames.append(self._frame())
        self._steps += 1
        self.t = t + 1.0

    def _frame(self) -> dict:
        links = {}
        for lid in self.network.links:
            n, q = self._count[lid], self._qcount[lid]
            if n:
                links[lid] = [n - q, q]
        evs = []
        for v in self._ev_vehicles:
            if v.t_done is None:
                lid = v.route[v.idx]
                frac = 1.0 if v.qt > 0 and v.ready <= self.t else \
                    1.0 - max(0.0, v.ready - self.t) / max(1e-9, self._ff[lid])
                evs.append({"link": lid, "frac": round(frac, 3)})
        return {"t": self.t, "links": links, "evs": evs,
                "signals": {n: (-1 if p is None else p) for n, p in self._phase.items()}}

    def run(self, stop=None) -> dict:
        """Run to the end. `stop`: an optional callable (e.g. threading.Event.is_set); when it returns True the
        run ends early with the result so far, so a server can give up on a simulation that takes too long."""
        end = self.horizon_s + self.drain_s
        while self.t < end:
            if stop is not None and stop():
                break
            if self.t >= self.horizon_s and self.in_network() == 0 and self._ei >= len(self._evs):
                break
            self.step()
        return self.result()

    def result(self) -> dict:
        t_end = self.t
        others = [v for v in self.vehicles if not v.ev]

        def delay(v):
            return max(0.0, (v.t_done if v.t_done is not None else t_end) - v.t_gen - v.fft)

        d = np.array([delay(v) for v in others]) if others else np.zeros(0)
        trips = np.array([(v.t_done if v.t_done is not None else t_end) - v.t_gen for v in others]) \
            if others else np.zeros(0)
        done_in_horizon = sum(1 for v in others if v.t_done is not None and v.t_done <= self.horizon_s)
        evs = [{"depart_s": v.t_gen, "finished": v.t_done is not None,
                "travel_time_s": (v.t_done if v.t_done is not None else t_end) - v.t_gen,
                "free_flow_s": v.fft, "delay_s": delay(v)} for v in self._ev_vehicles]
        steps = max(1, self._steps)
        return {"generated": self.generated, "finished": self.finished, "in_network": self.in_network(),
                "unfinished": sum(1 for v in others if v.t_done is None),
                "vehicles": len(others), "mean_delay_s": float(d.mean()) if d.size else 0.0,
                "p95_delay_s": float(np.percentile(d, 95)) if d.size else 0.0,
                "mean_trip_s": float(trips.mean()) if trips.size else 0.0,
                "throughput_vph": done_in_horizon * 3600.0 / self.horizon_s,
                "max_queue_veh": float(self._qmax), "mean_total_queue_veh": self._qsum / steps,
                "forced_releases": self.forced_releases, "ended_s": t_end, "ev": evs,
                "queue_series": self.queue_series, "frames": self.frames}


def simulate(network: Network, flows: list, controller, **kw) -> dict:
    """Build a Simulator, run it to the end, return its result."""
    return Simulator(network, flows, controller, **kw).run()
