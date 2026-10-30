"""Signal controllers: fixed-time, max pressure, and emergency-vehicle priority two ways.

Nothing here has controlled a real signal. These are controllers written against a small state interface
(`queue`, `movement_queue`, `evs`) that the simulator (sim.py) implements and that a camera-fed `QueueState` can
also satisfy, so the same code can be validated in simulation and later asked what it WOULD do on measured
queues. No learning and no GPU: the pressure idea is the one MPLight builds its reward on (Chen et al.), without
the learned policy; EMVLight (emergency routing with signal pre-emption) and Cao et al. (priority as a weighted
term) are the two designs for emergency vehicles, and both are implemented as plain rules.

Interface
    decide(node, state, t) -> phase index, or None while the junction is in its clearance interval (every
    approach red, nothing discharges). Called once per simulated second, in time order, for each signal.
    force(node, phase, t) tells a controller which phase is actually showing, so a wrapper can hand back.

FixedTimeController
    A cycle with a share of green per phase and a clearance interval after each: the baseline every signal in
    the city runs today. `from_flows` sets the shares in proportion to the expected flow on each phase's
    busiest approach, which is the strongest fixed-time plan the long-run average can justify; it still cannot
    follow a peak that is not in the average.

MaxPressureController (Varaiya, "Max pressure control of a network of signalized intersections", Transportation
Research Part C 36, 2013)
    Weight of a movement i->j: its queue minus the downstream link's queue weighted by that link's turn ratios,
        w(i,j) = x(i,j) - sum_k r(j,k) x(j,k)
    pressure of a phase = sum over its movements of saturation flow x w. Serve the phase of greatest pressure.
    Varaiya proves this stabilises any demand a fixed-time plan could, using only local queues and turn ratios;
    it needs no demand forecast, which is why it suits a city that can see queues but cannot predict them. Added
    here, as every deployed controller needs them: a minimum green (no flicker), a clearance interval (lost time
    is real), an optional maximum green (a small approach is not starved forever behind a saturated one), and
    optional exponential smoothing of the pressures (noisy camera queues). Limits of the theory: it assumes
    unbounded storage, and the stability proof is for the model, not for a street; pressures compare vehicle
    counts, not people, so a bus and a motorbike weigh the same.
    When only link queues are known (the camera case) the movement queue is the link queue times the turn ratio.

EmergencyCorridor (the EMVLight idea, as a rule)
    From each emergency vehicle's route, position and speed, predict when it reaches each junction ahead and
    serve its movement in a window around that arrival: early enough that the queue ahead of it has discharged
    (clearance + queue / saturation flow + margin), held until it has gone through, then handed back to the
    wrapped controller. A hard override: the ordinary traffic that loses the phase simply waits. EMVLight
    reports up to 42.6 % less emergency travel time and 23.5 % less for ordinary traffic with a LEARNED policy
    on its own benchmarks; that figure is the paper's, and whether a rule like this one costs the others
    anything is measured in tools/signal_sim.py, not assumed.

SoftPriority (Cao et al.: priority as a weighted term, not an override)
    The emergency vehicle adds `weight` vehicles to the queue of its movement, fading in as it approaches, and
    max pressure decides as usual. A heavily queued conflicting phase can still win, and minimum green is
    still respected, so the emergency vehicle may wait; in exchange the others are disturbed less.

recommend_timings: green seconds per phase over a cycle, proportional to the positive pressures, for a
dashboard to display. A suggestion from the current queues, not a plan anyone has installed.
"""

from __future__ import annotations

from typing import Optional

from .network import Network


# ---------------------------------------------------------------- state interface

class QueueState:
    """A state built from numbers rather than from a simulation: what a dashboard or a camera feed has.

    queues           {link id: vehicles waiting at its stop line}
    movement_queues  {movement id: vehicles}; where missing, the link queue times the movement's turn ratio
    evs              emergency vehicles, each {"route": [link ids], "idx": index of its current link,
                     "pos_m": metres along that link, "speed_mps": its speed}"""

    def __init__(self, network: Network, queues: Optional[dict] = None, movement_queues: Optional[dict] = None,
                 evs=()):
        self.network = network
        self._q = dict(queues or {})
        self._mq = dict(movement_queues or {})
        self._evs = list(evs)

    @classmethod
    def from_network(cls, network: Network) -> "QueueState":
        """The queues the network's camera measurements imply (link.state['queue_veh']); links with none are empty."""
        return cls(network, {lid: l.state.get("queue_veh", 0.0) for lid, l in network.links.items() if l.state})

    def queue(self, link: str) -> float:
        return float(self._q.get(link, 0.0))

    def movement_queue(self, movement: str) -> float:
        if movement in self._mq:
            return float(self._mq[movement])
        m = self.network.movements[movement]
        return self.queue(m.from_link) * m.ratio

    def evs(self) -> list:
        return self._evs


def ev_arrivals(network: Network, ev: dict, horizon_s: float = 120.0) -> list:
    """[(node, in_link, out_link, eta_s)] for the junctions an emergency vehicle will cross, soonest first.

    eta_s counts from now: the rest of the current link at the vehicle's speed (at least walking pace), then
    each later link at its free-flow time. Only junctions where the route continues are listed (the last link
    ends at the destination, which needs no green)."""
    route, i = ev["route"], int(ev["idx"])
    link = network.links[route[i]]
    v = max(float(ev.get("speed_mps") or 0.0), 1.0)
    eta = max(0.0, link.length_m - float(ev.get("pos_m") or 0.0)) / v
    out = []
    while i + 1 < len(route) and eta <= horizon_s:
        out.append((network.links[route[i]].to_node, route[i], route[i + 1], eta))
        i += 1
        eta += network.links[route[i]].free_flow_s
    return out


class Controller:
    """Base: decide(node, state, t) -> phase index or None (clearance)."""

    def decide(self, node: str, state, t: float) -> Optional[int]:
        raise NotImplementedError

    def force(self, node: str, phase: Optional[int], t: float) -> None:
        """The junction is showing `phase` as of time t (set by a wrapper that overrode this controller)."""


# ---------------------------------------------------------------- fixed time

class FixedTimeController(Controller):
    """Cycle of `cycle_s` seconds; each phase gets its split of the green (cycle minus clearances), then a
    clearance interval. splits: {node: [share per phase]} (normalised); default equal. offsets: {node: s}."""

    def __init__(self, network: Network, cycle_s: float = 80.0, splits: Optional[dict] = None,
                 clearance_s: float = 4.0, offsets: Optional[dict] = None, min_green: float = 5.0):
        self.network, self.cycle_s, self.clearance_s = network, float(cycle_s), float(clearance_s)
        self.offsets = offsets or {}
        self._plan: dict = {}
        for nid in network.signals():
            n = len(network.phases[nid])
            share = list((splits or {}).get(nid) or [1.0] * n)
            if len(share) != n or min(share) < 0 or sum(share) <= 0:
                raise ValueError(f"splits for {nid} must be {n} non-negative numbers")
            green = max(self.cycle_s - n * self.clearance_s, n * min_green)
            tot = sum(share)
            g = [max(min_green, green * s / tot) for s in share]
            k = sum(g) / green
            g = [x / k if k > 1 else x for x in g]
            bounds, t = [], 0.0
            for i, gi in enumerate(g):
                bounds.append((t, t + gi, i))
                t += gi + self.clearance_s
            self._plan[nid] = (bounds, t)           # actual cycle length = greens + clearances

    @classmethod
    def from_flows(cls, network: Network, link_flows_vph: dict, **kw) -> "FixedTimeController":
        """Splits proportional to the flow ratio (flow / saturation flow) of each phase's busiest approach."""
        splits = {}
        for nid in network.signals():
            ratios = []
            for ph in network.phases[nid]:
                links = {network.movements[m].from_link for m in ph.movements}
                ratios.append(max((link_flows_vph.get(l, 0.0) / network.links[l].sat_flow_vph for l in links),
                                  default=0.0))
            splits[nid] = [max(r, 0.02) for r in ratios]
        return cls(network, splits=splits, **kw)

    def decide(self, node: str, state, t: float) -> Optional[int]:
        bounds, cycle = self._plan[node]
        pos = (t - self.offsets.get(node, 0.0)) % cycle
        for a, b, i in bounds:
            if a <= pos < b:
                return i
        return None


# ---------------------------------------------------------------- max pressure

class _Node:
    __slots__ = ("cur", "since", "clear_until", "nxt", "smooth")

    def __init__(self):
        self.cur, self.since, self.clear_until, self.nxt, self.smooth = 0, None, None, None, None


class MaxPressureController(Controller):
    def __init__(self, network: Network, min_green: float = 15.0, clearance_s: float = 4.0,
                 max_green: Optional[float] = None, smoothing: float = 0.0):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.network, self.min_green, self.clearance_s = network, float(min_green), float(clearance_s)
        self.max_green, self.smoothing = max_green, float(smoothing)
        self._st: dict = {}

    # -- hooks
    def extra_queue(self, movement: str, state, t: float) -> float:
        """Vehicles added to a movement's queue (SoftPriority overrides this)."""
        return 0.0

    def movement_weight(self, mid: str, state, t: float) -> float:
        m = self.network.movements[mid]
        up = state.movement_queue(mid) + self.extra_queue(mid, state, t)
        down = sum(self.network.movements[k].ratio * state.movement_queue(k)
                   for k in self.network.movements_from(m.to_link))
        return up - down

    def movement_sat_flow(self, mid: str) -> float:
        m = self.network.movements[mid]
        return m.sat_flow_vph if m.sat_flow_vph is not None else self.network.links[m.from_link].sat_flow_vph

    def pressures(self, node: str, state, t: float = 0.0) -> list:
        """Pressure of every phase at `node` (before smoothing)."""
        return [sum(self.movement_sat_flow(m) * self.movement_weight(m, state, t) for m in ph.movements)
                for ph in self.network.phases[node]]

    # -- control
    def decide(self, node: str, state, t: float) -> Optional[int]:
        st = self._st.get(node)
        if st is None:
            st = self._st[node] = _Node()
            st.since = t
        if st.clear_until is not None:
            if t < st.clear_until:
                return None
            st.cur, st.since, st.clear_until = st.nxt, t, None
            return st.cur
        held = t - st.since
        if held < self.min_green:
            return st.cur
        p = self.pressures(node, state, t)
        if self.smoothing > 0 and st.smooth is not None and len(st.smooth) == len(p):
            p = [self.smoothing * a + (1 - self.smoothing) * b for a, b in zip(st.smooth, p)]
        st.smooth = p
        best = max(range(len(p)), key=lambda i: (p[i], i == st.cur, -i))
        if self.max_green is not None and held >= self.max_green and best == st.cur and len(p) > 1:
            best = max((i for i in range(len(p)) if i != st.cur), key=lambda i: (p[i], -i))
        if best == st.cur:
            return st.cur
        if self.clearance_s <= 0:
            st.cur, st.since = best, t
            return best
        st.nxt, st.clear_until = best, t + self.clearance_s
        return None

    def force(self, node: str, phase: Optional[int], t: float) -> None:
        st = self._st.setdefault(node, _Node())
        if phase is not None:
            st.cur = phase
        st.since, st.clear_until, st.smooth = t, None, None


class SoftPriority(MaxPressureController):
    """Max pressure where an approaching emergency vehicle counts as `weight` extra vehicles in its movement's
    queue, fading in linearly from `horizon_s` before it arrives to full weight at `lead_s`."""

    def __init__(self, network: Network, weight: float = 25.0, lead_s: float = 15.0, horizon_s: float = 60.0, **kw):
        super().__init__(network, **kw)
        self.weight, self.lead_s, self.horizon_s = float(weight), float(lead_s), float(horizon_s)

    def extra_queue(self, movement: str, state, t: float) -> float:
        m = self.network.movements[movement]
        total = 0.0
        for ev in state.evs():
            arr = ev_arrivals(self.network, ev, self.horizon_s)
            if arr and arr[0][1] == m.from_link and arr[0][2] == m.to_link:
                eta = arr[0][3]
                fade = 1.0 if eta <= self.lead_s else 1.0 - (eta - self.lead_s) / max(1e-9, self.horizon_s - self.lead_s)
                total += self.weight * max(0.0, min(1.0, fade))
        return total


# ---------------------------------------------------------------- hard emergency corridor

class EmergencyCorridor(Controller):
    """Pre-empt the wrapped controller for emergency vehicles (see the module docstring).

    margin_s       slack added to the time needed to clear the queue ahead
    max_override_s the longest one junction is held for one vehicle: a stuck ambulance must not freeze a junction
    horizon_s      how far ahead (in seconds of travel) vehicles are looked for"""

    def __init__(self, base: Controller, network: Network, clearance_s: float = 4.0, margin_s: float = 3.0,
                 horizon_s: float = 120.0, max_override_s: float = 90.0):
        self.base, self.network = base, network
        self.clearance_s, self.margin_s = float(clearance_s), float(margin_s)
        self.horizon_s, self.max_override_s = float(horizon_s), float(max_override_s)
        self._st: dict = {}
        self.overrides = 0          # seconds of junction time spent serving an emergency vehicle

    def _demand(self, node: str, state) -> Optional[tuple]:
        """(phase index, eta_s) of the soonest emergency vehicle that needs this junction now, or None."""
        best = None
        for ev in state.evs():
            for n, a, b, eta in ev_arrivals(self.network, ev, self.horizon_s):
                if n != node:
                    continue
                link = self.network.links[a]
                lead = self.clearance_s + state.queue(a) / (link.sat_flow_vph / 3600.0) + self.margin_s
                if eta <= lead:
                    phases = self.network.phases_with(f"{a}>{b}")
                    if phases and (best is None or eta < best[1]):
                        best = (phases[0], eta)
                break               # only the first time this vehicle reaches the junction counts
        return best

    def decide(self, node: str, state, t: float) -> Optional[int]:
        st = self._st.setdefault(node, {"shown": None, "mode": "base", "clear_until": None, "since": None,
                                        "spent": 0.0})
        want = self._demand(node, state) if t >= st.get("cool", 0.0) else None
        if want is not None and st["mode"] == "base":
            st.update(mode="ev", since=t)
        if want is not None and st["mode"] == "ev" and t - st["since"] > self.max_override_s:
            want = None                                      # give up on this one; do not freeze the junction
            st["cool"] = t + self.max_override_s
        if want is not None:
            self.overrides += 1
            phase = want[0]
            if st["shown"] == phase:
                return phase
            if st["shown"] is not None:                      # leaving another phase: clearance first
                st["shown"], st["clear_until"] = None, t + self.clearance_s
                return None
            if st["clear_until"] is not None and t < st["clear_until"]:
                return None
            st["shown"], st["clear_until"] = phase, None
            return phase
        if st["mode"] == "ev":                               # the vehicle has gone through: hand back
            st["mode"] = "base"
            self.base.force(node, st["shown"], t)
        out = self.base.decide(node, state, t)
        if out is not None and st["shown"] is not None and out != st["shown"]:
            # the base jumped from the phase we showed straight to another: insert the clearance it never saw
            st["shown"], st["clear_until"] = None, t + self.clearance_s
            return None
        if out is None:
            st["shown"] = None
            st["clear_until"] = None
            return None
        if st["clear_until"] is not None and t < st["clear_until"]:
            return None
        st["shown"], st["clear_until"] = out, None
        return out

    def force(self, node: str, phase: Optional[int], t: float) -> None:
        self.base.force(node, phase, t)


# ---------------------------------------------------------------- dashboard recommendation

def recommend_timings(network: Network, state=None, cycle_s: float = 90.0, clearance_s: float = 4.0,
                      min_green: float = 7.0) -> dict:
    """Suggested green seconds per phase over one cycle, per signal, from the current pressures.

    {node: {"cycle_s", "lost_s", "demand": "none|low|medium|high", "phases": [{"name", "green_s", "share",
    "pressure", "queue_veh"}]}}. Greens are proportional to the positive pressure of each phase (equal when no
    phase has any), never below `min_green`, and add up to the cycle minus the clearances. `state` defaults to
    the queues the network's camera measurements imply. This is what max pressure would favour right now,
    shown as a plan; Webster's cycle-length formula needs arrival rates, which queues alone do not give."""
    state = QueueState.from_network(network) if state is None else state
    mp = MaxPressureController(network)
    out = {}
    for nid in network.signals():
        phases = network.phases[nid]
        n = len(phases)
        pr = mp.pressures(nid, state)
        pos = [max(0.0, x) for x in pr]
        qv = [sum(state.queue(l) for l in {network.movements[m].from_link for m in ph.movements}) for ph in phases]
        lost = n * clearance_s
        green = max(cycle_s - lost, n * min_green)
        share = [x / sum(pos) for x in pos] if sum(pos) > 0 else [1.0 / n] * n
        g = [max(min_green, green * s) for s in share]
        # take the excess from the phases above the floor so the total stays the cycle
        for _ in range(n):
            excess = sum(g) - green
            free = [i for i in range(n) if g[i] > min_green + 1e-9]
            if excess <= 1e-9 or not free:
                break
            room = sum(g[i] - min_green for i in free)
            for i in free:
                g[i] -= min(g[i] - min_green, excess * (g[i] - min_green) / room)
        total_q = sum(qv)
        demand = "none" if total_q < 1 else "low" if total_q < 10 else "medium" if total_q < 30 else "high"
        out[nid] = {"cycle_s": round(sum(g) + lost, 1), "lost_s": lost, "demand": demand,
                    "phases": [{"name": ph.name, "green_s": round(gi, 1), "share": round(gi / sum(g), 3),
                                "pressure": round(p, 1), "queue_veh": round(q, 1)}
                               for ph, gi, p, q in zip(phases, g, pr, qv)]}
    return out
