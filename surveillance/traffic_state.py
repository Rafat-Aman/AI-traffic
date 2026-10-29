"""Rolling traffic state per approach, and the time-of-day baseline it is later compared against.

This is the measuring half of the project's impact assessment (report, SDP 3.1.4): "For each event, the system
compares the traffic state on the affected approach during the event window against the baseline for that
approach and time of day, over five quantities: queue length, mean speed, lane occupancy, traffic density, and
mean waiting time." Nothing in the code base measured any of those before this module, so each one is defined
here, with the reason for the definition, rather than borrowed from a traffic-engineering text whose detectors
and cameras are not ours.

What is measured (one sample per approach per second of video time)
    queue_m    Length in metres of the longest chain of stopped vehicles (< 5 km/h, bumper-to-bumper gaps of at most
               10 m), from the front of its head vehicle to the rear of its tail. At least two vehicles are needed:
               one vehicle standing on an empty road is not a queue. The chain is NOT required to start at the
               zone's downstream end. A signal queue does start there, but the case this project exists for - a bus
               standing in a running lane mid-block - has its head wherever the bus stopped, with free road
               downstream of it; anchoring at the exit would report zero queue for exactly that event. A signal
               queue is the special case where the head happens to be at the exit.
    speed_kph  Mean of the vehicles' smoothed speeds. The arithmetic mean over the vehicles present at one instant IS
               the space-mean speed (each vehicle counted once, however long it stays), which is the speed that
               travel time over a stretch is distance / speed of (Little's law). That is why excess delay in
               impact.py can use it as its baseline.
    occupancy  Space occupancy: sum of typical vehicle lengths / (zone length x lanes), capped at 1. The camera
               cannot see lane markings in general, so lanes = round(width / 3.2 m).
    density    Vehicles per km per lane.
    wait_s     Mean continuous time stationary (< 2 km/h) of the vehicles in the zone, 0 for vehicles that are moving.

    Motorbikes and bicycles (violations.base.FILTERING_CLASSES) weave through stopped traffic, so their movement says
    nothing about whether the traffic is flowing: they are left out of queue_m, speed_kph, wait_s and of the delay
    integral, but they do occupy road space, so they count towards occupancy and density.

Why one sample a second and nothing per frame
    The pipeline calls `observe` every processed frame, beside the detector, so the per-frame cost has to be close
    to zero. Between samples `observe` is one comparison. At a sample it is O(vehicles). Everything that needs a
    longer view (stationary time, travel time, queue growth) is derived from the stored samples, never kept as
    per-frame state, so memory is bounded by the horizon (15 min) and not by the length of the video.

Why every sample keeps the per-vehicle table
    The metrics are averaged over a window later, but the question the report asks is the delay an event caused TO
    OTHER VEHICLES. The vehicle that is itself the violation (a bus standing in the lane) is stationary for
    its whole dwell, so left in it would add its own waiting time and its own travel time to the "impact": a bus
    stopping on an empty road would score as if it had jammed the road. So each sample keeps a small array
    (tid, length, y, kph, wait, filtering, heading) per vehicle, 56 bytes per vehicle-sample (about 1 MB for a
    busy approach over the whole 15-minute horizon), and `state` and `excess_delay` can leave named vehicles out.
    That is also how the delay integral gets each vehicle's position at the window edges: `transits` and
    `excess_delay` are both read off these tables, so there is no second store of vehicle passes to keep
    consistent with them.

The baseline
    Per (camera, zone, 15-minute slot of the day, weekday or weekend) a bounded reservoir of window-sized block means
    (30 s by default) per quantity, from which a median and a MAD are read. Block means rather than per-second
    values, because the comparison is a window mean against the baseline: the noise that matters is the noise of a
    30-second mean, and per-second values would overstate it. A reservoir (algorithm R) bounds memory and the
    median / MAD pair bounds the influence of the events themselves, which are in the stream they are compared
    against. Once a reservoir has seen `history` values it keeps replacing at a fixed rate, so the baseline follows
    the last few weeks instead of averaging all time. The weekend is Friday and Saturday, Bangladesh's weekly
    holidays (configurable). A slot only becomes a baseline after `min_blocks` blocks on at least `min_days` different
    days, because one day can be anomalous and then "typical for this time" would not be true.

Limits, stated here so they are not found out later
    * Everything is validated on synthetic traffic (tests/test_impact.py) and nowhere else. No Dhaka footage has been
      run through it, and the thresholds (5 km/h, 2 km/h, 10 m, 3.2 m lanes, vehicle lengths) are judgement.
    * A state is only as good as the detector: missed vehicles lower occupancy and density; positions at the far end
      of a zone are coarse (xy_coarse) and have no speed, so those vehicles count for occupancy and density but not
      for speed or waiting.
    * A long line of parked vehicles is a queue to this module. The baseline absorbs it only if it is regular.
    * The cold-start baseline (the minutes just before the event) measures CHANGE, so it cannot see a jam that was
      already there when the window opened.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from violations.base import FILTERING_CLASSES

QUANTITIES = ("queue_m", "speed_kph", "occupancy", "density", "wait_s")
# +1: the quantity gets worse as it rises; -1: worse as it falls (only speed)
WORSE = {"queue_m": 1, "speed_kph": -1, "occupancy": 1, "density": 1, "wait_s": 1}
UNITS = {"queue_m": "m", "speed_kph": "km/h", "occupancy": "", "density": "veh/km/lane", "wait_s": "s"}

# Typical lengths in metres. Judgement, not measurement: they only set how much road a vehicle takes up.
VEHICLE_LENGTH_M = {"Bus": 11.0, "Truck": 9.0, "Car": 4.5, "MPV": 4.8, "Pickup": 5.0, "CNG": 2.7, "Easybike": 2.8,
                    "Leguna": 4.0, "Motorbike": 2.0, "Bicycle": 1.8, "Rickshaw": 2.2, "Van": 2.5,
                    "Bhotbhoti": 4.0, "PowerTiller": 3.5, "Wheelbarrow": 1.5}
DEFAULT_LENGTH_M = 4.0
LANE_WIDTH_M = 3.2
SLOW_KPH = 5.0        # at or below this a vehicle is part of a queue
STILL_KPH = 2.0       # below this it is standing, and the waiting clock runs
QUEUE_GAP_M = 10.0    # a bigger bumper-to-bumper gap ends the chain

# columns of the per-vehicle table kept with every sample
TID, LEN, Y, KPH, WAIT, FILT, HEAD = range(7)
NCOL = 7
_EMPTY = np.empty((0, NCOL))


def lane_count(road):
    return max(1, int(round(float(road.width_m) / LANE_WIDTH_M)))


def finite(x):
    """A real, finite number (numpy scalars included); False for None, NaN, inf and anything that is not a number."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def clean(x, nd=3):
    """A float that survives json.dumps(allow_nan=False): None for NaN / missing."""
    return round(float(x), nd) if finite(x) else None


def _nanmean_cols(a):
    """Column means that skip NaN; NaN where a column has nothing (no RuntimeWarning, unlike np.nanmean)."""
    ok = ~np.isnan(a)
    cnt = ok.sum(axis=0)
    return np.where(cnt > 0, np.where(ok, a, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)


def _queue_m(table, filt, kph, lens, flow):
    """The longest chain of stopped vehicles, front of the head to rear of the tail (see the module docstring)."""
    y, head = table[:, Y], table[:, HEAD]
    with np.errstate(invalid="ignore"):
        slow = ~filt & (kph < SLOW_KPH) & ~np.isnan(y)
    if flow:                                  # a vehicle known to go the other way is in the opposite carriageway
        slow &= (head == 0) | (head == flow)
    idx = np.flatnonzero(slow)
    if idx.size < 2:
        return 0.0
    order = idx[np.argsort(y[idx], kind="stable")]
    ys, ls = y[order], lens[order]
    gaps = np.diff(ys) - 0.5 * (ls[:-1] + ls[1:])            # bumper to bumper
    cuts = np.flatnonzero(gaps > QUEUE_GAP_M) + 1
    first = np.r_[0, cuts]
    last = np.r_[cuts, ys.size] - 1
    span = ys[last] - ys[first] + 0.5 * (ls[first] + ls[last])
    return float(np.where(last > first, span, 0.0).max())


def compute(table, length_m, lanes, flow=0):
    """(queue_m, speed_kph, occupancy, density, wait_s, contributors) of one table of vehicles. speed_kph is NaN with
    nobody to measure; the others are 0 then, because an empty road has no queue and nobody waiting. `contributors`
    is how many vehicles the two per-vehicle means (speed, wait) are means OF - NaN when none - which impact.py needs:
    one stuck vehicle makes a mean of 1 jump, a mean of 15 barely moves."""
    n = table.shape[0]
    if n == 0:
        return (0.0, math.nan, 0.0, 0.0, 0.0, math.nan)
    lens, kph, filt = table[:, LEN], table[:, KPH], table[:, FILT] > 0.5
    occ = min(1.0, float(lens.sum()) / (length_m * lanes))
    dens = n / (length_m / 1000.0) / lanes
    known = ~filt & ~np.isnan(kph)               # vehicles whose own movement says whether traffic flows
    k = int(known.sum())
    if k:
        speed, wait = float(kph[known].mean()), float(table[known, WAIT].mean())
    else:
        speed, wait = math.nan, 0.0
    return (_queue_m(table, filt, kph, lens, flow), speed, occ, dens, wait, float(k) if k else math.nan)


@dataclass
class Reference:
    """What the traffic on an approach normally looks like: a robust centre and spread per quantity."""
    source: str                       # "history" (time-of-day baseline) or "pre-event" (the minutes before)
    median: dict                      # quantity -> value (absent when nothing was measured)
    mad: dict                         # quantity -> raw median absolute deviation
    n: dict                           # quantity -> block means behind it
    detail: dict = field(default_factory=dict)


class _Zone:
    __slots__ = ("road", "name", "length_m", "lanes", "samples", "flow", "block_t0", "buf")

    def __init__(self, road, maxlen):
        self.road, self.name = road, road.name
        self.length_m, self.lanes = float(road.length_m), lane_count(road)
        self.samples = deque(maxlen=maxlen)          # (t, vals, n_vehicles, table)
        self.flow = int(getattr(road, "flow_sign", 0) or 0)
        self.block_t0 = None
        self.buf = []


class TrafficStateCollector:
    """Rolling state per approach, fed with the same Observation list the rule engine gets.

    `roads` must be every road zone of the camera: an approach nobody is on produces no observations, and its
    empty seconds have to be sampled too or the averages would only cover the times someone was there.
    """

    def __init__(self, camera_id, roads=(), baseline=None, window=30.0, horizon=900.0, period=1.0,
                 lost_sec=3.0, gap_sec=5.0):
        self.camera_id, self.baseline = camera_id, baseline
        self.window, self.horizon, self.period = float(window), float(horizon), float(period)
        self.lost_sec, self.gap_sec = float(lost_sec), float(gap_sec)
        self._lock = threading.RLock()
        self._zones, self._by_name = {}, {}
        for r in roads:
            self._register(r)
        self.reset()

    # ------------------------------------------------------------ feeding
    def reset(self):
        """Forget the samples (a new video, or the camera view was lost and what was measured no longer applies)."""
        with self._lock:
            self._next_t, self._last_t = -math.inf, -math.inf
            self._gap, self._wall_ref = False, None
            self._still = {}                     # tid -> [stationary since, last seen]
            for zs in self._zones.values():
                zs.samples.clear()
                zs.block_t0, zs.buf = None, []

    def _register(self, road):
        zs = _Zone(road, int(self.horizon / self.period) + 8)
        self._zones[id(road)] = zs
        self._by_name[road.name] = zs
        return zs

    def observe(self, observations, t, wall=None, valid=True):
        """One processed frame. `t`: video (or capture) seconds. `wall`: epoch seconds (a datetime is accepted),
        or None when the clock time of the video is not known. `valid=False`: the camera view is lost, so an
        empty zone means "cannot see", not "empty" - nothing is sampled and the stationary clocks stop."""
        if not valid:
            self._gap = True
            return
        if t < self._next_t:
            if t >= self._last_t - 0.5:
                return                                   # between samples: this is the per-frame cost
            self.reset()                                 # time went backwards
        with self._lock:
            self._sample(observations, t, wall)

    def _sample(self, observations, t, wall):
        if wall is not None:
            self._wall_ref = (t, wall.timestamp() if hasattr(wall, "timestamp") else float(wall))
        if self._gap or t - self._last_t > self.gap_sec:      # a hole in the stream: clocks and blocks restart
            self._still.clear()
            for zs in self._zones.values():
                zs.block_t0 = None
            self._gap = False
        self._last_t, self._next_t = t, t + self.period
        still, zones = self._still, self._zones
        for zs in zones.values():
            zs.buf = []
        nan = math.nan
        for o in observations:
            road = o.road
            if road is None:
                continue
            zs = zones.get(id(road)) or self._register(road)
            tid, cls, kph = o.tid, o.cls, o.kph
            xy = o.xy if o.xy is not None else o.xy_coarse
            sp = still.get(tid)
            if kph is not None and kph < STILL_KPH:
                if sp is None:
                    sp = still[tid] = [t, t]
                sp[1] = t
            elif kph is not None:
                if sp is not None:
                    del still[tid]
                sp = None
            elif sp is not None:                          # speed unknown: assume nothing changed
                sp[1] = t
            zs.buf += (tid, VEHICLE_LENGTH_M.get(cls, DEFAULT_LENGTH_M), nan if xy is None else xy[1],
                       nan if kph is None else kph, 0.0 if sp is None else t - sp[0],
                       1.0 if cls in FILTERING_CLASSES else 0.0, o.heading)
        for tid in [k for k, s in still.items() if t - s[1] > self.lost_sec]:
            del still[tid]
        cut = t - self.horizon
        for zs in zones.values():
            table = np.array(zs.buf, dtype=np.float64).reshape(-1, NCOL) if zs.buf else _EMPTY
            zs.buf = []
            zs.flow = self._flow(zs, table)
            zs.samples.append((t, compute(table, zs.length_m, zs.lanes, zs.flow), table.shape[0], table))
            while zs.samples[0][0] < cut:
                zs.samples.popleft()
            if self.baseline is not None:
                self._block(zs, t)

    @staticmethod
    def _flow(zs, table):
        """Which way traffic is going: the zone's rule, or for a two-way zone the majority heading of the vehicles that
        are moving (sticky: with nobody moving the last answer stands)."""
        sign = getattr(zs.road, "flow_sign", 0)
        if sign:
            return int(sign)
        if not table.shape[0]:
            return zs.flow
        with np.errstate(invalid="ignore"):
            moving = (table[:, FILT] < 0.5) & (table[:, KPH] >= SLOW_KPH) & (table[:, HEAD] != 0)
        s = float(table[moving, HEAD].sum())
        return 1 if s > 0 else -1 if s < 0 else zs.flow

    def _block(self, zs, t):
        """Hand a completed window-sized block of means to the baseline."""
        if zs.block_t0 is None:
            zs.block_t0 = t
            return
        if t - zs.block_t0 < self.window:
            return
        rows = []
        for s in reversed(zs.samples):
            if s[0] < zs.block_t0:
                break
            rows.append(s[1])
        t0, zs.block_t0 = zs.block_t0, t
        if len(rows) < 0.5 * self.window / self.period or self._wall_ref is None:
            return                                         # a block with half its seconds missing says too little
        means = _nanmean_cols(np.asarray(rows, dtype=np.float64))
        self.baseline.add(self.camera_id, zs.name, self.wall_at(0.5 * (t0 + t)),
                          {q: (float(m) if np.isfinite(m) else None) for q, m in zip(QUANTITIES, means)})

    def close(self):
        """End of the run: persist what the baseline has learned."""
        if self.baseline is not None:
            self.baseline.flush()

    # ------------------------------------------------------------ reading
    def wall_at(self, t):
        """Clock time (epoch seconds) of video time `t`, from the latest frame that carried one; None if none did."""
        return None if self._wall_ref is None else self._wall_ref[1] + (t - self._wall_ref[0])

    def zones(self):
        return list(self._by_name)

    def _find(self, zone):
        if isinstance(zone, str):
            return self._by_name.get(zone)
        return self._zones.get(id(zone)) or self._by_name.get(getattr(zone, "name", None))

    def _slice(self, zs, t0, t1):
        if t1 is None:
            t1 = zs.samples[-1][0] if zs.samples else 0.0
        if t0 is None:
            t0 = t1 - self.window
        out = []
        for s in reversed(zs.samples):
            if s[0] > t1:
                continue
            if s[0] < t0:
                break
            out.append(s)
        out.reverse()
        return out

    def _average(self, zs, samples, exclude):
        ex = np.asarray(sorted(exclude), dtype=np.float64) if exclude else None
        rows, ns = [], []
        for s in samples:
            vals, n = s[1], s[2]
            if ex is not None and n:
                hit = np.isin(s[3][:, TID], ex)
                if hit.any():
                    tab = s[3][~hit]
                    vals, n = compute(tab, zs.length_m, zs.lanes, zs.flow), tab.shape[0]
            rows.append(vals)
            ns.append(n)
        return _nanmean_cols(np.asarray(rows, dtype=np.float64)), float(np.mean(ns))

    def state(self, zone, t0=None, t1=None, exclude=()):
        """Averaged state of one approach over [t0, t1] (default: the last `window` seconds), or None when nothing
        was sampled there. `exclude`: track ids left out, so the vehicle that IS the event is not counted as its
        own victim. A quantity is None when it was never measurable in the window (speed with no vehicle moving
        in view). `contributors`: mean number of vehicles behind speed_kph / wait_s in the samples that had any."""
        zs = self._find(zone)
        if zs is None:
            return None
        with self._lock:
            samples = self._slice(zs, t0, t1)
            if not samples:
                return None
            means, n = self._average(zs, samples, exclude)
        out = {q: clean(m, 4) for q, m in zip(QUANTITIES, means)}
        out.update(contributors=clean(means[5], 2), samples=len(samples), vehicles=round(n, 2),
                   t0=samples[0][0], t1=samples[-1][0])
        return out

    def snapshot(self):
        """Current state of every approach, for a live display: {zone name: metrics + lanes, flow, vehicles now}."""
        out = {}
        with self._lock:
            for name, zs in self._by_name.items():
                st = self.state(zs.road)
                out[name] = {**{q: (st or {}).get(q) for q in QUANTITIES}, "samples": (st or {}).get("samples", 0),
                             "vehicles": zs.samples[-1][2] if zs.samples else 0,
                             "t": zs.samples[-1][0] if zs.samples else None, "lanes": zs.lanes, "flow": zs.flow}
        return out

    def reference(self, zone, t_a, t_b, block_sec=None):
        """Median and MAD of the block means between t_a and t_b: the cold-start baseline, "the same approach in the
        minutes before the event". Returns a Reference (source "pre-event") whose detail says how many seconds of
        samples stand behind it, or None when there are none."""
        zs = self._find(zone)
        if zs is None:
            return None
        block = float(block_sec or self.window)
        with self._lock:
            samples = self._slice(zs, t_a, t_b)
        if not samples:
            return None
        by_block = {}
        for s in samples:
            by_block.setdefault(int((s[0] - t_a) // block), []).append(s[1])
        means = np.array([_nanmean_cols(np.asarray(v, dtype=np.float64)) for v in by_block.values()])
        med, mad, n = {}, {}, {}
        for i, q in enumerate(QUANTITIES):
            col = means[:, i][~np.isnan(means[:, i])]
            if col.size:
                med[q] = float(np.median(col))
                mad[q] = float(np.median(np.abs(col - med[q])))
                n[q] = int(col.size)
        return Reference("pre-event", med, mad, n,
                         {"seconds": len(samples) * self.period, "blocks": len(by_block), "t0": t_a, "t1": t_b})

    def _transits(self, zs, t0, t1, exclude):
        """Per vehicle with a known position in [t0, t1]: when and where it was first and last seen, and the longest it
        stood still. FILTERING_CLASSES and `exclude` are left out. None when no vehicle qualifies."""
        with self._lock:
            samples = [s for s in self._slice(zs, t0, t1) if s[2]]
        if not samples:
            return None
        tab = np.concatenate([s[3] for s in samples])
        ts = np.concatenate([np.full(s[2], s[0]) for s in samples])
        keep = (tab[:, FILT] < 0.5) & ~np.isnan(tab[:, Y])
        if exclude:
            keep &= ~np.isin(tab[:, TID], np.asarray(sorted(exclude), dtype=np.float64))
        if not keep.any():
            return None
        tid, ts, y, wait = tab[keep, TID], ts[keep], tab[keep, Y], tab[keep, WAIT]
        order = np.lexsort((ts, tid))
        tid, ts, y, wait = tid[order], ts[order], y[order], wait[order]
        first = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
        last = np.r_[first[1:] - 1, tid.size - 1]
        return {"tid": tid[first], "t_in": ts[first], "t_out": ts[last], "y_in": y[first], "y_out": y[last],
                "wait": np.maximum.reduceat(wait, first), "t_end": samples[-1][0]}

    def transits(self, zone, t0=None, t1=None, exclude=()):
        """One record per vehicle that was on the approach in [t0, t1] (default: the last `window` seconds):
        {tid, entry_t, exit_t, seconds, distance_m, inside, longest_wait_s}. Times are video seconds of the first and
        last sample, `distance_m` is the net displacement along the road, `inside` says it was still there at the
        last sample. Samples are a second apart, so entry and exit are good to about a second - which is why travel
        time is always formed from the same stretch of time and distance, never from the zone's edges."""
        zs = self._find(zone)
        tr = None if zs is None else self._transits(zs, t0, t1, exclude)
        if tr is None:
            return []
        return [{"tid": int(tid), "entry_t": float(a), "exit_t": float(b), "seconds": float(b - a),
                 "distance_m": float(abs(yb - ya)), "inside": bool(b >= tr["t_end"] - 1e-9),
                 "longest_wait_s": float(w)}
                for tid, a, b, ya, yb, w in zip(tr["tid"], tr["t_in"], tr["t_out"], tr["y_in"], tr["y_out"],
                                                tr["wait"])]

    def excess_delay(self, zone, t0, t1, base_kph, exclude=(), held_s=3.0):
        """Delay, in vehicle-seconds, that vehicles on the approach suffered in [t0, t1] compared with travelling
        the same distance at `base_kph` (the baseline space-mean speed).

        Per vehicle: elapsed time between its first and last sample in the window, minus the distance it covered in
        that time divided by the baseline speed. Both are measured over the same stretch, so a vehicle that was
        already in the queue when the window opened, or is still in it when it closes, counts exactly the time it
        spent there ("vehicles still inside count time so far"). Distance is the net displacement along the road,
        not the path length: position jitter on a standing vehicle must not look like travel. Per-vehicle values are
        summed with their sign, because a clipped sum would add the positive half of the measurement noise of every
        free-flowing vehicle and report a delay where there is none; only the total is floored at 0.

        Returns {"veh_s", "raw_veh_s", "vehicles", "affected"} or None without a baseline speed; `affected` is the
        number of vehicles that stood still for at least `held_s` - a count of vehicles that were actually held up,
        not of those slower than the mean, because in mixed traffic a rickshaw is "slower than the baseline" all the
        time. FILTERING_CLASSES and `exclude` are left out."""
        zs = self._find(zone)
        if zs is None or not finite(base_kph) or base_kph <= 0:
            return None
        tr = self._transits(zs, t0, t1, exclude)
        if tr is None:
            return {"veh_s": 0.0, "raw_veh_s": 0.0, "vehicles": 0, "affected": 0}
        span, dist = tr["t_out"] - tr["t_in"], np.abs(tr["y_out"] - tr["y_in"])
        seen = span > 0
        raw = float((span - dist / (base_kph / 3.6))[seen].sum())
        return {"veh_s": max(0.0, raw), "raw_veh_s": raw, "vehicles": int(seen.sum()),
                "affected": int((tr["wait"][seen] >= held_s).sum())}


# ---------------------------------------------------------------- the time-of-day baseline

class _Reservoir:
    """Bounded uniform sample of the values seen, that slowly forgets (algorithm R with a floor on the keep rate)."""
    __slots__ = ("vals", "n", "days", "last_day")

    def __init__(self, vals=None, n=0, days=0, last_day=0):
        self.vals, self.n, self.days, self.last_day = vals or [], n, days, last_day

    def add(self, x, size, history, day, rng):
        self.n += 1
        if len(self.vals) < size:
            self.vals.append(x)
        else:
            # i-th value is kept with probability size / min(i, history): uniform until `history` values have
            # been seen, then a fixed rate, which makes the oldest values fade out exponentially
            if rng.random() * min(self.n, history) < size:
                self.vals[rng.randrange(size)] = x
        if day != self.last_day:
            self.days, self.last_day = self.days + 1, day

    def to_json(self):
        return {"n": self.n, "days": self.days, "last_day": self.last_day, "v": [round(v, 4) for v in self.vals]}

    @classmethod
    def from_json(cls, d):
        return cls([float(v) for v in d.get("v", ())], int(d.get("n", 0)), int(d.get("days", 0)),
                   int(d.get("last_day", 0)))


SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_baseline (
    camera_id INTEGER NOT NULL,
    zone TEXT NOT NULL,
    slot INTEGER NOT NULL,           -- 15-minute slot of the day, 0..95
    daytype TEXT NOT NULL,           -- weekday / weekend
    data TEXT NOT NULL,              -- {quantity: {n, days, last_day, v: [reservoir]}}
    updated_at TEXT,
    PRIMARY KEY (camera_id, zone, slot, daytype)
);
"""


def ensure_schema(store):
    """Create the baseline table (CREATE TABLE IF NOT EXISTS: safe to call on every start)."""
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


class Baseline:
    """Typical traffic per approach, per time of day, learned from the collector's own window means."""

    def __init__(self, store=None, size=64, history=640, min_blocks=12, min_days=2, slot_minutes=15,
                 weekend=(4, 5), flush_sec=300.0, max_cached=512, seed=0):
        """size: values kept per quantity per slot. history: after this many values the reservoir forgets at a fixed
        rate (640 blocks is about three weeks of a slot observed all the time, 30 blocks a day). min_blocks /
        min_days: what a slot needs before it counts as a baseline at all. weekend: weekday() numbers, Friday=4 and
        Saturday=5."""
        self.store, self.size, self.history = store, size, history
        self.min_blocks, self.min_days, self.slot_minutes = min_blocks, min_days, slot_minutes
        self.weekend, self.flush_sec, self.max_cached = tuple(weekend), flush_sec, max_cached
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._cache = OrderedDict()           # key -> {quantity: _Reservoir}
        self._dirty = set()
        self._last_flush = time.monotonic()
        if store is not None:
            ensure_schema(store)

    def slot(self, wall):
        """(slot of the day, "weekday" | "weekend") of an epoch time, in local time."""
        d = datetime.fromtimestamp(wall)
        return (d.hour * 60 + d.minute) // self.slot_minutes, ("weekend" if d.weekday() in self.weekend else "weekday")

    def _entry(self, key, create):
        e = self._cache.get(key)
        if e is not None:
            self._cache.move_to_end(key)
            return e
        if self.store is not None:
            rows = self.store.q("SELECT data FROM traffic_baseline WHERE camera_id=? AND zone=? AND slot=? AND "
                                "daytype=?", key)
            if rows:
                e = {q: _Reservoir.from_json(d) for q, d in json.loads(rows[0]["data"]).items()}
        if e is None:
            if not create:
                return None
            e = {}
        self._evict()                         # make room BEFORE inserting: the new entry is not yet marked dirty
        self._cache[key] = e
        return e

    def _evict(self):
        while len(self._cache) >= self.max_cached:
            victim = next((k for k in self._cache if k not in self._dirty), None)   # least recently used, unsaved last
            if victim is None:
                self.flush()
                victim = next(iter(self._cache))
            del self._cache[victim]

    def add(self, camera_id, zone, wall, means):
        """One completed block: `means` {quantity: window mean or None}, at clock time `wall` (epoch seconds)."""
        slot, daytype = self.slot(wall)
        key = (int(camera_id), str(zone), slot, daytype)
        day = datetime.fromtimestamp(wall).toordinal()
        with self._lock:
            e = self._entry(key, True)
            for q, x in means.items():
                if q in WORSE and finite(x):
                    e.setdefault(q, _Reservoir()).add(float(x), self.size, self.history, day, self._rng)
            self._dirty.add(key)
            if time.monotonic() - self._last_flush > self.flush_sec:
                self.flush()

    def stats(self, camera_id, zone, wall):
        """The baseline for this approach at this clock time, or None when the slot has too little behind it."""
        key = (int(camera_id), str(zone), *self.slot(wall))
        with self._lock:
            e = self._entry(key, False)
            if not e:
                return None
            med, mad, n = {}, {}, {}
            for q, r in e.items():
                if len(r.vals) >= self.min_blocks and r.days >= self.min_days:
                    v = np.asarray(r.vals)
                    med[q] = float(np.median(v))
                    mad[q] = float(np.median(np.abs(v - med[q])))
                    n[q] = int(r.n)
            if not med:
                return None
            days = max(r.days for r in e.values())
        return Reference("history", med, mad, n, {"slot": key[2], "daytype": key[3], "days": days})

    def flush(self):
        """Write every slot that changed since the last flush to the store (one transaction)."""
        with self._lock:
            if self.store is not None and self._dirty:
                stamp = datetime.now().isoformat(timespec="seconds")
                rows = [(*k, json.dumps({q: r.to_json() for q, r in self._cache[k].items()}), stamp)
                        for k in self._dirty if k in self._cache]
                with self.store.lock:
                    self.store.db.executemany("INSERT OR REPLACE INTO traffic_baseline (camera_id, zone, slot, "
                                              "daytype, data, updated_at) VALUES (?,?,?,?,?,?)", rows)
                    self.store.db.commit()
            self._dirty.clear()
            self._last_flush = time.monotonic()
