"""V9: a vehicle that may not use a bus lane, driving along it. The lanes are marked in the zone editor ("Bus lane"),
optionally with the hours in which they apply. Only buses are allowed by default; "allowed_classes" widens that
(add "Leguna" to let human-haulers use the lane, as for V1).

Being inside the polygon is not the offence. Dhaka vehicles cross the kerb lane all the time - to turn into a side
street, to reach a stop, to pick up or set down (CNGs, rickshaws), to pass a stopped vehicle - and that is legal.
Only USING the lane as a lane is the offence, so a vehicle is reported when one unbroken stretch inside the lane
covers BOTH:

* min_travel_m = 25 m along the road. A crossing covers the lane's width (3-3.5 m) at a shallow angle: even at 20
  degrees to the road that is about 10 m along it, and pulling in to the kerb adds a few car lengths. 25 m is more
  than twice that, yet a vehicle that drives the lane is still caught within a few seconds. It is measured along
  the road (road-plane y) between the first position in the lane and the latest, so sideways movement adds nothing
  and standing jitter adds nothing.
* min_sec = 4 s. At city speeds the distance is the binding limit (25 m at 20-30 km/h takes 3-4.5 s); at 60 km/h
  the time is. A fast vehicle that only clips the end of a lane drawn up to a junction mouth is inside for a
  second or two and is never reported.

Positions are the road-plane metres of the road-contact point, median-smoothed over 5 samples (xy, else xy_coarse:
the far end where positions are too coarse for speed is still good for tens of metres). A jump faster than max_kph
(120, as V4) AND longer than jump_m (6 m) means the tracker gave the ID to another vehicle: the stretch starts over.
The length condition matters because boxes at the far end jitter by metres from frame to frame, which alone must not
restart it. When there are no metres at all (the lane reaches outside the road zones, or the box is cut off by the
frame edge) the decision falls back to time alone and needs unmeasured_sec = 8 s, the time 25 m takes at a crawl of
11 km/h, because time alone cannot tell driving from standing; details["measured_by"] says which was used. A vehicle
that merely stands in the lane is not "driving along" it: waiting at a kerb is V3, standing in a running lane V7.

A jam is not an offence. When every other lane is stopped, drivers fill whatever space there is, the bus lane
included, and a notice would punish the jam. Like Scene.flowing_others, traffic is flowing when 3+ wheelers of the
same road zone move at >= flow_kph; unlike it, vehicles inside a bus lane do not count (they are the ones being
judged, and two cars cruising down an empty lane must not make a stopped road look as if it flowed). A moment is
jammed when at least jam_vehicles of the other vehicles are queueing (vehicles parked in a kerb strip or bay are
not) and none flows. The stretch is dropped when more than max_jam_share of it was jammed: 30%, the share the stand
rules allow for queueing. An empty road is not a jam. The price is that a car sneaking down an empty bus lane past
a standing queue is not reported.

A lane that applies only at some times has "hours" in the zone file, in the format of peak_hours (V3): ranges of
"HH:MM", which may cross midnight. Outside them the lane is just a lane and nobody is inside it. No hours = in
force all day. With hours but no wall-clock time (the recording's start time is unknown) the lane is enforced, as
V3 is, and the details say the hours were not checked; "enforce_if_time_unknown": false turns that round.

The class decides whether there is an offence, so it has to be reliable: a bus mislabelled as a truck would be
fined for using its own lane. min_purity is 0.75 (as V8 for look-alike classes). Junction mouths and U-turn
openings are where vehicles turn across the lane legally: entering one starts the stretch over (as V4).

Like the stand rules, one photo is taken once the vehicle is clearly driving in the lane (photo_after_sec), so the
case shows where it was before the moment it was reported."""

from collections import deque

import numpy as np

from .base import FILTERING_CLASSES, MOTORISED, NON_MOTORISED, BaseViolation
from .illegal_parking import in_peak_hours
from .stationary import MAX_GAP_SEC, stand_photo

VEHICLES = MOTORISED | NON_MOTORISED


class LaneRun:
    """One unbroken stretch of a vehicle inside a bus lane that is in force."""

    def __init__(self, start_s, smooth_samples):
        self.start_s = start_s                  # first sighting inside the lane
        self.last_in_s = start_s                # latest sighting inside the lane
        self.road = None                        # road zone the metres below were measured in
        self.raw = deque(maxlen=smooth_samples)  # latest measured positions (road-plane metres)
        self.first = self.last = None           # first / latest smoothed position
        self.last_pos_s = start_s
        self.n_pos = 0                          # positions measured so far
        self.jam_s = 0.0                        # seconds the other lanes of the road zone stood in a queue
        self.names = []                         # the lanes it was seen in
        self.photo = None                       # (time_s, JPEG) taken once it is clearly driving in the lane
        self.reported = False


class BusLaneViolation(BaseViolation):
    code = "V9"
    title = "Driving in a bus lane"
    DEFAULTS = {
        "allowed_classes": ["Bus"],  # add "Leguna" to also let human-haulers use the lane
        "min_travel_m": 25.0,        # along the road, inside the lane: more than twice a crossing manoeuvre
        "min_sec": 4.0,              # ...and this long: a fast vehicle clipping the end of the lane is not driving it
        "unmeasured_sec": 8.0,       # without metres, time alone: 25 m at a crawl of 11 km/h
        "min_purity": 0.75,          # the class decides: a bus read as a truck must not be fined
        "exit_grace_sec": 1.0,       # a gap in sightings / a step out of the polygon this short keeps the stretch
        "smooth_samples": 5,         # median of the latest positions: one jumpy box does not make metres
        "max_kph": 120.0,            # a faster jump means the track ID switched to another vehicle: start over...
        "jump_m": 6.0,               # ...if it is also this long: far-end boxes jitter by metres, no vehicle jumps
        "jam_vehicles": 3,           # queueing vehicles in the other lanes before the road counts as jammed...
        "max_jam_share": 0.3,        # ...and at most this share of the stretch may have been jammed
        "photo_after_sec": 2.0,      # "entered" photo once it has driven in the lane this long
        "enforce_if_time_unknown": True,  # lane with hours, wall-clock time unknown: enforce (V3 does) and say so
    }
    TURNING_AREAS = {"intersection", "uturn"}   # turning across the lane is legal: start over
    NOT_QUEUEING_AREAS = {"bus_lane", "no_parking", "parking"}   # standing here is not a traffic queue

    def __init__(self, settings=None, camera=None):
        super().__init__(settings, camera)
        self.offenders = VEHICLES - set(self.cfg["allowed_classes"])
        self.lanes = [a for a in camera.areas if a.kind == "bus_lane"] if camera is not None else []
        self.runs = {}  # tid -> LaneRun

    def _lanes_here(self, obs, wall_clock):
        """[(lane, why in force)] for the bus lanes in force at the vehicle's road-contact point. `lane` is None when
        only the tag is known (an observation built without the camera's geometry)."""
        if "bus_lane" not in obs.areas:
            return []
        inside = [a for a in self.lanes if a.contains(obs.point)]
        if not inside:
            return [(None, "")]
        found = []
        for lane in inside:
            if not lane.hours:
                found.append((lane, ""))
            elif wall_clock is None:
                if self.cfg["enforce_if_time_unknown"]:
                    found.append((lane, "time of day unknown: hours not checked"))
            elif in_peak_hours(lane.hours, wall_clock):
                found.append((lane, "inside the lane's hours"))
        return found

    def _advance(self, run, obs):
        """Adds the vehicle's measured position to the stretch. False when it jumped further than any vehicle can
        drive (the tracker gave the ID to another vehicle)."""
        pos = obs.xy if obs.xy is not None else obs.xy_coarse
        if pos is None or obs.road is None:
            return True   # not measurable now: the time keeps running, the distance waits
        run.road = run.road or obs.road
        run.raw.append(np.float64(pos))
        p = np.median(np.float64(run.raw), axis=0)
        if run.last is not None:
            step = float(np.linalg.norm(p - run.last))
            if step > self.cfg["jump_m"] and step / max(1e-3, obs.time_s - run.last_pos_s) * 3.6 > self.cfg["max_kph"]:
                return False
        if run.first is None:
            run.first = p
        run.last, run.last_pos_s = p, obs.time_s
        run.n_pos += 1
        return True

    def _jammed(self, obs, ctx):
        """True when the other lanes of this road zone are standing in a queue (see the module notes)."""
        queueing = 0
        for o in ctx.scene.by_road.get(id(obs.road), ()):
            if o.tid == obs.tid or "bus_lane" in o.areas or o.cls in FILTERING_CLASSES:
                continue
            if o.kph >= ctx.scene.flow_kph:
                return False
            if not o.areas & self.NOT_QUEUEING_AREAS:
                queueing += 1
        return queueing >= self.cfg["jam_vehicles"]

    def update(self, obs, ctx):
        if obs.cls not in self.offenders or obs.purity < self.cfg["min_purity"] or obs.areas & self.TURNING_AREAS:
            self.runs.pop(obs.tid, None)
            return None
        t, run = obs.time_s, self.runs.get(obs.tid)
        lanes = self._lanes_here(obs, ctx.wall_clock)
        if not lanes:
            if run is not None and t - run.last_in_s > self.cfg["exit_grace_sec"]:
                del self.runs[obs.tid]
            return None
        if run is not None and (t - run.last_in_s > self.cfg["exit_grace_sec"]
                                or (run.road is not None and obs.road is not None and obs.road is not run.road)):
            run = None   # back after a longer absence, or on another road zone (other metres): a new stretch
        if run is None:
            run = self.runs[obs.tid] = LaneRun(t, self.cfg["smooth_samples"])
        if not self._advance(run, obs):
            run = self.runs[obs.tid] = LaneRun(t, self.cfg["smooth_samples"])
            self._advance(run, obs)
        dt = min(MAX_GAP_SEC, t - run.last_in_s)
        run.last_in_s = t
        if run.reported:
            return None   # decided: the stretch is only kept so that the vehicle is not judged again
        if self._jammed(obs, ctx):
            run.jam_s += dt
        for lane, _ in lanes:
            if lane is not None and lane.name not in run.names:
                run.names.append(lane.name)

        secs = t - run.start_s
        measured = run.n_pos >= 2
        metres = abs(float(run.last[1] - run.first[1])) if measured else None
        if (run.photo is None and secs >= self.cfg["photo_after_sec"]
                and (not measured or metres >= 0.25 * self.cfg["min_travel_m"])
                and ctx.frame is not None and getattr(ctx.frame, "size", 0)):
            run.photo = (t, stand_photo(ctx.frame, obs.bbox))
        if measured:
            if metres < self.cfg["min_travel_m"] or secs < self.cfg["min_sec"]:
                return None
        elif secs < self.cfg["unmeasured_sec"]:
            return None
        if run.jam_s > self.cfg["max_jam_share"] * secs:
            return None

        run.reported = True
        details = {}
        if measured:
            details["metres_in_lane"] = round(metres, 1)
        details["seconds_in_lane"] = round(secs, 1)
        if measured:
            details["avg_kph"] = round(metres / secs * 3.6, 1)
        if run.names:
            details["area"] = ", ".join(run.names)
        details["measured_by"] = "distance" if measured else "time"
        details["other_lanes_queued_s"] = round(run.jam_s, 1)
        timed = [(lane, why) for lane, why in lanes if lane is not None and lane.hours]
        if timed:
            details["hours"] = " / ".join(sorted({lane.hours_text for lane, _ in timed}))
            details["enforced_because"] = " / ".join(sorted({why for _, why in timed}))
        ev = self.event(obs, details, started_s=run.start_s)
        if run.photo is not None:
            ev.extra_frames.append(("entered", run.photo[0], run.photo[1]))
            run.photo = None   # the event owns it now
        return ev

    def forget(self, tid):
        self.runs.pop(tid, None)
