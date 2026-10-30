"""V10: an abrupt swerve at real speed - a CANDIDATE for the vision-language model, never a citation on its own.

The report's Table 3.1 (#7, "unsafe or erratic manoeuvre") is decided by the VLM, not by geometry: whether a swerve was
reckless or a driver avoiding a pothole is something only the photographs show. This rule only nominates vehicles whose
road-plane track looks like an evasive swerve, so every event carries decided_by="vlm" and needs_review=True.

Dhaka traffic is dense, mixed and not lane-disciplined: lateral drift, weaving rickshaws and filtering motorbikes are
normal. The rule is therefore built around what normal driving cannot do, in terms that physics and the measured noise
justify, and it reads only precise positions (obs.xy; the far end of a zone, xy_coarse, is never used).

What is measured. The last window_sec = 2 s of (t, x, y) is fitted by least squares - x (across the road) with a cubic,
y (along it) with a quadratic - and everything below is read off the fitted curve, never off differences of raw
positions (a second difference of 0.2 m box jitter at 15 fps is about 50 m/s^2). A cubic is the smallest polynomial that
can follow an S-shaped lane change or an out-and-back swerve. From the interior 80% of the fit (the ends of a fitted
curve are its noisiest part):

* lateral_m   = range of the fitted x. The no-movement floor is below 0.7 m even for 0.5 m box jitter (simulated).
* peak speed  = largest |dx/dt|. Floor with 0.3 m jitter at 15 fps: 1.0 m/s (99th percentile of 400 straight drives).
* implied acceleration = 2 V^2 / lateral_m, the peak lateral acceleration of a smooth (raised-cosine) path with that
  excursion and peak speed. The second derivative of the fit is NOT used: with 0.2 m jitter its error is about 1 m/s^2
  and its straight-driving 99th percentile 2.7 m/s^2, as large as the thresholds themselves. Excursion and peak speed
  are measured to about 0.1 m and 0.2 m/s, so the acceleration derived from them is usable.
* reversal    = the lateral velocity has both signs, each at least half the speed limit (out and back);
* heading change = range of atan2(dx/dt, |dy/dt|) over the window, in degrees.

The trigger is: speed >= min_kph AND lateral_m >= D AND peak speed >= V AND implied acceleration >= A AND (reversal OR
heading change >= min_heading_deg). Why each part:

* min_kph = 15 along the road. Below that a car stops in under 2 m, so nobody swerves to avoid a hazard, while sideways
  nosing in a crawling queue is routine; and at a crawl the heading is mostly noise. The speed is the fitted along-road
  speed, so a vehicle crossing the road is not "at speed".
* Reversal is the signature of an evasive swerve (out, then back). A single lateral movement at speed is an ordinary lane
  change however quick, so it needs a large heading change instead (a dart across the traffic, which only happens at low
  speed: 15 degrees at 40 km/h needs 3 m/s sideways). A fast cut-in at 60 km/h is deliberately NOT reported.
* Excursion D is more than the free play a vehicle has inside a 3.3 m lane (lane minus vehicle width), so wander within
  the lane can never reach it. Peak speed V and acceleration A are set by the vehicle class:

  heavy (Bus, Truck)      D 1.5 m  V 1.6 m/s  A 2.5 m/s^2  - rollover limit of a loaded bus or truck is about 0.3-0.4 g,
                                  so even 0.25 g sideways is emphatic for one, and 1.5 m is over half the free play.
  standard (everything else motorised)
                          D 2.0 m  V 2.2 m/s  A 3.5 m/s^2  - a 3.3 m lane change takes 3-4 s (1-2 m/s^2); this needs it
                                  in about 2 s, 0.35 g, well under the tyre limit (7-9 m/s^2), so real swerves clear it.
  agile (Motorbike, Bicycle, Rickshaw, Van, Wheelbarrow)
                          D 2.4 m  V 3.0 m/s  A 5.0 m/s^2  - two-wheelers lean and filter: +-0.8 m weaving with a 3 s
                                  period (V 1.7, A 3.5) is ordinary in Dhaka and must not be reported; only violent
                                  zig-zags exceed this.

Fit quality: a fit whose rms residual exceeds max_fit_rms_m (0.45 m) is not trusted, and a lateral jump of 1.5 m between
two samples faster than 8 m/s (no road vehicle moves sideways at 29 km/h) means the tracker gave the ID to another
vehicle: the history starts over. Junction mouths, U-turn openings, bus stops and parking bays are exempt (vehicles turn
and pull in there legally).

Limits, honestly: the thresholds come from kinematic arithmetic and simulated noise, not from labelled Dhaka footage, so
the false-positive rate on real video is unmeasured. A bend inside a zone makes every vehicle drift sideways (smoothly,
so it stays below the speed and acceleration limits). A swerve that is quicker than the 2 s window's cubic can follow is
under-estimated, which errs toward missing it. Evaluation runs every eval_every_sec, so cost is bounded."""

import math
from collections import deque

import numpy as np

from .base import MOTORISED, NON_MOTORISED, BaseViolation

HEAVY = {"Bus", "Truck"}
AGILE = {"Motorbike", "Bicycle", "Rickshaw", "Van", "Wheelbarrow"}


def class_group(cls):
    return "heavy" if cls in HEAVY else "agile" if cls in AGILE else "standard"


def _poly_deriv(c, u, k):
    out = np.zeros_like(u)
    for p in range(k, len(c)):
        out += c[p] * math.factorial(p) / math.factorial(p - k) * u ** (p - k)
    return out


def fit_swerve(t, x, y, margin=0.1, grid=21):
    """Fits the window and returns its kinematics (see the module notes). t, x, y: 1-D arrays, t increasing."""
    mid, h = 0.5 * (t[0] + t[-1]), 0.5 * (t[-1] - t[0])
    u = (t - mid) / h
    vx_ = np.vander(u, 4, increasing=True)
    cx = np.linalg.lstsq(vx_, x, rcond=None)[0]
    cy = np.linalg.lstsq(np.vander(u, 3, increasing=True), y, rcond=None)[0]
    rms = float(np.sqrt(np.mean((vx_ @ cx - x) ** 2)))
    ug = np.linspace(-1 + 2 * margin, 1 - 2 * margin, grid)
    xg = np.vander(ug, 4, increasing=True) @ cx
    vx = _poly_deriv(cx, ug, 1) / h
    vy = _poly_deriv(cy, ug, 1) / h
    psi = np.degrees(np.arctan2(vx, np.abs(vy) + 1e-6))
    peak = float(np.abs(vx).max())
    lateral = float(xg.max() - xg.min())
    return {"rms": rms, "lateral_m": lateral, "peak_mps": peak,
            "accel_mps2": 2 * peak * peak / max(lateral, 1e-6),
            "heading_deg": float(psi.max() - psi.min()),
            "vpos": float(vx.max()), "vneg": float(-vx.min()),
            "along_mps": float(np.abs(vy).mean()), "speed_mps": float(np.hypot(vx, vy).mean()),
            "t_grid": mid + ug * h, "vx": vx}


class ErraticManoeuvre(BaseViolation):
    code = "V10"
    title = "Unsafe or erratic manoeuvre"
    DEFAULTS = {
        "classes": sorted(MOTORISED | NON_MOTORISED),
        "min_kph": 15.0,             # along the road; below this nobody swerves to avoid a hazard
        "window_sec": 2.0,           # the fitted window
        "min_span_sec": 1.4,         # data must cover this much of it...
        "min_samples": 8,            # ...with at least this many positions...
        "max_gap_sec": 0.6,          # ...and no hole longer than this
        "eval_every_sec": 0.2,       # bounded cost: one fit per vehicle per 0.2 s
        "min_dt_sec": 0.03,          # positions closer in time than this are dropped (60 fps video)
        "min_purity": 0.5,           # the class only picks the threshold set
        "max_fit_rms_m": 0.45,       # a fit this poor is a tracking glitch, not a swerve
        "jump_m": 1.5, "jump_mps": 8.0,   # sideways jump of this size and speed = the ID went to another vehicle
        "min_heading_deg": 15.0,     # one-way movement must also turn the heading this much
        "reversal_share": 0.5,       # both lateral-velocity lobes of an out-and-back must reach this share of V
        "lateral_m_heavy": 1.5, "peak_mps_heavy": 1.6, "accel_mps2_heavy": 2.5,
        "lateral_m_standard": 2.0, "peak_mps_standard": 2.2, "accel_mps2_standard": 3.5,
        "lateral_m_agile": 2.4, "peak_mps_agile": 3.0, "accel_mps2_agile": 5.0,
    }
    EXEMPT_AREAS = {"intersection", "uturn", "bus_stop", "bus_stop_unapproved", "parking"}

    def __init__(self, settings=None, camera=None):
        super().__init__(settings, camera)
        self.classes = set(self.cfg["classes"])
        self.state = {}   # tid -> [road, deque[(t, x, y)], last_eval_s, reported]

    def limits(self, cls):
        g = class_group(cls)
        return g, self.cfg[f"lateral_m_{g}"], self.cfg[f"peak_mps_{g}"], self.cfg[f"accel_mps2_{g}"]

    def update(self, obs, ctx):
        if (obs.cls not in self.classes or obs.purity < self.cfg["min_purity"] or obs.road is None
                or obs.areas & self.EXEMPT_AREAS):
            self.state.pop(obs.tid, None)
            return None
        if obs.xy is None:
            return None   # unreliable position: skipped, not guessed
        cfg, t = self.cfg, obs.time_s
        st = self.state.get(obs.tid)
        if st is None or st[0] is not obs.road:
            st = self.state[obs.tid] = [obs.road, deque(), t, False]
        if st[3]:
            return None
        hist = st[1]
        x, y = float(obs.xy[0]), float(obs.xy[1])
        if hist:
            t0, x0, y0 = hist[-1]
            dt = t - t0
            if dt < cfg["min_dt_sec"]:
                return None
            if abs(x - x0) >= cfg["jump_m"] and abs(x - x0) / dt > cfg["jump_mps"]:
                hist.clear()   # the tracker gave the ID to another vehicle
        hist.append((t, x, y))
        while hist[0][0] < t - cfg["window_sec"]:
            hist.popleft()
        if t - st[2] < cfg["eval_every_sec"] or len(hist) < cfg["min_samples"]:
            return None
        st[2] = t
        if t - hist[0][0] < cfg["min_span_sec"]:
            return None
        a = np.array(hist)
        if np.diff(a[:, 0]).max() > cfg["max_gap_sec"]:
            return None
        group, d_min, v_min, a_min = self.limits(obs.cls)
        if np.ptp(a[:, 1]) < 0.6 * d_min:       # cheap gate: raw range far too small for any swerve
            return None
        f = fit_swerve(a[:, 0], a[:, 1], a[:, 2])
        if f["rms"] > cfg["max_fit_rms_m"] or f["along_mps"] * 3.6 < cfg["min_kph"]:
            return None
        if f["lateral_m"] < d_min or f["peak_mps"] < v_min or f["accel_mps2"] < a_min:
            return None
        share = cfg["reversal_share"] * v_min
        reversal = f["vpos"] >= share and f["vneg"] >= share
        if not reversal and f["heading_deg"] < cfg["min_heading_deg"]:
            return None
        st[3] = True
        onset = f["t_grid"][np.abs(f["vx"]) >= 0.25 * v_min]
        details = {"lateral_m": round(f["lateral_m"], 2), "window_s": round(float(a[-1, 0] - a[0, 0]), 2),
                   "peak_lateral_mps": round(f["peak_mps"], 2), "lateral_accel_mps2": round(f["accel_mps2"], 1),
                   "heading_change_deg": round(f["heading_deg"], 1), "reversal": bool(reversal),
                   "speed_kph": round(f["speed_mps"] * 3.6, 1), "class_group": group,
                   "fit_rms_m": round(f["rms"], 2), "decided_by": "vlm", "needs_review": True}
        return self.event(obs, details, started_s=float(onset[0]) if len(onset) else float(a[0, 0]))

    def forget(self, tid):
        self.state.pop(tid, None)
