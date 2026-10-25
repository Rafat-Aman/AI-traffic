"""Recognising the same vehicle on a different camera (Layer 2 cross-camera linking).

Each camera's tracker numbers vehicles independently: track 7 on the Mirpur Road camera has nothing to do
with track 7 two junctions away. This turns those local tracks into city-level vehicle identities, so a
plate-indexed history, a journey across junctions, and the repeat-offender view are queries over one store
rather than separate per-camera logs.

Why this can work without a Re-ID network
    Matching happens in **metres on the road plane**, not in pixels, so it is the same physics at every
    camera. The surveillance pipeline already provides what that needs: a homography per road zone (so a
    position is metres across and along the carriageway), a speed in m/s, a wall clock on every frame, and a
    majority-voted vehicle class. What a tracklet adds is a colour signature taken from its clearest frames.

    The evidence per candidate pair is then:
      * class must be IDENTICAL - never Car to Pickup, Motorbike to Bicycle
      * transit time must match gap / speed, where the gap between two cameras is learned from the traffic
      * lateral position maps linearly between cameras (mirrored when they face opposite ways), also learned
      * speed and physical width must agree
      * colour: an HSV histogram and CIELAB mean/spread of the body - or, for a motorbike or bicycle, of the
        rider's clothing, which is what actually distinguishes two near-identical machines
    Each becomes a log-likelihood ratio fitted from the confident matches themselves, so the thresholds are
    not guessed; the assignment is then one-to-one and is repeated as the parameters are refitted.

    This is the approach of the team's two-camera footbridge study, reworked for N cameras and for the
    metric data this pipeline already produces, and without its ViT embedding - there that was only a
    tiebreaker, and there is no GPU here to pay for it.

Where the green CNG problem bites
    A large share of Dhaka's fleet is visually near-identical green CNG auto-rickshaws, so appearance alone
    is close to worthless for them. That is exactly why geometry carries the weight here and colour only
    adjusts it: two CNGs are separated by when and where they crossed, not by what they look like.

Two ways in
    online  - `match_online` links a tracklet as it leaves view, against tracklets other cameras produced
              in the last few minutes, using whatever parameters have been learned so far. Immediate, and
              wrong more often, because a vehicle still in view has no final speed yet.
    batch   - `link_all` reconsiders every tracklet in a window at once: it discovers the flows between
              cameras, fits the parameters, and solves a one-to-one assignment per flow. This is what
              corrects the online guesses, and is the result to quote.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

TWO_WHEEL = {"Motorbike", "Bicycle"}

# --- appearance -------------------------------------------------------------
LAB_SCALE = np.array([20.0, 8.0, 8.0, 10.0, 5.0, 5.0], np.float32)
CLEAR_FRAMES = 6            # crops kept per track: enough to average over, cheap to carry
MAX_OVERLAP = 0.15          # a crop with more of another vehicle in it is not this vehicle's colour
MIN_CROP_PX = 24

# --- matching ---------------------------------------------------------------
MAX_TERM = 8.0              # cap on any single cost term, so one bad cue cannot veto a pair
CALIB_ITERS = 4
# A route has to explain clearly more pairs by a fixed distance than a fixed delay can. This replaced a
# gate on the sharpness of the histogram peak, which was quietly biased against exactly the traffic this
# project runs on: when every vehicle is the same class every pair is a candidate, so the background rises
# and a real route scores few sigma. A single-class stream gave the correct 121 m route with 138 pairs
# against a delay's 68 - overwhelming evidence - and 4.1 sigma, and was thrown away.
MIN_ROUTE_RATIO = 1.5
MIN_ROUTE_PAIRS = 20
MIN_FLOW_SIGNIFICANCE = 0.0  # kept and reported, no longer a gate
ACCEPT_COST = 9.0
MIN_MARGIN = 1.0
MAX_GAP_M = 300.0
# Two cameras that are metres apart are the same camera. Without this floor, a dense stream of look-alikes
# produces a huge spurious peak near zero distance - periodic arrivals correlate at many offsets, and the
# strongest one is often the physically impossible one - which then outranks the real route and links every
# vehicle backwards.
MIN_GAP_M = 20.0
BIG = 1e3


# ---------------------------------------------------------------- appearance

def colour_region(crop, two_wheel):
    """The part of a crop whose colour identifies the vehicle: the rider's helmet and upper body on a
    two-wheeler (the machine itself is usually black), the body panels otherwise."""
    h, w = crop.shape[:2]
    if two_wheel:
        return crop[int(0.05 * h):int(0.55 * h), int(0.15 * w):int(0.85 * w)]
    return crop[int(0.2 * h):int(0.85 * h), int(0.12 * w):int(0.88 * w)]


def hsv_hist(bgr):
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 6:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return (h / (h.sum() + 1e-9)).astype(np.float32)


def lab_stats(bgr):
    """Mean and spread of the colour in CIELAB, scaled so the six numbers are comparable. Less sensitive to
    bin edges than a histogram, and the better cue for riders."""
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 6:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return (np.concatenate([lab.mean(0), lab.std(0)]) / LAB_SCALE).astype(np.float32)


def signature(crops, two_wheel):
    """One colour signature from several crops of the same vehicle: (hsv histogram, lab stats)."""
    hists, labs = [], []
    for c in crops:
        region = colour_region(c, two_wheel)
        h, l = hsv_hist(region), lab_stats(region)
        if h is not None:
            hists.append(h)
        if l is not None:
            labs.append(l)
    hist = np.mean(hists, 0).astype(np.float32) if hists else None
    if hist is not None:
        hist /= hist.sum() + 1e-9
    return hist, (np.median(labs, 0).astype(np.float32) if labs else None)


def colour_similarity(a, b):
    """(histogram, lab, embedding) similarity of two signatures; NaN where either side is missing."""
    nan = float("nan")
    hist = (float(np.sum(np.sqrt(a.hist * b.hist))) if a.hist is not None and b.hist is not None else nan)
    lab = (-float(np.linalg.norm(a.lab - b.lab)) if a.lab is not None and b.lab is not None else nan)
    emb = nan
    if getattr(a, "embed", None) is not None and getattr(b, "embed", None) is not None \
            and a.embed.shape == b.embed.shape:
        emb = float(np.dot(a.embed, b.embed))
    return hist, lab, emb


# ---------------------------------------------------------------- the tracklet

@dataclass
class Tracklet:
    """One vehicle's pass through one camera, summarised in road-plane metres and wall-clock seconds."""
    camera_id: int
    track_id: int
    cls: str
    t_cross: float                   # wall clock (seconds) when it crossed the zone's reference line
    lateral: float                   # metres across the carriageway at that moment
    speed: float                     # m/s along the road, from the whole track
    direction: str                   # "incoming" (toward the camera) / "outgoing" / "unknown"
    width: float = float("nan")      # physical width in metres
    extrap_m: float = 0.0            # metres the crossing had to be extrapolated (lost at the frame edge)
    zone: str = ""
    run_id: Optional[int] = None
    id: Optional[int] = None         # row id once stored
    hist: Optional[np.ndarray] = None
    lab: Optional[np.ndarray] = None
    embed: Optional[np.ndarray] = None     # appearance vector (surveillance.embed)
    plate: str = ""
    global_id: Optional[int] = None

    @property
    def group(self):
        return "2w" if self.cls in TWO_WHEEL else "other"

    @property
    def key(self):
        return (self.camera_id, self.direction)


# ---------------------------------------------------------------- flows

@dataclass
class Flow:
    """Traffic going from one camera's direction to another's, with the road distance between them."""
    name: str
    src: tuple                       # (camera_id, direction)
    dst: tuple
    gap_m: float = 0.0
    significance: float = 0.0
    params: dict = field(default_factory=dict)
    refused: set = field(default_factory=set)   # classes too alike to link on this flow


def transit_distances(src, dst, max_d=MAX_GAP_M):
    """(i, j, distance) for same-class pairs: how far a vehicle would have travelled at its average speed."""
    out = []
    for i, a in enumerate(src):
        for j, b in enumerate(dst):
            if a.cls != b.cls:
                continue
            dt = b.t_cross - a.t_cross
            if dt <= 0:
                continue
            d = dt * 0.5 * (a.speed + b.speed)
            if d < max_d:
                out.append((i, j, d))
    return out


def gap_peak(src, dst):
    """The distance at which same-class pairs pile up, and how far that peak stands above chance.

    Vehicles really travelling between two cameras all cover the same road distance, so a genuine route
    shows a sharp peak; unrelated cameras give a flat spread."""
    d = np.array([x[2] for x in transit_distances(src, dst)])
    if len(d) < 20:
        return 0.0, 0.0
    bins = np.arange(0, MAX_GAP_M + 2, 2.0)
    hist, _ = np.histogram(d, bins)
    smooth = np.convolve(hist, np.ones(3), "same")
    usable = bins[:-1] >= MIN_GAP_M                  # a peak closer than this is not two cameras
    if not usable.any() or not smooth[usable].any():
        return 0.0, 0.0
    k = int(np.flatnonzero(usable)[np.argmax(smooth[usable])])
    background = max(float(np.median(smooth[smooth > 0])), 1.0)
    return float(bins[k] + 1.0), float((smooth[k] - background) / math.sqrt(background))


def is_a_route(src, dst, gap, tol_t=0.6):
    """Does a fixed *distance* explain these pairs better than a fixed *delay*? Pure physics, and it is
    what separates a real route from a coincidence.

    Vehicles that really travel between two cameras all cover the same distance, so their time gaps scale
    as gap/speed - a fast one arrives sooner. Two unrelated streams that merely overlap in time produce a
    peak too, but there the *delay* is what is constant, and it does not care how fast anything was going.
    So the two hypotheses are fitted against each other and the pairs each explains are counted.

    Without this, a dense stream of look-alikes produces a stronger backwards peak than the real forward
    one - it did in testing, at 23 m against the true 121 m - and every vehicle is linked the wrong way.
    """
    rows = [(d / (0.5 * (src[i].speed + dst[j].speed)), 0.5 * (src[i].speed + dst[j].speed))
            for i, j, d in transit_distances(src, dst)]
    if len(rows) < 20:
        return True, 0, 0
    dt = np.array([r[0] for r in rows])
    v = np.array([r[1] for r in rows])
    route = int(np.sum(np.abs(dt - gap / v) < tol_t))            # fixed distance
    # best fixed delay, from the densest bin of the delay histogram
    hist, edges = np.histogram(dt, bins=np.arange(0, float(dt.max()) + 2 * tol_t, tol_t))
    delay = int(hist.max()) if len(hist) else 0
    return route >= delay, route, delay


def discover_flows(tracklets, min_significance=MIN_FLOW_SIGNIFICANCE,
                   log=lambda m: None):
    """Which camera-direction pairs actually carry traffic between them, found from the traffic itself.

    Nobody has to say how the cameras are laid out, which is the point: a city network is not going to be
    hand-described, and the junction between two cameras can change when a road is closed."""
    by = defaultdict(list)
    for t in tracklets:
        if t.direction != "unknown" and np.isfinite(t.t_cross) and t.speed > 0:
            by[t.key].append(t)
    keys = sorted(by)
    cands = []
    for a in keys:
        for b in keys:
            if a[0] == b[0]:                       # a camera cannot feed itself
                continue
            gap, sig = gap_peak(by[a], by[b])
            if sig <= 0:
                continue
            ok, route, delay = is_a_route(by[a], by[b], gap)
            if not ok:
                log(f"  ignoring camera {a[0]} {a[1]} -> camera {b[0]} {b[1]} at {gap:.0f} m: a fixed delay "
                    f"explains {delay} pairs against {route} for a fixed distance - overlapping traffic, "
                    f"not a route")
                continue
            if route < MIN_ROUTE_PAIRS or route < MIN_ROUTE_RATIO * max(delay, 1):
                continue
            # Strength is how much of the traffic a fixed distance actually explains, per vehicle present -
            # NOT the sharpness of the histogram peak. Sharpness ranks a narrow coincidence above a real
            # route: in testing a spurious backwards peak scored 12.5 sigma against the true flow's 8.6,
            # while explaining 88 pairs against its 232, and every vehicle was linked the wrong way.
            strength = route / max(1, min(len(by[a]), len(by[b])))
            cands.append((strength, sig, gap, a, b, route))
    cands.sort(key=lambda c: -c[0])
    used_src, used_dst, used_pairs, flows = set(), set(), set(), []
    for strength, sig, gap, a, b, route in cands:
        # Traffic cannot run both ways between the same two groups of vehicles, so once a direction is
        # taken its reverse is out: accepting both would link every vehicle backwards as well as forwards.
        if frozenset((a, b)) in used_pairs:
            continue
        if sig >= min_significance and a not in used_src and b not in used_dst:
            used_src.add(a)
            used_dst.add(b)
            used_pairs.add(frozenset((a, b)))
            flows.append(Flow(chr(65 + len(flows)), a, b, gap, sig))
            log(f"  flow {flows[-1].name}: camera {a[0]} {a[1]} -> camera {b[0]} {b[1]}, "
                f"{gap:.0f} m apart ({sig:.1f} sigma, explains {route} pairs, {strength:.1f} per vehicle)")
    return flows, by


# ---------------------------------------------------------------- cost model

class LLR:
    """How much a similarity score argues for a pair being the same vehicle.

    Fitted from the data itself: the distribution of a cue over confident matches against its distribution
    over the candidates that were rejected. A cue that does not separate the two is switched off rather than
    given a guessed weight."""

    def __init__(self, pos=None, neg=None, weight=1.0):
        self.ok = False
        if pos is None:
            return
        pos = np.asarray(pos, float)
        neg = np.asarray(neg, float)
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        if len(pos) < 8 or len(neg) < 20:
            return
        self.mp, self.sp = pos.mean(), max(pos.std(), 0.05)
        self.mn, self.sn = neg.mean(), max(neg.std(), 0.05)
        self.w = weight
        self.ok = self.mp > self.mn + 0.02      # the cue must point the right way to be used at all

    def cost(self, x):
        if not self.ok or not np.isfinite(x):
            return 0.0
        lp = -0.5 * ((x - self.mp) / self.sp) ** 2 - math.log(self.sp)
        ln = -0.5 * ((x - self.mn) / self.sn) ** 2 - math.log(self.sn)
        return float(np.clip(-(lp - ln) * self.w, -3.0, 4.0))

    def describe(self):
        return (f"match {self.mp:.2f}+-{self.sp:.2f} vs other {self.mn:.2f}+-{self.sn:.2f}"
                if self.ok else "off")


OFF = LLR()       # the "this cue is switched off" singleton used when a cue is missing entirely


def default_params(gap_m):
    return dict(gap=gap_m, t_sig0=0.4, t_sig1=0.1, l_a=0.0, l_b=-1.0, l_sig=1.0,
                v_mu=0.0, v_sig=0.3, w_mu=0.0, w_sig=0.3)


def pair_terms(a, b, P, llr=None):
    """Cost of claiming a and b are the same vehicle; None when the pair is impossible.

    Low is good. Each term is a squared residual in units of its own uncertainty, so they are comparable and
    can simply be added."""
    if a.cls != b.cls:                       # hard rule, never traded off against appearance
        return None
    dt = b.t_cross - a.t_cross
    if dt <= 0 or a.speed <= 0 or b.speed <= 0:
        return None
    v = 0.5 * (a.speed + b.speed)
    dt_pred = P["gap"] / v
    sig_t = math.sqrt(P["t_sig0"] ** 2 + (P["t_sig1"] * dt_pred) ** 2 +
                      (0.1 * (a.extrap_m + b.extrap_m) / v) ** 2)
    rt = (dt - dt_pred) / sig_t
    if abs(rt) > 6:
        return None
    rl = 0.0
    if np.isfinite(a.lateral) and np.isfinite(b.lateral):
        sig_l = P["l_sig"] * (1.5 if a.group == "2w" else 1.0)   # riders weave; cars hold a lane
        rl = (b.lateral - (P["l_a"] + P["l_b"] * a.lateral)) / sig_l
        if abs(rl) > 6:
            return None
    rv = (math.log(b.speed / a.speed) - P["v_mu"]) / P["v_sig"]
    rw = 0.0
    if np.isfinite(a.width) and np.isfinite(b.width) and a.width > 0 and b.width > 0:
        rw = (math.log(b.width / a.width) - P["w_mu"]) / P["w_sig"]

    def q(r):
        return min(0.5 * r * r, MAX_TERM)

    hist, lab, emb = colour_similarity(a, b)
    g = a.group
    llr = llr if llr is not None else defaultdict(LLR)
    t = dict(dt=dt, dt_pred=dt_pred, hist=hist, lab=lab, embed=emb,
             c_time=q(rt) + math.log(sig_t / P["t_sig0"]),
             c_lane=q(rl), c_speed=q(rv), c_size=q(rw),
             c_hist=llr[("hist", g)].cost(hist), c_lab=llr[("lab", g)].cost(lab),
             c_embed=llr[("embed", g)].cost(emb))
    t["c_motion"] = t["c_time"] + t["c_lane"] + t["c_speed"] + t["c_size"]
    t["c_colour"] = t["c_hist"] + t["c_lab"] + t["c_embed"]
    t["cost"] = t["c_motion"] + t["c_colour"]
    return t


def solve(cost):
    """One-to-one assignment minimising total cost. Returns [(row, col)]."""
    import lap
    n = cost.shape[0]
    _, x, _ = lap.lapjv(np.ascontiguousarray(cost, np.float64), extend_cost=False)
    return [(i, int(x[i])) for i in range(n) if x[i] >= 0]


def assign(src, dst, P, llr=None, accept=ACCEPT_COST, with_margin=False):
    """Match two sets of tracklets one-to-one.

    Leaving a vehicle unmatched costs accept/2, so a pair is only taken when it beats leaving both sides
    alone - most vehicles at a junction never reach the other camera, and forcing them into pairs is how a
    matcher invents journeys. With `with_margin`, each match also reports how much the best total would
    worsen if that pair were banned: a small margin means two look-alikes arrived together and the choice
    between them was close."""
    n, m = len(src), len(dst)
    if not n or not m:
        return [], {}
    c = np.full((n, m), BIG)
    terms = {}
    for i, a in enumerate(src):
        for j, b in enumerate(dst):
            t = pair_terms(a, b, P, llr)
            if t is not None:
                terms[(i, j)] = t
                c[i, j] = min(t["cost"], BIG)
    ext = np.full((n + m, m + n), BIG)
    ext[:n, :m] = c
    ext[:n, m:] = np.where(np.eye(n) > 0, accept / 2, BIG)
    ext[n:, :m] = np.where(np.eye(m) > 0, accept / 2, BIG)
    ext[n:, m:] = 0.0
    pairs = solve(ext)
    total = sum(ext[i, j] for i, j in pairs)
    matches = [(i, j, terms[(i, j)]) for i, j in pairs if i < n and j < m and c[i, j] < accept]
    if with_margin:
        for i, j, t in matches:
            keep = ext[i, j]
            ext[i, j] = BIG
            t["margin"] = float(sum(ext[r, cc] for r, cc in solve(ext)) - total)
            ext[i, j] = keep
    return matches, terms


# ---------------------------------------------------------------- self-calibration

def bootstrap(flow, src, dst):
    """First guess at a flow's parameters, from the pairs that agree with the transit-distance peak.

    The lateral mapping is found by RANSAC rather than least squares: most of those pairs are wrong, and a
    fit that averages them would point nowhere."""
    P = default_params(flow.gap_m)
    pts = np.array([(src[i].lateral, dst[j].lateral)
                    for i, j, d in transit_distances(src, dst)
                    if abs(d - flow.gap_m) < 3 and np.isfinite(src[i].lateral)
                    and np.isfinite(dst[j].lateral)])
    if len(pts) >= 10:
        rng = np.random.default_rng(0)
        best = None
        for _ in range(500):
            i, j = rng.choice(len(pts), 2, replace=False)
            if abs(pts[i, 0] - pts[j, 0]) < 0.5:
                continue
            b = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
            if not 0.3 < abs(b) < 2:                      # a plausible lane-to-lane mapping
                continue
            a = pts[i, 1] - b * pts[i, 0]
            n = int(np.sum(np.abs(pts[:, 1] - (a + b * pts[:, 0])) < 1.0))
            if best is None or n > best[0]:
                best = (n, a, b)
        if best:
            P["l_a"], P["l_b"] = float(best[1]), float(best[2])
    return P


def refit(src, dst, matches, P):
    """Re-estimate the flow's parameters and the colour LLRs from the matches just made."""
    P = dict(P)
    if len(matches) >= 8:
        v = np.array([0.5 * (src[i].speed + dst[j].speed) for i, j, _ in matches])
        dt = np.array([dst[j].t_cross - src[i].t_cross for i, j, _ in matches])
        gaps = dt * v
        P["gap"] = float(np.median(gaps))
        resid = dt - P["gap"] / v
        P["t_sig0"] = float(max(0.25, min(3.0, 1.4826 * np.median(np.abs(resid - np.median(resid))))))
        la = np.array([(src[i].lateral, dst[j].lateral) for i, j, _ in matches])
        la = la[np.isfinite(la).all(1)]
        if len(la) >= 8 and np.ptp(la[:, 0]) > 0.5:
            b, a = np.polyfit(la[:, 0], la[:, 1], 1)
            P["l_a"], P["l_b"] = float(a), float(b)
            r = la[:, 1] - (a + b * la[:, 0])
            P["l_sig"] = float(max(0.3, min(3.0, 1.4826 * np.median(np.abs(r - np.median(r))))))
        lv = np.array([math.log(dst[j].speed / src[i].speed) for i, j, _ in matches])
        P["v_mu"], P["v_sig"] = float(np.median(lv)), float(max(0.1, min(0.6, lv.std())))
        lw = np.array([math.log(dst[j].width / src[i].width) for i, j, _ in matches
                       if np.isfinite(src[i].width) and np.isfinite(dst[j].width)
                       and src[i].width > 0 and dst[j].width > 0])
        if len(lw) >= 8:
            P["w_mu"], P["w_sig"] = float(np.median(lw)), float(max(0.1, min(0.6, lw.std())))

    # colour: matched pairs are the positives, everything else considered is the background
    llr = defaultdict(LLR)
    matched = {(i, j) for i, j, _ in matches}
    pos, neg = defaultdict(list), defaultdict(list)
    for i, a in enumerate(src):
        for j, b in enumerate(dst):
            if a.cls != b.cls:
                continue
            hist, lab, emb = colour_similarity(a, b)
            where = pos if (i, j) in matched else neg
            where[("hist", a.group)].append(hist)
            where[("lab", a.group)].append(lab)
            where[("embed", a.group)].append(emb)
    for key in set(pos) | set(neg):
        llr[key] = LLR(pos.get(key), neg.get(key), weight=1.0)
    return P, llr


def unidentifiable(src, P, llr, ratio=0.5):
    """Classes whose vehicles cannot be told apart on this flow, whatever the matcher reports.

    Two conditions have to hold at once. The colour cue is uninformative for the group - which is the normal
    state of affairs for Dhaka's green CNG auto-rickshaws - and vehicles of that class arrive closer
    together than the uncertainty in their predicted transit time. When both hold, vehicle k and vehicle k+1
    are equally good explanations of the same arrival, and the assignment can be confidently, systematically
    wrong: in testing, a dense stream of identical vehicles produced links that were 0% correct while every
    individual pair looked unambiguous. Refusing the class is the only honest answer."""
    out = set()
    by_cls = defaultdict(list)
    for t in src:
        by_cls[t.cls].append(t.t_cross)
    for cls, times in by_cls.items():
        if len(times) < 4:
            continue
        group = "2w" if cls in TWO_WHEEL else "other"
        # .get, not [], so a plain dict works here as well as the defaultdict the matcher passes
        if any(llr.get((cue, group), OFF).ok for cue in ("hist", "lab", "embed")):
            continue                                   # some appearance cue can still separate them
        headway = float(np.median(np.diff(sorted(times))))
        sig_t = P["t_sig0"] + P["t_sig1"] * P["gap"] / 8.0    # at a typical 8 m/s
        if headway > 0 and sig_t > ratio * headway:
            out.add(cls)
    return out


def match_flow(flow, src, dst, accept=ACCEPT_COST, min_margin=MIN_MARGIN, log=lambda m: None):
    """Match one flow, refitting its parameters from its own confident matches a few times."""
    P = bootstrap(flow, src, dst)
    llr = defaultdict(LLR)
    matches = []
    for it in range(CALIB_ITERS + 1):
        last = it == CALIB_ITERS
        matches, _ = assign(src, dst, P, llr, accept, with_margin=last)
        if last:
            break
        P, llr = refit(src, dst, matches, P)
    kept = [m for m in matches if m[2].get("margin", min_margin) >= min_margin]
    refused = unidentifiable(src, P, llr)
    if refused:
        before = len(kept)
        kept = [m for m in kept if src[m[0]].cls not in refused]
        log(f"  flow {flow.name}: dropped {before - len(kept)} link(s) of {', '.join(sorted(refused))} - "
            f"look-alike vehicles arriving closer together than the timing can resolve")
    flow.params = P
    flow.refused = refused
    log(f"  flow {flow.name}: {len(src)}x{len(dst)} candidates -> {len(kept)} linked "
        f"({len(matches) - len(kept)} ambiguous or unidentifiable), transit {P['gap']:.0f} m / speed")
    return kept, P


# ---------------------------------------------------------------- global identities

class Union:
    """Union-find: merges pairwise links into one identity per vehicle across any number of cameras."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return self.find(a)

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return out


def link_all(tracklets, accept=ACCEPT_COST, min_margin=MIN_MARGIN, log=lambda m: None):
    """Give every tracklet a global vehicle id. Returns (links, groups).

    `links` is [(src tracklet, dst tracklet, terms)]; `groups` maps a global id to its tracklets. A vehicle
    seen by three cameras becomes one identity because A-B and B-C are merged transitively, which is what
    makes this a city-level identity rather than a set of pairwise guesses."""
    flows, by = discover_flows(tracklets, log=log)
    links = []
    for flow in flows:
        src, dst = by[flow.src], by[flow.dst]
        matches, _ = match_flow(flow, src, dst, accept, min_margin, log=log)
        links += [(src[i], dst[j], t) for i, j, t in matches]

    u = Union()
    for t in tracklets:
        u.find(id(t))
    for a, b, _ in links:
        u.union(id(a), id(b))
    by_id = {id(t): t for t in tracklets}
    groups = {}
    for gid, (root, members) in enumerate(sorted(u.groups().items(), key=lambda kv: -len(kv[1])), start=1):
        group = [by_id[m] for m in members]
        for t in group:
            t.global_id = gid
        groups[gid] = sorted(group, key=lambda t: t.t_cross)
    multi = sum(1 for g in groups.values() if len(g) > 1)
    log(f"  {len(tracklets)} tracklets -> {len(groups)} vehicles, {multi} seen by more than one camera")
    return links, groups


def match_online(new, recent, params_by_flow, accept=ACCEPT_COST, margin=MIN_MARGIN):
    """Best link for one tracklet that has just left view, against what other cameras saw recently.

    Deliberately strict: it only answers when the best candidate beats the runner-up by `margin`, because a
    provisional link shown on a dashboard is worse than no link, and `link_all` will revisit it anyway."""
    best = None
    second = BIG
    for other in recent:
        if other.camera_id == new.camera_id:
            continue
        a, b = (other, new) if other.t_cross <= new.t_cross else (new, other)
        P = params_by_flow.get((a.key, b.key))
        if P is None:
            continue
        t = pair_terms(a, b, P)
        if t is None or t["cost"] >= accept:
            continue
        if best is None or t["cost"] < best[2]["cost"]:
            second = best[2]["cost"] if best else second
            best = (a, b, t)
        else:
            second = min(second, t["cost"])
    if best is None or second - best[2]["cost"] < margin:
        return None
    return best


# ---------------------------------------------------------------- collecting tracklets online

class TrackletCollector:
    """Turns the pipeline's per-frame output into one tracklet per vehicle, as it is tracked.

    Runs alongside the detector on every processed frame, so it has to be cheap: it keeps a handful of
    road-plane samples and at most `keep` small crops per vehicle, and does the arithmetic once, when the
    vehicle leaves. Nothing here touches the violation path.

    `on_done(Tracklet)` is called for each vehicle that has left the view with enough of a track to be
    worth storing.
    """

    def __init__(self, camera_id, on_done, keep=CLEAR_FRAMES, min_samples=5, lost_sec=1.5,
                 sample_sec=0.2, max_crop_px=128, embed=True):
        self.camera_id, self.on_done, self.embed = camera_id, on_done, embed
        self.keep, self.min_samples, self.lost_sec = keep, min_samples, lost_sec
        self.sample_sec, self.max_crop = sample_sec, max_crop_px
        self.samples = defaultdict(list)     # tid -> [(t, x_m, y_m)]
        self.crops = defaultdict(list)       # tid -> [(score, jpeg-free BGR crop)]
        self.meta = {}                       # tid -> dict of the latest per-track facts
        self.last_crop = {}
        self.seen = {}
        self.done = set()

    def observe(self, tid, frame, box, cls, road, xy, kph, heading, t, wall_t, y_ref, boxes=()):
        """One frame of one tracked vehicle. `xy` is road-plane metres (None when not measurable)."""
        self.seen[tid] = t
        if tid in self.done or road is None:
            return
        m = self.meta.setdefault(tid, {})
        m.update(cls=cls, zone=road.name, y_ref=y_ref, wall_at=wall_t, t_at=t)
        m.setdefault("width_samples", [])
        if heading:
            m["heading"] = heading
        if xy is not None:
            self.samples[tid].append((t, float(xy[0]), float(xy[1])))
            w = self._width(road, box)
            if w is not None:
                m["width_samples"].append(w)
        if t - self.last_crop.get(tid, -1e9) >= self.sample_sec:
            self.last_crop[tid] = t
            self._keep_crop(tid, frame, box, boxes)

    @staticmethod
    def _width(road, box):
        """Physical width in metres, from the two bottom corners of the box on the road plane."""
        x1, _, x2, y2 = box
        try:
            left, right = road.to_metres((x1, y2)), road.to_metres((x2, y2))
        except Exception:
            return None
        w = abs(float(right[0]) - float(left[0]))
        return w if 0.2 < w < 6.0 else None      # outside that, the box bottom was not on the road

    def _keep_crop(self, tid, frame, box, boxes):
        """Keep the clearest crops: large, inside the frame, and not overlapping another vehicle."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        if x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3:
            return
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or min(x2 - x1, y2 - y1) < MIN_CROP_PX:
            return
        overlap = 0.0
        for o in boxes:
            if o is box or tuple(o) == tuple(box):
                continue
            iw = min(x2, o[2]) - max(x1, o[0])
            ih = min(y2, o[3]) - max(y1, o[1])
            if iw > 0 and ih > 0:
                overlap = max(overlap, iw * ih / area)
        if overlap > MAX_OVERLAP:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        scale = min(1.0, self.max_crop / max(crop.shape[:2]))   # stored small: only colour is read from it
        if scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA)
        best = self.crops[tid]
        best.append((area * (1 - overlap), crop))
        best.sort(key=lambda b: -b[0])
        del best[self.keep:]

    def tick(self, t, present):
        """Finish the vehicles that have left the view."""
        for tid in list(self.seen):
            if tid in present or tid in self.done:
                continue
            if t - self.seen[tid] >= self.lost_sec:
                self._finish(tid)

    def finish_all(self):
        for tid in list(self.seen):
            if tid not in self.done:
                self._finish(tid)

    def _finish(self, tid):
        self.done.add(tid)
        tl = self.build(tid)
        self._forget(tid)
        if tl is not None:
            self.on_done(tl)

    def _forget(self, tid):
        for d in (self.samples, self.crops, self.meta, self.last_crop, self.seen):
            d.pop(tid, None)

    def build(self, tid):
        """The finished tracklet, or None when the vehicle was never measured well enough to be useful."""
        pts = self.samples.get(tid) or []
        m = self.meta.get(tid) or {}
        if len(pts) < self.min_samples or "y_ref" not in m:
            return None
        a = np.asarray(pts, float)
        t, x, y = a[:, 0], a[:, 1], a[:, 2]
        if np.ptp(t) < 0.3:
            return None
        vy = float(np.polyfit(t, y, 1)[0])
        vx = float(np.polyfit(t, x, 1)[0])
        speed = math.hypot(vx, vy)
        if speed < 0.3:                       # parked or queued: it has no transit time to match on
            return None
        y_ref = m["y_ref"]
        # when it crossed the reference line, extrapolating along its own fitted motion if it never got there
        if abs(vy) < 1e-6:
            return None
        t_cross = float(t[0] + (y_ref - y[0]) / vy)
        inside = t.min() - 0.5 <= t_cross <= t.max() + 0.5
        extrap = 0.0 if inside else abs(y_ref - (y[-1] if t_cross > t.max() else y[0]))
        if extrap > 40.0:
            return None
        lateral = float(np.polyval(np.polyfit(t, x, 1), t_cross))
        widths = m.get("width_samples") or []
        two_wheel = m.get("cls") in TWO_WHEEL
        kept = [c for _, c in self.crops.get(tid, ())]
        hist, lab = signature(kept, two_wheel)
        heading = m.get("heading", 0)
        tl = Tracklet(
            camera_id=self.camera_id, track_id=tid, cls=m.get("cls", ""),
            t_cross=m["wall_at"] - (m["t_at"] - t_cross) if m.get("wall_at") is not None else t_cross,
            lateral=lateral, speed=speed,
            direction="incoming" if heading > 0 else "outgoing" if heading < 0 else "unknown",
            width=float(np.median(widths)) if widths else float("nan"),
            extrap_m=extrap, zone=m.get("zone", ""), hist=hist, lab=lab)
        if self.embed:
            from .embed import Embedder
            tl.embed = Embedder.encode(kept)
        tl.video_t = t_cross
        return tl
