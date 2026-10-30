"""Impact of a traffic event, how its weights are fitted and checked, and the priority shown to an operator.

The project report's central claim (SDP 3.1.4): for each event, compare the traffic state on the affected approach
during the event window with the baseline for that approach and time of day, over five quantities, and combine the
normalised departures as S = sum_i w_i * delta_i. "This design targets the excess vehicle-delay attributable to the
event, measured in vehicle-seconds ... Non-negative least squares then fits the weights. The non-negativity constraint
encodes a requirement: no component may reduce measured severity. Some event classes will carry too few annotated
instances to support a fit. There the system falls back to equal weights, and the report says so rather than
presenting an uncalibrated score as a calibrated one. Validation uses rank correlation." The state itself is measured
in surveillance/traffic_state.py; this module turns it into a score.

delta_i: one-sided, robust, in units of baseline noise
    delta_i = clip( worse_i * (during_i - median_i) / max(1.4826 * MAD_i, floor_i), 0, 10 ), where worse_i is +1 and
    -1 for speed. One-sided because S measures how much WORSE traffic is than normal: if improvements were credited,
    a quiet lane could cancel a jam in another component, which is the reverse of "no component may reduce measured
    severity" (and with non-negative weights it would make S negative, which means nothing to an operator). Robust
    (median / MAD) because the baseline is learned from a stream that contains the events being scored. The floor
    exists because a quantity that is always 0 in the baseline (a queue on a quiet road) has MAD 0, and without a
    floor the first parked car would be an infinite departure; the floors are judgement - one floor is "a small but
    noticeable change": 8 m of queue, 5 km/h, 5 points of occupancy, 5 veh/km/lane, 5 s of waiting. They set the unit of
    delta and so only matter before the weights are fitted. The cap stops one quantity from dominating.
    Speed and waiting time are means over vehicles, so with few vehicles they are mostly those vehicles' own
    variation: on an almost empty road ONE car stuck behind a stopped bus gives a mean speed of 0 and a mean wait of
    20 s, as extreme as a jam. Their delta is therefore scaled by n / (n + 3), n being the vehicles behind the mean
    (a shrinkage towards "no departure"). Queue, occupancy and density are totals, not means, and are not scaled.

Weights, and why the fallback is labelled
    Calibrated weights are fitted by non-negative least squares to the event's excess vehicle-delay, with no
    intercept: S = 0 must mean "nothing departed from the baseline", and then no delay can be predicted. A fitted S is
    therefore in vehicle-seconds. The fallback, equal weights 1/5, gives the mean delta, a unitless number that is only
    good for ranking events against each other. The two are not comparable, which is why every result carries
    `calibrated` and `weights_source`, and why `priority` has a separate band table for each. With the five
    quantities rising together (queue, occupancy, density and waiting all grow in a jam) NNLS usually gives some of
    them weight 0: individual weights are not a statement about which quantity "matters".
    A fit needs at least MIN_EVENTS (15) events of a code (three per weight); below that the code uses the pooled fit
    over all codes if there are 15 events in total, otherwise equal weights.

The target: delay caused to OTHER vehicles
    Per vehicle on the approach in the window, observed time minus distance / baseline space-mean speed, summed with
    sign (see TrafficStateCollector.excess_delay). The vehicle that is the event is left out of the state and of the
    delay: a bus standing on an empty road is stationary and so would otherwise look like the worst possible
    queue, waiting time and delay - made of itself.
    How good a target is it? Its error is dominated by the baseline speed, not by the integral: in the synthetic runs
    the traffic's own speed drifted by a factor of two between the minutes before an event and the event window, which
    moved the delay of a no-event control by -260 to +142 veh*s against event effects of 200-400. The integral itself is
    unbiased (the same control reads 5 veh*s against its own window's speed). So a single event's delay is a noisy
    number, quoted for ranking and for fitting - where the noise averages out and the signed value is used, since
    flooring at 0 would bias small events upwards - not as a measurement to the vehicle-second.

Baseline source, in order
    "history"    the time-of-day baseline for this camera, zone, 15-minute slot and weekday / weekend
    "pre-event"  cold start: the same approach in the four minutes before the window. It measures CHANGE, so it cannot
                 see a jam that was already there, and a second event inside the first one's window is compared with
                 traffic the first one disturbed.
    "none"       neither: the result says so, `meaningful` is False and `score` is None. A number would only mislead.

Validation
    `validate` is Spearman's rho between the score ordering and an independent severity ordering from human review,
    with a bootstrap interval and a permutation p-value because the sample will be small. NOTHING HAS BEEN VALIDATED
    YET: the only data so far is synthetic (tests/test_impact.py), which shows the machinery behaves; it says nothing
    about whether a real operator would agree with the ordering. The in-sample rho stored with a fit is optimistic.
    Fitted weights are only put to use when they win `compare_weights`: on the cases a reviewer graded, scored again
    under both weight sets from their stored departures, they must have a higher rho than equal weights, on at least
    MIN_GRADED cases. Anything less - too few graded cases, a tie, an ordering that cannot be correlated - keeps equal
    weights and says why. The fit saw excess delay and never a grade, so the grades are an independent check on it; the
    units a fit gives (vehicle-seconds) are not a reason to adopt it.

Limits
    * Synthetic only (see Validation). The floors, the shrinkage constant, the severity bands, the history cap and
      half-life, and the dangerous-driving floors in RULE_FLOOR are all judgement, put in so that the machinery has
      a defined behaviour; the report should present them as defaults to be set with the traffic authority.
    * The window is anchored to the moment the rule fires (`event_window`): a stop that lasts ten minutes is scored on
      its first dwell + 30 s, not on all ten. Re-schedule the case later if the whole duration matters.
    * A single event's excess delay is noisy (see "The target"); the score's five departures are steadier.
    * Intensive quantities are shrunk, extensive ones are not: with one victim on a near-empty road the score is
      still above a free-flow control (0.30 against 0.00 in the synthetic runs), by design - one vehicle was held up.
    * The five departures are strongly correlated, so a fit of five weights on a few dozen events is poorly
      conditioned; trust the fitted S for ranking and for units, not the individual weights.
    * Events close together share traffic: the later one's pre-event baseline contains the earlier one.

Priority
    Report: the priority combines the impact of the current event with the count and recency of the plate's
    confirmed prior events - "supporting context, never an automatic escalation trigger". The severity BAND comes from
    the impact alone (and, for a few dangerous-driving codes, a floor from the rule). History can only move a case
    within its band, by at most HISTORY_MAX_POINTS, so by construction it can never put a case above a high-impact
    first offence. A misread plate silently has no history; this is one more reason it is shown, not acted on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import traffic_state as ts
from .traffic_state import QUANTITIES, WORSE, clean

MAD_TO_SIGMA = 1.4826            # MAD -> standard deviation for normally distributed noise
SCALE_FLOOR = {"queue_m": 8.0, "speed_kph": 5.0, "occupancy": 0.05, "density": 5.0, "wait_s": 5.0}
DELTA_CAP = 10.0
PER_VEHICLE = ("speed_kph", "wait_s")   # means over vehicles: noisy when few vehicles stand behind them
SHRINK_N = 3.0                          # their delta is scaled by n / (n + 3), n = vehicles behind the mean
MIN_EVENTS = 15                  # events needed to fit the weights (of one code, or pooled)
MIN_GRADED = 10                  # reviewer-graded cases needed before fitted weights can be shown to beat equal ones
POST_SEC = 30.0                 # the event window runs this long after the rule fired
PRE_SEC = 240.0                  # cold-start baseline: this long before the window ...
PRE_GUARD_SEC = 2.0              # ... stopping just short of it
MIN_PRE_SEC = 60.0               # ... and it needs at least this much measured
MIN_WINDOW_SAMPLES = 5           # fewer seconds than this in the window is not a state

# priority bands: upper edges of "low" and "medium" in the unit of the score, per kind of score
SEVERITY_BANDS = {"calibrated": (30.0, 150.0),   # vehicle-seconds of excess delay (judgement until human labels exist)
                  "equal": (1.0, 3.0)}           # mean departure, in baseline-noise units
BAND_POINTS = {"low": (0.0, 33.0), "medium": (33.0, 66.0), "high": (66.0, 100.0)}
HISTORY_MAX_POINTS = 8.0         # the most that history can add (a quarter of a band)
HISTORY_COUNT_SCALE = 2.0        # 1 - exp(-count / 2): the 4th prior event adds little more than the 3rd
HISTORY_HALF_LIFE_DAYS = 60.0
# dangerous-driving codes are never shown as low priority just because the road happened to be quiet
RULE_FLOOR = {"V4": "medium", "V10": "medium", "V11": "medium"}
ORDER = ("low", "medium", "high")


# ---------------------------------------------------------------- the event window

def event_window(started_s, video_time_s, post_sec=POST_SEC):
    """(t0, t1) in video seconds: from when the offending behaviour began (the moment the rule fired, when the rule does
    not say) until `post_sec` after the rule fired, long enough to see the queue the event built."""
    t1 = float(video_time_s) + float(post_sec)
    t0 = float(video_time_s) if started_s is None else min(float(started_s), float(video_time_s))
    return t0, t1


# ---------------------------------------------------------------- weights

@dataclass(frozen=True)
class WeightSet:
    values: dict                          # quantity -> weight, all >= 0
    calibrated: bool
    source: str                           # "equal" | "pooled" | "code:V1"
    n_events: int = 0
    fit: dict = field(default_factory=dict)       # in-sample diagnostics: rmse, r2, spearman

    def score(self, deltas):
        return float(sum(self.values[q] * deltas.get(q, 0.0) for q in QUANTITIES))

    def to_dict(self):
        return {"values": dict(self.values), "calibrated": self.calibrated, "source": self.source,
                "n_events": self.n_events, "fit": dict(self.fit)}

    @classmethod
    def from_dict(cls, d):
        return cls({q: float(d["values"][q]) for q in QUANTITIES}, bool(d["calibrated"]), str(d["source"]),
                   int(d.get("n_events", 0)), dict(d.get("fit") or {}))


EQUAL_WEIGHTS = WeightSet({q: 1.0 / len(QUANTITIES) for q in QUANTITIES}, False, "equal")


@dataclass
class Weights:
    """The weights in force: one set per code that had enough events, a default for the rest."""
    default: WeightSet = EQUAL_WEIGHTS
    by_code: dict = field(default_factory=dict)
    min_events: int = MIN_EVENTS
    n_rows: int = 0
    skipped: int = 0

    def for_code(self, code):
        return self.by_code.get(code) or self.default

    def any_calibrated(self):
        return self.default.calibrated or any(w.calibrated for w in self.by_code.values())

    def to_dict(self):
        return {"default": self.default.to_dict(), "by_code": {c: w.to_dict() for c, w in self.by_code.items()},
                "min_events": self.min_events, "n_rows": self.n_rows, "skipped": self.skipped}

    @classmethod
    def from_dict(cls, d):
        return cls(WeightSet.from_dict(d["default"]), {c: WeightSet.from_dict(w) for c, w in d["by_code"].items()},
                   int(d.get("min_events", MIN_EVENTS)), int(d.get("n_rows", 0)), int(d.get("skipped", 0)))


# ---------------------------------------------------------------- non-negative least squares

def _least_squares_on(A, b, cols):
    s = np.zeros(A.shape[1])
    if cols.any():
        s[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
    return s


def nnls(A, b, max_iter=None):
    """min ||A x - b|| subject to x >= 0, by the Lawson-Hanson active-set method. Returns (x, residual norm).

    scipy.optimize.nnls does the same, but scipy is not a dependency of this project. The tests check this against
    brute force over every support (the optimum is the least-squares solution on some support whose coefficients are
    all >= 0, so with five columns all 32 supports can simply be tried)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    x = np.zeros(n)
    if m == 0 or n == 0:
        return x, float(np.linalg.norm(b))
    tol = 10 * max(m, n) * np.finfo(float).eps * np.linalg.norm(A, 1)
    passive = np.zeros(n, dtype=bool)                     # columns allowed to be positive
    w = A.T @ (b - A @ x)                                 # negative gradient
    limit = max_iter or 30 * n
    outer = 0
    while not passive.all() and w[~passive].max() > tol and outer < limit:
        outer += 1
        free = np.flatnonzero(~passive)
        passive[free[np.argmax(w[free])]] = True
        s = _least_squares_on(A, b, passive)
        inner = 0
        while passive.any() and s[passive].min() <= tol and inner < limit:
            inner += 1
            bad = passive & (s <= tol)
            den = x[bad] - s[bad]
            ratio = np.where(den > 0, x[bad] / np.where(den > 0, den, 1.0), 0.0)
            x = x + ratio.min() * (s - x)                 # walk to the boundary of the feasible region ...
            drop = passive & (x <= tol)
            if not drop.any():                            # (rounding: make sure something leaves)
                drop[np.flatnonzero(bad)[np.argmin(ratio)]] = True
            x[drop] = 0.0
            passive &= ~drop                              # ... and release the columns that hit zero
            s = _least_squares_on(A, b, passive)
        x = s
        w = A.T @ (b - A @ x)
    x = np.maximum(x, 0.0)                                # only matters if the iteration cap was hit
    return x, float(np.linalg.norm(A @ x - b))


def _fit(X, y, source):
    w, _ = nnls(X, y)
    if not np.any(w > 0):                                 # nothing explains the delay: not a calibration
        return None
    pred = X @ w
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return WeightSet({q: float(v) for q, v in zip(QUANTITIES, w)}, True, source, int(len(y)),
                     {"rmse_veh_s": clean(math.sqrt(float(((y - pred) ** 2).mean())), 3),
                      "r2": clean(1 - float(((y - pred) ** 2).sum()) / ss_tot, 3) if ss_tot > 0 else None,
                      "spearman": clean(spearman(pred, y), 3), "in_sample": True})


def _row(r):
    """(code, delta vector, target) from a compute_event_impact() result or a plain dict; None if unusable."""
    if r.get("meaningful") is False:
        return None
    d = r.get("deltas")
    # the SIGNED delay when there is one: the floored value is biased upwards for small events (noise that happens to
    # be negative is thrown away, noise that happens to be positive is kept), and a fit would learn that bias
    target = next((r[k] for k in ("excess_delay_signed_veh_s", "excess_delay_veh_s", "target")
                   if r.get(k) is not None), None)
    if d is None or not ts.finite(target):
        return None
    vec = [d[q] for q in QUANTITIES] if isinstance(d, dict) else list(d)
    if len(vec) != len(QUANTITIES) or not np.all(np.isfinite(vec)):
        return None
    return r.get("code"), np.asarray(vec, dtype=float), float(target)


def calibrate(rows, min_events=MIN_EVENTS):
    """Fit the weights from past events. `rows`: dicts with "code", "deltas" ({quantity: delta}) and
    "excess_delay_veh_s" - what compute_event_impact returns, or what calibration_rows(store) reads back.

    Per code with at least `min_events` events; otherwise pooled over every row if there are at least `min_events`
    in total; otherwise equal weights with calibrated=False. The Weights object says which applies to each code."""
    rows = list(rows)
    parsed = [p for p in map(_row, rows) if p is not None]
    out = Weights(min_events=min_events, n_rows=len(parsed), skipped=len(rows) - len(parsed))
    if len(parsed) >= min_events:
        X = np.vstack([p[1] for p in parsed])
        y = np.array([p[2] for p in parsed])
        pooled = _fit(X, y, "pooled")
        if pooled is not None:
            out.default = pooled
    codes = {}
    for code, vec, target in parsed:
        codes.setdefault(code, []).append((vec, target))
    for code, items in codes.items():
        if code is not None and len(items) >= min_events:
            ws = _fit(np.vstack([v for v, _ in items]), np.array([t for _, t in items]), f"code:{code}")
            if ws is not None:
                out.by_code[code] = ws
    return out


# ---------------------------------------------------------------- rank correlation

def ranks(x):
    """1-based ranks, tied values sharing the average of the ranks they span."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.r_[True, xs[1:] != xs[:-1]]
    first = np.flatnonzero(start)
    last = np.r_[first[1:], x.size] - 1
    out = np.empty(x.size)
    out[order] = ((first + last) / 2.0 + 1.0)[np.cumsum(start) - 1]
    return out


def spearman(a, b):
    """Spearman's rho: the Pearson correlation of the average ranks, which is the form that stays correct with ties.
    NaN when it is undefined (fewer than two pairs, or one side constant)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return math.nan
    ra, rb = ranks(a) - (a.size + 1) / 2.0, ranks(b) - (a.size + 1) / 2.0
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else math.nan


def validate(scores, human_severity, n_boot=2000, n_perm=5000, seed=0):
    """Agreement between the impact ordering and a human severity ordering (any scale where larger = worse).

    Pairs where either side is missing are left out and counted. Returns n, rho, a 95% bootstrap interval over
    pairs, a two-sided permutation p-value and the number of ties on each side - with a handful of cases the
    interval is the honest part of the answer."""
    scores, human_severity = list(scores), list(human_severity)
    pairs = [(float(s), float(h)) for s, h in zip(scores, human_severity) if ts.finite(s) and ts.finite(h)]
    n = len(pairs)
    out = {"n": n, "n_excluded": min(len(scores), len(human_severity)) - n, "rho": None, "ci95": None,
           "p_value": None, "ties_score": None, "ties_human": None}
    if n < 3:
        return out
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    rho = spearman(a, b)
    out.update(ties_score=int(n - np.unique(a).size), ties_human=int(n - np.unique(b).size), rho=clean(rho, 4))
    if not math.isfinite(rho):
        return out
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r = spearman(a[i], b[i])
        if math.isfinite(r):
            boot.append(r)
    if boot:
        out["ci95"] = [clean(v, 4) for v in np.percentile(boot, [2.5, 97.5])]
    ra = ranks(a)
    hits = sum(abs(spearman(ra, rng.permutation(ranks(b)))) >= abs(rho) - 1e-12 for _ in range(n_perm))
    out["p_value"] = clean((hits + 1) / (n_perm + 1), 4)
    return out


def score_graded(graded, weights):
    """(scores, grades) of reviewer-graded cases, each scored from its stored departures (`deltas`) under `weights`.
    Scoring every case again with one weight set puts them all on one scale: the score stored with a case was made
    with whichever weights were in force then, and calibrated and equal-weight scores cannot share one ordering."""
    scores, grades = [], []
    for impact, grade in graded:
        deltas = impact.get("deltas")
        if isinstance(deltas, dict) and ts.finite(grade):
            scores.append(weights.for_code(impact.get("code")).score(deltas))
            grades.append(float(grade))
    return scores, grades


def compare_weights(graded, candidate, min_graded=MIN_GRADED, **validate_kw):
    """Do the `candidate` weights order the reviewer-graded cases better than equal weights do?

    `graded`: [(impact, grade)] as graded_rows() reads them. The candidate was fitted to excess delay and never saw a
    grade, so the grades are an independent check on both. It only counts as better with at least `min_graded` cases
    and a strictly higher rho; a tie, too few cases, or an ordering that cannot be correlated (every grade or every
    score equal) keeps equal weights, because the burden of proof is on the calibration.
    Returns {adopt, reason, n_graded, min_graded, equal, candidate}; the last two are validate() results."""
    equal = validate(*score_graded(graded, Weights()), **validate_kw)
    fitted = validate(*score_graded(graded, candidate), **validate_kw)
    n, rho_c, rho_e = fitted["n"], fitted["rho"], equal["rho"]
    out = {"adopt": False, "n_graded": n, "min_graded": min_graded, "equal": equal, "candidate": fitted}
    if not candidate.any_calibrated():
        out["reason"] = ("no weights could be fitted (too few events, or none of them caused a measurable delay), "
                         "so equal weights stay")
    elif n < min_graded:
        out["reason"] = (f"{n} graded case(s): at least {min_graded} are needed to tell the weights apart, so equal "
                         "weights stay")
    elif rho_c is None or rho_e is None:
        out["reason"] = ("the grades or the scores are all equal, so the weights cannot be told apart and equal "
                         "weights stay")
    elif rho_c > rho_e:
        out["adopt"] = True
        out["reason"] = (f"on {n} graded cases the fitted weights agree with the reviewers better "
                         f"(rho {rho_c:+.2f}, against {rho_e:+.2f} for equal weights)")
    else:
        out["reason"] = (f"on {n} graded cases equal weights agree with the reviewers as well or better "
                         f"(rho {rho_e:+.2f}, against {rho_c:+.2f} for the fitted weights), so equal weights stay")
    return out


# ---------------------------------------------------------------- the impact of one event

def _deltas(during, ref):
    """(delta, scale, notes, shrink) per quantity: how much worse than the baseline, in units of baseline noise."""
    delta, scale, notes = {}, {}, []
    n = during.get("contributors") or 0.0
    shrink = n / (n + SHRINK_N)
    for q in QUANTITIES:
        x, m = during.get(q), ref.median.get(q)
        scale[q] = max(MAD_TO_SIGMA * ref.mad.get(q, 0.0), SCALE_FLOOR[q])
        if x is None or m is None:
            delta[q] = 0.0
            if m is None:
                notes.append(f"no baseline for {q}: counted as no departure")
            continue
        z = min(DELTA_CAP, max(0.0, WORSE[q] * (x - m) / scale[q]))
        delta[q] = z * shrink if q in PER_VEHICLE else z
    return delta, scale, notes, shrink


def compute_event_impact(collector, baseline, camera_id, zone, t0, t1, code, weights=None, *, exclude_tids=(),
                         pre_sec=PRE_SEC, min_pre_sec=MIN_PRE_SEC):
    """Impact of the event on `zone` (a RoadZone or its name) during [t0, t1] video seconds.

    exclude_tids: the vehicles that ARE the event (the rule's track id), left out of the state and the delay.
    Returns a JSON-serialisable dict: score (None when not meaningful), calibrated, weights_source, deltas, during,
    baseline, baseline_source, excess_delay_veh_s, window, vehicles_affected, plus meaningful / notes / the scales.
    excess_delay_veh_s is floored at 0 for display; excess_delay_signed_veh_s is the integral as measured and is what
    calibrate() fits. vehicles_affected: vehicles that stood still for 3 s or more in the window."""
    name = zone if isinstance(zone, str) else zone.name
    ws = weights.for_code(code) if hasattr(weights, "for_code") else (weights or EQUAL_WEIGHTS)
    exclude = tuple(int(t) for t in exclude_tids)
    out = {"score": None, "meaningful": False, "calibrated": ws.calibrated, "weights_source": ws.source,
           "weights": {q: clean(ws.values[q], 4) for q in QUANTITIES}, "code": code, "zone": name,
           "camera_id": camera_id, "deltas": None, "during": None, "baseline": None, "baseline_scale": None,
           "baseline_source": "none", "excess_delay_veh_s": None, "vehicles_affected": 0, "vehicles_seen": 0,
           "window": {"t0": clean(t0, 2), "t1": clean(t1, 2), "seconds": clean(t1 - t0, 2)},
           "excluded_tids": list(exclude), "notes": []}
    notes = out["notes"]
    during = collector.state(zone, t0, t1, exclude=exclude)
    if during is None or during["samples"] < MIN_WINDOW_SAMPLES:
        notes.append("the approach was sampled for fewer than %d seconds of the window" % MIN_WINDOW_SAMPLES)
        return out
    out["during"] = {k: (clean(v, 3) if k in QUANTITIES else v) for k, v in during.items() if k not in ("t0", "t1")}
    ref, wall = None, collector.wall_at(0.5 * (t0 + t1))
    if baseline is not None and wall is not None:
        ref = baseline.stats(camera_id, name, wall)
    if ref is None:
        pre = collector.reference(zone, t0 - pre_sec, t0 - PRE_GUARD_SEC)
        if pre is not None and pre.detail["seconds"] >= min_pre_sec:
            ref = pre
    if ref is None:
        notes.append("no baseline: no time-of-day history for this slot and fewer than %d s of traffic before the "
                     "event, so there is nothing to compare with" % int(min_pre_sec))
        return out
    out["baseline_source"] = ref.source
    out["baseline"] = {q: clean(ref.median.get(q), 3) for q in QUANTITIES}
    delta, scale, dnotes, shrink = _deltas(during, ref)
    notes += dnotes
    out["deltas"] = {q: round(v, 3) for q, v in delta.items()}
    out["evidence_factor"] = round(shrink, 3)
    out["baseline_scale"] = {q: clean(v, 3) for q, v in scale.items()}
    out["baseline_detail"] = ref.detail
    out["score"] = round(ws.score(delta), 3)
    out["meaningful"] = True
    base_kph = ref.median.get("speed_kph")
    if base_kph is None:
        notes.append("no baseline speed: excess delay not computed")
    else:
        ex = collector.excess_delay(zone, t0, t1, base_kph, exclude=exclude)
        if ex is not None:
            out["excess_delay_veh_s"] = round(ex["veh_s"], 2)
            out["excess_delay_signed_veh_s"] = round(ex["raw_veh_s"], 2)
            out["vehicles_affected"], out["vehicles_seen"] = ex["affected"], ex["vehicles"]
    return out


class ImpactTracker:
    """Waits for each event's window to close, then scores it.

    The window runs `post_sec` past the moment the rule fires, so an impact cannot be computed when the case is
    recorded: `schedule` it then, `poll` every frame with the video time, store what comes back."""

    def __init__(self, collector, baseline, camera_id, weights=None, post_sec=POST_SEC):
        self.collector, self.baseline, self.camera_id = collector, baseline, camera_id
        self.weights, self.post_sec = weights or Weights(), post_sec
        self.pending = []                       # (t1, violation id, zone, code, t0, offender tid)

    def schedule(self, violation_id, zone, code, started_s, video_time_s, tid=None):
        if zone is None:                        # not on a road zone: there is no approach to measure
            return False
        t0, t1 = event_window(started_s, video_time_s, self.post_sec)
        self.pending.append((t1, violation_id, zone, code, t0, tid))
        return True

    def _score(self, item):
        t1, vid, zone, code, t0, tid = item
        try:
            return vid, compute_event_impact(self.collector, self.baseline, self.camera_id, zone, t0, t1, code,
                                             self.weights, exclude_tids=() if tid is None else (tid,))
        except Exception as e:                  # a scoring bug must never stop the analysis, and must be visible
            return vid, {"score": None, "meaningful": False, "error": f"{type(e).__name__}: {e}", "code": code}

    def poll(self, t):
        """[(violation id, impact)] for every event whose window has closed by video time `t`."""
        due = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        return [self._score(p) for p in due]

    def flush(self):
        """End of the run: score what is still waiting with the part of the window that exists."""
        due, self.pending = self.pending, []
        return [self._score(p) for p in due]


# ---------------------------------------------------------------- priority

def _impact_band(score, bands):
    a, b = bands
    if score < a:
        return "low", BAND_POINTS["low"][0] + 33.0 * score / a
    if score < b:
        return "medium", BAND_POINTS["medium"][0] + 33.0 * (score - a) / (b - a)
    return "high", BAND_POINTS["high"][0] + 34.0 * (1.0 - math.exp(-(score - b) / b))


def history_points(prior_confirmed_count, days_since_last):
    """What a plate's confirmed prior events add, in priority points: saturating in the count, halving every
    HISTORY_HALF_LIFE_DAYS since the last one (unknown recency counts as one half-life)."""
    if not prior_confirmed_count or prior_confirmed_count <= 0:
        return 0.0
    days = HISTORY_HALF_LIFE_DAYS if days_since_last is None else max(0.0, float(days_since_last))
    return (HISTORY_MAX_POINTS * (1.0 - math.exp(-prior_confirmed_count / HISTORY_COUNT_SCALE))
            * 0.5 ** (days / HISTORY_HALF_LIFE_DAYS))


def priority(code, impact_score, prior_confirmed_count=0, days_since_last=None, *, calibrated=False):
    """Priority for the operator: {priority 0-100, severity low|medium|high, reason, ...}.

    impact_score: a number, None (unknown), or the dict compute_event_impact returned (then `calibrated` and
    `meaningful` are read from it). The band comes from impact (and RULE_FLOOR) only; history adds up to
    HISTORY_MAX_POINTS inside the band and is clipped at its top, so a repeat offender with a smaller impact can never
    outrank a high-impact first offence. It is context for a person, not a trigger for anything."""
    if isinstance(impact_score, dict):
        calibrated = bool(impact_score.get("calibrated"))
        impact_score = impact_score.get("score") if impact_score.get("meaningful", True) else None
    known = ts.finite(impact_score)
    reasons = []
    if known:
        s = max(0.0, float(impact_score))
        kind = "calibrated" if calibrated else "equal"
        band, base = _impact_band(s, SEVERITY_BANDS[kind])
        unit = "veh*s of excess delay, calibrated" if calibrated else "mean departure from baseline, uncalibrated"
        reasons.append(f"impact {s:.1f} ({unit}) is in the {band} band")
    else:
        band, base = "low", BAND_POINTS["low"][0]
        reasons.append("impact unknown (no baseline for this approach and time), so severity comes from the rule alone")
    floor = RULE_FLOOR.get(code)
    if floor and ORDER.index(floor) > ORDER.index(band):
        band, base = floor, BAND_POINTS[floor][0]
        reasons.append(f"{code} is never shown below {floor}")
    bonus = history_points(prior_confirmed_count, days_since_last)
    top = BAND_POINTS[band][1] - 0.1            # strictly inside the band even after the display rounding
    pts = min(base + bonus, top)
    if prior_confirmed_count and prior_confirmed_count > 0:
        when = "" if days_since_last is None else f", the last {days_since_last:.0f} days ago"
        clipped = " (held at the top of the band)" if base + bonus > top else ""
        reasons.append(f"{prior_confirmed_count} prior confirmed event(s) for this plate{when} add "
                       f"{pts - base:.1f} points{clipped}; history moves a case within its band, never into a "
                       f"higher one")
    return {"priority": round(pts, 1), "severity": band, "reason": "; ".join(reasons) + ".",
            "impact_points": round(base, 1), "history_points": round(pts - base, 1), "advisory": True}


# ---------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS event_impact (
    violation_id INTEGER PRIMARY KEY REFERENCES violations(id) ON DELETE CASCADE,
    camera_id INTEGER,
    zone TEXT,
    code TEXT,
    score REAL,                       -- NULL when not meaningful
    calibrated INTEGER,
    baseline_source TEXT,
    excess_delay REAL,
    impact TEXT NOT NULL,             -- the whole compute_event_impact() result as JSON
    human_severity REAL,              -- a reviewer's grade, larger = worse: the ordering rho is checked against
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS impact_weights (
    id INTEGER PRIMARY KEY,
    created_at TEXT,
    n_rows INTEGER,
    weights TEXT NOT NULL             -- Weights.to_dict() as JSON
);
"""


def ensure_schema(store):
    """Create every table this feature uses (baseline, event impacts, fitted weights). Safe on every start."""
    ts.ensure_schema(store)
    with store.lock:
        store.db.executescript(SCHEMA)
        store.db.commit()


def save_impact(store, violation_id, impact):
    """Store (or replace) the impact of one case; keeps a reviewer's grade if there already is one."""
    prev = store.q("SELECT human_severity FROM event_impact WHERE violation_id=?", (violation_id,))
    store.x("INSERT OR REPLACE INTO event_impact (violation_id, camera_id, zone, code, score, calibrated, "
            "baseline_source, excess_delay, impact, human_severity, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (violation_id, impact.get("camera_id"), impact.get("zone"), impact.get("code"), impact.get("score"),
             int(bool(impact.get("calibrated"))), impact.get("baseline_source"), impact.get("excess_delay_veh_s"),
             json.dumps(impact, allow_nan=False), prev[0]["human_severity"] if prev else None,
             datetime.now().isoformat(timespec="seconds")))


def load_impact(store, violation_id):
    rows = store.q("SELECT impact, human_severity FROM event_impact WHERE violation_id=?", (violation_id,))
    if not rows:
        return None
    impact = json.loads(rows[0]["impact"])
    impact["human_severity"] = rows[0]["human_severity"]
    return impact


def set_human_severity(store, violation_id, grade):
    """Record a reviewer's severity grade (larger = worse; None clears it) for a case that has an impact."""
    store.x("UPDATE event_impact SET human_severity=? WHERE violation_id=?",
            (None if grade is None else float(grade), violation_id))


def calibration_rows(store):
    """Every stored impact with a score and a measured delay, as calibrate() wants them."""
    return [json.loads(r["impact"]) for r in
            store.q("SELECT impact FROM event_impact WHERE score IS NOT NULL AND excess_delay IS NOT NULL")]


def validation_pairs(store):
    """(scores, human grades) of the cases a reviewer has graded, for validate()."""
    rows = store.q("SELECT score, human_severity FROM event_impact WHERE score IS NOT NULL AND "
                   "human_severity IS NOT NULL ORDER BY violation_id")
    return [r["score"] for r in rows], [r["human_severity"] for r in rows]


def graded_rows(store):
    """[(impact, grade)] of the cases a reviewer has graded, for compare_weights() (which scores them again)."""
    return [(json.loads(r["impact"]), r["human_severity"]) for r in
            store.q("SELECT impact, human_severity FROM event_impact WHERE score IS NOT NULL AND "
                    "human_severity IS NOT NULL ORDER BY violation_id")]


def save_weights(store, weights):
    return store.x("INSERT INTO impact_weights (created_at, n_rows, weights) VALUES (?,?,?)",
                   (datetime.now().isoformat(timespec="seconds"), weights.n_rows,
                    json.dumps(weights.to_dict(), allow_nan=False)))


def load_weights(store):
    """The most recently fitted weights, or equal weights (uncalibrated) when none were ever fitted."""
    rows = store.q("SELECT weights FROM impact_weights ORDER BY id DESC LIMIT 1")
    return Weights.from_dict(json.loads(rows[0]["weights"])) if rows else Weights()


def plate_history(store, plate_en, exclude_id=None, before=None):
    """(confirmed prior events for this plate, days since the latest) - events earlier than `before` (a datetime,
    default now), not counting `exclude_id`. The plate is matched exactly after dropping spaces and dashes, the same
    normalisation Store uses for plate search: a misread plate matches nothing."""
    key = "".join(ch for ch in (plate_en or "").upper() if ch not in " -")
    if not key:
        return 0, None
    before = before or datetime.now()
    rows = store.q("SELECT COALESCE(NULLIF(wall_time, ''), created_at) AS at FROM violations WHERE "
                   "status='confirmed' AND id != ? AND REPLACE(REPLACE(UPPER(plate_text_en), ' ', ''), '-', '') = ?",
                   (exclude_id if exclude_id is not None else -1, key))
    times = []
    for r in rows:
        try:
            when = datetime.fromisoformat(r["at"])
        except (TypeError, ValueError):
            continue
        if when < before:
            times.append(when)
    return len(times), (max(0.0, (before - max(times)).total_seconds() / 86400.0) if times else None)


def priority_for(store, v):
    """priority() for one case as store.violation() returns it: its stored impact plus its plate's confirmed history.
    Call it when the case is shown, not when it is recorded - the plate is read afterwards and the history changes."""
    impact = load_impact(store, v["id"])
    try:
        before = datetime.fromisoformat(v.get("wall_time") or v.get("created_at"))
    except (TypeError, ValueError):
        before = None
    count, days = plate_history(store, v.get("plate_text_en"), exclude_id=v["id"], before=before)
    p = priority(v.get("code"), impact if impact else None, count, days)
    return {**p, "prior_confirmed": count, "days_since_last": None if days is None else round(days, 1)}
