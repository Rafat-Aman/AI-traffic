"""Kinematic pre-screen in front of the two-tier VLM cascade (a 2B model screens, an 8B model reasons - TAU-R1).

    screen(candidate) -> {"class": ..., "score": 0..1, "send_to_vlm": bool, "reason": str}

`candidate` is a violations.ViolationEvent or a violation row dict (code, details, kph, cls). The class is the screening
hypothesis handed to the 2B model: "none", "direction_or_manoeuvre" (V10 swerve, V4 wrong way, V5 line crossing),
"near_collision_or_collision" (V11) or "abnormal_road_use" (V1 V3 V7 stopping in lanes, V8 banned class, V9 bus lane;
also V2 and V6 and any unknown code, the broadest class, so a rule-decided case is never labelled "none").

What it is for. It sits BEFORE the cascade and only saves the 2B model's time on obviously empty candidates. It never
discards a strong one, and it is not a decision: a candidate it sends is still judged by the models and a reviewer.
Rule-decided codes (V1-V9, V2, V6) come from geometry with conservative thresholds and carry no kinematic gradient to
weigh, so they always go through (score 1.0). Only the VLM-decided candidates V10 and V11 are weighed.

Asymmetric by design. The report (5.1.2) says that at the screening stage a missed anomaly costs more than an
over-flagged one. That is written as costs, not as a feel-good threshold:

    COST_WASTED  - sending an empty candidate on: one 2B pass (seconds of CPU) and a glance of reviewer attention  = 1
    COST_MISSED  - dropping a real anomaly: nothing downstream ever sees it again                                  = 10

Send iff the expected cost of dropping exceeds that of sending: p * COST_MISSED >= (1 - p) * COST_WASTED, i.e.

    p >= THRESHOLD = COST_WASTED / (COST_WASTED + COST_MISSED) = 1/11 = 0.091   (0.5 if the costs were equal)

The 10:1 ratio is a policy choice, not a measurement; change the two constants and the threshold follows (threshold()).

The score. Each kinematic measurement is divided by the level at which the rule fires (ratio r; r >= 1 for every
candidate the engine produced), the weakest ratio r_min decides (every criterion had to hold), and score = r_min/(1 +
r_min): a candidate exactly at the firing levels scores 0.5, twice over scores 0.67, a tenth of the level scores 0.09.
Besides the rule's own criteria there is a measurement-quality ratio (V10: excursion over fit residual against a
reference of 4), because a "swerve" that is mostly fit error is a tracking glitch. So a candidate is dropped only when
some criterion sits below r = COST_WASTED/COST_MISSED = 0.1 of its firing level, which the engine's own candidates
do not (they are >= 1 by construction, so with default settings only hand-built, loosened-rule or noisy rows are
dropped), and any candidate at or above its firing levels scores >= 0.5 and is sent for every cost ratio >= 1.

Limits: the score is an ordinal strength, not a calibrated probability - calibrating it needs labelled candidates,
which do not exist yet - so the derivation of the threshold holds only as far as it is monotone in the true chance of an
anomaly. Reference levels are the rules' DEFAULTS unless the camera's rule settings are passed as `settings`."""

import json

from violations.erratic import ErraticManoeuvre, class_group
from violations.near_collision import NearCollision

CLASSES = ("none", "direction_or_manoeuvre", "near_collision_or_collision", "abnormal_road_use")
COST_WASTED = 1.0
COST_MISSED = 10.0
SNR_REFERENCE = 4.0    # V10: excursion at least this many times the fit residual

CLASS_OF = {"V10": "direction_or_manoeuvre", "V4": "direction_or_manoeuvre", "V5": "direction_or_manoeuvre",
            "V11": "near_collision_or_collision"}
FALLBACK_CLASS = "abnormal_road_use"


def threshold(cost_missed=COST_MISSED, cost_wasted=COST_WASTED):
    """Smallest score worth sending on: the point where the expected costs of sending and dropping are equal."""
    return cost_wasted / (cost_wasted + cost_missed)


def score_of(ratio):
    return ratio / (1.0 + ratio)


def _parts(candidate):
    if isinstance(candidate, dict):
        code, details, cls = candidate.get("code"), candidate.get("details"), candidate.get("cls")
    else:
        code, details, cls = candidate.code, candidate.details, getattr(candidate.obs, "cls", None)
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = {}
    return (code or "").strip().upper(), details if isinstance(details, dict) else {}, cls


def _num(d, key):
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _ratios_v10(d, cls, cfg):
    group = d.get("class_group") or class_group(cls)
    out = {}
    for key, field, limit in (("lateral_m", "lateral_m", f"lateral_m_{group}"),
                              ("peak_lateral_mps", "peak_mps", f"peak_mps_{group}"),
                              ("lateral_accel_mps2", "accel_mps2", f"accel_mps2_{group}")):
        v = _num(d, key)
        if v is not None:
            out[field] = v / cfg[limit]
    if d.get("reversal") is not True:
        v = _num(d, "heading_change_deg")
        if v is not None:
            out["heading_deg"] = v / cfg["min_heading_deg"]
    lat, rms = _num(d, "lateral_m"), _num(d, "fit_rms_m")
    if lat is not None and rms is not None:
        out["signal_to_fit_error"] = lat / max(rms, 0.05) / SNR_REFERENCE
    return out


def _ratios_v11(d, cfg):
    out = {}
    close = _num(d, "closing_speed_mps")
    if close is not None:
        out["closing_speed"] = close / cfg["min_closing_mps"]
    if d.get("kind") == "collision":
        s = _num(d, "stationary_s")
        if s is not None:
            out["stationary"] = s / cfg["stasis_sec"]
    else:
        ttc, dec = _num(d, "ttc_s"), _num(d, "decel_mps2")
        if ttc is not None:
            out["ttc"] = cfg["ttc_max_s"] / max(ttc, 0.05)
        if dec is not None:
            out["decel"] = dec / cfg["min_decel_mps2"]
    return out


def screen(candidate, settings=None):
    """See the module notes. `settings`: the camera's {rule key: {setting: value}} overrides, if any."""
    code, d, cls = _parts(candidate)
    cls_name = CLASS_OF.get(code, FALLBACK_CLASS)
    if code not in ("V10", "V11"):
        why = "decided by the geometric rule engine" if code else "no rule code"
        return {"class": cls_name, "score": 1.0, "send_to_vlm": True,
                "reason": f"{why}; no kinematic evidence to weigh, passed on"}
    if code == "V10":
        ratios = _ratios_v10(d, cls, {**ErraticManoeuvre.DEFAULTS, **((settings or {}).get("erratic") or {})})
    else:
        ratios = _ratios_v11(d, {**NearCollision.DEFAULTS, **((settings or {}).get("near_collision") or {})})
    if not ratios:
        return {"class": cls_name, "score": 0.5, "send_to_vlm": True,
                "reason": f"{code} candidate without kinematic details; a missed anomaly costs more, so passed on"}
    weakest = min(ratios, key=ratios.get)
    r = ratios[weakest]
    score = round(score_of(r), 3)
    thr = threshold()
    send = score >= thr
    shown = ", ".join(f"{k} {v:.2f}x" for k, v in sorted(ratios.items(), key=lambda kv: kv[1])[:3])
    if send:
        reason = (f"weakest evidence is {weakest} at {r:.2f}x its firing level (score {score:.2f} >= {thr:.2f}, "
                  f"derived from missed:wasted cost {COST_MISSED:g}:{COST_WASTED:g}); {shown}")
    else:
        reason = (f"{weakest} is only {r:.2f}x its firing level, below {COST_WASTED / COST_MISSED:.2f}x "
                  f"(score {score:.2f} < {thr:.2f}): obviously empty; {shown}")
    return {"class": cls_name if send else "none", "score": score, "send_to_vlm": send, "reason": reason}
