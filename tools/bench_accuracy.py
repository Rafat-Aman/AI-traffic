"""Accuracy benchmark with ground truth: a simulated camera over a two-carriageway city road.

Real footage has no ground truth for speed or for "this car stood 25 s in a running lane", so this drives the real
pipeline (surveillance/pipeline.py + all rules) with detector output generated from a calibrated pin-hole camera
(the same geometry as the automatic calibration, surveillance/autozone.py) and known vehicle motion, with the
errors a real detector and tracker make:

  * box edges jitter (independent + slowly drifting, as YOLO boxes do), more for small far boxes
  * missed detections, and short bursts of them
  * vehicles partly hidden behind the one in front: the box covers only the visible part (bottom edge too high)
  * class flicker between look-alikes, and tracker ID switches (new ID after a gap, or at random)

It reports speed error (by distance), false movement of standing vehicles and the decisions of the speed, stopping
and wrong-side rules against the truth.

    .venv/bin/python tools/bench_accuracy.py                # default scenario, 3 minutes of traffic
    .venv/bin/python tools/bench_accuracy.py --fps 30 --seeds 3
    .venv/bin/python tools/bench_accuracy.py --clean         # no detector errors: the geometry floor
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surveillance.autozone import CLASS_SIZE, GroundModel, _cuboid_box  # noqa: E402
from surveillance.pipeline import Pipeline  # noqa: E402
from violations.zones import camera_config_from_dict  # noqa: E402

W, H = 1920, 1080
HFOV, ROLL, CAM_HEIGHT, VP = 62.0, 0.6, 7.5, (1010.0, 130.0)
# (across from, across to, direction of travel: +1 away from the camera, -1 toward it), metres
CARRIAGEWAYS = [(-13.0, -3.0, +1), (0.0, 10.0, -1)]
ZONE_ALONG = (13.0, 115.0)          # measured road zones: along-road extent (m from the camera's foot)
SPAWN_ALONG = (8.0, 150.0)          # vehicles exist between these
LIMIT = 60.0
SIZES = dict(CLASS_SIZE, Motorbike=(0.8, 2.0, 1.55, 0.55))
LOOKALIKE = {"Car": "MPV", "MPV": "Car", "CNG": "Leguna", "Leguna": "CNG", "Bus": "Truck", "Truck": "Bus",
             "Motorbike": "Bicycle", "Pickup": "Truck"}
CLASS_MIX = ["Car"] * 5 + ["CNG"] * 4 + ["Motorbike"] * 4 + ["Bus"] * 2 + ["MPV", "Pickup", "Leguna", "Truck"]


def camera():
    gm = GroundModel.from_hfov(W, H, HFOV, VP, ROLL)
    gm.scale = CAM_HEIGHT
    return gm


def zones_config(gm):
    zones = []
    for i, (a0, a1, sign) in enumerate(CARRIAGEWAYS):
        l0, l1 = ZONE_ALONG
        pts = gm.pixel(np.array([l1, l1, l0, l0]), np.array([a0, a1, a1, a0]))
        zones.append({"type": "road", "name": f"Carriageway {i + 1}", "points": pts.round(2).tolist(),
                      "width_m": a1 - a0, "length_m": l1 - l0, "road_category": "city",
                      "allowed_direction": "away_from_camera" if sign > 0 else "toward_camera"})
    return {"frame_size": [W, H], "zones": zones, "peak_hours": None}


# ------------------------------------------------------------------ traffic (ground truth)
class Vehicle:
    def __init__(self, vid, cls, size, cw, lane_across, sign, kph, kind, rng, t0):
        self.vid, self.cls, self.size, self.cw = vid, cls, size, cw
        self.sign, self.kind, self.t0 = sign, kind, t0
        self.across = lane_across
        self.along = SPAWN_ALONG[0] if sign > 0 else SPAWN_ALONG[1]
        self.v = self.v_set = kph / 3.6
        self.stop_at = rng.uniform(22, 48) if kind == "stop" else None
        self.stand_left = rng.uniform(24, 30) if kind == "stop" else 0.0
        self.lc = (rng.uniform(1, 5), rng.choice([-3.3, 3.3])) if kind == "lanechange" else None
        self.max_true_kph, self.stood_s, self.t_alive = 0.0, 0.0, 0.0
        self.hist = []          # (t, true kph, measurable): where the camera can measure it (truth geometry)
        self.stood_measurable = 0.0
        self.passed_by = set()  # motor vehicles that drove past it while it stood (it blocked them)
        self.visible = False    # detected this frame with its lower edge in view (a measurable ground point)
        self.done = False

    def step(self, dt):
        self.t_alive += dt
        if self.kind == "stop" and self.stand_left > 0:
            dist = (self.stop_at - self.along) * self.sign
            if dist <= 0.05 and self.v <= 0.3:
                self.v = 0.0
                self.stand_left -= dt
                self.stood_s += dt
            else:  # brake to stop exactly there
                need = math.sqrt(max(0.0, 2 * 2.5 * max(dist, 0.0)))
                self.v = max(0.0, min(self.v, need)) if dist < 40 else self.v
        elif self.v < self.v_set:
            self.v = min(self.v_set, self.v + 1.5 * dt)
        if self.lc and self.lc[0] <= self.t_alive <= self.lc[0] + 3.0:
            self.across += self.lc[1] / 3.0 * dt
        self.along += self.sign * self.v * dt
        if self.v * 3.6 > self.max_true_kph and ZONE_ALONG[0] + 20 < self.along < ZONE_ALONG[1] - 25:
            self.max_true_kph = self.v * 3.6
        self.done = not (SPAWN_ALONG[0] - 1 <= self.along <= SPAWN_ALONG[1] + 1)


class Scenario:
    def __init__(self, seed=0, seconds=180, fps=15, clean=False, id_switch_per_sec=0.02):
        self.rng = np.random.default_rng(seed)
        self.seconds, self.fps, self.clean = seconds, fps, clean
        self.id_switch = id_switch_per_sec / fps   # random tracker ID switches (besides those after a gap)
        self.gm = camera()
        self.vehicles, self.next_vid = [], 1
        self.track_of = {}             # vehicle id -> current tracker id
        self.next_tid = 1
        self.drift = {}                # vehicle id -> correlated box error (4,)
        self.gap = defaultdict(int)    # vehicle id -> frames missed in a row

    def spawn(self, t):
        rng = self.rng
        for ci, (a0, a1, sign) in enumerate(CARRIAGEWAYS):
            if rng.random() > 1.0 / self.fps * 0.55:   # ~0.55 vehicles / s per carriageway
                continue
            cls = str(rng.choice(CLASS_MIX))
            lanes = int((a1 - a0) // 3.3)
            lane = int(rng.integers(0, lanes))
            across = a0 + 1.65 + lane * 3.3 + rng.normal(0, 0.3)
            r = rng.random()
            kind = "stop" if r < 0.08 else "wrongway" if r < 0.10 else "lanechange" if r < 0.22 else "cruise"
            if kind == "stop" and any(o.kind == "stop" and o.cw == ci and not o.done for o in self.vehicles):
                kind = "cruise"   # one at a time per carriageway: a vehicle stopping behind another is queued
            kph = float(np.clip(rng.normal(38, 12), 12, 70)) if rng.random() > 0.25 else float(rng.uniform(60, 88))
            if cls == "Bus":
                kph = min(kph, 50)
            s = sign if kind != "wrongway" else -sign
            if kind == "wrongway":
                across = (a0 + 1.5) if sign > 0 else (a1 - 1.5)   # kerb-side lane, against the flow
                kph = min(kph, 25)
            if kind == "stop":
                across = a0 + 1.65 + 3.3 * (lanes // 2) if lanes > 2 else across   # a middle (running) lane
            k = 1 + rng.normal(0, 0.05)
            w, l, h, wt = SIZES[cls]
            v = Vehicle(self.next_vid, cls, (w * k, l * k, h * k, wt * k), ci, across, s, kph, kind, rng, t)
            # do not spawn into a vehicle of the same lane
            if any(o.cw == ci and abs(o.across - across) < 1.6 and abs(o.along - v.along) < 12 for o in self.vehicles):
                continue
            self.vehicles.append(v)
            self.next_vid += 1

    def follow(self):
        """Car following: nobody drives through the vehicle ahead in its lane (keeps stops realistic)."""
        for v in self.vehicles:
            ahead = [o for o in self.vehicles if o is not v and o.cw == v.cw and abs(o.across - v.across) < 1.6
                     and 0 < (o.along - v.along) * v.sign < 14 and o.sign == v.sign]
            if ahead:
                o = min(ahead, key=lambda o: (o.along - v.along) * v.sign)
                gap = (o.along - v.along) * v.sign - o.size[1]
                if gap < 4:
                    if o.v < 0.3 and v.kind != "stop":   # overtake a standing vehicle: switch lane
                        a0, a1, _ = CARRIAGEWAYS[v.cw]
                        alt = v.across + (3.3 if v.across + 3.3 < a1 - 0.8 else -3.3)
                        if a0 + 0.8 < alt < a1 - 0.8:
                            v.across = alt
                            continue
                    v.v = min(v.v, max(0.0, o.v - 0.2))

    def boxes(self, t):
        """Detector output for this frame: [(box, tracker id, class)] plus the truth for each id."""
        rng, out, truth = self.rng, [], {}
        for v in self.vehicles:
            v.visible = False
        vis = [v for v in self.vehicles if not v.done]
        if not vis:
            return out, truth
        al = np.array([v.along for v in vis])
        ac = np.array([v.across for v in vis])
        raw = []
        for v, a, c in zip(vis, al, ac):
            x1, y1, x2, y2 = _cuboid_box(self.gm, np.array([a]), np.array([c]), v.size)
            raw.append(np.array([x1[0], y1[0], x2[0], y2[0]]))
        order = np.argsort([-b[3] for b in raw])   # nearest (lowest bottom) first: they hide the ones behind
        shown = []
        for i in order:
            v, b = vis[i], raw[i].copy()
            if b[3] < VP[1] + 15 or b[2] < 0 or b[0] > W or b[1] > H:
                continue
            hidden = 0.0
            if not self.clean:
                for j in shown:    # a nearer box covering the lower part of this one
                    o = raw[j]
                    ov = min(b[2], o[2]) - max(b[0], o[0])
                    if ov > 0.35 * (b[2] - b[0]) and o[1] < b[3] and o[3] > b[3]:
                        cut = b[3] - max(o[1], b[1])
                        hidden = max(hidden, cut / (b[3] - b[1]))
                if hidden > 0.75:
                    self.gap[v.vid] += 1
                    shown.append(i)
                    continue
                if hidden > 0.1:   # the detector boxes the visible part
                    b[3] = b[3] - hidden * (b[3] - b[1]) + rng.normal(0, 1.0)
            shown.append(i)
            if not self.clean:
                size = max(b[2] - b[0], b[3] - b[1])
                d = self.drift.get(v.vid, np.zeros(4))
                d = 0.92 * d + rng.normal(0, 0.55, 4) * (1 + size / 300)
                self.drift[v.vid] = d
                b = b + d + rng.normal(0, 0.7, 4) * (1 + size / 400)
                if rng.random() < 0.05 or (self.gap[v.vid] and rng.random() < 0.5):   # misses come in bursts
                    self.gap[v.vid] += 1
                    continue
                if self.gap[v.vid] > 0.7 * self.fps or rng.random() < self.id_switch:   # tracker lost it: new ID
                    self.track_of.pop(v.vid, None)
                self.gap[v.vid] = 0
            if v.vid not in self.track_of:
                self.track_of[v.vid] = self.next_tid
                self.next_tid += 1
            cls = v.cls if self.clean or rng.random() > 0.06 else LOOKALIKE.get(v.cls, v.cls)
            x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            tid = self.track_of[v.vid]
            v.visible = hidden <= 0.1
            out.append(((float(x1), float(y1), float(x2), float(y2)), tid, cls))
            truth[tid] = v
        return out, truth


class FakeDetector:
    imgsz = 960

    def __init__(self):
        self.boxes, self.offset = [], (0, 0)

    def track(self, image):
        ox, oy = self.offset
        return [((x1 - ox, y1 - oy, x2 - ox, y2 - oy), tid, c) for (x1, y1, x2, y2), tid, c in self.boxes]


# ------------------------------------------------------------------ ablation: switch improvements off
def _legacy_speed_update(self, tid, t, zone, xy):
    """The original estimate: distance between the ends of the last second, median of the last 15 of those."""
    from collections import deque
    pos = (t, float(xy[0]), float(xy[1]))
    if self.zone.get(tid) is not zone:
        self.zone[tid], self.path[tid], self.recent[tid], self.first[tid] = zone, deque(), deque(maxlen=15), pos
    self.last[tid] = pos
    path = self.path[tid]
    path.append(pos)
    while len(path) > 2 and t - path[1][0] >= self.window_sec:
        path.popleft()
    t0, x0, y0 = path[0]
    if t - t0 >= self.min_span:
        kph = math.hypot(pos[1] - x0, pos[2] - y0) / (t - t0) * 3.6
        if kph <= self.max_kph:
            self.recent[tid].append(kph)
    return self.current(tid)


ABLATIONS = ("speedfit", "occlusion", "coarse")


def ablate(names):
    """Turn the named improvements off (monkeypatching), to measure what each one contributes."""
    import surveillance.pipeline as PL
    from surveillance.speed import SpeedEstimator
    if "speedfit" in names:
        SpeedEstimator.update = _legacy_speed_update
    if "occlusion" in names:
        PL.bottom_occluded = lambda boxes, *a, **k: [False] * len(boxes)
        PL.Pipeline._foot_hidden = lambda self, *a, **k: False
    return "coarse" in names


# ------------------------------------------------------------------ run + score
def run(seed=0, seconds=180, fps=15, clean=False, quiet=False, no_coarse=False, id_switch=0.02):
    sc = Scenario(seed, seconds, fps, clean, id_switch)
    cam = camera_config_from_dict(zones_config(sc.gm), W, H)
    det = FakeDetector()
    p = Pipeline(det, cam, (W, H), stabilize=False, display_width=320)
    if no_coarse:   # ablation: rules see no positions in the imprecise far part of a zone
        process = p.engine.process

        def without_coarse(observations, *a, **k):
            for o in observations:
                o.xy_coarse = None
            return process(observations, *a, **k)
        p.engine.process = without_coarse
    frame = np.zeros((H, W, 3), np.uint8)
    dt = 1.0 / fps
    errs = []              # (true kph, measured kph, along m)
    standing = []          # measured kph of vehicles truly standing (after 2 s of standing)
    owner = {}             # tracker id -> vehicle
    for k in range(int(seconds * fps)):
        t = k * dt
        sc.spawn(t)
        sc.follow()
        before = {v.vid: v.along for v in sc.vehicles}
        for v in sc.vehicles:
            v.step(dt)
        for s in sc.vehicles:   # who drove past a standing vehicle
            if s.kind == "stop" and s.v == 0:
                for o in sc.vehicles:
                    if (o is not s and o.cw == s.cw and o.cls not in ("Motorbike", "Bicycle") and o.vid in before
                            and (before[o.vid] - s.along) * (o.along - s.along) < 0):
                        s.passed_by.add(o.vid)
        det.boxes, truth = sc.boxes(t)
        det.offset = p.roi[:2]
        p.process(frame, t)
        owner.update(truth)
        for v in sc.vehicles:   # what an ideal system could judge: true position inside the measurable range
            x1, y1, x2, y2 = (float(a[0]) for a in _cuboid_box(sc.gm, np.array([v.along]), np.array([v.across]), v.size))
            road = cam.roads[v.cw]
            pt = ((x1 + x2) / 2, y2)
            ok = y2 < H - 2 and road.contains(pt) and road.to_metres(pt)[1] >= p.reliable_y[id(road)] + 1.0
            v.hist.append((t, v.v * 3.6, ok and v.visible))
            if ok and v.visible and v.v == 0:
                v.stood_measurable += dt
        for tid, v in truth.items():
            if p.speeds.zone.get(tid) is None or tid not in p.speeds.recent or not p.speeds.recent[tid]:
                continue
            if p.speeds.last.get(tid, (None,))[0] != t:   # not measured in this frame
                continue
            kph = p.speeds.current(tid)
            if v.v == 0 and v.stood_s > 2.0:
                standing.append(kph)
            elif v.v > 0 and v.t_alive > 3:
                errs.append((v.v * 3.6, kph, v.along))
        sc.vehicles = [v for v in sc.vehicles if not v.done]
    return score(p, sc, owner, errs, standing, quiet)


def score(p, sc, owner, errs, standing, quiet):
    e = np.array(errs) if errs else np.zeros((0, 3))
    res = {}
    if len(e):
        diff = e[:, 1] - e[:, 0]
        res["speed"] = {"n": int(len(e)), "mae_kph": round(float(np.mean(np.abs(diff))), 2),
                        "bias_kph": round(float(np.mean(diff)), 2),
                        "p95_abs_kph": round(float(np.percentile(np.abs(diff), 95)), 2),
                        "mape_pct": round(float(np.mean(np.abs(diff) / e[:, 0]) * 100), 1)}
        bands = {}
        for name, lo, hi in (("near <40 m", 0, 40), ("mid 40-75 m", 40, 75), ("far >75 m", 75, 999)):
            m = (e[:, 2] >= lo) & (e[:, 2] < hi)
            if m.sum() > 20:
                bands[name] = {"n": int(m.sum()), "mae_kph": round(float(np.mean(np.abs(diff[m]))), 2),
                               "bias_kph": round(float(np.mean(diff[m])), 2)}
        res["speed_by_distance"] = bands
    if standing:
        s = np.array(standing)
        res["standing"] = {"n": int(len(s)), "mean_kph": round(float(s.mean()), 2),
                           "p95_kph": round(float(np.percentile(s, 95)), 2)}
    # ---- decisions
    vehicles = {}
    for tid, v in owner.items():
        vehicles[v.vid] = v
    flagged = defaultdict(set)   # vehicle id -> codes
    for ev in p.engine.events:
        v = owner.get(ev.obs.tid)
        if v is not None:
            flagged[v.vid].add(ev.code)
    seen = list(vehicles.values())
    m = p.engine.rules
    v2 = next((r for r in m if r.code == "V2"), None)
    thr = v2.threshold(LIMIT, p.camera.roads[0]) if v2 else LIMIT * 1.05
    dec = {}

    def fast_run(v, above, need):
        """At least `need` seconds of measurable frames above `above` km/h, gaps of a few missed frames allowed."""
        run, start, last_ok = 0.0, None, None
        for t, k, ok in v.hist:
            if ok and k > above:
                if start is None or t - last_ok > 0.35:
                    start = t
                last_ok = t
                if t - start >= need:
                    return True
            elif k <= above:
                start = None
        return False

    fast = [v for v in seen if fast_run(v, thr + 2, 1.8) and v.kind != "wrongway"]
    legal = [v for v in seen if v.kind != "wrongway" and any(ok for _, _, ok in v.hist)
             and max((k for _, k, ok in v.hist if ok), default=0) < LIMIT]
    dec["V2 speeding"] = {"truth": len(fast), "caught": sum("V2" in flagged[v.vid] for v in fast),
                          "false_alarms": sum("V2" in flagged[v.vid] for v in legal),
                          "of_legal": len(legal), "threshold_kph": round(thr, 1)}
    # blocking (V1 / V7) means traffic had to go round it: at least 4 motor vehicles drove past
    stops = [v for v in seen if v.kind == "stop" and v.stood_measurable >= 22 and len(v.passed_by) >= 5]
    others = [v for v in seen if v.kind != "stop"]
    stop_codes = {"V1", "V3", "V7"}
    dec["V1/V7 standing in a lane"] = {
        "truth": len(stops), "caught": sum(bool(flagged[v.vid] & stop_codes) for v in stops),
        "false_alarms": sum(bool(flagged[v.vid] & stop_codes) for v in others), "of_moving": len(others)}
    wrong = [v for v in seen if v.kind == "wrongway" and sum(ok for _, _, ok in v.hist) / max(1, len(v.hist)) * v.t_alive > 3]
    right = [v for v in seen if v.kind != "wrongway"]
    dec["V4 wrong side"] = {"truth": len(wrong), "caught": sum("V4" in flagged[v.vid] for v in wrong),
                            "false_alarms": sum("V4" in flagged[v.vid] for v in right), "of_legal": len(right)}
    lc = [v for v in seen if v.kind == "lanechange"]
    res["decisions"] = dec
    res["tracks_per_vehicle"] = round(len(owner) / max(1, len(vehicles)), 2)
    res["vehicles"] = len(vehicles)
    res["other_codes"] = dict(sorted(((c, sum(c in f for f in flagged.values())) for c in
                                      {c for f in flagged.values() for c in f} - {"V1", "V2", "V3", "V4", "V7"})))
    if not quiet:
        print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--id-switch", type=float, default=0.02,
                    help="random tracker ID switches per vehicle per second (default 0.02; 0.15 = poor tracking)")
    ap.add_argument("--ablate", default="", help=f"comma-separated improvements to switch off: {', '.join(ABLATIONS)}"
                                                 " or 'all' (the pipeline before they were added)")
    a = ap.parse_args()
    names = set(ABLATIONS) if a.ablate == "all" else {n for n in a.ablate.split(",") if n}
    no_coarse = ablate(names)
    if a.seeds == 1:
        run(0, a.seconds, a.fps, a.clean, no_coarse=no_coarse, id_switch=a.id_switch)
        return
    allres = [run(s, a.seconds, a.fps, a.clean, quiet=True, no_coarse=no_coarse, id_switch=a.id_switch)
              for s in range(a.seeds)]
    sp = [r["speed"] for r in allres if "speed" in r]
    print("speed MAE %.2f  bias %+.2f  p95 %.2f  MAPE %.1f%%" % tuple(
        np.mean([s[k] for s in sp]) for k in ("mae_kph", "bias_kph", "p95_abs_kph", "mape_pct")))
    st = [r["standing"] for r in allres if "standing" in r]
    if st:
        print("standing: mean %.2f  p95 %.2f km/h" % (np.mean([s["mean_kph"] for s in st]),
                                                     np.mean([s["p95_kph"] for s in st])))
    for name in allres[0]["decisions"]:
        d = [r["decisions"][name] for r in allres]
        tot = {k: sum(x[k] for x in d) for k in d[0] if k != "threshold_kph"}
        print(name, tot)


if __name__ == "__main__":
    main()
