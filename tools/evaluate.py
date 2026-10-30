#!/usr/bin/env python3
"""Evaluation harness runner (SUPERVISION.md section 7): one subcommand per X-item, each writing results/<item>.json.

    ./.venv/bin/python tools/evaluate.py x2 --proxy-seconds 60      # ByteTrack vs BoT-SORT proxy (no GT needed)
    ./.venv/bin/python tools/evaluate.py x3 | x4 | x5 | x6 | x7 | optimize
    ./.venv/bin/python tools/evaluate.py x1 --data <dataset.yaml>   # needs a labelled Dhaka set
    ./.venv/bin/python tools/evaluate.py all                        # everything that can run on this CPU in minutes
    ./.venv/bin/python tools/evaluate.py report                     # REPORT_RESULTS.md from results/*.json only

Rule: a number appears only if a named command measured it here, or it is copied from a named file with its
location. Missing ground truth produces a "needs" entry, never a number. Status labels used everywhere:
  measured on Dhaka | synthetic | external | not measured
See docs/EVALUATION.md for the ground truth the team must produce.
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import detection, events, provenance, stats, tracking  # noqa: E402

RESULTS = ROOT / "results"
PY = sys.executable
GT_DIR = ROOT / "evaluation" / "ground_truth"
CLIP = ROOT / "data" / "videos" / "20261025_092449_side1_1.mp4"
M_DHAKA, SYN, EXT, NOT = "measured on Dhaka", "synthetic", "external", "not measured"


# ---------------------------------------------------------------- plumbing

def new_result(item, title, inputs=()):
    return {"item": item, "title": title, "provenance": provenance.make(inputs=inputs), "tables": [], "needs": [],
            "notes": [], "raw": {}}


def table(res, title, columns, rows, source):
    res["tables"].append({"title": title, "columns": columns, "rows": rows, "source": source})


def need(res, what, needs):
    res["needs"].append({"what": what, "status": f"{NOT} - needs {needs}"})


def save(res):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{res['item']}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {p.relative_to(ROOT)}")
    return p


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def run_cmd(args, timeout=900):
    t0 = time.time()
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {"cmd": " ".join(str(a) for a in args), "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "stdout": p.stdout, "stderr": p.stderr[-2000:]}


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- X1 detection

def x1(a):
    data = Path(a.data) if a.data else GT_DIR / "detection" / "data.yaml"
    res = new_result("x1_detection", "X1 Detection AP@0.5 / mAP@0.5", [data] if data.exists() else [])
    if not data.exists():
        need(res, "Detection mAP@0.5 per class on a held-out Dhaka set",
             f"a YOLO-format dataset yaml ({rel(data)}) with a val split of Dhaka frames NOT used to train "
             "bd_vehicles_best.pt, same 16 classes (docs/EVALUATION.md section 1). No vehicle-detector mAP exists "
             "anywhere in the repo (the only 'mAP50 0.968' in models/README.md is the plate-character model)")
        res["notes"].append("models/README.md is silent on how bd_vehicles_best.pt was validated; SUPERVISION D1 says "
                            "it is YOLOv8s, 80 epochs, 640 px, 16 classes.")
        return save(res)
    import yaml
    from ultralytics import YOLO
    cfg = yaml.safe_load(data.read_text())
    base = Path(cfg.get("path", data.parent))
    base = base if base.is_absolute() else (data.parent / base)
    split = a.split
    img_dir = base / cfg[split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    model = YOLO(str(ROOT / "bd_vehicles_best.pt"))
    t0 = time.time()
    v = model.val(data=str(data), split=split, imgsz=a.imgsz, conf=0.001, iou=0.6, device="cpu", plots=False, verbose=False)
    ul_secs = time.time() - t0
    names = model.names
    ul = {names[int(c)]: float(ap) for c, ap in zip(v.box.ap_class_index, v.box.ap50)}
    # own AP: predict on every val image, VOC all-point
    gt = detection.load_yolo_dir(lbl_dir, img_dir)
    preds = []
    for r in model.predict(str(img_dir), imgsz=a.imgsz, conf=0.001, iou=0.6, device="cpu", stream=True, verbose=False):
        stem = Path(r.path).stem
        for xy, cf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
            preds.append((stem, c, cf, tuple(xy)))
    own = detection.evaluate(gt, preds, 0.5, names)
    rows = []
    for c, d in own["per_class"].items():
        rows.append([d.get("name", c), d["n_gt"], d["n_pred"], fmt(d["ap"]), fmt(ul.get(d.get("name"))), M_DHAKA])
    rows.append(["**mAP@0.5**", sum(d["n_gt"] for d in own["per_class"].values()), "", fmt(own["map"]),
                 fmt(float(v.box.map50)), M_DHAKA])
    table(res, "Detection AP@0.5 per class", ["Class", "GT boxes", "Predictions", "AP@0.5 (VOC all-point, this harness)",
          "AP@0.5 (Ultralytics val, 101-pt)", "Status"], rows,
          f"{provenance.make()['command']}; dataset {rel(data)} split '{split}', imgsz {a.imgsz}, CPU; Ultralytics val took {ul_secs:.0f} s")
    return save(res)


# ---------------------------------------------------------------- X2 tracking

def percentiles(xs, ps=(25, 50, 75, 90)):
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{p}": s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] for p in ps}


def fragment_candidates(tracks, fps, W, H, max_gap_s=2.0, edge_frac=0.04):
    """Heuristic count of probable fragmentation: a track ends away from the image border, and another track of the
    same class starts within max_gap_s frames later with a box centre within 0.6 box-diagonals of where the first
    ended. Counts each ending at most once. It is an ESTIMATE, not a measurement: coincidences inflate it and
    vehicles that stop and restart hide in it."""
    ends = {t: (d["frames"][-1], d["boxes"][-1], d["cls"]) for t, d in tracks.items()}
    starts = {t: (d["frames"][0], d["boxes"][0], d["cls"]) for t, d in tracks.items()}

    def near_border(b):
        return b[0] < edge_frac * W or b[2] > (1 - edge_frac) * W or b[1] < edge_frac * H or b[3] > (1 - edge_frac) * H

    gap = max_gap_s * fps
    used, n, away = set(), 0, 0
    for t, (fe, be, ce) in ends.items():
        if near_border(be):
            continue
        away += 1
        best, bd = None, None
        cx, cy = (be[0] + be[2]) / 2, (be[1] + be[3]) / 2
        diag = ((be[2] - be[0]) ** 2 + (be[3] - be[1]) ** 2) ** 0.5
        for u, (fs, bs, cs) in starts.items():
            if u == t or u in used or cs != ce or not (0 < fs - fe <= gap):
                continue
            d = (((bs[0] + bs[2]) / 2 - cx) ** 2 + ((bs[1] + bs[3]) / 2 - cy) ** 2) ** 0.5
            if d <= 0.6 * diag and (bd is None or d < bd):
                best, bd = u, d
        if best is not None:
            used.add(best)
            n += 1
    return {"fragment_candidates": n, "track_ends_away_from_border": away}


def proxy_tracking(a):
    """Detect ONCE per sampled frame, then replay the identical detections into ByteTrack and BoT-SORT (the
    Ultralytics classes, configured exactly as surveillance.detector would: vehicle_tracker.build_tracker_cfg with
    the detector's defaults conf=None, 2.0 s buffer, no ReID). Identical detections make the comparison controlled;
    tracker-only time is measured separately from detection time."""
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from vehicle_tracker import build_tracker_cfg, select_classes

    cap = cv2.VideoCapture(str(a.video))
    native = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    step = max(1, round(native / a.fps))
    eff_fps = native / step
    max_frame = int(a.proxy_seconds * native)
    model = YOLO(str(ROOT / "bd_vehicles_best.pt"), task="detect")
    names = model.names
    cls_ids = select_classes(names, None, None)
    trackers, cfgs = {}, {}
    for name in a.trackers:
        path = build_tracker_cfg(name, None, eff_fps, 2.0, False)
        cfg = IterableSimpleNamespace(**YAML.load(path))
        cfg.device = "cpu"
        cfgs[name] = {k: v for k, v in vars(cfg).items() if k != "device"}
        trackers[name] = TRACKER_MAP[cfg.tracker_type](args=cfg)
        os.unlink(path)
    out = {n: defaultdict(lambda: {"frames": [], "boxes": [], "cls": None, "clsvotes": Counter()}) for n in trackers}
    t_track = {n: [] for n in trackers}
    t_det, n_frames, det_counts = [], 0, []
    idx = -1
    t_wall = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or idx + 1 >= max_frame:
            break
        idx += 1
        if idx % step:
            continue
        t0 = time.perf_counter()
        r = model.predict(frame, classes=cls_ids, conf=0.1, imgsz=a.imgsz, agnostic_nms=True, device="cpu", verbose=False)[0]
        t_det.append(time.perf_counter() - t0)
        det_counts.append(len(r.boxes))
        n_frames += 1
        for name, tr in trackers.items():
            det = r.boxes.cpu().numpy()
            t0 = time.perf_counter()
            tracks = tr.update(det, frame)
            t_track[name].append(time.perf_counter() - t0)
            for row in tracks:
                x1_, y1_, x2_, y2_, tid, sc, c = row[:7]
                d = out[name][int(tid)]
                d["frames"].append(idx + 1)          # 1-based NATIVE frame number
                d["boxes"].append((float(x1_), float(y1_), float(x2_), float(y2_)))
                d["clsvotes"][names[int(c)]] += 1
    cap.release()
    wall = time.time() - t_wall
    for name in out:
        for d in out[name].values():
            d["cls"] = d["clsvotes"].most_common(1)[0][0]
    return {"out": out, "t_track": t_track, "t_det": t_det, "n_frames": n_frames, "wall": wall, "native_fps": native,
            "eff_fps": eff_fps, "step": step, "W": W, "H": H, "det_counts": det_counts, "cfgs": cfgs,
            "load_avg": os.getloadavg()[0]}


def mot_rows(tracks):
    return [(f, tid, b) for tid, d in tracks.items() for f, b in zip(d["frames"], d["boxes"])]


def write_mot(path, tracks):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, b in sorted(mot_rows(tracks), key=lambda r: (r[0], r[1])):
            f.write(f"{fr},{tid},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},1,-1,-1,-1\n")


def x2(a):
    gt_path = Path(a.gt) if a.gt else GT_DIR / "tracking" / "gt.txt"
    res = new_result("x2_tracking", "X2 Tracking MOTA / IDF1 / ID switches", [CLIP, gt_path] if gt_path.exists() else [CLIP])
    a.video = a.video or CLIP
    a.trackers = a.trackers.split(",")
    P = proxy_tracking(a)
    outdir = RESULTS / "x2_tracks"
    for n in a.trackers:
        write_mot(outdir / f"{n}.txt", P["out"][n])
    frames = {f for f in range(1, int(a.proxy_seconds * P["native_fps"]) + 2)}
    # ---- proxy table
    rows, lens = [], {}
    min_len = 5
    for n in a.trackers:
        T = P["out"][n]
        L = [len(d["frames"]) for d in T.values()]
        sec = [x / P["eff_fps"] for x in L]
        long = [x for x in L if x >= min_len]
        fr = fragment_candidates({t: d for t, d in T.items() if len(d["frames"]) >= min_len}, P["eff_fps"], P["W"], P["H"])
        per_frame = Counter(f for d in T.values() for f in d["frames"])
        tt = P["t_track"][n]
        rows.append([n, len(T), len(long), fmt(statistics.median(sec), 2), fmt(percentiles(sec).get("p25"), 2),
                     fmt(percentiles(sec).get("p75"), 2), fmt(max(sec), 1), sum(1 for x in L if x < 10),
                     fr["fragment_candidates"], fmt(sum(per_frame.values()) / max(1, P["n_frames"]), 1),
                     fmt(1000 * statistics.mean(tt), 2), fmt(1000 * statistics.median(tt), 2), "proxy (no GT) on Dhaka clip"])
        lens[n] = {"n_tracks": len(T), "n_tracks_ge5": len(long), "len_seconds": percentiles(sec), "max_s": max(sec),
                   "short_lt1s": sum(1 for x in L if x < 10), **fr, "tracker_ms_mean": 1000 * statistics.mean(tt),
                   "tracker_ms_median": 1000 * statistics.median(tt), "tracker_ms_p95": 1000 * percentiles(tt, (95,))["p95"],
                   "boxes_per_frame_mean": sum(per_frame.values()) / max(1, P["n_frames"])}
    table(res, f"ByteTrack vs BoT-SORT proxy, first {a.proxy_seconds:.0f} s of {CLIP.name}",
          ["Tracker", "Tracks", "Tracks >=5 frames", "Median track length (s)", "p25 (s)", "p75 (s)", "Longest (s)",
           "Tracks < 1 s", "Fragment candidates (heuristic)", "Boxes/frame", "Tracker ms/frame (mean)",
           "Tracker ms/frame (median)", "Status"], rows,
          f"{provenance.make()['command']}; {P['n_frames']} frames sampled every {P['step']} native frames "
          f"({P['eff_fps']:.2f} fps effective of {P['native_fps']:.2f}); {P['W']}x{P['H']}; yolo imgsz {a.imgsz}; CPU i5-11400H; "
          f"load average at end {P['load_avg']:.1f}; configs in raw.tracker_configs")
    dt = P["t_det"]
    table(res, "Cost split on this CPU (same run)", ["Quantity", "Value", "Status"],
          [["Detector ms/frame (mean, imgsz %d)" % a.imgsz, fmt(1000 * statistics.mean(dt), 1), "measured on Dhaka clip (CPU timing)"],
           ["Detector ms/frame (median)", fmt(1000 * statistics.median(dt), 1), "measured on Dhaka clip (CPU timing)"]] +
          [[f"{n} end-to-end fps (detector + tracker)", fmt(1 / (statistics.mean(dt) + statistics.mean(P['t_track'][n])), 2),
            "derived from measured means (Dhaka clip, CPU)"] for n in a.trackers] +
          [["Wall time for whole run (both trackers + decoding)", f"{P['wall']:.0f} s", "measured on Dhaka clip (CPU timing)"]],
          "same run as above; per-frame times from time.perf_counter around model.predict and tracker.update")
    # ---- agreement of the two algorithms (NOT accuracy)
    if len(a.trackers) == 2:
        A, B = a.trackers
        ag = tracking.evaluate(mot_rows(P["out"][A]), mot_rows(P["out"][B]), 0.5)
        table(res, f"Agreement of {B} with {A} treated as reference (NOT accuracy: neither is ground truth)",
              ["Quantity", "Value", "Status"],
              [["IDF1 of %s against %s" % (B, A), fmt(ag["idf1"]), "proxy (agreement only, not accuracy)"],
               ["ID switches of %s relative to %s" % (B, A), ag["id_switches"], "proxy (agreement only, not accuracy)"],
               ["Fragmentations relative", ag["fragmentations"], "proxy (agreement only, not accuracy)"]],
              "evaluation.tracking.evaluate on the two tracker outputs, IoU 0.5")
        res["raw"]["agreement"] = ag
    res["raw"].update({"proxy": lens, "tracker_configs": P["cfgs"], "det_ms_mean": 1000 * statistics.mean(dt),
                       "frames": P["n_frames"], "step": P["step"], "effective_fps": P["eff_fps"],
                       "tracks_dir": rel(outdir)})
    res["notes"] += [
        "WHAT THE PROXY CAN SHOW: relative cost of the two association algorithms on identical detections on this "
        "CPU; how many tracks each makes and how long they live; whether one makes many more short tracks "
        "(a symptom of fragmentation) on the same footage.",
        "WHAT IT CANNOT SHOW: which tracker has fewer ID switches or higher MOTA/IDF1. Fewer, longer tracks can mean "
        "better continuity OR identities merged across different vehicles (an ID swap leaves no trace in these counts). "
        "'Fragment candidates' is a heuristic (end away from the border, same-class start within 2 s and 0.6 box diagonals) "
        "and can be inflated by coincidence. Choosing BoT-SORT over ByteTrack for accuracy needs MOT ground truth (DEC-3).",
        "BoT-SORT here is Ultralytics' configuration WITHOUT appearance ReID (with_reid False, the pipeline default); "
        "its difference from ByteTrack is camera-motion compensation (sparseOptFlow) and score fusion. This clip comes "
        "from a fixed camera, so camera-motion compensation has little to compensate. BoT-SORT with ReID was not run.",
    ]
    if gt_path.exists():
        hyp = {n: mot_rows(P["out"][n]) for n in a.trackers}
        gt_rows = tracking.load_mot(gt_path)
        processed = set(range(1, int(a.proxy_seconds * P["native_fps"]) + 2, P["step"]))
        # evaluate on sampled frames present in the GT only
        gt_f = {r[0] for r in gt_rows} & processed
        rows = []
        for n in a.trackers:
            m = tracking.evaluate(gt_rows, hyp[n], 0.5, frames=gt_f)
            rows.append([n, fmt(m["mota"]), fmt(m["motp_iou"]), m["id_switches"], m["fragmentations"], fmt(m["idf1"]),
                         fmt(m["idp"]), fmt(m["idr"]), m["num_gt"], M_DHAKA])
        table(res, "Tracking with ground truth", ["Tracker", "MOTA", "MOTP (mean IoU)", "ID switches", "Fragmentations",
              "IDF1", "IDP", "IDR", "GT boxes", "Status"], rows,
              f"GT {rel(gt_path)}; scored on {len(gt_f)} sampled frames that have GT; IoU 0.5")
    else:
        need(res, "MOTA / IDF1 / ID switches for ByteTrack vs BoT-SORT",
             f"MOT 1.1 ground truth of the clip ({rel(gt_path)}): CVAT track annotations of every vehicle, at least on "
             "every 3rd native frame, for a >=60 s stretch (docs/EVALUATION.md section 2). results/x2_tracks/*.txt "
             "can be imported into CVAT as pre-annotations to cut labelling time")
    return save(res)


# ---------------------------------------------------------------- X3 rules

def test_summary(modules):
    """Run unittest modules in-process; per-module counts and failures."""
    out = {}
    for m in modules:
        suite = unittest.defaultTestLoader.loadTestsFromName(m)
        t0 = time.time()
        r = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
        names = Counter()
        def walk(s):
            for t in s:
                if isinstance(t, unittest.TestSuite):
                    walk(t)
                else:
                    names[type(t).__name__] += 1
        walk(suite)
        out[m] = {"ran": r.testsRun, "failures": len(r.failures), "errors": len(r.errors), "skipped": len(r.skipped),
                  "seconds": round(time.time() - t0, 1), "classes": dict(names),
                  "failed": [str(f[0]) for f in r.failures + r.errors][:10]}
    return out


def x3(a):
    res = new_result("x3_rules", "X3 Rule precision / recall / coverage",
                     [ROOT / "tools" / "bench_accuracy.py", ROOT / "training" / "score_accuracy.py"])
    # 1. bench_accuracy: simulated camera with known truth
    runs = {}
    for label, extra in (("default (detector errors on)", []), ("clean (geometry floor)", ["--clean"])):
        r = run_cmd([PY, "tools/bench_accuracy.py"] + extra, timeout=600)
        try:
            runs[label] = json.loads(r["stdout"])
        except Exception:
            runs[label] = {"error": r["stderr"] or r["stdout"][-500:]}
        runs[label]["_cmd"], runs[label]["_seconds"] = r["cmd"], r["seconds"]
    rows = []
    for label, d in runs.items():
        for name, v in d.get("decisions", {}).items():
            tr, caught, fa = v["truth"], v["caught"], v["false_alarms"]
            neg = v.get("of_legal", v.get("of_moving"))
            rows.append([label, name, tr, caught, fa, fmt(caught / tr if tr else None, 2),
                         fmt(caught / (caught + fa) if caught + fa else None, 2), neg, SYN])
    table(res, "Rule decisions against simulated ground truth (tools/bench_accuracy.py)",
          ["Scenario", "Rule", "True events", "Caught", "False alarms", "Recall", "Precision", "Legal vehicles that could "
           "have been false alarms", "Status"], rows,
          "; ".join(f"`{d['_cmd']}` ({d['_seconds']} s)" for d in runs.values()) +
          "; seed 0, 180 s of simulated traffic at 15 fps through the real Pipeline with a simulated detector. "
          "Counts are tiny (2-6 events per rule): treat as a regression check, not an accuracy estimate")
    srows = []
    for label, d in runs.items():
        if "speed" in d:
            s = d["speed"]
            srows.append([label, s["n"], s["mae_kph"], s["bias_kph"], s["p95_abs_kph"], s["mape_pct"], SYN])
    table(res, "Speed estimation error vs simulated truth", ["Scenario", "Vehicle-samples", "MAE km/h", "Bias km/h",
          "p95 abs error km/h", "MAPE %", "Status"], srows, "same runs as above (bench_accuracy.py 'speed')")
    # 2. unit tests = synthetic coverage of each rule
    mods = ["tests.test_violations", "tests.test_bus_lane", "tests.test_accuracy", "tests.test_behaviour"]
    ts = test_summary(mods)
    table(res, "Synthetic scenario tests per rule module", ["Test module", "Tests", "Failures+errors", "Seconds", "Status"],
          [[m, v["ran"], v["failures"] + v["errors"], v["seconds"], SYN] for m, v in ts.items()],
          "unittest run in-process by tools/evaluate.py x3; these are scripted scenarios (pass/fail), not accuracy on real traffic")
    res["raw"]["bench"] = runs
    res["raw"]["tests"] = ts
    # 3. real-footage precision/recall: needs hand ground truth
    gt = GT_DIR / "violations" / "events.csv"
    if gt.exists() and a.cases and Path(a.cases).exists():
        G, Pc = events.load_events(gt), events.load_events(a.cases)
        ev = events.evaluate(G, Pc, mode=a.match, thr=a.iou, tol=a.tol, duration_s=a.duration)
        rows = [[c, v["n_gt"], v["n_pred"], v["tp"], fmt(v["precision"], 2), fmt(v["recall"], 2), fmt(v["f1"], 2), M_DHAKA]
                for c, v in ev["per_code"].items()]
        m = ev["micro"]
        rows.append(["all (micro)", sum(v["n_gt"] for v in ev["per_code"].values()),
                     sum(v["n_pred"] for v in ev["per_code"].values()), m["tp"], fmt(m["precision"], 2), fmt(m["recall"], 2),
                     fmt(m["f1"], 2), M_DHAKA])
        table(res, "Violation events vs hand-labelled truth", ["Rule", "True", "Cases", "Matched", "Precision", "Recall", "F1",
              "Status"], rows, f"GT {rel(gt)}, cases {rel(a.cases)}, matching {a.match} (tol {a.tol} s / IoU {a.iou})")
        res["raw"]["events"] = ev
    else:
        need(res, "Per-rule precision / recall / FP rate on real Dhaka footage",
             f"a hand-labelled violation CSV ({rel(gt)}; schema in docs/EVALUATION.md section 3) for the same video, plus the "
             "system's Export CSV of cases for that video passed as --cases. training/score_accuracy.py needs exactly this; "
             "only the blank training/ground_truth_template.csv exists")
    return save(res)


# ---------------------------------------------------------------- X4 impact

def x4(a):
    res = new_result("x4_impact", "X4 Impact score vs severity (Spearman rho)", [ROOT / "tests" / "test_impact.py"])
    sev = GT_DIR / "severity" / "ratings.csv"
    if sev.exists() and a.scores:
        rows = list(csv.DictReader(open(sev, encoding="utf-8-sig")))
        sc = {r["case_id"]: float(r["impact_score"]) for r in csv.DictReader(open(a.scores, encoding="utf-8-sig"))}
        ids = [r["case_id"] for r in rows if r["case_id"] in sc]
        r1 = [float(r["rater1"]) for r in rows if r["case_id"] in sc]
        r2 = [float(r["rater2"]) for r in rows if r["case_id"] in sc]
        mean = [(x + y) / 2 for x, y in zip(r1, r2)]
        v = stats.validate([sc[i] for i in ids], mean)
        table(res, "Impact score vs mean of two human severity ratings", ["Quantity", "Value", "Status"],
              [["cases", v["n"], M_DHAKA], ["Spearman rho", fmt(v["rho"]), M_DHAKA], ["95% bootstrap CI", str(v["ci95"]), M_DHAKA],
               ["permutation p", fmt(v["p_value"]), M_DHAKA], ["inter-rater rho", fmt(stats.spearman(r1, r2)), M_DHAKA]],
              f"{rel(sev)} and {rel(a.scores)}; surveillance.impact.validate")
    else:
        need(res, "Spearman rho between the impact score and human severity",
             f"human severity ratings of real cases ({rel(sev)}: case_id,rater1,rater2 on a 1-5 scale, >=30 cases covering "
             "several rule codes and impact levels, protocol in docs/EVALUATION.md section 4) and the system's impact scores "
             "for those cases (--scores CSV: case_id,impact_score). No human rating exists in the repo")
    name = "test_weights_fitted_to_simulated_events_predict_their_delay_in_vehicle_seconds"
    r = run_cmd([PY, "-m", "unittest", "-v", f"tests.test_impact.{_find_class('tests/test_impact.py', name)}.{name}"], timeout=600)
    line = next((l for l in (r["stdout"] + r["stderr"]).splitlines() if "27 simulated events" in l), "")
    m = re.search(r"held-out MAE (\d+) veh\*s \(mean delay (\d+)\), rho\(calibrated, delay\) ([\d.]+), rho\(equal weights, delay\) ([\d.]+)", line)
    if m:
        table(res, "Impact score vs SIMULATED excess delay (WP1 synthetic validation; not human severity)",
              ["Quantity", "Value", "Status"],
              [["Events (fit 18 / held-out 9)", "27", SYN], ["Spearman rho, calibrated score vs simulated delay", m.group(3), SYN],
               ["Spearman rho, equal-weight score vs simulated delay", m.group(4), SYN],
               ["Held-out MAE (veh*s) / mean delay (veh*s)", f"{m.group(1)} / {m.group(2)}", SYN]],
              f"`{r['cmd']}` ({r['seconds']} s) prints: {line.strip()[:260]}. The 'truth' is the simulator's own excess delay, so this "
              "checks the score tracks congestion in a model the team wrote, nothing about human judgement of severity")
    else:
        res["notes"].append("could not parse the synthetic validation line from tests.test_impact: " + (r["stderr"] or "")[-300:])
    res["raw"]["impact_test"] = {k: r[k] for k in ("cmd", "returncode", "seconds")} | {"line": line}
    return save(res)


def _find_class(path, test_name):
    cls = None
    for l in (ROOT / path).read_text().splitlines():
        mm = re.match(r"class (\w+)\(", l)
        if mm:
            cls = mm.group(1)
        if f"def {test_name}" in l:
            return cls
    raise SystemExit(f"{test_name} not found in {path}")


# ---------------------------------------------------------------- X5 VLM

def x5(a):
    res = new_result("x5_vlm", "X5 VLM explanation quality", [ROOT / "tools" / "vlm_probe.py", ROOT / "models" / "README.md"])
    src = (ROOT / "tools" / "vlm_probe.py").read_text()
    names = re.findall(r'^\s+\("([\w-]+)", "(\w+\.jpg)"', src, re.M)
    table(res, "Insufficient-evidence probe set as it exists (tools/vlm_probe.py)", ["Probe", "Photograph", "Must answer", "Status"],
          [["unrelated-evidence", "zidane.jpg (Ultralytics sample)", "insufficient / contradicted", "synthetic (probe definition)"],
           ["wrong-class", "bus.jpg (Ultralytics sample)", "insufficient / contradicted", "synthetic (probe definition)"],
           ["plausible-case", "bus.jpg", "any verdict; prose must read sensibly", "synthetic (probe definition)"]],
          f"parsed from tools/vlm_probe.py ({len(names)} probes found); NOT re-run: the harness never calls the live VLM server")
    table(res, "Last recorded result (copied, not re-run)", ["Run", "Result", "Status"],
          [["First attempt, Qwen2.5-VL-3B Q4_K_M", "1/3 passed: confirmed both impossible findings at 90% confidence", "synthetic (3 probes on non-Dhaka sample photographs; copied)"],
           ["After prompt change + verdict guard (to_explanation)", "3/3 passed", "synthetic (3 probes on non-Dhaka sample photographs; copied)"]],
          "models/README.md section 'Measured: the insufficient-evidence probe' and README.md (VLM section, 'It now passes 3/3'); "
          "3 probes cannot estimate a refusal rate and the three prompt/guard changes were not ablated")
    need(res, "Refusal rate on insufficient evidence; explanation accuracy",
         "a probe set of >=40 cases: ~20 insufficient-evidence (crop of the wrong vehicle class, blank/blurred crop, wrong "
         "time, other-lane vehicle) and ~20 supported, taken from real Dhaka cases, each with a hand label in {supported, "
         "contradicted, insufficient}; score = confusion matrix + 'false support' rate (docs/EVALUATION.md section 5). "
         "Free-text quality needs 2 human raters (1-5 faithfulness) on ~30 explanations; no model-based scoring exists")
    return save(res)


# ---------------------------------------------------------------- X6 cross-camera

def x6(a):
    res = new_result("x6_crosscamera", "X6 Cross-camera association", [ROOT / "research" / "cityflow" / "README.md", ROOT / "tools" / "reid_eval.py"])
    readme = (ROOT / "research" / "cityflow" / "README.md").read_text()
    chk = {"mAP **0.427**": "0.427", "rank-1 **0.560**": "0.560", "IDF1 **46.50**": "46.50", "IDF1 **40.54**": "40.54"}
    verified = {k: (k in readme) for k in chk}
    ext_rows = [["ReID retrieval, 145 held-out S02 vehicles, cross-camera (ResNet50-IBN-a @320)", "mAP 0.427, rank-1 0.560", EXT],
                ["MTMC S02, default settings", "IDF1 40.54 (IDP 34.91, IDR 48.34)", EXT],
                ["MTMC S02, tuned on S02 itself (optimistic)", "IDF1 47.84", EXT],
                ["MTMC S02, 2-fold tuned/scored on halves", "IDF1 46.50", EXT]]
    table(res, "CityFlowV2 (external benchmark; US roads, NOT Dhaka)", ["Quantity", "Value", "Status"], ext_rows,
          "copied from research/cityflow/README.md section 'What it scored' (all numbers there were printed by the team's notebook; "
          f"not re-run here). Verified present in README by string match: {verified}")
    # within-camera reid_eval: fresh run if asked, else copy of README table
    runs = {}
    if a.run_reid:
        for emb in ("reid", "mobilenet"):
            r = run_cmd([PY, "tools/reid_eval.py", str(CLIP), "--seconds", str(a.reid_seconds), "--embedder", emb], timeout=1800)
            runs[emb] = r
        res["raw"]["reid_eval"] = runs
        rowsf = []
        for emb, r in runs.items():
            for l in r["stdout"].splitlines():
                m = re.match(r"\s+(appearance \(\w+\)|colour histogram|CIELAB colour)\s+same .*separation\s+([\d.]+)\s+rank-1\s+([\d.]+)%", l)
                if m:
                    rowsf.append([emb, m.group(1), m.group(2), m.group(3) + "%", M_DHAKA])
        table(res, f"Within-camera appearance separation, fresh run (first {a.reid_seconds:.0f} s of the clip; first block = vs every vehicle, "
              "second = same class only)", ["Embedder run", "Cue", "Separation", "Rank-1", "Status"], rowsf,
              "; ".join(f"`{r['cmd']}` ({r['seconds']} s)" for r in runs.values()) + ". Early vs late crops of the SAME camera: "
              "this under-states cross-camera difficulty")
    table(res, "Within-camera appearance cues on this project's footage (copied)", ["Cue (same class only)", "Separation / rank-1", "Status"],
          [["Colour histogram", "1.79 / 64.5%", M_DHAKA], ["CIELAB colour", "1.79 / 43.5%", M_DHAKA],
           ["MobileNetV3-Small (ImageNet)", "1.76 / 75.8%", M_DHAKA], ["ReID network (CityFlow)", "3.35 / 98.4%", M_DHAKA]],
          "copied from research/cityflow/README.md 'Measured on this project's footage' (tools/reid_eval.py on 50 s of "
          "data/videos/20261025_092449_side1_1.mp4, 62 vehicles); one camera, early-vs-late halves of the same tracks")
    need(res, "Cross-camera pair precision/recall and MTMC IDF1 on Dhaka",
         "footage from >=2 cameras with an overlapping set of vehicles, and a hand-made table mapping (camera, local track id) -> "
         "global vehicle id for >=40 vehicles seen on both cameras (docs/EVALUATION.md section 6). The footbridge labels "
         "named in SUPERVISION X6 are not in the repo. Metrics are implemented and tested (evaluation/retrieval.py: "
         "pair_precision_recall, mtmc_idf1)")
    return save(res)


# ---------------------------------------------------------------- X7 grounding

def x7(a):
    res = new_result("x7_grounding", "X7 Grounding retrieval MRR / Recall@k")
    need(res, "MRR and Recall@k of the natural-language query panel",
         "the query panel (WP3, not built yet) and a set of >=30 queries (text) each with the ids of the true matching segments "
         "labelled by hand from the same footage (docs/EVALUATION.md section 7). Interface: evaluation.retrieval.mrr(rankings, relevant), "
         "recall_at_k(rankings, relevant, k); rankings = {query: [segment_id best-first]}, relevant = {query: {segment_id}}")
    return save(res)


# ---------------------------------------------------------------- OPTIMIZE

def optimize(a):
    res = new_result("optimize", "OPTIMIZE signal control in simulation", [ROOT / "tools" / "signal_sim.py"])
    out = RESULTS / "optimize_raw.json"
    r = run_cmd([PY, "tools/signal_sim.py", "--quick", "--out", str(out)], timeout=900)
    if r["returncode"] != 0 or not out.exists():
        res["notes"].append("tools/signal_sim.py failed: " + r["stderr"][-400:])
        return save(res)
    d = json.loads(out.read_text())

    def ms(xs):
        return f"{statistics.mean(xs):.1f} ± {statistics.pstdev(xs):.1f}"
    rows = []
    for sc, by in d["comparison"].items():
        for name, rs in by.items():
            rows.append([sc, name, ms([x["mean_delay_s"] for x in rs]), ms([x["p95_delay_s"] for x in rs]),
                         ms([x["throughput_vph"] for x in rs]), f"{statistics.mean(x['unfinished'] for x in rs):.1f}", SYN])
    table(res, "Fixed-time vs max-pressure on a synthetic 3x3 grid", ["Load", "Controller", "Mean delay (s)", "p95 delay (s)",
          "Throughput (veh/h)", "Unfinished", "Status"], rows,
          f"`{r['cmd']}` ({r['seconds']} s), seeds {d['seeds']}, horizon {d['horizon_s']:.0f} s; mean ± sd over seeds; "
          "mesoscopic simulator written for this project (surveillance/optimize/sim.py), not SUMO, not Dhaka data")
    rows = []
    for b, by in (d.get("ev") or {}).items():
        none = by["no priority"]
        for v, rs in by.items():
            e, o = [x["ev_travel_s"] for x in rs], [x["others_mean_delay_s"] for x in rs]
            rows.append([b, v, ms(e), ms(o), SYN])
    table(res, "Emergency vehicle corridor (5 ambulances along the middle street)", ["Base controller", "Priority",
          "EV travel time (s)", "Others' mean delay (s)", "Status"], rows, "same run; load x0.8")
    res["raw"]["signal_sim"] = d
    res["raw"]["seconds"] = r["seconds"]
    res["notes"].append("Synthetic results: demand levels were chosen to put the corridor near capacity and nothing here has driven a real "
                        "signal. The tool's own docstring states the min-green was swept on seeds 0-2 and these tables use seeds 10+.")
    return save(res)


# ---------------------------------------------------------------- report

def report(a):
    import evaluation.report as R
    R.write(RESULTS, ROOT / "REPORT_RESULTS.md")


def all_(a):
    for f, ns in ((x1, {}), (x2, {}), (x3, {}), (x4, {}), (x5, {}), (x6, {}), (x7, {}), (optimize, {})):
        t0 = time.time()
        f(a)
        print(f"  {f.__name__} took {time.time() - t0:.0f} s")
    report(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "optimize", "all", "report"])
    ap.add_argument("--data", help="x1: YOLO dataset yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--video")
    ap.add_argument("--gt", help="x2: MOT 1.1 gt.txt for the clip")
    ap.add_argument("--proxy-seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=10.0, help="x2: analysis rate (server default for files is 960 px)")
    ap.add_argument("--trackers", default="bytetrack,botsort")
    ap.add_argument("--cases", help="x3: system's violations CSV for the labelled video")
    ap.add_argument("--match", choices=["tolerance", "iou"], default="tolerance")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--duration", type=float, help="x3: video length in s, for false alarms per hour")
    ap.add_argument("--scores", help="x4: CSV case_id,impact_score")
    ap.add_argument("--run-reid", action="store_true", help="x6: re-run tools/reid_eval.py (minutes on CPU)")
    ap.add_argument("--reid-seconds", type=float, default=30.0)
    a = ap.parse_args()
    {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6, "x7": x7, "optimize": optimize, "all": all_,
     "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
