#!/usr/bin/env python3
"""Do the appearance cues actually tell one vehicle from another, on this footage?

The honest question behind cross-camera matching. Each tracked vehicle's crops are split into an early half
and a late half - different moments, different distances, different angles, the same vehicle - and we ask
whether the two halves of one vehicle look more alike than two different vehicles do.

    ./.venv/bin/python tools/reid_eval.py data/videos/<clip>.mp4 --seconds 60
    ./.venv/bin/python tools/reid_eval.py data/videos/<clip>.mp4 --seconds 60 --embedder reid
    ./.venv/bin/python tools/reid_eval.py data/videos/<clip>.mp4 --seconds 60 --embedder mobilenet

`--embedder` picks which appearance model is measured (the team's ReID network or the ImageNet MobileNet);
without it, whatever `surveillance.embed` would use. The colour cues do not depend on it. If the model asked
for cannot be loaded the tool stops rather than report another model's numbers under its name.

Reported per cue:
  separation   (mean same-vehicle score - mean different-vehicle score) / spread. Above ~1 is useful.
  rank-1       how often a vehicle's other half is its nearest neighbour out of all candidates.
The table is printed twice. First against every other vehicle, which is how this tool has always reported
and what earlier numbers were measured with. Then against vehicles of the same class only, which is the
question the matcher actually faces, since it never links a Car to a CNG: telling a bus from a motorbike is
the easy part of the first table and none of the work. Then the appearance vector class by class, and what
one vehicle costs to embed on this CPU.

`--max-crop-px 128` shrinks every crop the way `reid.TrackletCollector` stores them (long side at most 128)
before any cue sees it. The default keeps full resolution, which is the best case for a model that wants
detail; the difference between the two is what the collector's 128 px would cost it. `--reid-size` and
`--reid-flip` change the ReID network's input size and switch on the notebook's mirror averaging, to see what
a cheaper or a more faithful setting is worth.

A cue that scores near zero is not broken - the matcher switches such cues off by itself - but it does mean
geometry is carrying the identification on this footage, which is worth knowing before trusting a link.
One camera cannot show the other half of the problem: both halves of a vehicle come from the same viewpoint
and background, so this under-states how hard cross-camera matching is - it cannot say whether a model copes
with the change of angle, light and background between two cameras.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surveillance import embed, reid              # noqa: E402
from surveillance.detector import Detector        # noqa: E402
from surveillance.embed import Embedder, similarity  # noqa: E402
from surveillance.sources import FileSource       # noqa: E402

MIN_CROPS = 6
MIN_CLASS = 6          # vehicles a class needs before its own row means anything
COST_CROPS = 10        # crops per vehicle in the cost line: what the notebook (and a long pass) keeps
COST_REPEATS = 3


def shrink(crop, max_px):
    """The crop as `reid.TrackletCollector` would have stored it: long side at most `max_px`."""
    scale = min(1.0, max_px / max(crop.shape[:2])) if max_px else 1.0
    if scale >= 1.0:
        return crop
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def collect(video, seconds, imgsz, fps, min_px, max_crop_px=0):
    src = FileSource(video, fps, 0, False).open()
    det = Detector("bd_vehicles_best.pt", imgsz, "auto", tracker="bytetrack", fps=src.fps, log=lambda m: None)
    crops, cls = defaultdict(list), {}
    t0 = None
    try:
        while True:
            got = src.read()
            if got is None:
                break
            frame, t = got[0], got[1]
            t0 = t if t0 is None else t0
            tracks = det.track(frame)
            boxes = [b for b, _, _ in tracks]
            for box, tid, name in tracks:
                x1, y1, x2, y2 = [int(v) for v in box]
                h, w = frame.shape[:2]
                if min(x2 - x1, y2 - y1) < min_px or x1 < 3 or y1 < 3 or x2 > w - 3 or y2 > h - 3:
                    continue
                area = (x2 - x1) * (y2 - y1)
                overlap = max([max(0, min(x2, o[2]) - max(x1, o[0])) * max(0, min(y2, o[3]) - max(y1, o[1]))
                               / area for o in boxes if tuple(o) != tuple(box)] or [0])
                if overlap > reid.MAX_OVERLAP:
                    continue
                c = frame[y1:y2, x1:x2]
                if c.size:
                    crops[tid].append(shrink(c.copy(), max_crop_px))
                    cls[tid] = name
            if t - t0 >= seconds:
                break
    finally:
        src.close()
        det.close()
    return {k: v for k, v in crops.items() if len(v) >= MIN_CROPS}, cls


def halves(crops):
    """(early half, late half) of one vehicle's crops, evenly spread."""
    n = len(crops)
    step = max(1, n // 12)
    sel = crops[::step][:12]
    half = len(sel) // 2
    return sel[:half], sel[half:2 * half]


def report(name, pos, neg, rank1, n, width=22):
    pos, neg = np.array(pos, float), np.array(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        print(f"  {name:<{width}} not enough data")
        return
    spread = max(np.sqrt(0.5 * (pos.std() ** 2 + neg.std() ** 2)), 1e-6)
    sep = (pos.mean() - neg.mean()) / spread
    verdict = "useful" if sep > 1.0 else "weak" if sep > 0.4 else "no real signal"
    print(f"  {name:<{width}} same {pos.mean():+.3f}  different {neg.mean():+.3f}  "
          f"separation {sep:5.2f}  rank-1 {rank1 / max(n, 1):5.1%}   {verdict}")


def compare(ids, A, B, score, same_class=False):
    """(same-vehicle scores, different-vehicle scores, vehicles whose other half ranked first, vehicles
    that had a rival at all). With `same_class`, only vehicles of the same class count as rivals."""
    pos, neg, rank1, n = [], [], 0, 0
    for i in ids:
        s_self = score(A[i], B[i])
        if not np.isfinite(s_self):
            continue
        others = [score(A[i], B[j]) for j in ids if j != i and (not same_class or A[i]["cls"] == A[j]["cls"])]
        others = [s for s in others if np.isfinite(s)]
        if not others:
            continue
        pos.append(s_self)
        neg += others
        n += 1
        rank1 += s_self > max(others)
    return pos, neg, rank1, n


def cost_of_a_vehicle(crops):
    """(median, best) milliseconds to embed one vehicle of `COST_CROPS` crops, after a warm-up call. Vehicles
    with fewer crops are padded by repeating theirs, which costs the same. The median is what a busy machine
    delivers; the best is nearer what an idle one would."""
    runs = []
    for cl in list(crops.values())[:6]:
        ten = (cl * COST_CROPS)[:COST_CROPS]
        Embedder.encode(ten)
        for _ in range(COST_REPEATS):
            t = time.perf_counter()
            Embedder.encode(ten)
            runs.append(1000 * (time.perf_counter() - t))
    return (float(np.median(runs)), float(min(runs))) if runs else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--embedder", choices=("reid", "mobilenet"),
                    help="appearance model to measure (default: whichever surveillance.embed picks)")
    ap.add_argument("--max-crop-px", type=int, default=0,
                    help="shrink crops to this long side first, as the tracklet collector does (128); 0 = full size")
    ap.add_argument("--reid-size", type=int, help="ReID input size in px (default: the 320 it was trained at)")
    ap.add_argument("--reid-flip", action="store_true",
                    help="ReID: also run each crop mirrored and add the vectors, as the notebook does (2x the CPU)")
    a = ap.parse_args()

    if a.reid_size:
        embed.REID_SIZE = a.reid_size
    if a.reid_flip:
        embed.REID_FLIP = True
    if a.embedder:
        Embedder.use(a.embedder)
    model = Embedder.name                       # loads it now, so a failure shows before the slow part
    if model is None or (a.embedder and model != a.embedder):
        print(f"--embedder {a.embedder or 'default'} is not available here"
              f"{'' if model is None else f' (surveillance.embed fell back to {model})'}; nothing measured.")
        return 2
    options = f", input {embed.REID_SIZE or 320} px, mirror averaging {'on' if embed.REID_FLIP else 'off'}" \
        if model == "reid" else ""
    print(f"Appearance model: {model} ({Embedder.dim} numbers, {Embedder.tag}{options}); crops "
          f"{'at full size' if not a.max_crop_px else f'shrunk to {a.max_crop_px} px'}")
    print(f"Tracking {a.seconds:.0f}s of {Path(a.video).name} ...")
    crops, cls = collect(a.video, a.seconds, a.imgsz, a.fps, a.min_px, a.max_crop_px)
    print(f"{len(crops)} vehicles with at least {MIN_CROPS} clean crops\n")
    if len(crops) < 8:
        print("Too few vehicles to measure. Try a longer --seconds or a smaller --min-px.")
        return 1

    A, B = {}, {}
    for tid, cl in crops.items():
        first, second = halves(cl)
        if not first or not second:
            continue
        two = cls.get(tid) in reid.TWO_WHEEL
        A[tid] = dict(zip(("hist", "lab"), reid.signature(first, two)))
        B[tid] = dict(zip(("hist", "lab"), reid.signature(second, two)))
        A[tid]["embed"] = Embedder.encode(first)
        B[tid]["embed"] = Embedder.encode(second)
        A[tid]["cls"] = B[tid]["cls"] = cls.get(tid, "")
    ids = [t for t in A if A[t]["hist"] is not None and B[t]["hist"] is not None]
    print(f"comparing {len(ids)} vehicles, each against every other\n")

    scorers = {
        "hist": lambda x, y: float(np.sum(np.sqrt(x["hist"] * y["hist"]))),
        "lab": lambda x, y: -float(np.linalg.norm(x["lab"] - y["lab"])),
        "embed": lambda x, y: similarity(x["embed"], y["embed"]),
    }
    names = {"hist": "colour histogram", "lab": "CIELAB colour", "embed": f"appearance ({model})"}

    for title, same_class in (("", False), ("Against vehicles of the same class only - the matcher never links "
                                            "across classes:", True)):
        if title:
            print(f"\n{title}")
        for cue in ("hist", "lab", "embed"):
            pos, neg, rank1, n = compare(ids, A, B, scorers[cue], same_class)
            report(names[cue], pos, neg, rank1, n)

    by_class = defaultdict(list)
    for t in ids:
        by_class[A[t]["cls"]].append(t)
    print(f"\nAppearance vector by class (same-class rivals; classes with fewer than {MIN_CLASS} vehicles "
          f"are too few to read):")
    for name, members in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        if len(members) < MIN_CLASS:
            print(f"  {name:<12} {len(members):>3} vehicles  too few")
            continue
        for label, cue in (("appearance", "embed"), ("colour histogram", "hist")):
            pos, neg, rank1, n = compare(members, A, B, scorers[cue], same_class=True)
            report(f"{name} ({len(members)}) {label}", pos, neg, rank1, n, width=32)

    # and what geometry alone would do: class agreement is the cue the matcher treats as absolute
    same_cls = sum(1 for i in ids for j in ids if i != j and A[i]["cls"] == A[j]["cls"])
    total = len(ids) * (len(ids) - 1)
    print(f"\n  two random vehicles share a class {same_cls / max(total, 1):.0%} of the time "
          f"- that is how much work class alone does")
    med, best = cost_of_a_vehicle(crops)
    print(f"  cost: {med:.0f} ms (best {best:.0f}) to embed one vehicle of {COST_CROPS} crops with {model} "
          f"on this CPU, load average {os.getloadavg()[0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
