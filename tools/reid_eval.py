#!/usr/bin/env python3
"""Do the appearance cues actually tell one vehicle from another, on this footage?

The honest question behind cross-camera matching. Each tracked vehicle's crops are split into an early half
and a late half - different moments, different distances, different angles, the same vehicle - and we ask
whether the two halves of one vehicle look more alike than two different vehicles do.

    ./.venv/bin/python tools/reid_eval.py data/videos/<clip>.mp4 --seconds 60

Reported per cue:
  separation   (mean same-vehicle score - mean different-vehicle score) / spread. Above ~1 is useful.
  rank-1       how often a vehicle's other half is its nearest neighbour out of all candidates.
A cue that scores near zero is not broken - the matcher switches such cues off by itself - but it does mean
geometry is carrying the identification on this footage, which is worth knowing before trusting a link.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surveillance import reid                     # noqa: E402
from surveillance.detector import Detector        # noqa: E402
from surveillance.embed import Embedder, similarity  # noqa: E402
from surveillance.sources import FileSource       # noqa: E402

MIN_CROPS = 6


def collect(video, seconds, imgsz, fps, min_px):
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
                    crops[tid].append(c.copy())
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


def report(name, pos, neg, rank1, n):
    pos, neg = np.array(pos, float), np.array(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        print(f"  {name:<22} not enough data")
        return
    spread = max(np.sqrt(0.5 * (pos.std() ** 2 + neg.std() ** 2)), 1e-6)
    sep = (pos.mean() - neg.mean()) / spread
    verdict = "useful" if sep > 1.0 else "weak" if sep > 0.4 else "no real signal"
    print(f"  {name:<22} same {pos.mean():+.3f}  different {neg.mean():+.3f}  "
          f"separation {sep:5.2f}  rank-1 {rank1 / max(n, 1):5.1%}   {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--min-px", type=int, default=40)
    a = ap.parse_args()

    print(f"Tracking {a.seconds:.0f}s of {Path(a.video).name} ...")
    crops, cls = collect(a.video, a.seconds, a.imgsz, a.fps, a.min_px)
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

    def score(cue, x, y):
        if cue == "hist":
            return float(np.sum(np.sqrt(x["hist"] * y["hist"])))
        if cue == "lab":
            return -float(np.linalg.norm(x["lab"] - y["lab"]))
        return similarity(x["embed"], y["embed"])

    for cue in ("hist", "lab", "embed"):
        pos, neg, rank1 = [], [], 0
        for i in ids:
            s_self = score(cue, A[i], B[i])
            if not np.isfinite(s_self):
                continue
            pos.append(s_self)
            others = [(score(cue, A[i], B[j]), j) for j in ids if j != i]
            others = [(s, j) for s, j in others if np.isfinite(s)]
            neg += [s for s, _ in others]
            if others and s_self > max(s for s, _ in others):
                rank1 += 1
        report({"hist": "colour histogram", "lab": "CIELAB colour",
                "embed": "appearance vector"}[cue], pos, neg, rank1, len(pos))

    # and what geometry alone would do: class agreement is the cue the matcher treats as absolute
    same_cls = sum(1 for i in ids for j in ids if i != j and A[i]["cls"] == A[j]["cls"])
    total = len(ids) * (len(ids) - 1)
    print(f"\n  two random vehicles share a class {same_cls / max(total, 1):.0%} of the time "
          f"- that is how much work class alone does")
    return 0


if __name__ == "__main__":
    sys.exit(main())
