"""Tune the plate acceptance thresholds on the VALIDATION plates (the test plates stay untouched)."""
import sys, glob, random, itertools, pickle
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/isti115/CSE Projects/CSE499B-Implimentation")
sys.path.insert(0, str(Path(__file__).parent))
from surveillance import anpr
import calibrate_anpr as C   # reuses degrade / truth / read (loads the model given on the command line)

split = sys.argv[3] if len(sys.argv) > 3 else "val"
pattern = "*_o.jpg" if split == "val" else "*.jpg"
plates = []
for p in sorted(glob.glob(str(C.DS / split / "images" / pattern))):
    img = cv2.imread(p); gt = C.truth(p, img.shape[1], img.shape[0])
    if gt.complete: plates.append((img, gt))
random.seed(5)
cache = {}
for W in (70, 80, 90, 100, 120, 150, 200):
    cache[W] = [([C.read(C.degrade(img, W + random.randint(-6, 6)), W) for _ in range(4)], gt) for img, gt in plates if img.shape[1] >= W]
pickle.dump(cache, open(f"reads_{split}.pkl", "wb"))
print(f"{split}: {len(plates)} plates")
print(f"{'min_px':>6} {'agree':>5} {'conf':>5} | " + " ".join(f"{W:>11}" for W in cache) + " | worst wrong")
for min_px, agree, conf in itertools.product((70, 80, 90), (2, 3), (0.4, 0.5, 0.6, 0.7)):
    cells, worst = [], 0
    for W, items in cache.items():
        rep = wrong = 0
        for reads, gt in items:
            r = anpr.vote(reads, min_plate_px=min_px, min_agree=agree, min_digit_conf=conf)
            if r.status in ("read", "partial"):
                rep += 1; wrong += r.text_en.split(" ")[-1].replace("-", "") != "".join(gt.digits)
        n = len(items)
        cells.append(f"{100*rep/n:4.0f}%/{100*wrong/max(1,rep):4.1f}%")
        if rep >= 5: worst = max(worst, 100 * wrong / rep)
    print(f"{min_px:>6} {agree:>5} {conf:>5} | " + " ".join(f"{c:>11}" for c in cells) + f" | {worst:.1f}%")
