"""Per-crop precision / recall of the bare-head class on the held-out test crops."""
import sys, glob
from ultralytics import YOLO
for w in sys.argv[2:]:
    m = YOLO(w, task="classify"); bare = next(i for i, n in m.names.items() if "no" in n)
    ps = []
    for cls in ("helmet", "no_helmet"):
        for p in glob.glob(f"{sys.argv[1]}/test/{cls}/*.jpg"):
            r = m.predict(p, imgsz=224, verbose=False)[0]
            ps.append((cls == "no_helmet", float(r.probs.data[bare])))
    pos = sum(y for y, _ in ps)
    print(f"\n{w.split('/')[-1]}: {len(ps)} test crops, {pos} bare-headed")
    print(f"{'threshold':>9} {'precision':>10} {'recall':>7} {'helmeted wrongly flagged':>25}")
    for t in (0.5, 0.6, 0.75, 0.85, 0.9):
        tp = sum(y and p >= t for y, p in ps); fp = sum((not y) and p >= t for y, p in ps)
        print(f"{t:>9} {100*tp/max(1,tp+fp):9.1f}% {100*tp/pos:6.1f}% {100*fp/(len(ps)-pos):24.1f}%")
