"""Helmet classifier on the labelled Dhaka test motorbikes (H = helmets on, N = someone bare-headed)."""
import sys, json
from ultralytics import YOLO
D = sys.argv[1]
meta = {m["idx"]: m for m in json.load(open(f"{D}/meta.json"))}
labels = {int(a): b for a, b in (l.split() for l in open(f"{D}/labels.txt") if l.strip() and not l.startswith("#"))}
test = [(meta[i], y) for i, y in labels.items() if meta[i]["split"] == "test" and y in "HN"]
for w in sys.argv[2:]:
    m = YOLO(w, task="classify"); bare = next(i for i, n in m.names.items() if "no" in n.lower())
    ps = [(y == "N", float(m.predict(f"{D}/crops/{mm['file']}", imgsz=224, verbose=False)[0].probs.data[bare])) for mm, y in test]
    pos = sum(y for y, _ in ps); neg = len(ps) - pos
    print(f"\n{w.split('/')[-1]}  ({neg} helmeted, {pos} bare-headed)")
    print(f"{'threshold':>9} {'bare caught':>12} {'helmeted wrongly flagged':>25} {'precision':>10}")
    for t in (0.5, 0.75, 0.9, 0.95):
        tp = sum(y and p >= t for y, p in ps); fp = sum((not y) and p >= t for y, p in ps)
        print(f"{t:>9} {tp:>5}/{pos:<6} {fp:>12}/{neg:<11} {100 * tp / max(1, tp + fp):9.1f}%")
