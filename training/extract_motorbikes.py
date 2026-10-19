"""One best (largest, uncut) crop per tracked motorbike from TC_00006.MOV, tagged train/test by time."""
import sys, json
sys.path.insert(0, "/Users/isti115/CSE Projects/CSE499B-Implimentation")
import cv2
from surveillance.sources import FileSource
from surveillance.detector import Detector
OUT = sys.argv[1]
src = FileSource("/Users/isti115/Downloads/TC_00006.MOV", analysis_fps=10, max_width=0).open()
det = Detector("bd_vehicles_best.pt", imgsz=960, fps=10, track_buffer_sec=1.5, log=lambda *a: None)
best = {}  # tid -> (height, t, crop)
n = 0
while True:
    got = src.read()
    if got is None: break
    f, t = got; n += 1
    H, W = f.shape[:2]
    for (x1, y1, x2, y2), tid, name in det.track(f):
        if name != "Motorbike": continue
        h, w = y2 - y1, x2 - x1
        if h < 110 or y2 > H - 4 or x1 < 4 or x2 > W - 4: continue
        if tid in best and best[tid][0] >= h: continue
        crop = f[max(0, int(y1 - 0.45 * h)):y2, max(0, int(x1 - 0.08 * w)):min(W, int(x2 + 0.08 * w))].copy()
        best[tid] = (h, t, crop)
    if n % 600 == 0: print(f"t={t:.0f}s tracks={len(best)}", flush=True)
meta = []
for tid, (h, t, crop) in best.items():
    split = "test" if 122 <= t <= 242 else "train" if (t < 109 or t > 255) else None
    if split is None: continue
    name = f"{split}_{tid}_{int(t)}s.jpg"
    cv2.imwrite(f"{OUT}/{name}", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta.append({"file": name, "split": split, "t": round(t, 1), "h": h})
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=0)
print("train", sum(m["split"] == "train" for m in meta), "test", sum(m["split"] == "test" for m in meta))
