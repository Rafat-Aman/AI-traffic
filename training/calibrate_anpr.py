"""Multi-frame ANPR on held-out plates: each plate is 'seen' in 4 frames at a given width with different blur /
compression, read by the character model, parsed and voted exactly like the live pipeline.

Reports, per plate width: how often a plate is reported, and how often a reported plate is wrong."""
import sys, glob, random
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/isti115/CSE Projects/CSE499B-Implimentation")
from surveillance import anpr
from ultralytics import YOLO

DS, MODEL = Path(sys.argv[1]), sys.argv[2]
NAMES = [l.split(": ", 1)[1] for l in (DS / "plate_chars.yaml").read_text().splitlines() if l.startswith("  ")]
m = YOLO(MODEL, task="detect")
random.seed(1); np.random.seed(1)


def degrade(img, width):
    h, w = img.shape[:2]
    im = cv2.resize(img, (width, max(8, round(h * width / w))), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        k = random.choice([3, 5]); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1 / k
        im = cv2.filter2D(im, -1, ker)
    q = random.randint(40, 85)
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def truth(p, w, h):
    dets = []
    for l in open(Path(p.replace("/images/", "/labels/")).with_suffix(".txt")):
        c, x, y, bw, bh = map(float, l.split())
        dets.append((NAMES[int(c)], 1.0, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return anpr.parse_plate(dets, w, h)


def read(img, px):
    r = m.predict(img, imgsz=320, conf=0.25, verbose=False)[0]
    dets = [(m.names[int(k)], float(cf), *b) for k, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return anpr.parse_plate(dets, img.shape[1], img.shape[0], px)


def main():
    plates = []
    for p in sorted(glob.glob(str(DS / "test" / "images" / "*.jpg"))):
        img = cv2.imread(p)
        gt = truth(p, img.shape[1], img.shape[0])
        if gt.complete:
            plates.append((img, gt))
    print(f"{len(plates)} test plates with a complete label")
    for min_px in ([anpr.MIN_PLATE_PX] if len(sys.argv) < 4 else [int(v) for v in sys.argv[3].split(",")]):
        print(f"\nMIN_PLATE_PX = {min_px}")
        print(f"{'width':>6} {'reported':>9} {'number wrong':>13} {'full plate wrong':>17} {'unreadable':>11}")
        for W in (50, 60, 70, 80, 90, 100, 120, 150, 200):
            rep = num_wrong = full_wrong = 0
            n = 0
            for img, gt in plates:
                if img.shape[1] < W:
                    continue
                n += 1
                reads = [read(degrade(img, W + random.randint(-6, 6)), W) for _ in range(4)]
                res = anpr.vote(reads, min_plate_px=min_px)
                if res.status in ("read", "partial"):
                    rep += 1
                    got_num = res.text_en.split(" ")[-1].replace("-", "")
                    num_wrong += got_num != "".join(gt.digits)
                    full_wrong += res.status == "read" and res.text_en != gt.text("en")
            if n:
                print(f"{W:>6} {100 * rep / n:8.1f}% {100 * num_wrong / max(1, rep):12.1f}% {100 * full_wrong / max(1, rep):16.1f}% "
                      f"{100 * (n - rep) / n:10.1f}%")


if __name__ == "__main__":
    main()
