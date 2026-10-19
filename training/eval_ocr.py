"""Accuracy of plate reading on the held-out NightLPBD test plates, by plate width (pixels)."""
import sys, glob, re, json
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/isti115/CSE Projects/CSE499B-Implimentation")
from surveillance.anpr import parse_plate, DIGITS

DS = Path(sys.argv[1]); MODE = sys.argv[2]; MODEL = sys.argv[3] if len(sys.argv) > 3 else None
NAMES = [l.split(": ", 1)[1] for l in (DS / "plate_chars.yaml").read_text().splitlines() if l.startswith("  ")]
WIDTHS = [50, 60, 70, 80, 100, 130, 160, 220, 0]

def truth(lab, w, h):
    dets = []
    for l in open(lab):
        c, x, y, bw, bh = map(float, l.split())
        dets.append((NAMES[int(c)], 1.0, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return parse_plate(dets, w, h)

if MODE == "yolo":
    from ultralytics import YOLO
    m = YOLO(MODEL, task="detect")
    def read(img):
        r = m.predict(img, imgsz=320, conf=0.25, verbose=False)[0]
        dets = [(m.names[int(k)], float(cf), *b) for k, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        p = parse_plate(dets, img.shape[1], img.shape[0])
        return p.digits, p
else:
    import easyocr
    reader = easyocr.Reader(["bn"], gpu=False, verbose=False)
    def read(img):
        txt = "".join(reader.readtext(img, detail=0))
        ds = [str(DIGITS.index(ch)) for ch in txt if ch in DIGITS]
        return ds[-6:] if len(ds) >= 6 else ds, None

rows = []
for p in sorted(glob.glob(str(DS / "test" / "images" / "*.jpg"))):
    img = cv2.imread(p); h, w = img.shape[:2]
    gt = truth(Path(p.replace("/images/", "/labels/")).with_suffix(".txt"), w, h)
    if not gt.number_ok:
        continue
    for W in WIDTHS:
        if W and W >= w:
            continue
        im = img if not W else cv2.resize(img, (W, round(h * W / w)), interpolation=cv2.INTER_AREA)
        if W:  # what a camera stream delivers: JPEG/H.264 compression
            im = cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_COLOR)
        digits, pr = read(im)
        full = pr is not None and pr.number_ok and pr.digits == gt.digits and pr.city == gt.city and pr.letter == gt.letter and pr.metro == gt.metro
        chars = sum(a == b for a, b in zip(digits, gt.digits)) / 6 if len(digits) == 6 else 0
        rows.append((W or "orig", digits == gt.digits, full, chars, len(digits) == 6, min(pr.digit_conf) if pr and pr.number_ok else 0))
print(f"{'plate width':>11} {'n':>4} {'number exact':>13} {'full plate':>11} {'digit acc':>10} {'6 digits found':>15}")
for W in [*WIDTHS[:-1], "orig"]:
    rs = [r for r in rows if r[0] == W]
    if rs:
        n = len(rs)
        print(f"{str(W):>11} {n:>4} {100*sum(r[1] for r in rs)/n:12.1f}% {100*sum(r[2] for r in rs)/n:10.1f}% "
              f"{100*sum(r[3] for r in rs)/n:9.1f}% {100*sum(r[4] for r in rs)/n:14.1f}%")
json.dump(rows, open(f"{DS}/../eval_{MODE}.json", "w"))
