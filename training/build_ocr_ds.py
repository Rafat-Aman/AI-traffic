"""Plate-crop dataset for the Bangla plate character detector, from NightLPBD (CC BY 4.0).

Each plate is cropped with padding; training crops are also degraded to the resolutions a traffic camera
actually delivers (60-400 px wide, blur, JPEG artefacts, brightness), so the model learns low-res plates and
the test set can measure accuracy against plate width."""
import sys, glob, random, json
from pathlib import Path
import cv2, numpy as np

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
random.seed(0); np.random.seed(0)

def degrade(img, width):
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, max(8, round(h * width / w))), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:  # motion blur along a random near-horizontal direction
        k = random.choice([3, 5])
        kernel = np.zeros((k, k), np.float32); kernel[k // 2, :] = 1 / k
        small = cv2.filter2D(small, -1, kernel)
    if random.random() < 0.3:
        small = cv2.GaussianBlur(small, (3, 3), 0)
    q = random.randint(30, 85)
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    a = random.uniform(0.6, 1.4); b = random.uniform(-40, 40)  # day / night brightness and contrast
    small = np.clip(small.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return small

def crops(split):
    for img_path in sorted(glob.glob(str(SRC / split / "images" / "*"))):
        lab = Path(img_path.replace("/images/", "/labels/")).with_suffix(".txt")
        if not lab.exists():
            continue
        im = cv2.imread(img_path)
        if im is None:
            continue
        H, W = im.shape[:2]
        rows = [list(map(float, l.split())) for l in open(lab) if l.strip()]
        boxes = [(int(r[0]), (r[1] - r[3] / 2) * W, (r[2] - r[4] / 2) * H, (r[1] + r[3] / 2) * W, (r[2] + r[4] / 2) * H) for r in rows]
        for k, (c, x1, y1, x2, y2) in enumerate(b for b in boxes if b[0] == 0):
            pw, ph = x2 - x1, y2 - y1
            px = random.uniform(0.04, 0.15) if split == "train" else 0.08
            py = random.uniform(0.06, 0.2) if split == "train" else 0.1
            X1, Y1 = int(max(0, x1 - px * pw)), int(max(0, y1 - py * ph))
            X2, Y2 = int(min(W, x2 + px * pw)), int(min(H, y2 + py * ph))
            crop = im[Y1:Y2, X1:X2]
            cw, ch = X2 - X1, Y2 - Y1
            chars = []
            for cc, a1, b1, a2, b2 in boxes:
                if cc == 0:
                    continue
                cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
                if X1 <= cx <= X2 and Y1 <= cy <= Y2:
                    a1, b1, a2, b2 = max(a1, X1), max(b1, Y1), min(a2, X2), min(b2, Y2)
                    chars.append((cc - 1, ((a1 + a2) / 2 - X1) / cw, ((b1 + b2) / 2 - Y1) / ch, (a2 - a1) / cw, (b2 - b1) / ch))
            if chars:
                yield f"{Path(img_path).stem}_{k}", crop, chars, pw

def save(split_out, name, img, chars):
    (OUT / split_out / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split_out / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / split_out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(OUT / split_out / "labels" / f"{name}.txt", "w") as f:
        for c in chars:
            f.write("%d %.6f %.6f %.6f %.6f\n" % c)

meta = {"test": []}
n = {"train": 0, "val": 0, "test": 0}
for name, crop, chars, pw in crops("train"):
    h, w = crop.shape[:2]
    base = cv2.resize(crop, (640, round(h * 640 / w))) if w > 640 else crop
    save("train", name + "_o", base, chars); n["train"] += 1
    for v in range(3):  # three degraded copies per plate
        width = random.choice([60, 75, 90, 110, 130, 160, 200, 260, 330, 400])
        save("train", f"{name}_d{v}", degrade(crop, min(width, w)), chars); n["train"] += 1
for name, crop, chars, pw in crops("valid"):
    save("val", name + "_o", crop if crop.shape[1] <= 640 else cv2.resize(crop, (640, round(crop.shape[0] * 640 / crop.shape[1]))), chars)
    save("val", name + "_d", degrade(crop, min(random.choice([80, 120, 180]), crop.shape[1])), chars); n["val"] += 2
for name, crop, chars, pw in crops("test"):
    save("test", name, crop, chars); n["test"] += 1
names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'bo', 'co', 'Dha', 'Dhaka', 'go', 'ho', 'ka', 'khu', 'khulona',
         'lo', 'me', 'meTrO', 'na', 'ro', 'shO', 'To', 'TrO', 'zo', 'zoshOro']
(OUT / "plate_chars.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n" +
                                      "".join(f"  {i}: {s}\n" for i, s in enumerate(names)))
print(n)
