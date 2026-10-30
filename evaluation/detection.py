"""Detection AP / mAP @ IoU 0.5 from YOLO-format labels and predictions.

Definition (stated because AP has several): VOC2010+ "all-point interpolation". Per class, predictions are sorted
by confidence; each is a true positive if its IoU with a not-yet-matched ground-truth box of that class in the
same image is >= 0.5 (greedy, highest-IoU unmatched GT), else a false positive. Precision is made monotonically
non-increasing from the right and AP is the exact area under that step curve. mAP@0.5 is the unweighted mean of AP
over classes that have at least one ground-truth box (classes with none are reported as None, not 0).
This is NOT the COCO 101-point average over IoU 0.5:0.95 that Ultralytics prints as mAP50-95; its mAP50 differs
slightly from this because it uses a 101-point interpolation.
"""
from pathlib import Path

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ap_all_point(recall, precision):
    """Area under the precision envelope (VOC2010+). Inputs are cumulative-by-rank arrays."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([0.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def average_precision(gt, preds, iou_thr=0.5):
    """gt: {image: [box,...]} for one class; preds: [(image, conf, box), ...] for that class. Boxes xyxy."""
    n_gt = sum(len(v) for v in gt.values())
    if n_gt == 0:
        return None
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used = {k: [False] * len(v) for k, v in gt.items()}
    tp = np.zeros(len(order))
    for rank, i in enumerate(order):
        img, _, box = preds[i]
        best, bj = 0.0, -1
        for j, g in enumerate(gt.get(img, [])):
            if used[img][j]:
                continue
            o = iou_xyxy(box, g)
            if o > best:
                best, bj = o, j
        if bj >= 0 and best >= iou_thr:
            used[img][bj] = True
            tp[rank] = 1
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    if len(order) == 0:
        return 0.0
    return ap_all_point(rec, prec)


def evaluate(gt_boxes, pred_boxes, iou_thr=0.5, class_names=None):
    """gt_boxes: [(image, cls, xyxy)]; pred_boxes: [(image, cls, conf, xyxy)].
    Returns {'map': float|None, 'per_class': {cls: {'ap','n_gt','n_pred'}}}."""
    classes = sorted({c for _, c, _ in gt_boxes} | {c for _, c, _, _ in pred_boxes})
    per = {}
    for c in classes:
        g = {}
        for img, cc, b in gt_boxes:
            if cc == c:
                g.setdefault(img, []).append(b)
        p = [(img, conf, b) for img, cc, conf, b in pred_boxes if cc == c]
        per[c] = {"ap": average_precision(g, p, iou_thr), "n_gt": sum(len(v) for v in g.values()), "n_pred": len(p)}
        if class_names is not None:
            per[c]["name"] = class_names.get(c, str(c)) if isinstance(class_names, dict) else class_names[c]
    aps = [v["ap"] for v in per.values() if v["ap"] is not None]
    return {"map": float(np.mean(aps)) if aps else None, "iou_thr": iou_thr, "per_class": per,
            "interpolation": "VOC2010+ all-point"}


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def load_yolo_dir(labels_dir, images_dir=None, with_conf=False):
    """Read a directory of YOLO .txt files -> list. Ground truth: 'cls cx cy w h'; predictions (with_conf=True):
    'cls cx cy w h conf' (Ultralytics --save-txt --save-conf layout). Image sizes come from images_dir (needed to
    compare in pixels; without it coordinates are scaled by 1x1, only valid if every image has one aspect ratio)."""
    out = []
    for f in sorted(Path(labels_dir).glob("*.txt")):
        w = h = 1.0
        if images_dir:
            for ext in IMG_EXT:
                im = Path(images_dir) / (f.stem + ext)
                if im.exists():
                    from PIL import Image
                    with Image.open(im) as I:
                        w, h = I.size
                    break
        for line in f.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:5])
            box = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            if with_conf:
                out.append((f.stem, c, float(t[5]) if len(t) > 5 else 1.0, box))
            else:
                out.append((f.stem, c, box))
    return out
