"""Event-level precision / recall / F1 for violations, and loaders for the team's CSV schema.

Matching rule (stated): a predicted event matches a ground-truth event only if the rule code is the same (and the
vehicle class too when both give one). Time criterion, one of
  mode='iou'       temporal IoU of [start,end] >= `thr` (default 0.3 - violations are short and hand-labelled
                   boundaries are loose); a zero-length (point) event is first widened to +-0.5 s.
  mode='tolerance' intervals match if they overlap after the ground-truth interval is widened by `tol` seconds
                   each side (default 3 s, the same default as training/score_accuracy.py --window).
Matching is one-to-one, greedy by best time score (highest IoU / smallest centre distance first). TP = matched
pairs, FP = unmatched predictions, FN = unmatched ground truth.
Note on training/score_accuracy.py: that script matches on |time difference| between two single timestamps; this
one generalises it to intervals but the tolerance mode with point events reduces to nearly the same rule.
"""
import csv
from collections import defaultdict


def _interval(e, min_len=1.0):
    s, t = float(e["start"]), float(e["end"])
    if t - s < min_len:
        m = (s + t) / 2
        s, t = m - min_len / 2, m + min_len / 2
    return s, t


def _score(g, p, mode, thr, tol):
    gs, ge = _interval(g)
    ps, pe = _interval(p)
    if mode == "iou":
        inter = max(0.0, min(ge, pe) - max(gs, ps))
        union = (ge - gs) + (pe - ps) - inter
        v = inter / union if union > 0 else 0.0
        return v if v >= thr else None
    if pe >= gs - tol and ps <= ge + tol:
        return -abs((gs + ge) / 2 - (ps + pe) / 2)
    return None


def match_events(gt, pred, mode="tolerance", thr=0.3, tol=3.0):
    cand = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if g["code"] != p["code"]:
                continue
            if g.get("cls") and p.get("cls") and g["cls"].lower() != p["cls"].lower():
                continue
            s = _score(g, p, mode, thr, tol)
            if s is not None:
                cand.append((s, i, j))
    cand.sort(reverse=True)
    ug, up, pairs = set(), set(), []
    for s, i, j in cand:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        pairs.append((i, j))
    return pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def evaluate(gt, pred, mode="tolerance", thr=0.3, tol=3.0, duration_s=None):
    """Per-code and micro-averaged P/R/F1. gt/pred: dicts with code, start, end (s), optional cls."""
    codes = sorted({e["code"] for e in gt} | {e["code"] for e in pred})
    per, T = {}, [0, 0, 0]
    for c in codes:
        g = [e for e in gt if e["code"] == c]
        p = [e for e in pred if e["code"] == c]
        tp = len(match_events(g, p, mode, thr, tol))
        fp, fn = len(p) - tp, len(g) - tp
        pr, rc, f = prf(tp, fp, fn)
        per[c] = {"n_gt": len(g), "n_pred": len(p), "tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": f}
        T[0] += tp; T[1] += fp; T[2] += fn
    pr, rc, f = prf(*T)
    out = {"per_code": per, "micro": {"tp": T[0], "fp": T[1], "fn": T[2], "precision": pr, "recall": rc, "f1": f},
           "mode": mode, "iou_thr": thr if mode == "iou" else None, "tolerance_s": tol if mode == "tolerance" else None}
    if duration_s:
        out["fp_per_hour"] = T[1] / (duration_s / 3600.0)
    return out


def load_events(path, point_half_width=0.0):
    """CSV with header. Ground truth schema (docs/EVALUATION.md): video,code,start_s,end_s,class,note.
    The system's export has video_time_s instead of start/end: it becomes a point event."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            code = (r.get("code") or "").strip().upper()
            if not code:
                continue
            s = r.get("start_s") or r.get("video_time_s") or r.get("time_s")
            if s in (None, ""):
                continue
            e = r.get("end_s") or s
            out.append({"video": (r.get("video") or "").strip(), "code": code, "start": float(s) - point_half_width,
                        "end": float(e or s) + point_half_width,
                        "cls": (r.get("class") or r.get("cls") or "").strip(), "note": r.get("note", "")})
    return out
