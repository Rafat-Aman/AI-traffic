"""Multi-object tracking metrics: CLEAR-MOT (MOTA, MOTP, ID switches, fragmentations) and IDF1/IDP/IDR.

Input format: MOTChallenge text, one row per box: frame,id,x,y,w,h[,conf,...]  (frame 1-based, tlwh in pixels) -
the layout of CVAT's "MOT 1.1" export (gt/gt.txt) and of the tracker output written by tools/evaluate.py.

CLEAR-MOT (Bernardin & Stiefelhagen 2008), IoU >= `thr` (default 0.5) as the match criterion:
  * a ground-truth/hypothesis pair matched in the previous frame stays matched while IoU >= thr;
  * the remaining GT and hypotheses are matched by a minimum-cost bipartite assignment on 1-IoU (lap);
  * ID switch: a GT object matched to a different hypothesis id than the last one it was matched to;
  * fragmentation: a GT object whose matched status goes tracked -> not tracked -> tracked again (counted at the
    resumption), the usual "Frag" of the MOT benchmarks;
  * MOTA = 1 - (FN + FP + IDSW) / num_GT;  MOTP = mean IoU of matched pairs (higher is better; the 1-IoU
    "distance" form is 1 - MOTP).
IDF1 (Ristani et al. 2016): GT and hypothesis *trajectories* are matched one-to-one globally (not frame by frame)
to maximise the number of frames where the pair overlaps with IoU >= thr, via one min-cost assignment over the
(G+H)x(G+H) augmented matrix. IDTP = matched overlap frames, IDFN = GT boxes - IDTP, IDFP = hypothesis boxes -
IDTP, IDP = IDTP/(IDTP+IDFP), IDR = IDTP/(IDTP+IDFN), IDF1 = 2 IDTP/(2 IDTP + IDFP + IDFN).
Frame keys may be any hashable, so multi-camera scoring (MTMC) passes (camera, frame).
"""
from collections import defaultdict

import numpy as np

from .assign import min_cost_assign

BIG = 1e9


def load_mot(path):
    """Rows (frame, id, (x1,y1,x2,y2)) from a MOT txt/csv file."""
    out = []
    with open(path) as f:
        for line in f:
            t = line.strip().replace(";", ",").split(",")
            if len(t) < 6 or not t[0].strip().lstrip("-").isdigit():
                continue
            fr, i = int(float(t[0])), int(float(t[1]))
            x, y, w, h = map(float, t[2:6])
            out.append((fr, i, (x, y, x + w, y + h)))
    return out


def _iou_matrix(A, B):
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _by_frame(rows):
    d = defaultdict(list)
    for fr, i, b in rows:
        d[fr].append((i, b))
    return d


def _frame_order(frames):
    try:
        return sorted(frames)
    except TypeError:
        return sorted(frames, key=repr)


def clear_mot(gt_rows, hyp_rows, thr=0.5):
    G, H = _by_frame(gt_rows), _by_frame(hyp_rows)
    tp = fp = fn = idsw = frag = 0
    iou_sum = 0.0
    n_gt = 0
    last_match = {}          # gt id -> last hyp id matched
    prev = {}                # gt id -> hyp id matched in the previous frame
    was_tracked = {}         # gt id -> tracked in its previous visible frame
    ever = set()
    for fr in _frame_order(set(G) | set(H)):
        g, h = G.get(fr, []), H.get(fr, [])
        n_gt += len(g)
        M = _iou_matrix([b for _, b in g], [b for _, b in h])
        gi = {gid: k for k, (gid, _) in enumerate(g)}
        hi = {hid: k for k, (hid, _) in enumerate(h)}
        matches = {}
        for gid, hid in prev.items():
            if gid in gi and hid in hi and M[gi[gid], hi[hid]] >= thr:
                matches[gid] = hid
        used_h = set(matches.values())
        fg = [k for k, (gid, _) in enumerate(g) if gid not in matches]
        fh = [k for k, (hid, _) in enumerate(h) if hid not in used_h]
        if fg and fh:
            sub = 1.0 - M[np.ix_(fg, fh)]
            for a, b in min_cost_assign(sub, limit=1.0 - thr):
                if M[fg[a], fh[b]] >= thr:
                    matches[g[fg[a]][0]] = h[fh[b]][0]
        for gid, hid in matches.items():
            if gid in last_match and last_match[gid] != hid:
                idsw += 1
            last_match[gid] = hid
            iou_sum += M[gi[gid], hi[hid]]
        tp += len(matches)
        fn += len(g) - len(matches)
        fp += len(h) - len(matches)
        for gid, _ in g:
            tracked = gid in matches
            if gid in was_tracked and tracked and not was_tracked[gid] and gid in ever:
                frag += 1
            if tracked:
                ever.add(gid)
            was_tracked[gid] = tracked
        prev = matches
    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else None
    return {"num_gt": n_gt, "tp": tp, "fp": fp, "fn": fn, "id_switches": idsw, "fragmentations": frag,
            "mota": mota, "motp_iou": (iou_sum / tp) if tp else None, "iou_thr": thr}


def idf1(gt_rows, hyp_rows, thr=0.5):
    G, H = _by_frame(gt_rows), _by_frame(hyp_rows)
    gids = sorted({i for _, i, _ in gt_rows}, key=repr)
    hids = sorted({i for _, i, _ in hyp_rows}, key=repr)
    gk, hk = {i: k for k, i in enumerate(gids)}, {i: k for k, i in enumerate(hids)}
    ng, nh = len(gids), len(hids)
    n_gt_boxes, n_hyp_boxes = len(gt_rows), len(hyp_rows)
    if n_gt_boxes == 0:
        return {"idf1": None, "idp": None, "ir": None, "idr": None, "idtp": 0, "idfp": n_hyp_boxes, "idfn": 0}
    tpm = np.zeros((ng, nh))
    for fr in set(G) & set(H):
        g, h = G[fr], H[fr]
        M = _iou_matrix([b for _, b in g], [b for _, b in h])
        for a in range(len(g)):
            for b in range(len(h)):
                if M[a, b] >= thr:
                    tpm[gk[g[a][0]], hk[h[b][0]]] += 1
    glen, hlen = np.zeros(ng), np.zeros(nh)
    for _, i, _ in gt_rows:
        glen[gk[i]] += 1
    for _, i, _ in hyp_rows:
        hlen[hk[i]] += 1
    n = ng + nh
    C = np.zeros((n, n))
    C[:ng, :nh] = glen[:, None] + hlen[None, :] - 2 * tpm
    C[:ng, nh:] = BIG
    C[ng:, :nh] = BIG
    for a in range(ng):
        C[a, nh + a] = glen[a]
    for b in range(nh):
        C[ng + b, b] = hlen[b]
    C[ng:, nh:] = 0.0
    idtp = 0.0
    for r, c in min_cost_assign(C):
        if r < ng and c < nh:
            idtp += tpm[r, c]
    idfn, idfp = n_gt_boxes - idtp, n_hyp_boxes - idtp
    idp = idtp / (idtp + idfp) if (idtp + idfp) else None
    idr = idtp / (idtp + idfn) if (idtp + idfn) else None
    f = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else None
    return {"idf1": f, "idp": idp, "idr": idr, "idtp": int(idtp), "idfp": int(idfp), "idfn": int(idfn), "iou_thr": thr}


def evaluate(gt_rows, hyp_rows, thr=0.5, frames=None):
    """CLEAR-MOT + IDF1. `frames` (optional set) restricts both sides to frames that were actually processed."""
    if frames is not None:
        gt_rows = [r for r in gt_rows if r[0] in frames]
        hyp_rows = [r for r in hyp_rows if r[0] in frames]
    out = clear_mot(gt_rows, hyp_rows, thr)
    out.update({k: v for k, v in idf1(gt_rows, hyp_rows, thr).items() if k != "iou_thr"})
    return out
