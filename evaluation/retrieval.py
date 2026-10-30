"""Retrieval metrics (X7 grounding) and cross-camera association metrics (X6)."""
from collections import defaultdict
from itertools import combinations

from .tracking import idf1


def mrr(rankings, relevant):
    """Mean reciprocal rank. rankings: {query: [item,...] best first}; relevant: {query: set(items)}.
    A query with no relevant item in its list contributes 0; a query with no relevant items at all is skipped."""
    rr = []
    for q, ranked in rankings.items():
        rel = set(relevant.get(q, ()))
        if not rel:
            continue
        rr.append(next((1.0 / (k + 1) for k, it in enumerate(ranked) if it in rel), 0.0))
    return sum(rr) / len(rr) if rr else None


def recall_at_k(rankings, relevant, k):
    """Fraction of queries with at least one relevant item in the top k (hit-rate form: 'Recall@k' for the
    one-answer queries typical of grounding). Queries with no relevant items are skipped."""
    hits = []
    for q, ranked in rankings.items():
        rel = set(relevant.get(q, ()))
        if rel:
            hits.append(1.0 if any(it in rel for it in ranked[:k]) else 0.0)
    return sum(hits) / len(hits) if hits else None


def recall_at_k_fraction(rankings, relevant, k):
    """Mean over queries of |relevant in top k| / |relevant| (the multi-answer form)."""
    v = []
    for q, ranked in rankings.items():
        rel = set(relevant.get(q, ()))
        if rel:
            v.append(len(rel & set(ranked[:k])) / len(rel))
    return sum(v) / len(v) if v else None


def pair_precision_recall(pred_gid, gt_gid, cross_camera_only=True):
    """Association-pair metrics. pred_gid/gt_gid: {(camera, local_track_id): global_id}. Over items present in
    both, a pair is positive when both items share a global id. With cross_camera_only (default) only pairs from
    different cameras count - the question cross-camera linking answers."""
    items = sorted(set(pred_gid) & set(gt_gid), key=repr)
    tp = fp = fn = 0
    for a, b in combinations(items, 2):
        if cross_camera_only and a[0] == b[0]:
            continue
        p = pred_gid[a] == pred_gid[b]
        g = gt_gid[a] == gt_gid[b]
        tp += p and g
        fp += p and not g
        fn += g and not p
    pr = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f = 2 * pr * rc / (pr + rc) if pr and rc else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": f, "cross_camera_only": cross_camera_only}


def mtmc_idf1(gt_rows, hyp_rows, thr=0.5):
    """MTMC IDF1: IDF1 over all cameras at once with global ids as identities. Rows are (camera, frame, global_id,
    box_xyxy). A global identity must be kept across cameras to score; same-camera ID switches also count."""
    g = [((c, f), i, b) for c, f, i, b in gt_rows]
    h = [((c, f), i, b) for c, f, i, b in hyp_rows]
    return idf1(g, h, thr)
