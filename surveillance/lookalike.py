"""Look-alike suggestions: the vehicles that look most like an open one, anywhere in the network (WP17, P5).

Suggestions only. This module reads `tracklets` and writes nothing: it never links, never sets a global id, never
asks link.py anything. A person decides whether two sightings are the same vehicle.

Where the vectors come from
    The appearance vectors the pipeline already stored on every tracklet: the ReID vector (`embed_reid`, 2048 floats)
    when the vehicle has one, otherwise the MobileNet vector (`embed`, 576 floats). Vectors made by different
    models, or of different lengths, are never compared: one in-memory space per (column, model tag). The vehicle
    is searched in the best space it has a vector in.

How the search works (no hnswlib / faiss in the venv, and no new dependency)
    Brute force: every vector is unit length, so the dot product with the query IS the cosine similarity, and one
    matrix-vector product scores the whole space. float32 on purpose: numpy has no half-precision BLAS, and on this
    machine a float16 matrix was 10-30x slower per query (and stays that way after converting in chunks) for half
    the memory. Measured query times are in docs/VEHICLES.md.

    The matrix is kept in memory, built lazily on the first query and refreshed (new rows appended, vanished rows
    dropped) when a query arrives more than `ttl_s` after the last check. Late ReID vectors (written seconds after
    the row) are picked up because the check looks at which ids HAVE a vector, not at the highest id.

The cap
    Each space holds at most MAX_VECTORS vectors and at most MAX_MB megabytes: 100,000 vectors for MobileNet
    (230 MB) and about 31,000 for ReID (256 MB). When there are more, the OLDEST are left out (older rows are also the
    ones storage_mgr strips of their vectors after `embed_days`). The answer says how many vectors were searched
    and whether the cap left some out.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from . import passed as passed_mod

MAX_VECTORS = 100_000
MAX_MB = 256
DEFAULT_K, MAX_K = 10, 50
CHUNK = 2000                       # rows fetched per SQL call while loading

# (column holding the vector, column holding the model that made it); best space first
COLUMNS = (("embed_reid", "embed_reid_model"), ("embed", "embed_model"))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


class Space:
    """The vectors of one (column, model tag): ids ascending, a unit-length float32 matrix, the class of each."""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids = np.empty(0, np.int64)
        self.mat = np.empty((0, dim), np.float32)
        self.cls = np.empty(0, object)
        self.truncated = 0         # vectors left out by the cap at the last refresh

    def append(self, ids, mat, cls):
        self.ids = np.concatenate([self.ids, np.asarray(ids, np.int64)])
        self.mat = np.vstack([self.mat, mat]) if len(self.mat) else mat
        self.cls = np.concatenate([self.cls, np.asarray(cls, object)])

    def keep(self, mask):
        self.ids, self.mat, self.cls = self.ids[mask], self.mat[mask], self.cls[mask]

    def cap(self, limit: int):
        extra = len(self.ids) - limit
        self.truncated = max(0, extra)
        if extra > 0:                                  # ids ascend: the front is the oldest
            self.ids, self.mat, self.cls = self.ids[extra:], self.mat[extra:], self.cls[extra:]


class LookalikeIndex:
    def __init__(self, store, max_vectors: int = MAX_VECTORS, max_mb: int = MAX_MB, ttl_s: float = 30.0,
                 now_fn: Callable = time.time):
        self.store, self.max_vectors, self.max_mb, self.ttl, self.now = store, max_vectors, max_mb, ttl_s, now_fn
        self.spaces: dict = {}
        self._checked = -1e18
        self._lock = threading.Lock()
        self.last_build_ms = 0.0
        passed_mod.ensure_schema(store)       # _describe joins the passed-vehicle records

    def limit_for(self, dim: int) -> int:
        return max(1, min(self.max_vectors, int(self.max_mb * 1e6 // (4 * dim))))

    # ---- keeping the matrices current
    def refresh(self, force: bool = False) -> None:
        with self._lock:
            if not force and self.now() - self._checked < self.ttl:
                return
            t0 = time.perf_counter()
            from .embed import MOBILENET_TAG
            seen_keys = set()
            for col, tagcol in COLUMNS:
                rows = self.store.q(f"SELECT id, {tagcol} AS tag, cls FROM tracklets WHERE {col} IS NOT NULL ORDER BY id")
                by_tag: dict = {}
                for r in rows:
                    tag = r["tag"] or (MOBILENET_TAG if col == "embed" else "")
                    by_tag.setdefault(tag, []).append(r)
                for tag, items in by_tag.items():
                    key = (col, tag)
                    seen_keys.add(key)
                    # older rows beyond the cap are not even loaded (else they would be re-read at every refresh)
                    limit = self.limit_for(2048 if col == "embed_reid" else 576)
                    self._sync(key, items[-limit:], len(items) - min(len(items), limit))
            for key in [k for k in self.spaces if k not in seen_keys]:
                del self.spaces[key]                      # every vector of that model is gone
            self._checked = self.now()
            self.last_build_ms = (time.perf_counter() - t0) * 1000.0

    def _sync(self, key, items, left_out=0):
        col = key[0]
        space = self.spaces.get(key)
        current = {r["id"] for r in items}
        if space is not None:
            gone = ~np.isin(space.ids, np.fromiter(current, np.int64, len(current)))
            if gone.any():
                space.keep(~gone)
            known = set(space.ids.tolist())
        else:
            known = set()
        fresh = [r for r in items if r["id"] not in known]
        dim = space.dim if space is not None else None
        for i in range(0, len(fresh), CHUNK):
            part = fresh[i:i + CHUNK]
            ids = [r["id"] for r in part]
            blobs = {r["id"]: r[col] for r in self.store.q(
                f"SELECT id, {col} FROM tracklets WHERE id IN ({','.join('?' * len(ids))})", ids)}
            cls = {r["id"]: r["cls"] for r in part}
            vecs, kept = [], []
            for tid in ids:
                b = blobs.get(tid)
                if not b or len(b) % 4:
                    continue
                v = np.frombuffer(b, np.float32)
                if dim is None:
                    dim = len(v)
                if len(v) != dim or not np.isfinite(v).all():
                    continue
                vecs.append(_unit(v))
                kept.append(tid)
            if kept:
                if space is None:
                    space = self.spaces[key] = Space(dim)
                space.append(kept, np.vstack(vecs), [cls[t] or "" for t in kept])
        if space is not None:
            space.cap(self.limit_for(space.dim))
            space.truncated += left_out

    def stats(self) -> dict:
        self.refresh()
        return {"spaces": [{"column": k[0], "model": k[1], "dim": s.dim, "vectors": int(len(s.ids)),
                            "left_out_by_cap": s.truncated, "mb": round(s.mat.nbytes / 1e6, 1)}
                           for k, s in self.spaces.items()],
                "cap_vectors": self.max_vectors, "cap_mb": self.max_mb, "last_build_ms": round(self.last_build_ms, 1)}

    # ---- the query
    def search(self, vec: np.ndarray, key, k: int = DEFAULT_K, cls: Optional[str] = None, exclude=()) -> list:
        """[(tracklet id, cosine)] best first, from one space. `cls` keeps only that class (the linker never joins
        different classes either); `exclude` are tracklet ids to leave out."""
        space = self.spaces.get(key)
        if space is None or not len(space.ids):
            return []
        scores = space.mat @ _unit(vec)
        if cls:
            scores = np.where(space.cls == cls, scores, -2.0)
        if exclude:
            scores[np.isin(space.ids, list(exclude))] = -2.0
        k = max(1, min(k, MAX_K, len(scores)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(space.ids[i]), float(scores[i])) for i in top if scores[i] > -1.5]

    def similar(self, tracklet_id: int, k: int = DEFAULT_K, same_class: bool = True) -> dict:
        """The k vehicles most like tracklet `tracklet_id`. Never writes anything."""
        t0 = time.perf_counter()
        self.refresh()
        row = self.store.q("SELECT id, cls, global_id, embed, embed_model, embed_reid, embed_reid_model"
                           " FROM tracklets WHERE id=?", (tracklet_id,))
        if not row:
            return {"items": [], "reason": "no such sighting"}
        row = row[0]
        from .embed import MOBILENET_TAG
        pick = None
        for col, tagcol in COLUMNS:
            if row[col] is not None:
                tag = row[tagcol] or (MOBILENET_TAG if col == "embed" else "")
                if (col, tag) in self.spaces:
                    pick = (col, tag)
                    break
        if pick is None:
            return {"items": [], "reason": "no appearance vector is stored for this vehicle (older than the "
                                           "retention of vectors, or it was never embedded)"}
        v = np.frombuffer(row[pick[0]], np.float32)
        if len(v) != self.spaces[pick].dim:
            return {"items": [], "reason": "its vector has a different length from the rest of its model"}
        hits = self.search(v, pick, k, row["cls"] if same_class else None, exclude=(tracklet_id,))
        info = self._describe([h[0] for h in hits])
        items = []
        for tid, score in hits:
            d = info.get(tid)
            if d is None:
                continue
            items.append({**d, "score": round(score, 4),
                          "same_global_id": bool(row["global_id"] and d["global_id"] == row["global_id"])})
        sp = self.spaces[pick]
        return {"items": items, "space": {"column": pick[0], "model": pick[1], "dim": sp.dim},
                "searched": int(len(sp.ids)), "left_out_by_cap": sp.truncated, "suggestions_only": True,
                "ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def _describe(self, ids) -> dict:
        if not ids:
            return {}
        rows = self.store.q(
            "SELECT t.id AS tracklet_id, t.camera_id, c.name AS camera_name, c.location AS camera_location, t.run_id,"
            " t.track_id, t.global_id, t.cls, t.t_cross, t.speed, p.id AS passed_id, p.colour, p.thumb,"
            " p.plate_status, p.plate_en, p.plate_bn"
            " FROM tracklets t LEFT JOIN cameras c ON c.id=t.camera_id"
            " LEFT JOIN passed_vehicles p ON p.camera_id=t.camera_id AND p.run_id=t.run_id AND p.track_id=t.track_id"
            f" WHERE t.id IN ({','.join('?' * len(ids))})", list(ids))
        return {r["tracklet_id"]: r for r in rows}
