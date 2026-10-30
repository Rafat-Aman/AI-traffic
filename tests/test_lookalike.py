"""Look-alike suggestions (WP17): the planted near-duplicate comes first, nothing is ever written, and the index
keeps up with the table. Synthetic unit vectors only."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import lookalike as la  # noqa: E402
from surveillance.embed import MOBILENET_TAG  # noqa: E402
from surveillance.reid import Tracklet  # noqa: E402
from surveillance.store import Store  # noqa: E402

RTAG = "reid:test"


def unit(n, seed):
    v = np.random.default_rng(seed).normal(size=n)
    return (v / np.linalg.norm(v)).astype(np.float32)


def near(v, seed, noise=0.05):
    w = v + noise * unit(len(v), seed)
    return (w / np.linalg.norm(w)).astype(np.float32)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Store(Path(self._tmp.name) / "s.db")
        self.addCleanup(self.store.db.close)
        self.c1 = self.store.add_camera("A", "file", "x")
        self.c2 = self.store.add_camera("B", "file", "y")
        self.run = self.store.start_run(self.c1, None, "full")
        self.n = 0

    def add(self, cam=None, cls="Car", gid=None, mobile=None, reid=None, t=0.0):
        self.n += 1
        tl = Tracklet(cam or self.c1, self.n, cls, 1000.0 + t, 1.0, 8.0, "outgoing", 1.7, run_id=self.run)
        tl.global_id = gid
        if mobile is not None:
            tl.embed, tl.embed_model = mobile, MOBILENET_TAG
        if reid is not None:
            tl.embed_reid, tl.embed_reid_model = reid, RTAG
        return self.store.add_tracklet(tl.camera_id, self.run, tl)

    def dump(self):
        return self.store.q("SELECT * FROM tracklets ORDER BY id")


class SimilarTests(Base):
    def test_planted_near_duplicate_comes_first(self):
        q = unit(576, 0)
        me = self.add(mobile=q)
        twin = self.add(cam=self.c2, mobile=near(q, 1), t=60)
        for i in range(2, 60):
            self.add(mobile=unit(576, i))
        out = la.LookalikeIndex(self.store).similar(me, k=5)
        self.assertEqual(out["items"][0]["tracklet_id"], twin)
        self.assertGreater(out["items"][0]["score"], 0.95)
        self.assertLess(out["items"][1]["score"], 0.5)
        self.assertNotIn(me, [i["tracklet_id"] for i in out["items"]])        # never itself
        self.assertEqual(out["items"][0]["camera_name"], "B")
        self.assertTrue(out["suggestions_only"])

    def test_never_writes_or_links(self):
        q = unit(576, 0)
        me = self.add(mobile=q)
        self.add(cam=self.c2, mobile=near(q, 1))
        before = self.dump()
        changes = self.store.db.total_changes
        idx = la.LookalikeIndex(self.store)
        idx.similar(me, k=5)
        idx.stats()
        self.assertEqual(self.dump(), before)
        self.assertEqual(self.store.db.total_changes, changes)               # not one UPDATE / INSERT
        self.assertTrue(all(r["global_id"] is None for r in self.dump()))

    def test_same_global_id_is_flagged_not_hidden(self):
        q = unit(576, 0)
        me = self.add(mobile=q, gid=5)
        twin = self.add(cam=self.c2, mobile=near(q, 1), gid=5)
        other = self.add(mobile=near(q, 2, 0.4), gid=9)
        items = {i["tracklet_id"]: i for i in la.LookalikeIndex(self.store).similar(me)["items"]}
        self.assertTrue(items[twin]["same_global_id"])
        self.assertFalse(items[other]["same_global_id"])

    def test_class_filter_and_switch(self):
        q = unit(576, 0)
        me = self.add(mobile=q, cls="Car")
        bus = self.add(mobile=near(q, 1), cls="Bus")
        car = self.add(mobile=near(q, 2, 0.3), cls="Car")
        idx = la.LookalikeIndex(self.store)
        self.assertEqual([i["tracklet_id"] for i in idx.similar(me)["items"]], [car])
        self.assertEqual(idx.similar(me, same_class=False)["items"][0]["tracklet_id"], bus)

    def test_reid_space_is_preferred_and_models_are_never_mixed(self):
        r = unit(2048, 0)
        me = self.add(mobile=unit(576, 100), reid=r)
        twin = self.add(mobile=unit(576, 101), reid=near(r, 1))
        mob_only = self.add(mobile=near(unit(576, 100), 2, 0.01))          # looks like me in MobileNet space only
        out = la.LookalikeIndex(self.store).similar(me)
        self.assertEqual(out["space"]["column"], "embed_reid")
        self.assertEqual([i["tracklet_id"] for i in out["items"]], [twin])
        self.assertNotIn(mob_only, [i["tracklet_id"] for i in out["items"]])

    def test_no_vector_explains_itself(self):
        me = self.add()
        self.add(mobile=unit(576, 1))
        out = la.LookalikeIndex(self.store).similar(me)
        self.assertEqual(out["items"], [])
        self.assertIn("no appearance vector", out["reason"])
        self.assertEqual(la.LookalikeIndex(self.store).similar(99999)["items"], [])

    def test_matrix_follows_the_table(self):
        q = unit(576, 0)
        me = self.add(mobile=q)
        idx = la.LookalikeIndex(self.store, ttl_s=0)
        self.assertEqual(idx.similar(me)["items"], [])
        late = self.add(mobile=near(q, 1))
        self.assertEqual(idx.similar(me)["items"][0]["tracklet_id"], late)    # appended without a rebuild
        self.store.x("DELETE FROM tracklets WHERE id=?", (late,))
        self.assertEqual(idx.similar(me)["items"], [])                        # gone rows are dropped

    def test_a_late_reid_vector_is_picked_up(self):
        r = unit(2048, 0)
        me = self.add(mobile=unit(576, 1), reid=r)
        twin = self.add(mobile=unit(576, 2))                                  # its ReID vector has not arrived yet
        idx = la.LookalikeIndex(self.store, ttl_s=0)
        self.assertEqual(idx.similar(me)["items"], [])
        self.store.set_tracklet_reid(twin, near(r, 3), RTAG)                  # the background worker delivers it
        self.assertEqual(idx.similar(me)["items"][0]["tracklet_id"], twin)

    def test_cap_leaves_out_the_oldest_and_says_so(self):
        q = unit(576, 0)
        old = self.add(mobile=near(q, 1, 0.01))                               # the best match, but the oldest
        me = self.add(mobile=q)
        for i in range(4):
            self.add(mobile=unit(576, 10 + i))
        idx = la.LookalikeIndex(self.store, max_vectors=3)
        out = idx.similar(me, k=10)
        self.assertNotIn(old, [i["tracklet_id"] for i in out["items"]])
        self.assertEqual(out["searched"], 3)
        self.assertEqual(out["left_out_by_cap"], 3)
        self.assertEqual(idx.stats()["spaces"][0]["left_out_by_cap"], 3)
        self.assertEqual(idx.limit_for(2048), 3)
        self.assertEqual(la.LookalikeIndex(self.store).limit_for(2048), 256_000_000 // (4 * 2048))

    def test_odd_vectors_are_skipped(self):
        me = self.add(mobile=unit(576, 0))
        self.add(mobile=np.full(576, np.nan, np.float32))
        self.add(mobile=np.zeros(576, np.float32))                            # zero length: kept as zeros, scores 0
        out = la.LookalikeIndex(self.store).similar(me)
        self.assertTrue(all(np.isfinite(i["score"]) for i in out["items"]))

    def test_latency_at_ten_thousand_is_small(self):
        rng = np.random.default_rng(0)
        idx = la.LookalikeIndex(self.store)
        m = rng.normal(size=(10_000, 576)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        sp = la.Space(576)
        sp.append(range(10_000), m, ["Car"] * 10_000)
        idx.spaces[("embed", MOBILENET_TAG)] = sp
        import time
        t = time.perf_counter()
        hits = idx.search(m[7], ("embed", MOBILENET_TAG), 10)
        ms = (time.perf_counter() - t) * 1000
        self.assertEqual(hits[0][0], 7)
        self.assertLess(ms, 250)                                              # a generous bound; real figure in docs


if __name__ == "__main__":
    unittest.main()
