"""The ReID vector arrives late and from a background thread: tagging, queueing, storage and matching.

None of this needs the ReID weights; the model is replaced by a stand-in that sleeps.
"""

import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import embed, link, reid  # noqa: E402
from surveillance.embed import Embedder, MOBILENET_TAG  # noqa: E402
from surveillance.reid import ReIDWorker, Tracklet  # noqa: E402
from surveillance.store import Store  # noqa: E402

RTAG = "reid:test-weights"


def unit(n, seed=0):
    v = np.random.default_rng(seed).normal(size=n)
    return (v / np.linalg.norm(v)).astype(np.float32)


def tl(**kw):
    t = Tracklet(1, 1, "Car", 0.0, 2.0, 8.0, "outgoing", 1.7)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


class AppearanceChoiceTests(unittest.TestCase):
    def test_reid_is_used_when_both_have_it(self):
        v, w = unit(2048, 1), unit(2048, 2)
        a = tl(embed=unit(576), embed_model=MOBILENET_TAG, embed_reid=v, embed_reid_model=RTAG)
        b = tl(embed=unit(576, 5), embed_model=MOBILENET_TAG, embed_reid=w, embed_reid_model=RTAG)
        s, tag = reid.appearance(a, b)
        self.assertEqual(tag, "reid:" + RTAG)
        self.assertAlmostEqual(s, float(np.dot(v, w)), places=6)

    def test_mobilenet_when_only_one_has_reid(self):
        a = tl(embed=unit(576), embed_model=MOBILENET_TAG, embed_reid=unit(2048), embed_reid_model=RTAG)
        b = tl(embed=unit(576, 5), embed_model=MOBILENET_TAG)
        s, tag = reid.appearance(a, b)
        self.assertEqual(tag, MOBILENET_TAG)
        self.assertTrue(np.isfinite(s))

    def test_different_weights_or_models_are_never_compared(self):
        a = tl(embed_reid=unit(2048), embed_reid_model="reid:old")
        b = tl(embed_reid=unit(2048, 3), embed_reid_model="reid:new")
        self.assertTrue(np.isnan(reid.appearance(a, b)[0]))
        c = tl(embed=unit(576), embed_model="other:model")
        d = tl(embed=unit(576, 4), embed_model=MOBILENET_TAG)
        self.assertTrue(np.isnan(reid.appearance(c, d)[0]))
        self.assertTrue(np.isnan(reid.appearance(tl(), d)[0]))

    def test_untagged_legacy_vectors_count_as_mobilenet(self):
        a, b = tl(embed=unit(576)), tl(embed=unit(576, 1), embed_model=MOBILENET_TAG)
        self.assertTrue(np.isfinite(reid.appearance(a, b)[0]))

    def test_missing_vector_is_no_evidence(self):
        P = dict(reid.default_params(120.0), l_a=10.0)
        a, b = tl(), tl(camera_id=2, t_cross=15.0, lateral=8.0, direction="incoming")
        t = reid.pair_terms(a, b, P)
        self.assertEqual(t["c_embed"], 0.0)


class PerModelLLRTests(unittest.TestCase):
    def test_each_model_gets_its_own_fit(self):
        rng = np.random.default_rng(0)
        src, dst, matches = [], [], []
        for i in range(40):
            v = unit(2048, i)
            w = v + rng.normal(0, 0.01, 2048)
            w = (w / np.linalg.norm(w)).astype(np.float32)
            m, m2 = unit(576, 1000 + i), unit(576, 2000 + i)    # MobileNet vectors carry no signal here
            a = tl(track_id=i, t_cross=float(i), embed=m, embed_model=MOBILENET_TAG,
                   embed_reid=v, embed_reid_model=RTAG)
            b = tl(camera_id=2, track_id=100 + i, t_cross=float(i) + 15, embed=m2, embed_model=MOBILENET_TAG,
                   embed_reid=w, embed_reid_model=RTAG)
            src.append(a)
            dst.append(b)
            matches.append((i, i, None))
        _, llr = reid.refit(src, dst, matches, reid.default_params(120.0))
        keys = {k[0] for k in llr}
        self.assertIn("embed:reid:" + RTAG, keys)
        self.assertTrue(llr[("embed:reid:" + RTAG, "other")].ok)
        # the same pairs, ReID vectors stripped from half: MobileNet is fitted on its own, never ReID's numbers
        for t in src[::2] + dst[::2]:
            t.embed_reid = None
        _, llr2 = reid.refit(src, dst, matches, reid.default_params(120.0))
        self.assertIn("embed:" + MOBILENET_TAG, {k[0] for k in llr2})
        self.assertFalse(llr2[("embed:" + MOBILENET_TAG, "other")].ok)     # no signal there: switched off

    def test_a_reid_score_is_not_judged_by_a_mobilenet_distribution(self):
        P = dict(reid.default_params(120.0), l_a=10.0)
        v = unit(2048)
        a = tl(embed_reid=v, embed_reid_model=RTAG)
        b = tl(camera_id=2, t_cross=15.0, lateral=8.0, direction="incoming", embed_reid=v,
               embed_reid_model=RTAG)
        mob = reid.LLR(np.full(30, 0.9) + np.linspace(0, 0.01, 30), np.full(60, 0.1) + np.linspace(0, 0.01, 60))
        self.assertTrue(mob.ok)
        from collections import defaultdict
        llr = defaultdict(reid.LLR)
        llr[("embed:" + MOBILENET_TAG, "other")] = mob
        self.assertEqual(reid.pair_terms(a, b, P, llr)["c_embed"], 0.0)    # no ReID fit yet: no evidence


class WorkerTests(unittest.TestCase):
    def patched(self, seconds):
        def fake(name, crops):
            time.sleep(seconds)
            return unit(2048), RTAG
        return mock.patch.object(Embedder, "encode_with", side_effect=fake), \
            mock.patch.object(Embedder, "reid_wanted", return_value=True)

    def test_a_full_queue_drops_and_never_blocks(self):
        w = ReIDWorker(maxsize=2)
        a, b = self.patched(0.15)
        crops = [np.zeros((40, 40, 3), np.uint8)]
        with a, b:
            t0 = time.perf_counter()
            results = [w.submit(tl(track_id=i), crops) for i in range(10)]
            self.assertLess(time.perf_counter() - t0, 0.1, "submit waited for the model")
            self.assertGreater(w.stats()["dropped"], 0)
            self.assertEqual(sum(results), w.stats()["queued"])
            self.assertTrue(w.drain(10))
        st = w.stats()
        self.assertEqual(st["done"], st["queued"])
        self.assertEqual(st["queued"] + st["dropped"], 10)
        self.assertGreater(st["mean_ms"], 100)

    def test_the_vector_is_attached_and_the_callback_runs_on_another_thread(self):
        w, seen = ReIDWorker(), []
        a, b = self.patched(0.0)
        t = tl()
        with a, b:
            w.submit(t, [np.zeros((40, 40, 3), np.uint8)], lambda x: seen.append(threading.current_thread()))
            self.assertTrue(w.drain(5))
        self.assertEqual(t.embed_reid_model, RTAG)
        self.assertEqual(t.embed_reid.shape, (2048,))
        self.assertNotEqual(seen[0], threading.current_thread())

    def test_nothing_is_queued_when_reid_is_off(self):
        w = ReIDWorker()
        with mock.patch.object(Embedder, "reid_wanted", return_value=False):
            self.assertFalse(w.submit(tl(), [np.zeros((40, 40, 3), np.uint8)]))
        self.assertEqual(w.stats()["queued"], 0)
        self.assertEqual(w.stats()["dropped"], 0)

    def test_a_failing_model_is_counted_not_raised(self):
        w = ReIDWorker()
        with mock.patch.object(Embedder, "encode_with", return_value=(None, None)), \
                mock.patch.object(Embedder, "reid_wanted", return_value=True):
            w.submit(tl(), [np.zeros((40, 40, 3), np.uint8)])
            self.assertTrue(w.drain(5))
        self.assertEqual(w.stats()["failed"], 1)


class CollectorTests(unittest.TestCase):
    def collector(self, worker):
        done, late = [], []
        c = reid.TrackletCollector(1, done.append, on_reid=late.append, reid_queue=worker)
        tid = 7
        c.meta[tid] = dict(cls="Car", zone="z", y_ref=10.0, wall_at=100.0, t_at=5.0, width_samples=[1.7])
        c.samples[tid] = [(0.5 * k, 2.0, 2.0 + 2.0 * k) for k in range(8)]
        c.crops[tid] = [(1.0, np.random.default_rng(k).integers(0, 255, (60, 40, 3), dtype=np.uint8))
                        for k in range(3)]
        c.seen[tid] = 4.0
        return c, tid, done, late

    def test_mobilenet_now_reid_later(self):
        w = ReIDWorker()
        c, tid, done, late = self.collector(w)

        gate = threading.Event()        # holds the ReID job until the "not there yet" check has run

        def fake(name, crops):
            if name == "mobilenet":
                return unit(576), MOBILENET_TAG
            gate.wait(5)
            return unit(2048), RTAG
        with mock.patch.object(Embedder, "encode_with", side_effect=fake), \
                mock.patch.object(Embedder, "reid_wanted", return_value=True):
            c._finish(tid)
            self.assertEqual(len(done), 1)
            t = done[0]
            self.assertEqual(t.embed_model, MOBILENET_TAG)          # present at once
            self.assertIsNone(t.embed_reid)                         # not there yet
            self.assertNotIn("pending_crops", t.__dict__)
            gate.set()
            self.assertTrue(w.drain(5))
        self.assertEqual(t.embed_reid_model, RTAG)
        self.assertEqual(late, [t])

    def test_mobilenet_only_policy_queues_nothing(self):
        w = ReIDWorker()
        c, tid, done, late = self.collector(w)
        with mock.patch.object(Embedder, "encode_with", return_value=(unit(576), MOBILENET_TAG)), \
                mock.patch.object(Embedder, "reid_wanted", return_value=False):
            c._finish(tid)
        self.assertEqual(w.stats()["queued"], 0)
        self.assertEqual(done[0].embed_model, MOBILENET_TAG)
        self.assertEqual(late, [])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "s.db"

    def test_round_trip_and_late_reid(self):
        s = Store(self.path)
        t = tl(zone="z", embed=unit(576), embed_model=MOBILENET_TAG)
        t.hist = t.lab = None
        cam_id = s.add_camera("c", "file", "x.mp4")
        t.id = s.add_tracklet(cam_id, None, t)
        row = s.tracklets()[0]
        back = link.to_tracklet(row)
        self.assertEqual(back.embed_model, MOBILENET_TAG)
        self.assertIsNone(back.embed_reid)
        s.set_tracklet_reid(t.id, unit(2048), RTAG)
        back = link.to_tracklet(s.tracklets()[0])
        self.assertEqual(back.embed_reid.shape, (2048,))
        self.assertEqual(back.embed_reid_model, RTAG)
        self.assertEqual(back.embed.shape, (576,))

    def test_an_old_database_is_migrated_in_place(self):
        db = sqlite3.connect(str(self.path))
        from surveillance import store as store_mod
        old = "\n".join(line for line in store_mod.SCHEMA.splitlines()
                        if not line.strip().startswith(("embed_model", "embed_reid")))
        db.executescript(old)       # the schema as it was before the ReID columns
        db.execute("INSERT INTO tracklets (camera_id, track_id, t_cross, embed, created_at) VALUES (1, 1, 1.0, ?, 'x')",
                   (unit(576).tobytes(),))
        db.execute("INSERT INTO tracklets (camera_id, track_id, t_cross, created_at) VALUES (1, 2, 2.0, 'x')")
        db.commit()
        db.close()
        s = Store(self.path)
        rows = s.q("SELECT id, embed, embed_model, embed_reid, embed_reid_model FROM tracklets ORDER BY id")
        self.assertEqual(rows[0]["embed_model"], MOBILENET_TAG)
        self.assertEqual(len(rows[0]["embed"]), 576 * 4)
        self.assertIsNone(rows[1]["embed_model"])
        self.assertIsNone(rows[0]["embed_reid"])
        Store(self.path)        # a second open changes nothing and does not fail


if __name__ == "__main__":
    unittest.main()
