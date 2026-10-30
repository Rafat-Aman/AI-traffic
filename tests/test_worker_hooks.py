import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TRAFFIC_DATA_DIR", tempfile.mkdtemp())

from surveillance import segment_index
from surveillance.clips import ClipRecorder
from surveillance.worker import CameraWorker


def worker():
    w = CameraWorker.__new__(CameraWorker)
    w.store, w.cam = object(), {"id": 3}
    w._index_last, w._index_thread = 0.0, None
    return w


class IndexHooks(unittest.TestCase):
    def test_run_hook_called_once_and_never_raises(self):
        w = worker()
        with mock.patch.object(segment_index, "on_run_finished", side_effect=RuntimeError("boom")) as m:
            w._index_run(7)
        m.assert_called_once_with(w.store, 7)

    def test_live_build_does_not_overlap(self):
        w = worker()
        gate, calls = threading.Event(), []

        def slow(store, cid, since=None):
            calls.append(cid)
            gate.wait(5)
        with mock.patch.object(segment_index, "build_for_camera", side_effect=slow):
            w._index_live(True)
            time.sleep(0.1)
            w._index_last = 0.0             # interval elapsed, but the first build is still running
            w._index_live(True)
            w._index_live(True)
            self.assertEqual(calls, [3])
            gate.set()
            w._index_thread.join(2)
            w._index_last = 0.0
            w._index_live(True)
            w._index_thread.join(2)
        self.assertEqual(calls, [3, 3])

    def test_interval_and_files_skipped(self):
        w = worker()
        with mock.patch.object(segment_index, "build_for_camera") as m:
            w._index_live(False)
            w._index_last = time.time()
            w._index_live(True)
        m.assert_not_called()


class ClipStart(unittest.TestCase):
    def rec(self):
        r = ClipRecorder(pre_sec=5, post_sec=5, workers=1)
        for i in range(0, 200):                # 20 s at 10 fps
            r.add(b"x", i / 10)
        return r

    def first(self, r):
        return r.pending[-1][2][0][0]

    def test_ring_holds_12s(self):
        r = self.rec()
        self.assertAlmostEqual(r.ring[-1][0] - r.ring[0][0], 12.0, delta=0.11)
        r.close()

    def test_early_started_reaches_back(self):
        r = self.rec()
        t = r.ring[-1][0]
        r.trigger("a.mp4", t, started_s=t - 8)
        self.assertAlmostEqual(self.first(r), t - 10, delta=0.11)
        r.trigger("b.mp4", t, started_s=t - 30)       # beyond the ring: capped
        self.assertAlmostEqual(self.first(r), t - 12, delta=0.11)
        r.pending = []
        r.close()

    def test_normal_unchanged(self):
        r = self.rec()
        t = r.ring[-1][0]
        r.trigger("c.mp4", t)
        self.assertAlmostEqual(self.first(r), t - 5, delta=0.11)
        r.trigger("d.mp4", t, started_s=t - 1)        # started inside the normal window
        self.assertAlmostEqual(self.first(r), t - 5, delta=0.11)
        r.pending = []
        r.close()

    def test_worker_passes_started_only_for_early_codes(self):
        self.assertEqual(ClipRecorder.EARLY_CODES, ("V10", "V11"))


if __name__ == "__main__":
    unittest.main()
