"""Measurement accuracy: speed fitting, hidden ground points, stand handoff after an ID switch, passing vehicles
in the imprecise far part of a zone, and an end-to-end ground-truth benchmark (tools/bench_accuracy.py).

    python -m unittest tests.test_accuracy -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from surveillance.pipeline import Pipeline, bottom_occluded  # noqa: E402
from surveillance.speed import SpeedEstimator, fit_kph  # noqa: E402
from test_violations import KERB, Sim, codes, flowing_traffic, make_camera  # noqa: E402
from violations import Observation, ViolationEngine  # noqa: E402


class SpeedFitTests(unittest.TestCase):
    def test_fit_is_more_precise_than_end_points(self):
        """Noisy positions of a car at a steady 40 km/h: the least-squares fit is much closer than the distance
        between the window's end points."""
        rng = np.random.default_rng(1)
        v = 40 / 3.6
        fit_err, end_err = [], []
        for _ in range(300):
            t = np.arange(16) / 15
            y = v * t + rng.normal(0, 0.25, len(t))
            x = 3 + rng.normal(0, 0.1, len(t))
            path = list(zip(t, x, y))
            fit_err.append(abs(fit_kph(path) - 40))
            end_err.append(abs(math.hypot(x[-1] - x[0], y[-1] - y[0]) / (t[-1] - t[0]) * 3.6 - 40))
        self.assertLess(np.mean(fit_err), 0.6 * np.mean(end_err))

    def test_one_jumped_position_is_ignored(self):
        t = np.arange(16) / 15
        y = 10 * t
        y[9] += 3.0   # one box edge jumped by 3 m
        self.assertAlmostEqual(fit_kph(list(zip(t, np.zeros(16), y))), 36.0, delta=1.0)

    def test_estimator_reports_the_fit(self):
        s = SpeedEstimator()
        kph = None
        for i in range(30):
            kph = s.update(1, i / 15, "zone", (2.0, 20 * i / 15))
        self.assertAlmostEqual(kph, 72.0, delta=0.5)


class HiddenGroundPointTests(unittest.TestCase):
    def test_nearer_box_covering_the_bottom(self):
        far = (100, 100, 200, 180)        # its bottom edge (y 180) is covered by...
        near = (120, 160, 260, 300)       # ...this nearer vehicle (bottom lower in the picture)
        side = (400, 100, 500, 180)
        self.assertEqual(bottom_occluded([far, near, side]), [True, False, False])

    def test_vehicles_side_by_side_do_not_hide_each_other(self):
        self.assertEqual(bottom_occluded([(100, 100, 200, 200), (190, 105, 290, 202)]), [False, False])

    def test_suddenly_flatter_box_without_a_detected_occluder(self):
        """The vehicle in front was missed by the detector: the box of the one behind is suddenly much shorter but
        as wide - its bottom edge is not the ground point."""
        p = Pipeline.__new__(Pipeline)
        from collections import defaultdict, deque
        p.shapes, p.flat_since = defaultdict(deque), {}
        for i in range(10):
            self.assertFalse(p._foot_hidden(1, i / 15, 100, 100, 200, 180, False))
        self.assertTrue(p._foot_hidden(1, 0.7, 100, 100, 200, 140, False))      # bottom half hidden
        # still that flat 2 s later: it really looks like that now (turned) - accepted again
        self.assertFalse(p._foot_hidden(1, 2.8, 100, 100, 200, 140, False))


class StandHandoffTests(unittest.TestCase):
    def test_new_id_first_seen_off_position_takes_over_the_timer(self):
        """A CNG stands in a lane; at 12 s the tracker loses it and gives it a new ID whose first positions are
        3 m off (it is still partly hidden). Once its position settles the new ID continues the 12 s timer, so it
        is reported at 20 s of standing, not 20 s after the ID switch."""
        sim = Sim(make_camera(areas=[KERB]), only=["V7"])

        def scene(t):
            if t < 12:
                cng = (1, "CNG", (5.0, 60.0), 0.0)
            elif t < 12.4:
                return flowing_traffic(t)                               # not detected for a moment
            else:
                cng = (2, "CNG", (5.0, 57.0 if t < 12.8 else 60.0), 0.0)  # new ID, first positions off
            return [cng] + flowing_traffic(t)
        ev = sim.run(26, scene)
        self.assertEqual(codes(ev), ["V7"])
        self.assertLess(ev[0].obs.time_s, 22.5)


class CoarsePositionTests(unittest.TestCase):
    def test_vehicles_driving_on_into_the_far_end_still_pass(self):
        """A CNG stands right where precise measurement starts. Every car passing it drives on into the far end,
        where only coarse positions exist: they have still gone round it."""
        cam = make_camera(areas=[KERB])
        engine = ViolationEngine(cam, only=["V7"])
        road = cam.roads[0]
        reliable = 58.0   # positions nearer the far edge than this are coarse only
        events = []
        for i in range(30 * 30):
            t = i / 30
            obs = []
            # traffic away from the camera (y flipped): from the near end past the CNG into the far end. Without
            # coarse positions no car is ever clearly (1.5 m) ahead of it, so none would count as passing.
            away = [(tid, cls, (x, 89 - y), kph) for tid, cls, (x, y), kph in flowing_traffic(t)]
            for tid, cls, (x, y), kph in [(1, "CNG", (5.0, 58.5), 0.0)] + away:
                px = road.to_pixels([[x, y]])[0]
                pt = (float(px[0]), float(px[1]))
                precise = y >= reliable
                obs.append(Observation(tid, i, t, cls, 1.0, 100, (0, 0, 1, 1), pt, road,
                                       (x, y) if precise else None, kph, None, cam.areas_at(pt),
                                       xy_coarse=(x, y)))
            events += engine.process(obs, None, i, t)
        self.assertEqual(codes(events), ["V7"])


class BenchmarkTests(unittest.TestCase):
    def test_simulated_camera_end_to_end(self):
        """Two minutes of simulated traffic with realistic detector and tracker errors: accurate speeds, and no
        legal vehicle reported (the full benchmark is tools/bench_accuracy.py)."""
        import bench_accuracy
        r = bench_accuracy.run(seed=2, seconds=120, quiet=True)
        self.assertLess(r["speed"]["mae_kph"], 3.0)
        self.assertLess(r["speed"]["p95_abs_kph"], 10.0)
        self.assertLess(r["standing"]["p95_kph"], 4.0)    # standing vehicles never look like they move
        for name, d in r["decisions"].items():
            self.assertEqual(d["false_alarms"], 0, name)
            if d["truth"]:
                self.assertGreaterEqual(d["caught"] / d["truth"], 0.75, name)


if __name__ == "__main__":
    unittest.main()
