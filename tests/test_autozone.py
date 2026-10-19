"""Automatic zone setup (surveillance/autozone.py) on a synthetic road seen by a camera with known lens, roll,
height and tilt: does it recover the camera, the layout and - what matters for speed - real distances?

    python -m unittest tests.test_autozone -v
"""

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.autozone import (CLASS_SIZE, GroundModel, TrafficObserver, _cuboid_box, lens_from_verticals,  # noqa: E402
                                   propose, vertical_segments)
from violations.zones import RoadZone, camera_config_from_dict  # noqa: E402

W, H = 1920, 1080
TRUE_HFOV, TRUE_ROLL, TRUE_HEIGHT = 62.0, 0.6, 7.5
VP = (1010.0, 130.0)
FPS = 10
# Carriageways in true road coordinates (metres across; negative = image left). Median 3 m.
LEFT = (-13.0, -3.0)    # traffic away from the camera
RIGHT = (0.0, 10.0)     # traffic toward the camera


def true_camera():
    gm = GroundModel.from_hfov(W, H, TRUE_HFOV, VP, TRUE_ROLL)
    gm.scale = TRUE_HEIGHT
    return gm


def vehicle_box(gm, along, across, size):
    x1, y1, x2, y2 = _cuboid_box(gm, np.array([along]), np.array([across]), size)
    return float(x1[0]), float(y1[0]), float(x2[0]), float(y2[0])


def inside(b):
    return b[0] > 0 and b[1] > 0 and b[2] < W - 1 and b[3] < H - 1


class Traffic:
    """Builds the observation: moving traffic, kerbside parking, a side road and buses stopping after it."""

    def __init__(self, seed=0):
        self.gm = true_camera()
        self.rng = np.random.default_rng(seed)
        self.frames = {}          # t -> list of ((box), tid, cls)
        self.tid = 0

    def add_path(self, cls, pts, size_noise=0.04):
        """pts: [(t, along, across)] of the vehicle's nearest ground edge."""
        self.tid += 1
        w, l, h, wt = CLASS_SIZE[cls]
        k = 1 + self.rng.normal(0, size_noise)
        size = (w * k, l * k, h * k, wt * k)
        for t, al, ac in pts:
            b = vehicle_box(self.gm, al, ac, size)
            if not inside(b):
                continue
            b = tuple(v + self.rng.normal(0, 1.0) for v in b)
            self.frames.setdefault(round(t, 2), []).append((b, self.tid, cls))

    def drive(self, t0, cls, across, sign, kph, al_from=12, al_to=140):
        v = kph / 3.6
        n = int((al_to - al_from) / v * FPS)
        pts = []
        for i in range(n):
            d = i / FPS * v
            al = al_from + d if sign > 0 else al_to - d
            pts.append((t0 + i / FPS, al, across + self.rng.normal(0, 0.15)))
        self.add_path(cls, pts)

    def build(self, seconds=150):
        rng = self.rng
        classes = ["Car"] * 6 + ["CNG"] * 3 + ["Bus"] * 1 + ["MPV"]
        t = 0.0
        while t < seconds:
            for sign, (a0, a1) in ((1, LEFT), (-1, RIGHT)):
                lane = rng.integers(0, 3)
                across = a0 + 1.7 + lane * 3.3
                self.drive(t, rng.choice(classes), across, sign, rng.uniform(20, 50))
            t += rng.uniform(0.7, 1.3)
        # kerbside parking along the right kerb (several sections), standing 40 s each
        for k, al in enumerate(range(25, 115, 12)):
            t0 = 5 + 9 * k
            self.add_path("Car", [(t0 + i / FPS, al, RIGHT[1] + 0.9) for i in range(40 * FPS)])
        # side road joining the left carriageway from the left at along 60-70 m
        for k in range(14):
            t0 = 3 + 10 * k
            pts = []
            for i in range(9 * FPS):
                s = i / FPS
                if s < 3:   # comes in across the kerb line
                    al, ac = 62 + s * 1.5, LEFT[0] - 12 + s * 4.0
                else:       # then drives away along the road
                    al, ac = 66.5 + (s - 3) * 8, LEFT[0] + 1.7
                pts.append((t0 + s, al, ac))
            self.add_path("Car", pts)
        # buses stopping at the left kerb just after that junction (traffic moving away: farther along)
        for k in range(4):
            t0 = 20 + 30 * k
            pts = [(t0 + i / FPS, 30 + i / FPS * 8, LEFT[0] + 1.3) for i in range(int(55 / 8 * FPS))]
            t1 = pts[-1][0]
            pts += [(t1 + i / FPS, 85.0, LEFT[0] + 1.3) for i in range(14 * FPS)]
            t2 = pts[-1][0]
            pts += [(t2 + i / FPS, 85 + i / FPS * 8, LEFT[0] + 1.3) for i in range(5 * FPS)]
            self.add_path("Bus", pts)
        return self

    def pole_frame(self):
        """A frame with vertical poles / building edges (and some road markings, which must be ignored)."""
        img = np.full((H, W, 3), 70, np.uint8)
        for al in (20, 35, 55, 80, 110, 150):
            for ac in (-24, -18, 15, 21):
                a, b = self.gm.pixel(np.array([al]), np.array([ac]), 0.0)[0], \
                    self.gm.pixel(np.array([al]), np.array([ac]), 9.0)[0]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (235, 235, 235), 4, cv2.LINE_AA)
        for ac in (-9.7, -6.4, 3.3, 6.6):  # lane markings: they point at the road's vanishing point
            a, b = self.gm.pixel(np.array([10, 140]), np.array([ac, ac]))
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 3, cv2.LINE_AA)
        return img

    def observer(self, with_frames=True):
        obs = TrafficObserver((W, H))
        frame = self.pole_frame() if with_frames else None
        for t in sorted(self.frames):
            obs.add(t, self.frames[t])
            if with_frames:
                obs.add_frame(frame, t, every_sec=20)
        return obs


class GroundModelTests(unittest.TestCase):
    def test_pixel_and_road_coords_are_inverse(self):
        gm = true_camera()
        al, ac = np.array([15, 40, 90, 140.0]), np.array([-12, -3, 4, 9.5])
        px = gm.pixel(al, ac)
        al2, ac2 = gm.road_coords(px[:, 0], px[:, 1])
        np.testing.assert_allclose(al2, al, rtol=1e-6)
        np.testing.assert_allclose(ac2, ac, atol=1e-6)

    def test_road_vanishing_point(self):
        gm = true_camera()
        far = gm.pixel(np.array([1e6]), np.array([0.0]))[0]
        self.assertAlmostEqual(far[0], VP[0], delta=2)
        self.assertAlmostEqual(far[1], VP[1], delta=2)


class LensTests(unittest.TestCase):
    def test_lens_and_roll_from_vertical_lines(self):
        tr = Traffic()
        img = tr.pole_frame()
        segs = vertical_segments([(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.0)], VP)
        lens = lens_from_verticals(segs, W, H, VP)
        self.assertIsNotNone(lens)
        hfov = 2 * math.degrees(math.atan(W / 2 / lens["f"]))
        self.assertAlmostEqual(hfov, TRUE_HFOV, delta=1.5)
        self.assertAlmostEqual(lens["roll_deg"], TRUE_ROLL, delta=0.25)


class ProposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traffic = Traffic().build()
        cls.cfg, cls.rep = propose(cls.traffic.observer())

    def test_camera_recovered(self):
        cal = self.rep["calibration"]
        self.assertAlmostEqual(cal["hfov_deg"], TRUE_HFOV, delta=2.0)
        self.assertAlmostEqual(cal["camera_height_m"], TRUE_HEIGHT, delta=0.08 * TRUE_HEIGHT)
        self.assertLessEqual(cal["speed_uncertainty_pct"], 15)

    def test_two_carriageways_with_directions(self):
        roads = [z for z in self.cfg["zones"] if z["type"] == "road"]
        self.assertEqual(len(roads), 2)
        self.assertEqual(roads[0]["allowed_direction"], "away_from_camera")   # image-left one
        self.assertEqual(roads[1]["allowed_direction"], "toward_camera")
        for z in roads:
            self.assertEqual(z["scale_source"], "auto")
        # widths: traffic uses ~8.6 m of each 10 m carriageway; the right one also has cars parked at its kerb, so
        # it must reach out to them (~11 m)
        self.assertGreater(roads[0]["width_m"], 8.0)
        self.assertLess(roads[0]["width_m"], 11.5)
        self.assertGreater(roads[1]["width_m"], 10.0)
        self.assertLess(roads[1]["width_m"], 13.0)

    def test_distances_on_learned_zones_match_reality(self):
        """What speed enforcement depends on: 20 m driven along the road must measure 20 m on the learned zone."""
        cam = camera_config_from_dict(self.cfg, W, H)
        gm = true_camera()
        for road, across in ((cam.roads[0], -8.0), (cam.roads[1], 5.0)):
            a, b = gm.pixel(np.array([45.0, 65.0]), np.array([across, across]))
            self.assertTrue(road.contains(a) and road.contains(b))
            d = float(np.linalg.norm(road.to_metres(b) - road.to_metres(a)))
            self.assertAlmostEqual(d, 20.0, delta=20.0 * 0.08, msg=road.name)

    def test_kerbs_junction_and_bus_stop(self):
        kinds = [z["type"] for z in self.cfg["zones"]]
        self.assertEqual(kinds.count("no_parking"), 2)        # outer kerb of each carriageway
        self.assertGreaterEqual(kinds.count("intersection"), 1)
        self.assertGreaterEqual(kinds.count("bus_stop"), 1)
        cam = camera_config_from_dict(self.cfg, W, H)
        self.assertTrue(all(a.valid for a in cam.areas if a.kind == "bus_stop"), cam.check_bus_stops())

    def test_config_is_valid_for_the_pipeline(self):
        cam = camera_config_from_dict(self.cfg, W, H)
        self.assertEqual(len(cam.roads), 2)
        for r in cam.roads:
            self.assertIsInstance(r, RoadZone)
            self.assertGreater(r.length_m, 30)


class UTurnTests(unittest.TestCase):
    def test_u_turn_through_the_median_gap(self):
        """Vehicles that drive away, cross the median opening and come back: a U-turn area, no crash (the gap
        between the carriageways belongs to neither of them)."""
        tr = Traffic(seed=3).build(seconds=120)
        for k in range(6):
            t0, pts = 4 + 15 * k, []
            for i in range(int(8 * FPS)):        # away along the left carriageway, 30 -> 94 m
                pts.append((t0 + i / FPS, 30 + i / FPS * 8, -8.0))
            t1 = pts[-1][0]
            for i in range(int(4 * FPS)):        # swing across the median at ~96 m
                s = i / FPS / 4
                pts.append((t1 + i / FPS, 94 + 4 * math.sin(math.pi * s), -8.0 + 13.0 * s))
            t2 = pts[-1][0]
            for i in range(int(8 * FPS)):        # and back toward the camera on the right one
                pts.append((t2 + i / FPS, 94 - i / FPS * 8, 5.0))
            tr.add_path("Car", pts)
        cfg, rep = propose(tr.observer(with_frames=False))
        uturns = [z for z in cfg["zones"] if z["type"] == "uturn"]
        self.assertEqual(len(uturns), 1, rep["notes"])


class FailureTests(unittest.TestCase):
    def test_too_little_traffic_says_so(self):
        tr = Traffic()
        tr.drive(0, "Car", -8, 1, 30)
        with self.assertRaises(ValueError) as e:
            propose(tr.observer(with_frames=False))
        self.assertIn("observe longer", str(e.exception))

    def test_without_vertical_lines_the_lens_is_assumed_and_said(self):
        tr = Traffic().build(seconds=60)
        cfg, rep = propose(tr.observer(with_frames=False))
        self.assertIn("assumed", rep["calibration"]["lens"])
        self.assertTrue(any("vertical" in n for n in rep["notes"]))


class AutoSpeedMarginTests(unittest.TestCase):
    def test_auto_calibrated_zone_flags_only_well_above_the_limit(self):
        from violations.speed_violation import SpeedViolation
        rule = SpeedViolation()
        measured = RoadZone("m", [[0, 0], [100, 0], [100, 100], [0, 100]], 10, 50)
        auto = RoadZone("a", [[0, 0], [100, 0], [100, 100], [0, 100]], 10, 50, scale_source="auto",
                        speed_uncertainty_pct=8)
        self.assertAlmostEqual(rule.threshold(60, measured), 63.0)
        self.assertAlmostEqual(rule.threshold(60, auto), 60 * 1.12)   # 1.5 x 8 % margin


if __name__ == "__main__":
    unittest.main()
