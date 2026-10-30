"""Tests for the rules ported from Shihab's Kakoli check (2026-10-28): V6 direction / kerb-strip / stricter vote,
kerb areas attaching to a road, kerb queues (creep) and duplicate suppression for the standing rules.

    python -m unittest tests.test_kerb_helmet
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tests.test_violations import FPS, POINTS, Sim, codes, flowing_traffic, make_camera  # noqa: E402
from violations import AreaZone, CameraConfig, RoadZone, ViolationEngine  # noqa: E402
from violations.zones import KERB_AREAS  # noqa: E402

FRAME_W = 2560


def bike(t, x=3.0, v=20 / 3.6):
    return [(1, "Motorbike", (x, 20 + v * t), 20.0)] if t < 3 else []


class HelmetRuleTests(unittest.TestCase):
    def run_bike(self, probs, heading=1, x=3.0, camera=None, overrides=None):
        seq = iter(probs)
        sim = Sim(camera or make_camera(), only=["V6"], overrides=overrides, heading=heading,
                  helmet=lambda crop: next(seq, probs[-1]))
        return sim.run(5, lambda t: bike(t, x))

    def test_defaults(self):
        from violations.helmet import NoHelmet
        d = NoHelmet.DEFAULTS
        self.assertEqual((d["helmet_prob"], d["min_votes"], d["min_ratio"], d["directions"]),
                         (0.5, 4, 0.75, ["toward_camera"]))

    def test_toward_camera_is_judged(self):
        self.assertEqual(codes(self.run_bike([0.9] * 10)), ["V6"])

    def test_away_from_camera_is_not_judged(self):
        self.assertEqual(self.run_bike([0.9] * 10, heading=-1), [])

    def test_unknown_heading_is_not_judged_yet(self):
        self.assertEqual(self.run_bike([0.9] * 10, heading=0), [])

    def test_directions_are_configurable(self):
        ov = {"helmet": {"directions": ["away_from_camera"]}}
        self.assertEqual(codes(self.run_bike([0.9] * 10, heading=-1, overrides=ov)), ["V6"])
        self.assertEqual(self.run_bike([0.9] * 10, heading=1, overrides=ov), [])

    def test_three_of_four_rule(self):
        # 4 bare + 1 helmet-ish = 0.8 >= 0.75: reported. 3 bare + 1 helmet = only 3 votes: not.
        self.assertEqual(codes(self.run_bike([0.9, 0.9, 0.1, 0.9, 0.9] + [0.6] * 20)), ["V6"])
        self.assertEqual(self.run_bike([0.9, 0.9, 0.9, 0.1] + [0.6] * 20), [])

    def test_two_helmets_in_five_is_below_three_quarters(self):
        self.assertEqual(self.run_bike([0.9, 0.9, 0.9, 0.9, 0.2, 0.2] + [0.6] * 20), [])

    def test_p_up_to_half_counts_as_helmet(self):
        # 4 bare + 2 sightings at exactly 0.5 (a helmet vote now) = 4/6 < 3/4; with one it is 4/5 = 0.8
        self.assertEqual(self.run_bike([0.9] * 4 + [0.5] * 2 + [0.6] * 20), [])
        self.assertEqual(codes(self.run_bike([0.9] * 4 + [0.5] + [0.6] * 20)), ["V6"])

    def test_rider_in_a_kerb_strip_is_not_judged(self):
        cam = make_camera(areas=[("no_parking", (-3, 0, -0.1, 90))])
        self.assertEqual(self.run_bike([0.9] * 10, x=-1.5, camera=cam), [])
        # the same bike on the road itself is judged
        self.assertEqual(codes(self.run_bike([0.9] * 10, x=3.0, camera=cam)), ["V6"])


def two_road_camera(extra_kind="no_parking"):
    """Road A (the usual zone) and road B far to the left; an area of `extra_kind` beside A and one beside B."""
    a = RoadZone("A", POINTS, 10.5, 90.0)
    b_pts = [[100.0, 200.0], [260.0, 196.0], [600.0, 932.0], [150.0, 908.0]]
    b = RoadZone("B", b_pts, 10.5, 90.0)
    near_a = AreaZone(extra_kind, "kerb A", a.to_pixels([[-3, 10], [-0.1, 10], [-0.1, 80], [-3, 80]]))
    return a, b, near_a, CameraConfig([a, b], [near_a], frame_size=[FRAME_W, 1440])


class KerbAreaTests(unittest.TestCase):
    def test_area_attaches_to_the_nearest_road_not_a_far_one(self):
        a, b, area, cam = two_road_camera()
        self.assertIs(area.road, a)
        centre = tuple(area.poly.mean(axis=0))
        self.assertIsNone(b.contains(centre) or None)
        self.assertIs(cam.road_at(centre), a)

    def test_far_area_attaches_to_nothing(self):
        a = RoadZone("A", POINTS, 10.5, 90.0)
        far = AreaZone("parking", "far", [[100, 500], [300, 500], [300, 700], [100, 700]])
        cam = CameraConfig([a], [far], frame_size=[FRAME_W, 1440])
        self.assertIsNone(far.road)
        self.assertIsNone(cam.road_at((200, 600)))

    def test_kinds_and_bus_lane_is_not_a_kerb_area(self):
        self.assertEqual(KERB_AREAS, ("no_parking", "bus_stop", "parking"))
        a, b, area, cam = two_road_camera("bus_lane")
        self.assertIsNone(area.road)
        self.assertIsNone(cam.road_at(tuple(area.poly.mean(axis=0))))   # outside the road: no road, as before

    def test_point_inside_road_still_gets_that_road(self):
        a, b, area, cam = two_road_camera()
        self.assertIs(cam.road_at(tuple(a.to_pixels([[5, 40]])[0])), a)

    def test_car_in_a_kerb_strip_is_measured_on_the_road_and_reported_by_v3(self):
        cam = make_camera(areas=[("no_parking", (-3, 0, -0.1, 90))])
        ev = Sim(cam, only=["V3"]).run(70, lambda t: [(1, "CNG", (-1.5, 60.0), 0.0)] + flowing_traffic(t))
        self.assertEqual(codes(ev), ["V3"])
        self.assertIsNotNone(ev[0].obs.road)

    def test_bus_lane_still_reports_a_car(self):
        from tests.test_bus_lane import make_camera as bus_camera, Sim as BusSim
        sim = BusSim(bus_camera())
        v = 30 / 3.6
        ev = sim.run(80 / v, lambda t: [(1, "Car", (1.5, 2 + v * t), 30.0)])
        self.assertIn("V9", codes(ev))


class CreepAndDuplicateTests(unittest.TestCase):
    KERB = ("no_parking", (0, 0, 3, 90))

    def run_stand(self, positions, **kw):
        """positions(t) -> (x, y) of one CNG with a fixed ID at the kerb."""
        cam = make_camera(areas=[self.KERB])
        return Sim(cam, only=["V3"], **kw).run(60, lambda t: [(1, "CNG", positions(t), 0.0)] + flowing_traffic(t))

    @staticmethod
    def inching(step):
        # stands 8 s, then moves up `step` m every 8 s
        return lambda t: (1.5, 40.0 + step * int(t // 8))

    def test_inching_up_within_creep_keeps_the_stand(self):
        ev = self.run_stand(self.inching(3.0))     # 3 m a time: 6 m after 16 s, 21 m at the end is beyond 10 m
        self.assertEqual(codes(ev), ["V3"])
        self.assertLess(ev[0].started_s or 0, 12)  # the stand began at the start, not after the first move

    def test_creep_is_reported_in_the_evidence(self):
        ev = self.run_stand(lambda t: (1.5, 40.0 if t < 8 else 46.0))
        self.assertEqual(codes(ev), ["V3"])
        self.assertAlmostEqual(ev[0].details["moved_up_m"], 6.0, delta=0.6)

    def test_moving_beyond_creep_starts_a_new_stand(self):
        # 12 m jump after 12 s: neither stand lasts 20 s on its own within 24 s of video
        cam = make_camera(areas=[self.KERB])
        ev = Sim(cam, only=["V3"]).run(24, lambda t: [(1, "CNG", (1.5, 30.0 if t < 12 else 42.0), 0.0)]
                                       + flowing_traffic(t))
        self.assertEqual(ev, [])

    def test_creep_is_off_by_default_in_the_monitor_and_10_m_for_parking(self):
        from violations.illegal_parking import IllegalParking
        from violations.stationary import StationaryMonitor
        self.assertIsNone(StationaryMonitor().creep_m)
        self.assertEqual(IllegalParking.DEFAULTS["creep_m"], 10.0)

    def test_same_vehicle_under_a_new_id_is_not_reported_twice(self):
        # Reported at 25 s; at 40 s a second ID appears at the same spot, still alive alongside the first.
        def scene(t):
            v = [(1, "CNG", (1.5, 60.0), 0.0)] if t < 45 else []
            if t >= 40:
                v.append((2, "CNG", (1.5, 60.4), 0.0))
            return v + flowing_traffic(t)
        ev = Sim(make_camera(areas=[self.KERB]), only=["V3"]).run(80, scene)
        self.assertEqual([e.obs.tid for e in ev], [1])

    def test_different_vehicle_3_m_away_is_reported(self):
        def scene(t):
            v = [(1, "CNG", (1.5, 60.0), 0.0)] if t < 45 else []
            if t >= 40:
                v.append((2, "CNG", (1.5, 63.0), 0.0))
            return v + flowing_traffic(t)
        ev = Sim(make_camera(areas=[self.KERB]), only=["V3"]).run(80, scene)
        self.assertEqual(sorted(e.obs.tid for e in ev), [1, 2])

    def test_base_class_covers_v1_v3_v7(self):
        from violations.bus_blocking import BusBlocking
        from violations.illegal_parking import IllegalParking
        from violations.road_blocking import RoadBlocking
        from violations.stationary import StandRule
        for cls in (BusBlocking, IllegalParking, RoadBlocking):
            self.assertTrue(issubclass(cls, StandRule), cls)
            self.assertTrue(hasattr(cls.__mro__[0], "_same_as_reported"))


if __name__ == "__main__":
    unittest.main()
