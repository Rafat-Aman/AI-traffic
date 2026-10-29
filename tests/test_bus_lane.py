"""Synthetic-trajectory tests for the bus-lane rule V9 and the "bus_lane" area it needs (no model or video needed).

    python -m unittest tests.test_bus_lane -v

Vehicles are placed on a real road zone, seen through real area polygons, and fed to the real rule engine the way
surveillance/pipeline.py does it. The lane is 3.5 m wide along the left edge of the zone, so x = 1.75 m is the middle
of the lane and x = 8.5 m a general lane on the far side of the road.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import vlm  # noqa: E402
from violations import AreaZone, CameraConfig, Observation, RoadZone, ViolationEngine  # noqa: E402
from violations.engine import RULES  # noqa: E402
from violations.zones import AREA_COLORS, AREA_TYPES, camera_config_from_dict  # noqa: E402

FPS = 30
POINTS = [[1340.0, 200.0], [1505.0, 196.0], [2508.0, 932.0], [1458.0, 908.0]]   # 10.5 m x 90 m, as test_violations
FRAME = np.zeros((1440, 2560, 3), np.uint8)
LANE = "Left bus lane"
LANE_X, GENERAL_X = 1.75, 8.5


def make_camera(lane=(0, 0, 3.5, 90), hours=None, extra=(), rules=None):
    """One road zone, a bus lane drawn as a rectangle in road metres (x0, y0, x1, y1) - None for no bus lane - and
    any other areas as (kind, rectangle)."""
    road = RoadZone("Right side", POINTS, 10.5, 90.0)

    def polygon(x0, y0, x1, y1):
        return road.to_pixels([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    areas = [AreaZone("bus_lane", LANE, polygon(*lane), hours=hours)] if lane else []
    for kind, rect in extra:
        areas.append(AreaZone(kind, f"{kind} {len(areas) + 1}", polygon(*rect)))
    return CameraConfig([road], areas, rules=rules)


class Sim:
    """Feeds vehicles moving on the road plane through the rule engine, one frame at a time."""

    def __init__(self, camera, start=None, keep_every=1, metres="precise", overrides=None, noise=0.0):
        """metres: "precise" = xy is measured, "coarse" = only xy_coarse (the far end of a zone), "none" = the box
        is cut off by the frame edge. keep_every > 1 simulates a live camera dropping frames. noise = standard
        deviation, in metres, of the box jitter added to every measured position."""
        self.cam, self.start, self.keep_every, self.metres = camera, start, keep_every, metres
        self.engine = ViolationEngine(camera, overrides, only=["V9"])
        self.noise, self.rng = noise, np.random.default_rng(0)
        self.frame = 0

    def step(self, vehicles, purity=1.0):
        """vehicles: (tid, cls, (x_m, y_m), kph)."""
        self.frame += 1
        if self.frame % self.keep_every:
            return []
        t = self.frame / FPS
        road = self.cam.roads[0]
        obs = []
        for tid, cls, xy, kph in vehicles:
            px = road.to_pixels([xy])[0]
            point = (float(px[0]), float(px[1]))
            r = self.cam.road_at(point)
            metres = tuple(float(v) for v in r.to_metres(point)) if r else None
            if metres is not None and self.noise:
                metres = tuple(float(v) for v in np.float64(metres) + self.rng.normal(0, self.noise, 2))
            bbox = (int(point[0] - 60), int(point[1] - 120), int(point[0] + 60), int(point[1]))
            obs.append(Observation(
                tid=tid, frame_idx=self.frame, time_s=t, cls=cls, purity=purity, n_frames=100, bbox=bbox,
                point=point, road=r, xy=metres if self.metres == "precise" else None, kph=kph,
                areas=self.cam.areas_at(point), xy_coarse=None if self.metres == "none" else metres))
        wall = self.start + timedelta(seconds=t) if self.start else None
        return self.engine.process(obs, FRAME, self.frame, t, wall)

    def run(self, seconds, vehicles_fn, **kw):
        events = []
        for i in range(int(seconds * FPS)):
            events += self.step(vehicles_fn(i / FPS), **kw)
        return events


def drive(t, x=LANE_X, y0=5.0, kph=30.0, cls="Car", tid=1):
    """One vehicle driving toward the camera along road position x."""
    return [(tid, cls, (x, y0 + kph / 3.6 * t), kph)]


def flowing(t, kph=30.0, xs=(5.5, GENERAL_X), per_lane=3):
    """Cars in the general lanes, re-spawned at the far end with a new ID when they leave the zone."""
    v, out = kph / 3.6, []
    for i, x in enumerate(xs):
        for k in range(per_lane):
            dist = v * t + 28 * k + 9 * i
            out.append((100 + 100 * i + 10 * k + int(dist // 85), "Car", (x, 2 + dist % 85), kph))
    return out


def queue(kph=0.0, xs=(5.5, GENERAL_X), per_lane=3):
    """Cars standing (or crawling at kph) in the general lanes: a jam or a red light."""
    return [(300 + 100 * i + k, "Car", (x, 30 + 9 * k + 4 * i), kph) for i, x in enumerate(xs) for k in range(per_lane)]


def codes(events):
    return sorted(e.code for e in events)


class DrivingTests(unittest.TestCase):
    LANE_40M = (0, 10, 3.5, 50)

    def test_car_driving_40_m_along_the_bus_lane_is_reported(self):
        ev = Sim(make_camera(self.LANE_40M)).run(6, lambda t: drive(t, y0=10) + flowing(t))
        self.assertEqual(codes(ev), ["V9"])
        d = ev[0].details
        self.assertGreaterEqual(d["metres_in_lane"], 25)
        self.assertLessEqual(d["metres_in_lane"], 40)
        self.assertGreaterEqual(d["seconds_in_lane"], 4)
        self.assertAlmostEqual(d["avg_kph"], 30, delta=3)
        self.assertEqual(d["measured_by"], "distance")
        self.assertEqual(d["area"], LANE)
        self.assertNotIn("hours", d)   # the lane has no hours: it is in force all day

    def test_the_stretch_starts_when_the_car_entered_the_lane(self):
        ev = Sim(make_camera()).run(6, lambda t: drive(t) + flowing(t))
        self.assertAlmostEqual(ev[0].started_s, 1 / FPS, delta=0.1)
        self.assertAlmostEqual(ev[0].obs.time_s - ev[0].started_s, ev[0].details["seconds_in_lane"], delta=0.1)

    def test_one_event_per_vehicle(self):
        self.assertEqual(len(Sim(make_camera()).run(9, lambda t: drive(t) + flowing(t))), 1)

    def test_a_photo_from_inside_the_lane_comes_with_the_case(self):
        ev = Sim(make_camera()).run(6, lambda t: drive(t) + flowing(t))
        label, t, jpeg = ev[0].extra_frames[0]
        self.assertEqual(label, "entered")
        self.assertAlmostEqual(t, ev[0].started_s + 2.0, delta=0.1)   # taken photo_after_sec into the stretch
        self.assertLess(t, ev[0].obs.time_s)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_a_vehicle_in_a_general_lane_is_never_reported(self):
        self.assertEqual(Sim(make_camera()).run(9, lambda t: drive(t, x=GENERAL_X) + flowing(t)), [])

    def test_a_bus_may_use_the_lane(self):
        self.assertEqual(Sim(make_camera()).run(9, lambda t: drive(t, cls="Bus") + flowing(t)), [])

    def test_every_class_that_is_not_a_bus_is_an_offender(self):
        for cls in ("Car", "CNG", "Motorbike", "Rickshaw", "Truck", "Leguna"):
            with self.subTest(cls=cls):
                ev = Sim(make_camera()).run(6, lambda t: drive(t, cls=cls))
                self.assertEqual(codes(ev), ["V9"])

    def test_allowed_classes_are_configurable_per_camera(self):
        cam = make_camera(rules={"bus_lane": {"allowed_classes": ["Bus", "Leguna"]}})
        self.assertEqual(Sim(cam).run(6, lambda t: drive(t, cls="Leguna")), [])
        self.assertEqual(codes(Sim(cam).run(6, lambda t: drive(t, cls="Car"))), ["V9"])

    def test_a_vehicle_whose_class_is_uncertain_is_not_reported(self):
        # A bus read as a truck must not be fined for using its own lane.
        self.assertEqual(Sim(make_camera()).run(6, lambda t: drive(t, cls="Truck"), purity=0.6), [])

    def test_a_vehicle_standing_in_the_lane_is_not_driving_along_it(self):
        # Waiting at a kerb is V3 and standing in a running lane V7; V9 is about driving along the lane.
        self.assertEqual(Sim(make_camera()).run(60, lambda t: [(1, "Car", (LANE_X, 40.0), 0.0)] + flowing(t)), [])

    def test_only_the_part_of_the_drive_inside_the_lane_counts(self):
        # The lane starts at 40 m; the car has already driven 38 m (4.6 s) on the road by the time it enters it.
        ev = Sim(make_camera((0, 40, 3.5, 90))).run(11, lambda t: drive(t, y0=2) + flowing(t))
        self.assertEqual(codes(ev), ["V9"])
        self.assertAlmostEqual(ev[0].started_s, 38 / (30 / 3.6), delta=0.2)
        self.assertAlmostEqual(ev[0].details["seconds_in_lane"], 4.0, delta=0.2)
        self.assertLess(ev[0].details["metres_in_lane"], 40)


class CrossingTests(unittest.TestCase):
    def test_crossing_the_lane_to_turn_is_not_reported(self):
        def cross(t):   # from the middle of the road to the kerb: 3.5 m across the lane at about 14 degrees
            return [(1, "Car", (6.0 - 2.0 * t, 40.0 + 8.0 * t), 30.0)]
        self.assertEqual(Sim(make_camera()).run(3.4, lambda t: cross(t) + flowing(t)), [])

    def test_a_slow_merge_to_the_kerb_is_not_driving_along_the_lane(self):
        # In the lane for over 20 s - long enough - but only about 5 m along the road: not enough distance.
        def merge(t):
            return [(1, "Car", (max(0.5, 6.3 - 0.8 * t), 40.0 + 2.0 * t if t < 6 else 52.0), 7.0 if t < 6 else 0.0)]
        self.assertEqual(Sim(make_camera()).run(30, lambda t: merge(t) + flowing(t)), [])

    def test_pulling_in_to_the_kerb_and_waiting_is_not_reported(self):
        def pull_in(t):
            return [(1, "CNG", (max(0.8, 6.0 - 1.5 * t), 40.0 + 4.0 * min(t, 3.5)), 5.0 if t < 3.5 else 0.0)]
        self.assertEqual(Sim(make_camera()).run(40, lambda t: pull_in(t) + flowing(t)), [])

    def test_a_fast_car_clipping_the_end_of_the_lane_is_not_reported(self):
        # 30 m of lane at 100 km/h: far enough, but inside for barely a second.
        cam = make_camera((0, 40, 3.5, 70))
        self.assertEqual(Sim(cam).run(3, lambda t: drive(t, y0=10, kph=100)), [])

    def test_turning_at_a_junction_starts_the_stretch_over(self):
        # Two junction mouths cut the drive into pieces of 17 m, 23 m and 30 m (3.6 s): none is a stretch.
        for kind in ("intersection", "uturn"):
            with self.subTest(kind=kind):
                cam = make_camera(extra=[(kind, (0, 22, 10.5, 27)), (kind, (0, 50, 10.5, 55))])
                self.assertEqual(Sim(cam).run(9.5, lambda t: drive(t) + flowing(t)), [])
        self.assertEqual(codes(Sim(make_camera()).run(9.5, lambda t: drive(t) + flowing(t))), ["V9"])   # without them


class JamTests(unittest.TestCase):
    def test_a_jam_across_the_road_is_not_a_bus_lane_offence(self):
        # Every other lane is stopped; a car goes down the empty bus lane.
        self.assertEqual(Sim(make_camera()).run(9, lambda t: drive(t, kph=20) + queue()), [])

    def test_a_jam_where_every_lane_crawls_is_not_reported(self):
        # The car in the bus lane crawls too (5 km/h, over 25 m in 18 s) and nothing anywhere reaches flow speed.
        self.assertEqual(Sim(make_camera()).run(40, lambda t: drive(t, kph=5) + queue(kph=3)), [])

    def test_the_same_drive_is_reported_when_the_other_lanes_flow(self):
        self.assertEqual(codes(Sim(make_camera()).run(9, lambda t: drive(t, kph=20) + flowing(t))), ["V9"])

    def test_motorbikes_filtering_through_a_jam_do_not_make_it_a_flow(self):
        bikes = lambda t: [(700 + k, "Motorbike", (5.0, 10 + 15 * k + 5 * t), 18.0) for k in range(3)]
        self.assertEqual(Sim(make_camera()).run(9, lambda t: drive(t, kph=20) + queue() + bikes(t)), [])

    def test_cars_cruising_in_the_bus_lane_do_not_make_a_jam_look_like_flowing_traffic(self):
        both = lambda t: drive(t, x=1.2, kph=25, tid=1) + drive(t, x=2.6, kph=25, tid=2, y0=20)
        self.assertEqual(Sim(make_camera()).run(9, lambda t: both(t) + queue()), [])

    def test_one_stopped_car_next_door_is_not_a_jam(self):
        one = [(300, "CNG", (GENERAL_X, 40.0), 0.0)]
        self.assertEqual(codes(Sim(make_camera()).run(9, lambda t: drive(t, kph=20) + one)), ["V9"])

    def test_cars_parked_in_a_kerb_strip_are_not_a_queue(self):
        cam = make_camera(extra=[("no_parking", (7.5, 0, 10.5, 90))])
        parked = [(300 + k, "Car", (9.0, 20 + 12 * k), 0.0) for k in range(4)]
        self.assertEqual(codes(Sim(cam).run(9, lambda t: drive(t, kph=20) + parked)), ["V9"])
        # The same cars standing in a running lane, with no kerb strip drawn, ARE a queue.
        self.assertEqual(Sim(make_camera()).run(9, lambda t: drive(t, kph=20) + parked), [])

    def test_an_empty_road_is_not_a_jam(self):
        self.assertEqual(codes(Sim(make_camera()).run(9, lambda t: drive(t, kph=20))), ["V9"])

    def test_a_queue_that_clears_lets_a_long_drive_through_later(self):
        # Jammed for the first 3 s of a 10 s+ drive: reported once the jam is under 30% of the time in the lane.
        ev = Sim(make_camera()).run(14, lambda t: drive(t, y0=2, kph=20) + (queue() if t < 3 else flowing(t)))
        self.assertEqual(codes(ev), ["V9"])
        self.assertGreater(ev[0].obs.time_s, 9)
        self.assertGreater(ev[0].details["other_lanes_queued_s"], 2)


class HoursTests(unittest.TestCase):
    PEAK = [["07:00", "10:00"]]

    def run_at(self, hour, minute=0, hours=PEAK, second=0, overrides=None):
        start = None if hour is None else datetime(2026, 10, 16, hour, minute, second)
        return Sim(make_camera(hours=hours), start=start, overrides=overrides).run(
            6, lambda t: drive(t) + flowing(t))

    def test_outside_the_lanes_hours_nothing_is_reported(self):
        self.assertEqual(self.run_at(13), [])

    def test_inside_the_lanes_hours_the_lane_is_enforced(self):
        ev = self.run_at(8)
        self.assertEqual(codes(ev), ["V9"])
        self.assertEqual(ev[0].details["hours"], "07:00-10:00")
        self.assertEqual(ev[0].details["enforced_because"], "inside the lane's hours")

    def test_a_lane_without_hours_is_enforced_at_any_time(self):
        self.assertEqual(codes(self.run_at(13, hours=None)), ["V9"])

    def test_hours_may_cross_midnight(self):
        night = [["22:00", "06:00"]]
        self.assertEqual(codes(self.run_at(23, 30, hours=night)), ["V9"])
        self.assertEqual(codes(self.run_at(3, 0, hours=night)), ["V9"])
        self.assertEqual(self.run_at(12, 0, hours=night), [])

    def test_several_ranges(self):
        peaks = [["07:00", "10:00"], ["16:00", "20:00"]]
        self.assertEqual(codes(self.run_at(17, hours=peaks)), ["V9"])
        self.assertEqual(self.run_at(12, hours=peaks), [])

    def test_unknown_time_of_day_is_enforced_and_says_so(self):
        ev = self.run_at(None)
        self.assertEqual(codes(ev), ["V9"])
        self.assertEqual(ev[0].details["enforced_because"], "time of day unknown: hours not checked")

    def test_unknown_time_of_day_can_be_made_not_enforced(self):
        self.assertEqual(self.run_at(None, overrides={"bus_lane": {"enforce_if_time_unknown": False}}), [])

    def test_only_the_time_the_lane_was_in_force_counts(self):
        # The hours end 2 s into the drive: 2 s in force is not a stretch, and the drive goes on outside the hours.
        self.assertEqual(self.run_at(9, 59, second=58), [])
        # With 10 s of hours left the same drive is reported.
        self.assertEqual(codes(self.run_at(9, 59, second=50)), ["V9"])


class MeasurementTests(unittest.TestCase):
    def test_without_metres_the_decision_falls_back_to_time(self):
        sim = lambda: Sim(make_camera(), metres="none")
        self.assertEqual(sim().run(7, lambda t: drive(t)), [])   # 7 s: under the 8 s needed without distance
        ev = sim().run(9, lambda t: drive(t))
        self.assertEqual(codes(ev), ["V9"])
        self.assertEqual(ev[0].details["measured_by"], "time")
        self.assertNotIn("metres_in_lane", ev[0].details)
        self.assertGreaterEqual(ev[0].details["seconds_in_lane"], 8)

    def test_far_end_positions_count_when_precise_ones_are_missing(self):
        ev = Sim(make_camera(), metres="coarse").run(6, lambda t: drive(t))
        self.assertEqual(codes(ev), ["V9"])
        self.assertEqual(ev[0].details["measured_by"], "distance")

    def test_far_end_box_jitter_does_not_keep_restarting_the_stretch(self):
        # Beyond the reliable line a box jitters by metres from frame to frame (1.5 m here): a speed check on
        # consecutive positions alone would call every other frame an ID switch and the rule would never decide.
        ev = Sim(make_camera(), metres="coarse", noise=1.5).run(6, lambda t: drive(t) + flowing(t))
        self.assertEqual(codes(ev), ["V9"])
        self.assertAlmostEqual(ev[0].details["seconds_in_lane"], 4.0, delta=0.3)   # one unbroken stretch
        self.assertAlmostEqual(ev[0].details["metres_in_lane"], 33, delta=5)

    def test_a_track_id_jumping_to_another_vehicle_is_not_distance(self):
        # Something stands at 10 m for 4.5 s, then the ID is on another vehicle standing at 60 m.
        def switch(t):
            return [(1, "Car", (LANE_X, 10.0 if t < 4.5 else 60.0), 0.0)]
        self.assertEqual(Sim(make_camera()).run(11, switch), [])

    def test_sideways_movement_adds_no_distance(self):
        # Weaving across the whole lane width while crawling forward 1 m/s: 20 m along the road in 20 s.
        def weave(t):
            return [(1, "Car", (1.75 + 1.4 * np.sin(2 * t), 30.0 + 1.0 * t), 4.0)]
        self.assertEqual(Sim(make_camera()).run(20, weave), [])

    def test_same_results_at_10_fps(self):
        full = Sim(make_camera()).run(6, lambda t: drive(t) + flowing(t))
        dropped = Sim(make_camera(), keep_every=3).run(6, lambda t: drive(t) + flowing(t))
        self.assertEqual(codes(full), ["V9"])
        self.assertEqual(codes(dropped), ["V9"])
        self.assertAlmostEqual(full[0].obs.time_s, dropped[0].obs.time_s, delta=0.2)

    def test_a_track_that_is_forgotten_leaves_no_state_behind(self):
        sim = Sim(make_camera())
        sim.run(2, lambda t: drive(t))
        rule = sim.engine.rules[0]
        self.assertEqual(list(rule.runs), [1])
        sim.engine.reset()
        self.assertEqual(rule.runs, {})


class AreaAndEngineTests(unittest.TestCase):
    def config(self, **lane_extra):
        road = RoadZone("Right side", POINTS, 10.5, 90.0)
        lane = road.to_pixels([[0, 0], [3.5, 0], [3.5, 90], [0, 90]]).tolist()
        return road, {"frame_size": [2560, 1440], "peak_hours": None, "rules": {}, "lane_lines": [], "zones": [
            {"type": "road", "name": "Right side", "points": POINTS, "width_m": 10.5, "length_m": 90},
            {"type": "bus_lane", "name": LANE, "points": lane, **lane_extra},
            {"type": "bus_lane", "name": "Right bus lane", "points": road.to_pixels(
                [[7, 0], [10.5, 0], [10.5, 90], [7, 90]]).tolist()}]}

    def test_the_rule_is_inactive_without_a_bus_lane_area(self):
        engine = ViolationEngine(make_camera(lane=None))
        self.assertEqual(engine.inactive["V9"], "no bus lane drawn")
        self.assertNotIn("V9", [r.code for r in engine.rules])
        engine = ViolationEngine(make_camera())
        self.assertNotIn("V9", engine.inactive)
        self.assertIn("V9", [r.code for r in engine.rules])

    def test_the_rule_can_be_switched_off_in_the_camera_settings(self):
        engine = ViolationEngine(make_camera(rules={"bus_lane": {"enabled": False}}))
        self.assertEqual(engine.inactive["V9"], "disabled in settings")

    def test_a_zone_config_with_a_bus_lane_round_trips(self):
        road, cfg = self.config(hours=[["07:00", "10:00"], ["16:00", "20:00"]])
        cam = camera_config_from_dict(cfg, 2560, 1440)
        self.assertTrue(cam.has_area("bus_lane"))
        left, right = cam.areas
        self.assertEqual(left.hours, [["07:00", "10:00"], ["16:00", "20:00"]])
        self.assertEqual(left.hours_text, "07:00-10:00, 16:00-20:00")
        self.assertIsNone(right.hours)
        self.assertEqual(cam.areas_at(tuple(road.to_pixels([[1.75, 40]])[0])), frozenset({"bus_lane"}))
        self.assertEqual(cam.areas_at(tuple(road.to_pixels([[5.0, 40]])[0])), frozenset())
        self.assertEqual(cam.area_names_at(tuple(road.to_pixels([[1.75, 40]])[0])), [LANE])
        saved = cam.to_dict()
        self.assertEqual(saved["zones"][1]["hours"], [["07:00", "10:00"], ["16:00", "20:00"]])
        self.assertNotIn("hours", saved["zones"][2])   # no hours = all day, nothing written
        self.assertEqual(camera_config_from_dict(json.loads(json.dumps(saved)), 2560, 1440).to_dict(), saved)

    def test_a_bus_lane_is_scaled_with_the_frame_like_every_other_area(self):
        _, cfg = self.config()
        full = camera_config_from_dict(cfg, 2560, 1440).areas[0]
        half = camera_config_from_dict(cfg, 1280, 720).areas[0]
        self.assertTrue(np.allclose(half.poly, full.poly / 2, atol=0.1))

    def test_bad_hours_are_refused_with_a_readable_message(self):
        for bad in ([["7:00", "10:00"]], [["07:00", "07:00"]], [["07:00"]], "07:00-10:00", [["25:00", "26:00"]],
                    [[7, 10]], [["07:00", "10:00", "12:00"]], [["07:00", None]]):
            with self.subTest(hours=bad):
                _, cfg = self.config(hours=bad)
                with self.assertRaises(ValueError) as caught:
                    camera_config_from_dict(cfg, 2560, 1440)
                self.assertIn(LANE, str(caught.exception))
                self.assertIn("HH:MM", str(caught.exception))

    def test_hours_belong_to_bus_lanes_only(self):
        kerb = AreaZone("no_parking", "Kerb", [[0, 0], [10, 0], [10, 10]], hours=[["07:00", "10:00"]])
        self.assertIsNone(kerb.hours)
        self.assertNotIn("hours", kerb.to_dict())

    def test_a_bus_lane_is_a_known_area_type_with_its_own_colour(self):
        self.assertIn("bus_lane", AREA_TYPES)
        self.assertEqual(set(AREA_COLORS), set(AREA_TYPES))                 # every area type has a colour
        self.assertEqual(len(set(AREA_COLORS.values())), len(AREA_COLORS))   # and none shares it with another

    def test_a_bus_lane_is_drawn_in_its_colour_with_its_hours(self):
        img = np.zeros((1440, 2560, 3), np.uint8)
        cam = make_camera(hours=[["07:00", "10:00"]])
        cam.draw(img)
        x, y = (int(v) for v in cam.roads[0].to_pixels([[1.75, 55]])[0])   # between the 10 m cross lines
        self.assertTrue(np.allclose(img[y, x], np.array(AREA_COLORS["bus_lane"]) * 0.18, atol=2))   # the 18% tint

    def test_the_rules_settings_can_be_sent_to_the_editor_as_json(self):
        # /api/meta returns every rule's DEFAULTS to the zone editor.
        json.dumps(RULES["bus_lane"].DEFAULTS)
        self.assertEqual((RULES["bus_lane"].code, RULES["bus_lane"].title), ("V9", "Driving in a bus lane"))

    def test_the_vlm_is_told_what_v9_claims_and_what_would_overturn_it(self):
        claim, innocent = vlm.RULE_CLAIM["V9"]
        self.assertIn("bus", claim)
        for word in ("turn", "kerb", "authorised", "time of day", "markings"):
            self.assertIn(word, innocent)
        prompt = vlm.build_prompt({"id": 5, "code": "V9", "title": "Driving in a bus lane", "cls": "Car",
                                   "details": {"metres_in_lane": 33.3, "seconds_in_lane": 4.0}}, [])
        self.assertIn(claim, prompt)
        self.assertIn(innocent, prompt)
        self.assertIn("metres in lane: 33.3", prompt)


if __name__ == "__main__":
    unittest.main()
