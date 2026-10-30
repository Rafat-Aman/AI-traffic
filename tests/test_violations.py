"""Synthetic-trajectory tests for the violation rules (no model or video needed).

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from violations import AreaZone, CameraConfig, LaneLine, Observation, RoadZone, ViolationEngine  # noqa: E402

FPS = 30
# The "Right side" zone from speed_zones.json: 10.5 m x 90 m.
POINTS = [[1340.0, 200.0], [1505.0, 196.0], [2508.0, 932.0], [1458.0, 908.0]]
FRAME = np.zeros((1440, 2560, 3), np.uint8)


def make_camera(direction="both", banned=(), areas=(), lines=(), peak_hours=None, category="city"):
    road = RoadZone("Right side", POINTS, 10.5, 90.0, road_category=category, allowed_direction=direction,
                    banned_classes=banned)
    area_objs = []
    for kind, (x0, y0, x1, y1), *opt in areas:  # rectangles given in road metres, optional {"approved": True}
        area_objs.append(AreaZone(kind, f"{kind} {len(area_objs) + 1}",
                                  road.to_pixels([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), **(opt[0] if opt else {})))
    cam = CameraConfig([road], area_objs, peak_hours=peak_hours)
    cam.lines = [LaneLine(name, road.to_pixels(pts_m), cam.roads) for name, pts_m in lines]
    return cam


class Sim:
    """Feeds vehicles moving on the road plane through the engine exactly as violation_tracker.py does."""

    def __init__(self, camera, only=None, overrides=None, helmet=None, start=None, keep_every=1, heading=0):
        self.cam = camera
        self.engine = ViolationEngine(camera, overrides, only, helmet)
        self.frame = 0
        self.first = {}
        self.start = start
        self.heading = heading  # +1 toward the camera, -1 away, 0 unknown (what the pipeline would measure)
        self.keep_every = keep_every  # >1 simulates a live camera dropping frames when the computer is behind

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
            xy_m = tuple(float(v) for v in r.to_metres(point)) if r else None
            section = None
            if r:
                t0, p0 = self.first.setdefault(tid, (t, xy_m))
                section = (float(np.hypot(xy_m[0] - p0[0], xy_m[1] - p0[1])), t - t0)
            w = 60
            bbox = (int(point[0] - w), int(point[1] - 2 * w), int(point[0] + w), int(point[1]))
            obs.append(Observation(tid, self.frame, t, cls, purity, 100, bbox, point, r, xy_m,
                                   kph, section, self.cam.areas_at(point), self.heading))
        wall = None
        if self.start:
            from datetime import timedelta
            wall = self.start + timedelta(seconds=t)
        return self.engine.process(obs, FRAME, self.frame, t, wall)

    def run(self, seconds, vehicles_fn, **kw):
        events = []
        for i in range(int(seconds * FPS)):
            events += self.step(vehicles_fn(i / FPS), **kw)
        return events


def flowing_traffic(t, kph=30, n=3, first_tid=1000, x=8.5):
    """n cars in the far lane, re-spawned at the far end with a new ID when they leave the zone."""
    v = kph / 3.6
    out = []
    for k in range(n):
        dist = v * t + k * 30
        lap = int(dist // 85)
        out.append((first_tid + 10 * k + lap, "Car", (x, 2 + dist % 85), kph))
    return out


def codes(events):
    return sorted(e.code for e in events)


class SpeedTests(unittest.TestCase):
    def drive(self, cls, kph, camera=None):
        sim = Sim(camera or make_camera(), only=["V2"])
        v = kph / 3.6
        return sim.run(85 / v, lambda t: [(1, cls, (3.0, 2 + v * t), kph)])

    def test_car_over_city_limit(self):
        ev = self.drive("Car", 75)
        self.assertEqual(codes(ev), ["V2"])
        self.assertEqual(ev[0].details["limit_kph"], 60)

    def test_city_limit_is_60_for_every_motor_vehicle(self):
        self.assertEqual(self.drive("Car", 58), [])
        self.assertEqual(self.drive("Motorbike", 62), [])  # limit 60, flagged above 63 (measurement margin)
        self.assertEqual(codes(self.drive("Motorbike", 70)), ["V2"])
        self.assertEqual(codes(self.drive("Bus", 70)), ["V2"])

    def test_rickshaw_has_no_speed_limit(self):
        self.assertEqual(self.drive("Rickshaw", 25), [])

    def test_expressway_and_highway_limit_is_80(self):
        for cat in ("expressway", "highway"):
            cam = make_camera(category=cat)
            self.assertEqual(self.drive("Car", 78, cam), [], cat)
            ev = self.drive("Car", 95, cam)
            self.assertEqual(codes(ev), ["V2"], cat)
            self.assertEqual(ev[0].details["limit_kph"], 80)

    def test_single_noisy_spike_is_ignored(self):
        sim = Sim(make_camera(), only=["V2"])
        v = 30 / 3.6
        ev = sim.run(8, lambda t: [(1, "Car", (3.0, 2 + v * t), 70 if 3 < t < 3.5 else 30)])
        self.assertEqual(ev, [])


def bus_alone(t):
    return [(1, "Bus", (2.0, 45.0), 0.0)] + flowing_traffic(t)


class BusBlockingTests(unittest.TestCase):
    def test_bus_stands_while_traffic_flows(self):
        sim = Sim(make_camera(), only=["V1"])
        ev = sim.run(25, bus_alone)
        self.assertEqual(codes(ev), ["V1"])
        self.assertGreaterEqual(ev[0].details["dwell_s"], 10)
        self.assertGreaterEqual(ev[0].details["vehicles_passed"], 3)
        self.assertEqual(ev[0].details["where"], "running lane, no bus stop")

    def test_stand_has_a_photo_from_when_it_stopped(self):
        ev = Sim(make_camera(), only=["V1"]).run(25, bus_alone)
        label, t, jpeg = ev[0].extra_frames[0]
        self.assertEqual(label, "stopped")
        self.assertAlmostEqual(t, ev[0].started_s + 2.0, delta=0.1)  # taken 2 s after it stopped
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_red_light_queue_is_not_a_violation(self):
        sim = Sim(make_camera(), only=["V1"])
        queue = [(1000 + k, "Car", (8.0, 50 + 6 * k), 0.0) for k in range(4)]
        # 2-wheelers filtering through the queue must not count as "traffic flowing"
        bikes = lambda t: [(2000 + k, "Motorbike", (5.0, (10 + 5 * t + 20 * k) % 85 + 2), 18.0) for k in range(3)]
        ev = sim.run(120, lambda t: [(1, "Bus", (2.0, 45.0), 0.0)] + queue + bikes(t))
        self.assertEqual(ev, [])

    def test_waiting_behind_a_standing_vehicle_is_a_queue(self):
        # One-way road toward the camera: a car stands 6 m in front of the bus in its lane, the far lane flows.
        cam = make_camera(direction="toward_camera")
        ev = Sim(cam, only=["V1"]).run(40, lambda t: bus_alone(t) + [(5, "Car", (2.0, 51.0), 0.0)])
        self.assertEqual(ev, [])
        # The same bus with a free lane in front of it is the blocker.
        self.assertEqual(codes(Sim(cam, only=["V1"]).run(40, bus_alone)), ["V1"])

    def test_signal_area_is_exempt(self):
        sim = Sim(make_camera(areas=[("signal", (0, 35, 10.5, 60))]), only=["V1"])
        self.assertEqual(sim.run(30, bus_alone), [])

    def test_bus_stop_after_an_intersection_is_exempt(self):
        # Traffic flows toward the camera (y grows): junction at 20-30 m, stop just after it at 38-55 m.
        cam = make_camera(direction="toward_camera",
                          areas=[("intersection", (0, 20, 10.5, 30)), ("bus_stop", (0, 38, 4, 55))])
        self.assertTrue(cam.areas[1].valid, cam.areas[1].reason)
        self.assertEqual(Sim(cam, only=["V1"]).run(60, bus_alone), [])

    def test_bus_stop_on_a_straight_road_is_not_a_bus_stop(self):
        cam = make_camera(direction="toward_camera", areas=[("bus_stop", (0, 38, 4, 55))])
        self.assertFalse(cam.areas[0].valid)
        ev = Sim(cam, only=["V1"]).run(25, bus_alone)
        self.assertEqual(codes(ev), ["V1"])
        self.assertIn("unapproved", ev[0].details["where"])

    def test_bus_stop_before_the_intersection_is_not_valid(self):
        # Junction at 60-70 m is AFTER the stop (38-55 m) in the direction of traffic: a near-side stop.
        cam = make_camera(direction="toward_camera",
                          areas=[("intersection", (0, 60, 10.5, 70)), ("bus_stop", (0, 38, 4, 55))])
        self.assertFalse(cam.areas[1].valid)
        self.assertEqual(codes(Sim(cam, only=["V1"]).run(25, bus_alone)), ["V1"])

    def test_approved_mid_block_bus_bay(self):
        cam = make_camera(areas=[("bus_stop", (0, 38, 4, 55), {"approved": True})])
        self.assertTrue(cam.areas[0].valid)
        self.assertEqual(Sim(cam, only=["V1"]).run(60, bus_alone), [])

    def test_crawling_bus_in_slow_traffic(self):
        sim = Sim(make_camera(), only=["V1"])
        v = 4 / 3.6  # Dhaka peak-hour crawl
        ev = sim.run(60, lambda t: [(1, "Bus", (2.0, 5 + v * t), 4.0)] + flowing_traffic(t))
        self.assertEqual(ev, [])

    def test_box_jitter_and_id_switch(self):
        rng = np.random.default_rng(0)
        sim = Sim(make_camera(), only=["V1"])

        def scene(t):
            tid = 1 if t < 8 else 2  # occluded at 8 s, tracker gives it a new ID
            if 7.8 < t < 8.3:
                return flowing_traffic(t)  # not detected while occluded
            jitter = rng.normal(0, 0.4, 2)
            return [(tid, "Bus", (2.0 + jitter[0], 45.0 + jitter[1]), 1.5)] + flowing_traffic(t)
        ev = sim.run(20, scene)
        self.assertEqual(codes(ev), ["V1"])
        self.assertEqual(ev[0].obs.tid, 2)


KERB = ("no_parking", (0, 0, 3, 90))  # 3 m kerb strip along the left edge of the zone


def cng_at_kerb(t):
    return [(1, "CNG", (1.5, 60.0), 0.0)] + flowing_traffic(t)


def heavy_traffic(t):
    """22 vehicles in the zone: the far lanes packed but moving."""
    return flowing_traffic(t) + flowing_traffic(t, kph=15, n=19, first_tid=5000, x=6.5)


class ParkingTests(unittest.TestCase):
    def test_cng_stand_at_the_kerb(self):
        ev = Sim(make_camera(areas=[KERB]), only=["V3"]).run(70, cng_at_kerb)
        self.assertEqual(codes(ev), ["V3"])
        self.assertEqual(ev[0].details["enforced_because"], "no-parking all day")

    def test_needs_a_kerb_strip(self):
        engine = ViolationEngine(make_camera())
        self.assertIn("V3", engine.inactive)

    def test_outside_peak_hours_light_traffic(self):
        sim = Sim(make_camera(areas=[KERB], peak_hours=[["07:00", "10:00"]]), only=["V3"],
                  start=datetime(2026, 10, 16, 13, 0, 0))
        self.assertEqual(sim.run(70, cng_at_kerb), [])

    def test_outside_peak_hours_heavy_traffic(self):
        sim = Sim(make_camera(areas=[KERB], peak_hours=[["07:00", "10:00"]]), only=["V3"],
                  start=datetime(2026, 10, 16, 13, 0, 0))
        ev = sim.run(70, lambda t: [(1, "CNG", (1.5, 60.0), 0.0)] + heavy_traffic(t))
        self.assertEqual(codes(ev), ["V3"])
        self.assertIn("heavy traffic", ev[0].details["enforced_because"])

    def test_inside_peak_hours(self):
        sim = Sim(make_camera(areas=[KERB], peak_hours=[["07:00", "10:00"]]), only=["V3"],
                  start=datetime(2026, 10, 16, 8, 0, 0))
        ev = sim.run(70, cng_at_kerb)
        self.assertEqual(codes(ev), ["V3"])
        self.assertEqual(ev[0].details["enforced_because"], "peak hours")

    def test_legal_parking_area(self):
        sim = Sim(make_camera(areas=[KERB, ("parking", (0, 55, 3, 65))]), only=["V3"])
        self.assertEqual(sim.run(70, cng_at_kerb), [])

    def test_bus_is_left_to_v1(self):
        sim = Sim(make_camera(areas=[KERB]), only=["V1", "V3", "V7"])
        ev = sim.run(70, lambda t: [(1, "Bus", (1.5, 60.0), 0.0)] + flowing_traffic(t))
        self.assertEqual(codes(ev), ["V1"])
        self.assertEqual(ev[0].details["where"], "kerb of a straight road, no bus stop")


    def test_row_of_parked_bikes_is_not_a_queue(self):
        # One-way road: three motorbikes parked nose to tail at the kerb, the far lanes flow past.
        cam = make_camera(direction="toward_camera", areas=[KERB])
        bikes = [(1 + k, "Motorbike", (1.0, 60.0 + 2.5 * k), 0.0) for k in range(3)]
        ev = Sim(cam, only=["V3"]).run(30, lambda t: bikes + flowing_traffic(t))
        self.assertEqual(codes(ev), ["V3", "V3", "V3"])

    def test_parked_bike_hidden_by_pedestrians_keeps_its_timer(self):
        # Lost for 6 s behind people, comes back with a new ID - and the detector now calls it a Bicycle.
        def scene(t):
            if 8 < t < 14:
                return flowing_traffic(t)
            bike = (1, "Motorbike", (1.0, 60.0), 0.0) if t <= 8 else (2, "Bicycle", (1.2, 60.3), 0.0)
            return [bike] + flowing_traffic(t)
        ev = Sim(make_camera(areas=[KERB]), only=["V3"]).run(30, scene)
        self.assertEqual(codes(ev), ["V3"])
        self.assertEqual(ev[0].obs.tid, 2)


class RoadBlockingTests(unittest.TestCase):
    def test_no_timer_without_a_measured_position(self):
        # A CNG stands 8 s, then drives on into the far end where positions are not measured (xy = None):
        # its standing timer must not keep running out there.
        sim = Sim(make_camera(areas=[KERB]), only=["V7"])
        real = sim.engine.process

        def process(obs, *a, **k):
            for o in obs:
                if o.tid == 1 and o.time_s >= 8.3:
                    o.xy = None  # beyond the reliable line
            return real(obs, *a, **k)
        sim.engine.process = process

        def scene(t):
            moving = t >= 8
            return [(1, "CNG", (5.0, 60.0 - 3 * (t - 8) if moving else 60.0), 11.0 if moving else 0.0)] + flowing_traffic(t)
        self.assertEqual(sim.run(30, scene), [])

    def test_cng_picking_up_in_a_running_lane(self):
        sim = Sim(make_camera(areas=[KERB]), only=["V3", "V7"])
        ev = sim.run(30, lambda t: [(1, "CNG", (5.0, 60.0), 0.0)] + flowing_traffic(t))
        self.assertEqual(codes(ev), ["V7"])
        self.assertEqual(ev[0].details["where"], "running lane")
        self.assertGreaterEqual(ev[0].details["vehicles_passed"], 4)

    def test_jam_is_not_blocking(self):
        # Every lane stands (a jam / red light): nobody passes it.
        jam = [(1000 + k, "Car", (x, 40 + 7 * k), 0.0) for k in range(5) for x in (2.0, 8.0)]
        jam = [(tid + (100 if x == 8.0 else 0), c, (x, y), v) for tid, c, (x, y), v in jam]
        sim = Sim(make_camera(areas=[KERB]), only=["V7"])
        self.assertEqual(sim.run(90, lambda t: [(1, "CNG", (5.0, 60.0), 0.0)] + jam), [])

    def test_kerb_strip_is_v3_not_v7(self):
        sim = Sim(make_camera(areas=[KERB]), only=["V7"])
        self.assertEqual(sim.run(40, cng_at_kerb), [])

    def test_junction_is_exempt(self):
        sim = Sim(make_camera(areas=[KERB, ("intersection", (3, 50, 10.5, 70))]), only=["V7"])
        self.assertEqual(sim.run(40, lambda t: [(1, "CNG", (5.0, 60.0), 0.0)] + flowing_traffic(t)), [])


class WrongSideTests(unittest.TestCase):
    def test_pushing_a_bike_at_walking_pace(self):
        sim = Sim(make_camera(direction="toward_camera"), only=["V4"])
        v = 3.5 / 3.6
        self.assertEqual(sim.run(15, lambda t: [(1, "Motorbike", (1.0, 70 - v * t), 3.5)]), [])

    def test_noisy_edge_of_the_measured_part_is_ignored(self):
        cam = make_camera(direction="toward_camera")
        cam.roads[0].reliable_from_m = 50.0
        v = 10 / 3.6  # from 60 m to 45 m: only 6 m of it lies beyond the reliable line + 4 m
        sim = Sim(cam, only=["V4"])
        self.assertEqual(sim.run(5.4, lambda t: [(1, "Rickshaw", (1.0, 60 - v * t), 10.0)]), [])

    def test_rickshaw_against_one_way(self):
        sim = Sim(make_camera(direction="toward_camera"), only=["V4"])
        v = 10 / 3.6
        ev = sim.run(6, lambda t: [(1, "Rickshaw", (1.0, 70 - v * t), 10.0)])
        self.assertEqual(codes(ev), ["V4"])

    def test_correct_direction(self):
        sim = Sim(make_camera(direction="toward_camera"), only=["V4"])
        v = 30 / 3.6
        self.assertEqual(sim.run(9, lambda t: [(1, "Car", (3.0, 5 + v * t), 30.0)]), [])

    def test_standing_jitter(self):
        rng = np.random.default_rng(1)
        sim = Sim(make_camera(direction="toward_camera"), only=["V4"])
        self.assertEqual(sim.run(60, lambda t: [(1, "Car", tuple(np.array([3.0, 40.0]) + rng.normal(0, 0.5, 2)),
                                                 0.0)]), [])

    def test_uturn_opening(self):
        sim = Sim(make_camera(direction="toward_camera", areas=[("uturn", (0, 20, 10.5, 60))]), only=["V4"])
        v = 10 / 3.6
        self.assertEqual(sim.run(4, lambda t: [(1, "CNG", (1.0, 55 - v * t), 10.0)]), [])


class LaneLineTests(unittest.TestCase):
    LINE = [("lane 1|2", [[5.25, 10.0], [5.25, 80.0]])]

    def test_crossing_solid_line(self):
        sim = Sim(make_camera(lines=self.LINE), only=["V5"])
        v = 30 / 3.6
        ev = sim.run(6, lambda t: [(1, "Car", (3.0 + min(4.5, max(0, t - 2) * 2), 15 + v * t), 30.0)])
        self.assertEqual(codes(ev), ["V5"])

    def test_straddling_the_line(self):
        rng = np.random.default_rng(2)
        sim = Sim(make_camera(lines=self.LINE), only=["V5"])
        v = 20 / 3.6
        self.assertEqual(sim.run(10, lambda t: [(1, "Car", (5.25 + rng.normal(0, 0.25), 12 + v * t), 20.0)]), [])

    def test_change_beyond_the_solid_part(self):
        sim = Sim(make_camera(lines=[("short", [[5.25, 50.0], [5.25, 80.0]])]), only=["V5"])
        v = 30 / 3.6
        ev = sim.run(4, lambda t: [(1, "Car", (3.0 + min(4.5, max(0, t - 1) * 3), 5 + v * t), 30.0)])
        self.assertEqual(ev, [])


class RestrictedVehicleTests(unittest.TestCase):
    def test_cng_on_flyover(self):
        sim = Sim(make_camera(banned=["CNG", "Rickshaw"]), only=["V8"])
        ev = sim.run(3, lambda t: [(1, "CNG", (3.0, 10 + 8 * t), 30.0), (2, "Car", (7.0, 10 + 8 * t), 30.0)])
        self.assertEqual([(e.code, e.obs.tid) for e in ev], [("V8", 1)])

    def test_uncertain_class(self):
        sim = Sim(make_camera(banned=["CNG"]), only=["V8"])
        self.assertEqual(sim.run(3, lambda t: [(1, "CNG", (3.0, 10 + 8 * t), 30.0)], purity=0.6), [])


class HelmetTests(unittest.TestCase):
    """The helmet classifier gives P(someone on the motorbike is bare-headed) per sighting."""

    def run_with(self, probs, seconds=3):
        """The motorbike is seen for `seconds`, then leaves the view (the rule decides when it is gone)."""
        seq = iter(probs)
        sim = Sim(make_camera(), only=["V6"], helmet=lambda crop: next(seq, probs[-1]), heading=1)
        v = 20 / 3.6
        return sim.run(seconds + 2, lambda t: [(1, "Motorbike", (3.0, 20 + v * t), 20.0)] if t < seconds else [])

    def test_decided_when_it_leaves_with_the_best_frame(self):
        ev = self.run_with([0.9, 0.97, 0.85, 0.9])
        self.assertEqual(codes(ev), ["V6"])
        self.assertIsNotNone(ev[0].frame)  # evidence = the frame of the strongest sighting
        self.assertEqual(ev[0].details["max_probability"], 0.97)

    def test_unsure_vehicle_class(self):
        sim = Sim(make_camera(), only=["V6"], helmet=lambda c: 0.99, heading=1)
        v = 20 / 3.6
        ev = sim.run(5, lambda t: [(1, "Motorbike", (3.0, 20 + v * t), 20.0)] if t < 3 else [], purity=0.7)
        self.assertEqual(ev, [])

    def test_bare_head(self):
        self.assertEqual(codes(self.run_with([0.9] * 4)), ["V6"])

    def test_helmet(self):
        self.assertEqual(self.run_with([0.05]), [])

    def test_uncertain_sightings_are_ignored(self):
        self.assertEqual(self.run_with([0.6]), [])

    def test_two_bad_frames_do_not_make_a_case(self):
        self.assertEqual(self.run_with([0.9, 0.9] + [0.1] * 20), [])

    def test_mostly_bare(self):
        self.assertEqual(codes(self.run_with([0.9, 0.9, 0.1, 0.9, 0.95])), ["V6"])

    def test_too_small_to_judge(self):
        sim = Sim(make_camera(), only=["V6"], overrides={"helmet": {"min_box_px": 200}}, helmet=lambda c: 0.99, heading=1)
        self.assertEqual(sim.run(5, lambda t: [(1, "Motorbike", (3.0, 20 + 5 * t), 20.0)] if t < 3 else []), [])

    def test_parked_motorbike_is_not_riding(self):
        sim = Sim(make_camera(), only=["V6"], helmet=lambda c: 0.99, heading=1)
        self.assertEqual(sim.run(5, lambda t: [(1, "Motorbike", (3.0, 70.0), 0.0)] if t < 3 else []), [])

    def test_bike_right_behind_makes_sightings_ambiguous(self):
        # A second motorbike just behind (above in the image) - its riders' heads would be in the crop.
        sim = Sim(make_camera(), only=["V6"], helmet=lambda c: 0.99, heading=1)
        v = 20 / 3.6
        ev = sim.run(5, lambda t: [(1, "Motorbike", (3.0, 40 + v * t), 20.0), (2, "Motorbike", (3.0, 37 + v * t), 20.0)]
                     if t < 3 else [])
        self.assertNotIn(1, [e.obs.tid for e in ev])


class DroppedFramesTests(unittest.TestCase):
    """A live camera processes only the newest frame when the computer falls behind (here 1 of every 3 frames,
    i.e. 10 FPS from a 30 FPS camera). Rules work on timestamps, so the results must not change."""

    def scenarios(self):
        v_fast, v_rick = 80 / 3.6, 10 / 3.6
        return [
            (["V2"], make_camera(), 5, lambda t: [(1, "Car", (3.0, 2 + v_fast * t), 80.0)], ["V2"]),
            (["V1"], make_camera(), 25, lambda t: [(1, "Bus", (2.0, 45.0), 0.0)] + flowing_traffic(t), ["V1"]),
            (["V4"], make_camera(direction="toward_camera"), 6,
             lambda t: [(1, "Rickshaw", (1.0, 70 - v_rick * t), 10.0)], ["V4"]),
            (["V3"], make_camera(areas=[KERB]), 70, cng_at_kerb, ["V3"]),
            (["V7"], make_camera(areas=[KERB]), 30,
             lambda t: [(1, "CNG", (5.0, 60.0), 0.0)] + flowing_traffic(t), ["V7"]),
        ]

    def test_same_results_at_10_fps(self):
        for only, cam, secs, fn, want in self.scenarios():
            full = Sim(cam, only=only).run(secs, fn)
            dropped = Sim(cam, only=only, keep_every=3).run(secs, fn)
            self.assertEqual(codes(full), want, only)
            self.assertEqual(codes(dropped), want, only)
            self.assertAlmostEqual(full[0].obs.time_s, dropped[0].obs.time_s, delta=0.2)


class InactiveRulesTests(unittest.TestCase):
    def test_reports_why_rules_are_off(self):
        engine = ViolationEngine(make_camera())
        self.assertIn("V4", engine.inactive)  # no one-way zone
        self.assertIn("V5", engine.inactive)  # no lane lines
        self.assertIn("V6", engine.inactive)  # no helmet model
        self.assertIn("V8", engine.inactive)  # no banned classes
        self.assertIn("V3", engine.inactive)  # no kerb strip


class ReportOnceTests(unittest.TestCase):
    def test_one_event_per_vehicle_and_rule(self):
        sim = Sim(make_camera(), only=["V2"])
        v = 70 / 3.6
        ev = sim.run(80 / v, lambda t: [(1, "Car", (3.0, 2 + v * t), 70.0)])
        self.assertEqual(len(ev), 1)


if __name__ == "__main__":
    unittest.main()
