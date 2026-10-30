"""Synthetic-trajectory tests for the behavioural candidates V10 (erratic manoeuvre) and V11 (near-collision or collision)
and for the kinematic pre-screen that precedes the VLM cascade (no model or video needed).

    python -m unittest tests.test_behaviour -v

Vehicles move on a real road zone (10.5 m x 90 m, y grows toward the camera) and are fed through the real rule engine the
way surveillance/pipeline.py does it: every vehicle every frame, with measurement noise on the road-plane positions.
"""

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import screening, vlm  # noqa: E402
from violations import CameraConfig, Observation, RoadZone, ViolationEngine  # noqa: E402
from violations.engine import RULES  # noqa: E402
from violations.near_collision import FOOTPRINT  # noqa: E402

FPS = 15
POINTS = [[1340.0, 200.0], [1505.0, 196.0], [2508.0, 932.0], [1458.0, 908.0]]
FRAME = np.zeros((1440, 2560, 3), np.uint8)


def make_camera(areas=()):
    return CameraConfig([RoadZone("Right side", POINTS, 10.5, 90.0)], list(areas))


class Sim:
    """vehicles: (tid, cls, (x_m, y_m), kph). Positions get Gaussian noise `noise` metres (box jitter)."""

    def __init__(self, only, noise=0.12, fps=FPS, cam=None, seed=0):
        self.cam = cam or make_camera()
        self.engine = ViolationEngine(self.cam, only=only)
        self.noise, self.fps, self.frame = noise, fps, 0
        self.rng = np.random.default_rng(seed)

    def step(self, vehicles, purity=1.0, xy_ok=True):
        self.frame += 1
        t = self.frame / self.fps
        road = self.cam.roads[0]
        obs = []
        for tid, cls, xy, kph in vehicles:
            px = road.to_pixels([xy])[0]
            point = (float(px[0]), float(px[1]))
            m = tuple(float(v) for v in np.float64(xy) + self.rng.normal(0, self.noise, 2))
            bbox = (int(point[0] - 60), int(point[1] - 120), int(point[0] + 60), int(point[1]))
            obs.append(Observation(tid, self.frame, t, cls, purity, 100, bbox, point, road, m if xy_ok else None,
                                   kph, None, self.cam.areas_at(point), xy_coarse=m))
        return self.engine.process(obs, FRAME, self.frame, t)

    def run(self, seconds, fn, **kw):
        events = []
        for i in range(int(seconds * self.fps)):
            events += self.step(fn(i / self.fps), **kw)
        return events


def codes(events):
    return sorted(e.code for e in events)


def drive(t, x_of, kph, y0=5.0, cls="Car", tid=1):
    return [(tid, cls, (float(x_of(t)), y0 + kph / 3.6 * t), kph)]


def lane(x):
    return lambda t: x


def out_and_back(x0, E, t0, T):
    """Raised-cosine swerve out by E metres and back, starting at t0, lasting T seconds."""
    def f(t):
        s = min(max((t - t0) / T, 0.0), 1.0)
        return x0 + E / 2 * (1 - np.cos(2 * np.pi * s))
    return f


def lane_change(x0, D, t0, T):
    def f(t):
        s = min(max((t - t0) / T, 0.0), 1.0)
        return x0 + D / 2 * (1 - np.cos(np.pi * s))
    return f


class SwerveTests(unittest.TestCase):
    def run_v10(self, x_of, kph=40.0, cls="Car", seconds=6.0, **kw):
        return Sim(["V10"], **kw).run(seconds, lambda t: drive(t, x_of, kph, cls=cls))

    def test_straight_driving_is_never_a_candidate(self):
        self.assertEqual(self.run_v10(lane(5.0), seconds=7), [])

    def test_ordinary_lateral_drift_in_the_lane_is_not_a_candidate(self):
        # +-0.4 m wander with an 5 s period: a driver correcting in a wide Dhaka lane
        self.assertEqual(self.run_v10(lambda t: 5.0 + 0.4 * np.sin(2 * np.pi * t / 5), seconds=8), [])

    def test_a_normal_lane_change_is_not_a_candidate(self):
        # 3.3 m in 4 s: peak sideways speed 1.3 m/s, 1 m/s^2
        self.assertEqual(self.run_v10(lane_change(3.0, 3.3, 2.0, 4.0), seconds=8), [])

    def test_a_sharp_swerve_out_and_back_at_speed_is_a_candidate(self):
        ev = self.run_v10(out_and_back(5.0, 3.0, 2.0, 2.4))
        self.assertEqual(codes(ev), ["V10"])
        d = ev[0].details
        self.assertGreaterEqual(d["lateral_m"], 2.0)
        self.assertGreaterEqual(d["peak_lateral_mps"], 2.2)
        self.assertTrue(d["reversal"] or d["heading_change_deg"] >= 15)   # the shape condition
        self.assertAlmostEqual(d["speed_kph"], 40, delta=4)
        self.assertLessEqual(d["window_s"], 2.1)
        self.assertEqual((d["decided_by"], d["needs_review"]), ("vlm", True))
        self.assertGreaterEqual(ev[0].started_s, 1.5)
        self.assertLessEqual(ev[0].started_s, ev[0].obs.time_s)

    def test_a_candidate_is_reported_once_per_vehicle(self):
        x = lambda t: out_and_back(5.0, 3.0, 2.0, 2.4)(t) if t < 4.4 else out_and_back(5.0, 3.0, 5.0, 2.4)(t)
        self.assertEqual(len(self.run_v10(x, seconds=8)), 1)

    def test_the_same_swerve_at_a_crawl_is_not_a_candidate(self):
        # below 15 km/h sideways nosing in a queue is routine and a driver would brake instead
        self.assertEqual(self.run_v10(out_and_back(5.0, 3.0, 2.0, 2.4), kph=10.0), [])

    def test_a_swerve_needs_precise_positions(self):
        sim = Sim(["V10"])
        x = out_and_back(5.0, 3.0, 2.0, 2.4)
        self.assertEqual(sim.run(6, lambda t: drive(t, x, 40.0), xy_ok=False), [])

    def test_a_moderately_weaving_motorbike_is_not_a_candidate(self):
        # +-0.8 m, 3 s period: ordinary filtering in Dhaka (peak 1.7 m/s)
        weave = lambda t: 5.0 + 0.8 * np.sin(2 * np.pi * t / 3.0)
        self.assertEqual(self.run_v10(weave, kph=25.0, cls="Motorbike", seconds=10), [])

    def test_two_wheelers_have_looser_limits_than_heavy_vehicles(self):
        swerve = out_and_back(5.0, 1.8, 2.0, 2.4)   # 1.8 m out and back, peak about 2.4 m/s
        self.assertEqual(codes(self.run_v10(swerve, cls="Bus", kph=35.0)), ["V10"])
        self.assertEqual(self.run_v10(swerve, cls="Motorbike", kph=35.0), [])

    def test_a_vehicle_turning_at_a_junction_is_exempt(self):
        road = RoadZone("Right side", POINTS, 10.5, 90.0)
        from violations import AreaZone
        pts = road.to_pixels([[0, 0], [10.5, 0], [10.5, 90], [0, 90]])
        cam = make_camera([AreaZone("intersection", "Junction", pts)])
        x = out_and_back(5.0, 3.0, 2.0, 2.4)
        self.assertEqual(Sim(["V10"], cam=cam).run(6, lambda t: drive(t, x, 40.0)), [])

    def test_a_tracker_id_jump_between_lanes_is_not_a_swerve(self):
        x = lambda t: 3.0 if t < 3.0 else 6.5   # one frame: the ID went to another vehicle
        self.assertEqual(self.run_v10(x), [])

    def test_a_swerve_is_found_at_10_fps_too(self):
        x = out_and_back(5.0, 3.0, 2.0, 2.4)
        ev = Sim(["V10"], fps=10).run(6, lambda t: drive(t, x, 40.0))
        self.assertEqual(codes(ev), ["V10"])

    def test_forget_clears_the_history(self):
        sim = Sim(["V10"])
        sim.run(2, lambda t: drive(t, lane(5.0), 40.0))
        rule = sim.engine.rules[0]
        self.assertEqual(list(rule.state), [1])
        sim.engine.reset()
        self.assertEqual(rule.state, {})


def approach(t, v0_kph, brake_at_gap, decel, stop_y, cls="Car", tid=2):
    """A vehicle driving toward the camera at v0 that brakes at `decel` once its front is `brake_at_gap` metres short of
    stop_y (the y of the stopped vehicle's near edge less its length is handled by the caller); returns (y, kph)."""
    v0 = v0_kph / 3.6
    y_brake = stop_y - brake_at_gap
    y0 = 5.0
    t_brake = (y_brake - y0) / v0
    if t < t_brake:
        return y0 + v0 * t, v0_kph
    dt = t - t_brake
    tb = v0 / decel
    if dt >= tb:
        return y_brake + v0 * tb / 2, 0.0
    return y_brake + v0 * dt - decel * dt * dt / 2, (v0 - decel * dt) * 3.6


class ConflictTests(unittest.TestCase):
    CAR_L = FOOTPRINT["Car"][1]

    def run_v11(self, fn, seconds, **kw):
        return Sim(["V11"], **kw).run(seconds, fn)

    def test_safe_following_is_not_a_conflict(self):
        pair = lambda t: drive(t, lane(3.0), 30.0, y0=5.0, tid=1) + drive(t, lane(3.0), 30.0, y0=17.0, tid=2)
        self.assertEqual(self.run_v11(pair, 8), [])

    def test_gentle_braking_up_to_a_stopped_queue_is_not_a_conflict(self):
        stopped_y = 70.0
        def scene(t):
            y, kph = approach(t, 30.0, 14.0, 2.0, stopped_y - self.CAR_L)
            return [(1, "Car", (3.0, stopped_y), 0.0), (2, "Car", (3.0, y), kph)]
        self.assertEqual(self.run_v11(scene, 9), [])

    def test_side_by_side_passing_in_dense_slow_traffic_is_not_a_conflict(self):
        # two cars 0.45 m apart (centres 2.2 m) passing each other at 4 km/h difference in a crawl, plus a crowd
        def scene(t):
            crowd = [(10 + k, "Car", (8.5, 20.0 + 9 * k + 2.0 * t), 7.0) for k in range(5)]
            return (drive(t, lane(2.0), 12.0, y0=40.0, tid=1) + drive(t, lane(4.2), 8.0, y0=44.0, tid=2) + crowd)
        self.assertEqual(self.run_v11(scene, 8), [])

    def test_fast_closing_on_a_stopped_car_with_hard_braking_is_a_near_collision(self):
        stopped_y = 70.0
        def scene(t):
            y, kph = approach(t, 45.0, 14.0, 6.0, stopped_y - self.CAR_L)
            return [(1, "Car", (3.0, stopped_y), 0.0), (2, "Car", (3.0, y), kph)]
        ev = self.run_v11(scene, 9)
        self.assertEqual(codes(ev), ["V11"])
        d = ev[0].details
        self.assertEqual(d["kind"], "near_collision")
        self.assertLessEqual(d["ttc_s"], 1.5)
        self.assertGreaterEqual(d["closing_speed_mps"], 2.5)
        self.assertGreaterEqual(d["decel_mps2"], 4.0)
        self.assertEqual({ev[0].obs.tid, d["other_tid"]}, {1, 2})
        self.assertEqual(d["other_cls"], "Car")
        self.assertEqual((d["decided_by"], d["needs_review"]), ("vlm", True))

    def collision_scene(self, t):
        """Car 2 at 36 km/h drives into the standing car 1; both are still within 1.5 s and stay so."""
        stopped_y = 60.0
        hit = (stopped_y - self.CAR_L - 5.0) / 10.0   # time the front reaches the stopped car's rear (10 m/s)
        if t < hit:
            return [(1, "Car", (3.0, stopped_y), 0.0), (2, "Car", (3.0, 5.0 + 10.0 * t), 36.0)]
        s = min(t - hit, 1.0)
        shove = 0.5 * s * (1 - 0.5 * s)   # car 1 is pushed 0.25 m, car 2 stops 0.2 m inside it
        stop2 = 5.0 + 10.0 * hit + 0.2
        return [(1, "Car", (3.0, stopped_y + shove * 0.5), 0.0 if s >= 1 else 2.0),
                (2, "Car", (3.0, stop2 + shove * 0.5 if s < 1 else stop2 + 0.25), 0.0 if s >= 0.3 else 8.0)]

    def test_contact_then_both_stationary_is_a_collision_and_one_event(self):
        ev = self.run_v11(self.collision_scene, 14)
        self.assertEqual(codes(ev), ["V11"])
        d = ev[0].details
        self.assertEqual(d["kind"], "collision")
        self.assertGreaterEqual(d["stationary_s"], 3.0)
        self.assertGreaterEqual(d["closing_speed_mps"], 2.5)
        self.assertEqual(ev[0].extra_frames[0][0], "contact")
        self.assertLess(ev[0].started_s, ev[0].obs.time_s - 3.0)

    def test_a_collision_in_which_one_vehicle_drives_away_is_not_reported(self):
        def scene(t):
            cars = self.collision_scene(t)
            return cars if t < 6.5 else [cars[0], (2, "Car", (3.0, cars[1][2][1] + 2.0 * (t - 6.5)), 7.0)]
        self.assertEqual(self.run_v11(scene, 14), [])

    def test_two_cars_that_only_touch_while_creeping_are_not_a_collision(self):
        def scene(t):   # 1.5 m/s creep into a stopped car, stop touching
            y = min(5.0 + 1.5 * t, 60.0 - self.CAR_L + 0.1)
            return [(1, "Car", (3.0, 60.0), 0.0), (2, "Car", (3.0, y), 1.5 if y < 55 else 0.0)]
        self.assertEqual(self.run_v11(scene, 40), [])

    def test_bus_footprint_counts_the_body_behind_the_box_edge(self):
        # a car 6 m behind a bus's bottom edge is still 4 m short of its rear only because the bus is 10.5 m long
        from violations.near_collision import NearCollision
        self.assertAlmostEqual(NearCollision.DEFAULTS["body_behind_frac"], 1.0)
        self.assertAlmostEqual(FOOTPRINT["Bus"][1], 10.5)

    def test_footprints_match_the_calibration_table(self):
        from surveillance.autozone import CLASS_SIZE
        for cls, (w, length, *_) in CLASS_SIZE.items():
            self.assertEqual(FOOTPRINT[cls], (w, length), cls)

    def test_forget_clears_pair_state(self):
        sim = Sim(["V11"])
        sim.run(2, lambda t: drive(t, lane(3.0), 30.0, tid=1) + drive(t, lane(3.0), 30.0, y0=20.0, tid=2))
        rule = sim.engine.rules[0]
        sim.engine.reset()
        self.assertEqual(rule.rings.slot, {})


class CostTests(unittest.TestCase):
    def test_60_vehicles_for_300_frames_cost_well_under_a_second(self):
        cam = make_camera()
        engine = ViolationEngine(cam, only=["V10", "V11"])
        road = cam.roads[0]
        rng = np.random.default_rng(3)
        n, frames = 60, 300
        xs = rng.uniform(1.0, 9.5, n)
        y0 = rng.uniform(0, 85, n)
        speed = rng.uniform(3.0, 9.0, n)
        classes = rng.choice(["Car", "CNG", "Motorbike", "Bus", "Rickshaw"], n)
        built = []
        for f in range(frames):
            t = f / 30.0
            ys = (y0 + speed * t) % 88 + 1.0
            px = road.to_pixels(np.stack([xs, ys], 1))
            xy = np.stack([xs, ys], 1) + rng.normal(0, 0.1, (n, 2))
            built.append([Observation(i, f, t, str(classes[i]), 1.0, 100, (0, 0, 10, 10), (float(px[i, 0]), float(px[i, 1])),
                                      road, (float(xy[i, 0]), float(xy[i, 1])), float(speed[i] * 3.6), None, frozenset(),
                                      xy_coarse=(float(xy[i, 0]), float(xy[i, 1]))) for i in range(n)])
        # CPU time of this thread, not wall clock: the bound is about the rules' cost, and a loaded machine
        # (other jobs, other test runs) would otherwise fail it without the rules having got any slower.
        t0 = time.thread_time()
        for f, obs in enumerate(built):
            engine.process(obs, None, f, f / 30.0)
        elapsed = time.thread_time() - t0
        self.assertLess(elapsed, 0.9, f"{elapsed:.2f} s CPU")


def v10_row(**over):
    d = {"lateral_m": 3.0, "peak_lateral_mps": 3.3, "lateral_accel_mps2": 7.0, "heading_change_deg": 25.0, "reversal": True,
         "speed_kph": 40.0, "class_group": "standard", "fit_rms_m": 0.2, "decided_by": "vlm", "needs_review": True}
    d.update(over.pop("details", {}))
    return {"code": "V10", "details": d, "kph": 40.0, "cls": "Car", **over}


def v11_row(**details):
    d = {"kind": "near_collision", "ttc_s": 0.8, "closing_speed_mps": 6.0, "decel_mps2": 6.0, "other_tid": 1,
         "other_cls": "Car", "decided_by": "vlm", "needs_review": True}
    d.update(details)
    return {"code": "V11", "details": d, "kph": 20.0, "cls": "Car"}


class ScreeningTests(unittest.TestCase):
    def test_classes_follow_the_rule_code(self):
        self.assertEqual(screening.screen(v10_row())["class"], "direction_or_manoeuvre")
        self.assertEqual(screening.screen(v11_row())["class"], "near_collision_or_collision")
        self.assertEqual(screening.screen({"code": "V7", "details": {}, "cls": "Car"})["class"], "abnormal_road_use")
        self.assertEqual(screening.screen({"code": "V4", "details": {}, "cls": "Car"})["class"], "direction_or_manoeuvre")

    def test_every_answer_has_the_four_keys(self):
        r = screening.screen(v10_row())
        self.assertEqual(set(r), {"class", "score", "send_to_vlm", "reason"})
        self.assertIn(r["class"], screening.CLASSES)
        self.assertTrue(0.0 <= r["score"] <= 1.0 and r["reason"])

    def test_the_threshold_is_derived_from_the_costs_and_asymmetric(self):
        self.assertAlmostEqual(screening.threshold(1, 1), 0.5)
        self.assertAlmostEqual(screening.threshold(10, 1), 1 / 11)
        self.assertLess(screening.threshold(), 0.5)
        self.assertGreater(screening.COST_MISSED, screening.COST_WASTED)

    def test_a_candidate_exactly_at_the_firing_levels_is_sent(self):
        r = screening.screen(v10_row(details={"lateral_m": 2.0, "peak_lateral_mps": 2.2, "lateral_accel_mps2": 3.5,
                                              "heading_change_deg": 15.0, "reversal": False, "fit_rms_m": 0.1}))
        self.assertTrue(r["send_to_vlm"])
        self.assertAlmostEqual(r["score"], 0.5, delta=0.01)

    def test_a_strong_candidate_is_sent_whatever_the_cost_ratio(self):
        for cost in (1, 2, 5, 10, 100):
            thr = screening.threshold(cost, 1)
            for row in (v10_row(), v11_row(), v11_row(kind="collision", stationary_s=4.0)):
                self.assertGreaterEqual(screening.screen(row)["score"], 0.5)
                self.assertGreaterEqual(screening.screen(row)["score"], thr)

    def test_an_obviously_empty_candidate_is_not_sent(self):
        weak = v10_row(details={"lateral_m": 0.15, "peak_lateral_mps": 0.2, "lateral_accel_mps2": 0.3, "reversal": False,
                                "heading_change_deg": 1.0, "fit_rms_m": 0.4})
        r = screening.screen(weak)
        self.assertFalse(r["send_to_vlm"])
        self.assertEqual(r["class"], "none")
        self.assertLess(r["score"], screening.threshold())

    def test_a_swerve_that_is_mostly_fit_error_scores_lower(self):
        clean = screening.screen(v10_row(details={"fit_rms_m": 0.1}))["score"]
        glitch = screening.screen(v10_row(details={"fit_rms_m": 0.45, "lateral_m": 2.0}))["score"]
        self.assertLess(glitch, clean)

    def test_the_weakest_criterion_decides(self):
        strong_but_slow = screening.screen(v11_row(decel_mps2=0.2))
        self.assertLess(strong_but_slow["score"], screening.screen(v11_row())["score"])
        self.assertFalse(strong_but_slow["send_to_vlm"])

    def test_accepts_a_violation_event_and_a_json_details_string(self):
        import json
        from violations.base import ViolationEvent
        row = v10_row()
        ev = ViolationEvent("V10", "x", Observation(1, 1, 1.0, "Car", 1, 9, (0, 0, 1, 1), (0, 0)), row["details"])
        self.assertEqual(screening.screen(ev)["score"], screening.screen(row)["score"])
        row["details"] = json.dumps(row["details"])
        self.assertEqual(screening.screen(row)["score"], screening.screen(ev)["score"])

    def test_rule_decided_codes_and_unknown_codes_always_pass(self):
        for code in ("V1", "V2", "V6", "V9", "V99", ""):
            r = screening.screen({"code": code, "details": None, "cls": "Bus"})
            self.assertTrue(r["send_to_vlm"], code)
            self.assertNotEqual(r["class"], "none")

    def test_a_candidate_without_details_is_sent(self):
        r = screening.screen({"code": "V11", "details": {}, "cls": "Car"})
        self.assertTrue(r["send_to_vlm"])


class RegistrationTests(unittest.TestCase):
    def test_rules_are_registered_and_always_active(self):
        self.assertEqual((RULES["erratic"].code, RULES["near_collision"].code), ("V10", "V11"))
        engine = ViolationEngine(make_camera())
        self.assertNotIn("V10", engine.inactive)
        self.assertNotIn("V11", engine.inactive)
        self.assertEqual({"V10", "V11"} - {r.code for r in engine.rules}, set())

    def test_settings_can_be_sent_to_the_editor_as_json(self):
        import json
        json.dumps(RULES["erratic"].DEFAULTS)
        json.dumps(RULES["near_collision"].DEFAULTS)

    def test_rule_claims_exist_and_name_the_innocent_explanations(self):
        for code in ("V10", "V11"):
            claim, innocent = vlm.RULE_CLAIM[code]
            self.assertTrue(claim and innocent)
        self.assertIn("pothole", vlm.RULE_CLAIM["V10"][1])
        self.assertIn("swap", vlm.RULE_CLAIM["V11"][1])


if __name__ == "__main__":
    unittest.main()
