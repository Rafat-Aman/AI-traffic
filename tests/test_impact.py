"""Traffic state, baseline and impact scoring (report SDP 3.1.4).

Everything here runs on SYNTHETIC traffic, because the project has no Dhaka footage with a measured delay and no
human severity ranking yet. That makes these tests a floor, not evidence: what they pin down is the behaviour that
must never regress - each quantity means what its docstring says, an empty road with a stopped bus is not scored as a
jam, the weights are never negative, an uncalibrated score says so, and history never outranks impact.
"""

import json
import math
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import impact as im  # noqa: E402
from surveillance.impact import (EQUAL_WEIGHTS, ImpactTracker, Weights, calibrate, compute_event_impact,  # noqa: E402
                                 event_window, nnls, priority, ranks, spearman, validate)
from surveillance.store import Store  # noqa: E402
from surveillance.traffic_state import (QUANTITIES, Baseline, TrafficStateCollector,  # noqa: E402
                                        ensure_schema as ensure_baseline_schema)
from violations.base import Observation  # noqa: E402
from violations.zones import RoadZone  # noqa: E402

PX = [[900, 300], [1100, 300], [1500, 1000], [500, 1000]]    # far-left, far-right, near-right, near-left


def road(name="Test Rd", width=10.0, length=90.0, direction="toward_camera"):
    return RoadZone(name, PX, width, length, allowed_direction=direction)


def ob(rd, tid, y, kph, cls="Car", heading=1, x=5.0, t=0.0, coarse=False):
    """One tracked vehicle at `y` metres from the far edge. coarse=True: the far end of a zone, where the pipeline
    gives only a coarse position and no speed."""
    return Observation(tid=tid, frame_idx=int(t * 30), time_s=t, cls=cls, purity=1.0, n_frames=30,
                       bbox=(0, 0, 10, 10), point=(0.0, 0.0), road=rd, xy=None if coarse else (x, y),
                       kph=None if coarse else kph, heading=heading, xy_coarse=(x, y))


def feed(col, frames, t0=0.0, step=1.0, wall0=None):
    """frames: a list of observation lists, one per second."""
    for i, obs in enumerate(frames):
        t = t0 + i * step
        col.observe(obs, t, None if wall0 is None else wall0 + t)


def at(rd, vehicles, t=0.0):
    return [ob(rd, tid, y, kph, cls, t=t) for tid, y, kph, cls in vehicles]


def local(y, mo, d, h=8, mi=5):
    return datetime(y, mo, d, h, mi).timestamp()


class QuantityTests(unittest.TestCase):
    """Each of the five quantities, against a frame small enough to work out by hand."""

    def setUp(self):
        self.rd = road()                      # 10 m x 90 m -> 3 lanes
        self.col = TrafficStateCollector(1, [self.rd])

    def test_each_quantity_matches_a_hand_computed_frame(self):
        v = [(1, 80, 0.0, "Car"), (2, 73, 1.0, "Car"), (3, 66, 0.5, "Car"),     # a queue at the exit
             (4, 30, 40.0, "Car"),                                               # free-flowing
             (5, 79, 0.0, "Motorbike")]                                          # weaving: not part of any of it
        feed(self.col, [at(self.rd, v, 0.0)] + [at(self.rd, v, float(t)) for t in range(1, 11)])
        s = self.col.state(self.rd, 10.0, 10.0)
        # chain: front of the head (y=66) to rear of the tail (y=80): 14 m between centres + half a car at each end
        self.assertAlmostEqual(s["queue_m"], 18.5, places=3)
        self.assertAlmostEqual(s["speed_kph"], (0 + 1.0 + 0.5 + 40.0) / 4, places=3)    # no motorbike
        self.assertAlmostEqual(s["occupancy"], (4 * 4.5 + 2.0) / (90 * 3), places=3)     # the motorbike takes room
        self.assertAlmostEqual(s["density"], 5 / 0.09 / 3, places=2)
        self.assertAlmostEqual(s["wait_s"], (10 + 10 + 10 + 0) / 4, places=3)             # standing since t=0

    def test_one_stopped_vehicle_is_not_a_queue(self):
        feed(self.col, [at(self.rd, [(1, 80, 0.0, "Bus")])] * 5)
        self.assertEqual(self.col.state(self.rd)["queue_m"], 0.0)

    def test_a_gap_over_ten_metres_ends_the_chain(self):
        def queue(extra_y):
            v = [(1, 80, 0, "Car"), (2, 73, 0, "Car"), (3, 66, 0, "Car"), (4, extra_y, 0, "Car")]
            col = TrafficStateCollector(1, [self.rd])
            feed(col, [at(self.rd, v)])
            return col.state(self.rd)["queue_m"]
        self.assertAlmostEqual(queue(52.0), 80 - 52 + 4.5, places=3)     # bumper gap 9.5 m: still one chain
        self.assertAlmostEqual(queue(50.0), 18.5, places=3)              # bumper gap 11.5 m: a second chain, shorter

    def test_motorbikes_and_bicycles_never_define_the_queue_or_the_flow(self):
        v = [(i, 80 - 3 * i, 0.0, "Motorbike") for i in range(4)] + [(9, 20, 0.0, "Bicycle")]
        feed(self.col, [at(self.rd, v)] * 4)
        s = self.col.state(self.rd)
        self.assertEqual((s["queue_m"], s["speed_kph"], s["wait_s"]), (0.0, None, 0.0))
        self.assertGreater(s["occupancy"], 0)            # ... but they do take up road
        self.assertGreater(s["density"], 0)

    def test_a_queue_forms_where_its_head_stopped_not_only_at_the_exit(self):
        v = [(1, 40, 0.0, "Bus"), (2, 31, 0.0, "Car"), (3, 25.5, 0.0, "Car"),   # behind a bus standing mid-block
             (4, 70, 35.0, "Car")]                                              # free road downstream of it
        feed(self.col, [at(self.rd, v)])
        self.assertAlmostEqual(self.col.state(self.rd)["queue_m"], 40 - 25.5 + 0.5 * (11 + 4.5), places=3)

    def test_vehicles_going_the_other_way_are_not_in_this_queue(self):
        two_way = road("Both", direction="both")
        col = TrafficStateCollector(1, [two_way])
        moving = [ob(two_way, i, 20 + 10 * i, 30.0, heading=-1) for i in range(3)]             # majority: away
        opposing = [ob(two_way, 10, 80, 0.0, heading=1), ob(two_way, 11, 73, 0.0, heading=1)]  # stopped, toward
        col.observe(moving + opposing, 0.0)
        self.assertEqual(col.state(two_way)["queue_m"], 0.0)
        same_way = [ob(two_way, 10, 80, 0.0, heading=-1), ob(two_way, 11, 73, 0.0, heading=-1)]
        col2 = TrafficStateCollector(1, [two_way])
        col2.observe(moving + same_way, 0.0)
        self.assertGreater(col2.state(two_way)["queue_m"], 0.0)

    def test_the_waiting_clock_runs_while_stationary_and_resets_when_it_moves(self):
        for t in range(0, 21):
            self.col.observe([ob(self.rd, 1, 50, 0.4, t=t)], float(t))
        self.assertAlmostEqual(self.col.state(self.rd, 20.0, 20.0)["wait_s"], 20.0)
        self.col.observe([ob(self.rd, 1, 52, 15.0, t=21)], 21.0)
        self.assertEqual(self.col.state(self.rd, 21.0, 21.0)["wait_s"], 0.0)
        for t in (22.0, 23.0, 24.0):
            self.col.observe([ob(self.rd, 1, 53, 0.0, t=t)], t)
        self.assertAlmostEqual(self.col.state(self.rd, 24.0, 24.0)["wait_s"], 2.0)       # a new spell, not 24

    def test_a_vehicle_whose_speed_is_unknown_keeps_its_waiting_clock(self):
        for t in range(0, 6):
            self.col.observe([ob(self.rd, 1, 50, 0.0, t=t)], float(t))
        o = ob(self.rd, 1, 50, 0.0, t=6)
        o.kph = None
        self.col.observe([o], 6.0)
        self.assertAlmostEqual(self.col.state(self.rd, 6.0, 6.0)["wait_s"], 0.0)       # unknown speed: not in the mean
        self.col.observe([ob(self.rd, 1, 50, 0.0, t=7)], 7.0)
        self.assertAlmostEqual(self.col.state(self.rd, 7.0, 7.0)["wait_s"], 7.0)         # ... and the spell went on

    def test_far_end_vehicles_count_for_occupancy_and_density_but_not_speed(self):
        self.col.observe([ob(self.rd, 1, 5, None, coarse=True), ob(self.rd, 2, 6, None, "Bus", coarse=True)], 0.0)
        s = self.col.state(self.rd)
        self.assertIsNone(s["speed_kph"])
        self.assertAlmostEqual(s["occupancy"], (4.5 + 11.0) / 270, places=3)
        self.assertAlmostEqual(s["density"], 2 / 0.09 / 3, places=2)

    def test_occupancy_is_capped_at_one(self):
        v = [(i, 5 + i, 0.0, "Bus") for i in range(40)]
        self.col.observe(at(self.rd, v), 0.0)
        self.assertEqual(self.col.state(self.rd)["occupancy"], 1.0)


class SamplingTests(unittest.TestCase):
    def test_a_sample_is_taken_at_most_once_a_second_of_video(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        for k in range(101):                                    # 10 fps for 10 s
            col.observe(at(rd, [(1, 50, 30.0, "Car")]), k / 10)
        times = [s[0] for s in col._zones[id(rd)].samples]
        self.assertEqual(len(times), 11)
        self.assertGreaterEqual(min(np.diff(times)), 1.0 - 1e-9)

    def test_an_approach_nobody_is_on_is_still_sampled_as_empty(self):
        a, b = road("A"), road("B")
        col = TrafficStateCollector(1, [a, b])
        feed(col, [at(a, [(1, 50, 30.0, "Car")])] * 3)
        snap = col.snapshot()
        self.assertEqual(snap["B"]["samples"], 3)
        self.assertEqual((snap["B"]["vehicles"], snap["B"]["queue_m"], snap["B"]["occupancy"]), (0, 0.0, 0.0))
        self.assertIsNone(snap["B"]["speed_kph"])                    # nobody to measure is not a speed of 0
        self.assertEqual(snap["A"]["vehicles"], 1)

    def test_the_snapshot_is_plain_json(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        feed(col, [at(rd, [(1, 80, 0.0, "Car"), (2, 73, 0.0, "Car")])] * 3)
        text = json.dumps(col.snapshot(), allow_nan=False)
        self.assertIn("queue_m", text)
        json.dumps(TrafficStateCollector(1, [rd]).snapshot(), allow_nan=False)      # nothing sampled yet

    def test_memory_is_bounded_by_the_horizon_not_the_length_of_the_video(self):
        rd = road()
        col = TrafficStateCollector(1, [rd], horizon=600.0)
        for k in range(2400):                                           # 40 minutes
            vehicles = [(i, 10 + (k + 3 * i) % 70, 0.0 if i % 2 else 30.0, "Car") for i in range(20)]
            col.observe(at(rd, vehicles if k < 2300 else []), float(k))
        zs = col._zones[id(rd)]
        self.assertLessEqual(len(zs.samples), 600 + 8)
        self.assertGreaterEqual(zs.samples[0][0], 2399 - 600)
        self.assertEqual(col._still, {})                                # clocks of vehicles long gone are dropped

    def test_a_lost_view_is_not_sampled_as_an_empty_road(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        for t in range(0, 6):
            col.observe(at(rd, [(1, 50, 0.0, "Car")]), float(t))
        for t in range(6, 30):
            col.observe([], float(t), valid=False)                      # the pipeline cannot see the zones
        self.assertEqual(len(col._zones[id(rd)].samples), 6)
        col.observe(at(rd, [(1, 50, 0.0, "Car")]), 30.0)
        self.assertEqual(col.state(rd, 30.0, 30.0)["wait_s"], 0.0)      # a clock across a blind gap proves nothing

    def test_time_going_backwards_starts_over(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        feed(col, [at(rd, [(1, 50, 30.0, "Car")])] * 20)
        col.observe(at(rd, [(1, 50, 30.0, "Car")]), 2.0)
        self.assertEqual(len(col._zones[id(rd)].samples), 1)

    def test_the_wall_clock_is_mapped_from_the_latest_frame_that_had_one(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        self.assertIsNone(col.wall_at(5.0))
        col.observe([], 4.0, 1_000_000.0)
        self.assertEqual(col.wall_at(7.5), 1_000_003.5)
        col2 = TrafficStateCollector(1, [rd])
        col2.observe([], 0.0, datetime(2026, 3, 2, 8, 0))             # the pipeline's own wall is a datetime
        self.assertEqual(col2.wall_at(60.0), datetime(2026, 3, 2, 8, 0).timestamp() + 60.0)

    def test_the_offender_can_be_left_out_of_its_own_impact(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        for t in range(21):                                             # a bus standing alone for 20 s
            col.observe(at(rd, [(99, 50, 0.0, "Bus")]), float(t))
        with_bus = col.state(rd, 20.0, 20.0)
        without = col.state(rd, 20.0, 20.0, exclude={99})
        self.assertEqual(with_bus["wait_s"], 20.0)
        self.assertEqual((without["wait_s"], without["occupancy"], without["queue_m"], without["vehicles"]),
                         (0.0, 0.0, 0.0, 0.0))
        self.assertIsNone(without["speed_kph"])

    def test_a_window_with_nothing_sampled_has_no_state(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        self.assertIsNone(col.state(rd))
        feed(col, [at(rd, [(1, 50, 30.0, "Car")])] * 3)
        self.assertIsNone(col.state(rd, 100.0, 120.0))
        self.assertIsNone(col.state("no such zone"))


class TransitTests(unittest.TestCase):
    def test_each_vehicle_gets_an_entry_an_exit_a_distance_and_whether_it_is_still_inside(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        for t in range(0, 11):
            vehicles = [(1, 10 + 8 * t, 29.0, "Car")]                     # drives 8 m/s through, gone after t=10
            if t >= 4:
                vehicles.append((2, 30.0, 0.0, "Car"))                    # arrives and stands still
            vehicles.append((3, 50.0 - t, 0.0, "Motorbike"))              # filtering: not a transit
            col.observe(at(rd, vehicles), float(t))
        tr = {x["tid"]: x for x in col.transits(rd, 0.0, 10.0)}
        self.assertEqual(set(tr), {1, 2})
        self.assertEqual((tr[1]["entry_t"], tr[1]["exit_t"], tr[1]["distance_m"]), (0.0, 10.0, 80.0))
        self.assertEqual((tr[2]["entry_t"], tr[2]["seconds"], tr[2]["distance_m"]), (4.0, 6.0, 0.0))
        self.assertTrue(tr[2]["inside"] and tr[1]["inside"])
        self.assertAlmostEqual(tr[2]["longest_wait_s"], 6.0)
        self.assertFalse({x["tid"]: x for x in col.transits(rd, 0.0, 7.0)}[1]["seconds"] == 10.0)
        self.assertEqual([x["tid"] for x in col.transits(rd, 0.0, 10.0, exclude={2})], [1])
        self.assertEqual(col.transits("no such zone"), [])

    def test_a_vehicle_that_left_is_not_inside_and_a_standing_one_costs_exactly_its_time(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        for t in range(0, 20):
            vehicles = [(2, 30.0, 0.0, "Car")]                            # stands still the whole time
            if t < 5:
                vehicles.append((1, 10 + 8 * t, 29.0, "Car"))             # passes and is gone
            col.observe(at(rd, vehicles), float(t))
        tr = {x["tid"]: x for x in col.transits(rd, 0.0, 19.0)}
        self.assertFalse(tr[1]["inside"])
        self.assertTrue(tr[2]["inside"])
        # 19 s standing and 0 m covered, against 36 km/h: 19 s of delay, exactly
        self.assertAlmostEqual(col.excess_delay(rd, 0.0, 19.0, 36.0, exclude={1})["raw_veh_s"], 19.0)
        # vehicle 1 covers 32 m in 4 s at 8 m/s against 36 km/h = 10 m/s: it is 0.8 s FASTER than the baseline
        self.assertAlmostEqual(col.excess_delay(rd, 0.0, 19.0, 36.0, exclude={2})["raw_veh_s"], 4 - 32 / 10)
        self.assertEqual(col.excess_delay(rd, 0.0, 19.0, 36.0)["vehicles"], 2)
        self.assertIsNone(col.excess_delay(rd, 0.0, 19.0, None))
        self.assertIsNone(col.excess_delay(rd, 0.0, 19.0, 0.0))

    def test_a_road_with_one_sample_has_no_delay_to_integrate(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        col.observe(at(rd, [(1, 30.0, 0.0, "Car")]), 0.0)
        self.assertEqual(col.excess_delay(rd, 0.0, 10.0, 30.0)["raw_veh_s"], 0.0)


class CostTests(unittest.TestCase):
    def test_eighty_vehicles_for_six_hundred_frames_cost_well_under_a_second(self):
        rd = road()
        frames = [[ob(rd, i, 5 + (i * 7 + k // 5) % 80, 0.0 if i % 3 == 0 else 25.0,
                      "Motorbike" if i % 4 == 0 else "Car", t=k / 25) for i in range(80)] for k in range(600)]
        col = TrafficStateCollector(1, [rd])
        t0 = time.perf_counter()
        for k, obs in enumerate(frames):                                # 25 fps for 24 s
            col.observe(obs, k / 25, 1_000_000.0 + k / 25)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.25, f"{dt * 1e3:.0f} ms for 600 frames of 80 vehicles")
        t0 = time.perf_counter()
        col2 = TrafficStateCollector(1, [rd])
        for k, obs in enumerate(frames):                                # worst case: every frame is a sample
            col2.observe(obs, float(k))
        dt2 = time.perf_counter() - t0
        self.assertLess(dt2, 1.0, f"{dt2 * 1e3:.0f} ms when every one of 600 frames is sampled")
        print(f"\n  cost: 600 frames x 80 vehicles: {dt * 1e3:.0f} ms at 25 fps, {dt2 * 1e3:.0f} ms if every frame "
              f"were a sample ({dt2 / 600 * 1e3:.2f} ms per sample)")


class BaselineTests(unittest.TestCase):
    MON = local(2026, 3, 2)

    def setUp(self):
        self.assertEqual(datetime.fromtimestamp(self.MON).weekday(), 0)

    def fill(self, base, n, day_offsets, level=10.0, wall0=None, zone="Z"):
        rng = np.random.default_rng(0)
        for d in day_offsets:
            for k in range(n):
                wall = (wall0 or self.MON) + d * 86400 + k * 30
                base.add(1, zone, wall, {"queue_m": level + rng.normal(0, 1), "speed_kph": 30 + rng.normal(0, 2)})

    def test_a_slot_only_counts_after_enough_blocks_on_enough_days(self):
        base = Baseline(min_blocks=12, min_days=2)
        self.fill(base, 20, [0])
        self.assertIsNone(base.stats(1, "Z", self.MON), "one day is not 'typical for this time'")
        self.fill(base, 5, [1])
        ref = base.stats(1, "Z", self.MON + 86400)
        self.assertEqual(ref.source, "history")
        self.assertAlmostEqual(ref.median["queue_m"], 10.0, delta=0.6)
        self.assertNotIn("occupancy", ref.median)                  # never measured: no baseline for it
        base2 = Baseline(min_blocks=12, min_days=2)
        self.fill(base2, 3, [0, 1, 2])
        self.assertIsNone(base2.stats(1, "Z", self.MON), "nine blocks on three days: enough days, too few blocks")
        self.fill(base2, 3, [3])
        self.assertIsNotNone(base2.stats(1, "Z", self.MON))

    def test_the_median_and_mad_ignore_the_events_in_the_stream(self):
        base = Baseline(min_blocks=12, min_days=1)
        self.fill(base, 40, [0])
        for k in range(6):                                          # six blocks of a long jam
            base.add(1, "Z", self.MON + 40 * 30 + k * 30, {"queue_m": 150.0})
        ref = base.stats(1, "Z", self.MON)
        self.assertAlmostEqual(ref.median["queue_m"], 10.0, delta=0.7)
        self.assertLess(ref.mad["queue_m"], 2.0)

    def test_weekday_weekend_and_each_slot_are_kept_apart(self):
        base = Baseline(min_blocks=12, min_days=1)
        sat = local(2026, 3, 7)
        self.assertEqual(datetime.fromtimestamp(sat).weekday(), 5)
        self.fill(base, 20, [0], level=10.0)
        self.fill(base, 20, [0], level=40.0, wall0=sat)
        self.fill(base, 20, [0], level=70.0, wall0=local(2026, 3, 2, 8, 20))
        self.assertAlmostEqual(base.stats(1, "Z", self.MON).median["queue_m"], 10.0, delta=1)
        self.assertAlmostEqual(base.stats(1, "Z", sat).median["queue_m"], 40.0, delta=1)
        self.assertAlmostEqual(base.stats(1, "Z", local(2026, 3, 2, 8, 25)).median["queue_m"], 70.0, delta=1)
        self.assertIsNone(base.stats(2, "Z", self.MON), "another camera has its own baseline")
        self.assertIsNone(base.stats(1, "Other", self.MON))

    def test_the_reservoir_is_bounded_and_follows_the_recent_weeks(self):
        base = Baseline(size=32, history=100, min_blocks=12, min_days=1)
        rng = np.random.default_rng(1)
        for day, level in ((0, 10.0), (1, 30.0)):                    # the road changed
            for _ in range(700):                                    # every value in the same slot
                base.add(1, "Z", self.MON + day * 86400, {"queue_m": level + rng.normal(0, 1)})
        key = (1, "Z", *base.slot(self.MON))
        r = base._cache[key]["queue_m"]
        self.assertEqual(len(r.vals), 32)
        self.assertEqual(r.n, 1400)
        self.assertGreater(base.stats(1, "Z", self.MON).median["queue_m"], 25.0)

    def test_it_survives_reopening_the_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            store = Store(path)
            base = Baseline(store, min_blocks=12, min_days=2)
            self.fill(base, 30, [0, 1])
            before = base.stats(1, "Z", self.MON)
            base.flush()
            store.db.close()
            store2 = Store(path)
            after = Baseline(store2, min_blocks=12, min_days=2).stats(1, "Z", self.MON)
            for q in before.median:                                    # stored to 4 decimals
                self.assertAlmostEqual(before.median[q], after.median[q], places=3)
                self.assertAlmostEqual(before.mad[q], after.mad[q], places=3)
            self.assertEqual(after.detail["days"], 2)
            ensure_baseline_schema(store2)                             # idempotent
            # 30 blocks from 08:05 run past 08:15, so they fill two 15-minute slots
            self.assertEqual(store2.q("SELECT COUNT(*) AS n FROM traffic_baseline")[0]["n"], 2)

    def test_a_write_in_progress_is_lost_not_corrupted_when_nothing_was_flushed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            store = Store(path)
            base = Baseline(store, min_blocks=1, min_days=1, flush_sec=1e9)
            self.fill(base, 30, [0])
            store.db.close()
            self.assertIsNone(Baseline(Store(path), min_blocks=1, min_days=1).stats(1, "Z", self.MON))

    def test_the_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "s.db")
            base = Baseline(store, max_cached=8, min_blocks=1, min_days=1)
            for k in range(40):                                        # 40 different slots
                base.add(1, "Z", self.MON + k * 900, {"queue_m": 1.0 * k})
            self.assertLessEqual(len(base._cache), 8)
            base.flush()
            self.assertEqual(store.q("SELECT COUNT(*) AS n FROM traffic_baseline")[0]["n"], 40)
            self.assertAlmostEqual(base.stats(1, "Z", self.MON + 3 * 900).median["queue_m"], 3.0)   # reloaded

    def test_the_collector_feeds_its_window_means_to_the_baseline(self):
        rd = road()
        base = Baseline(min_blocks=10, min_days=2)
        col = TrafficStateCollector(1, [rd], baseline=base, window=30.0)
        for day in (0, 1):
            wall0 = self.MON + day * 86400
            col = TrafficStateCollector(1, [rd], baseline=base, window=30.0)
            for k in range(0, 400):
                col.observe(at(rd, [(1, 20 + k % 60, 36.0, "Car"), (2, 60, 0.0, "Car"), (3, 53, 0.0, "Car")]),
                            float(k), wall0 + k)
        ref = base.stats(1, rd.name, self.MON)
        self.assertIsNotNone(ref)
        self.assertAlmostEqual(ref.median["queue_m"], 60 - 53 + 4.5, places=2)
        self.assertAlmostEqual(ref.median["speed_kph"], (36 + 0 + 0) / 3, places=2)
        self.assertEqual(ref.detail["days"], 2)
        self.assertEqual(set(ref.median), set(QUANTITIES))




# ---------------------------------------------------------------- synthetic traffic

LENGTH = {"Car": 4.5, "CNG": 2.7, "Rickshaw": 2.2, "Motorbike": 2.0, "Truck": 9.0, "Bus": 11.0}
MIX = (("Car", .35, 10.0), ("CNG", .20, 8.5), ("Rickshaw", .10, 3.0), ("Motorbike", .25, 11.0),
       ("Truck", .05, 7.5), ("Bus", .05, 8.0))          # (class, share, desired speed in m/s)


class ApproachSim:
    """A small car-following model of one approach (three lanes, vehicles heading toward the camera) that emits the
    Observations the pipeline would: noisy road-plane positions, speeds in km/h, no speed at the far end. It stands in
    for footage and is deliberately simple - but it has the two things the score must respond to: a queue that
    grows behind a vehicle that stopped, and the other lanes slowing to squeeze past it. Motorbikes ignore the queue."""

    def __init__(self, rd, rate, seed=0, lanes=3):
        self.rd, self.rate, self.lanes = rd, rate, lanes
        self.rng = np.random.default_rng([seed, 1])                 # arrivals and speeds
        self.noise = np.random.default_rng([seed, 2])               # measurement noise: kept apart so that two
        self.veh, self.t, self.next_tid = [], 0.0, 1                # runs with one seed have the same traffic
        self.stopper = self.waiting = None

    def _new(self, cls, vd, lane):
        L = LENGTH[cls]
        if any(u["lane"] == lane and not u["filter"] and u["y"] < L + 8 for u in self.veh):
            return None                                  # no room at the entrance: that arrival never enters view
        v = dict(tid=self.next_tid, cls=cls, lane=lane, y=L / 2, v=0.8 * vd, vd=vd, L=L, filter=cls == "Motorbike",
                 hold=None)
        self.next_tid += 1
        self.veh.append(v)
        return v

    def add_stopper(self, lane, y_stop, hold, cls="Bus"):
        """A bus that enters, stops at `y_stop` metres and stands for `hold` seconds."""
        self.waiting = (cls, lane, y_stop, hold)

    def step(self, dt):
        self.t += dt
        if self.waiting is not None:
            cls, lane, y_stop, hold = self.waiting
            v = self._new(cls, 8.0, lane)
            if v is not None:
                v["hold"] = dict(y_stop=y_stop, dwell=hold, stopped_at=None, released=False)
                self.stopper, self.waiting = v, None
        if self.rng.random() < self.rate * dt:
            k = self.rng.choice(len(MIX), p=[m[1] for m in MIX])
            cls, _, vd0 = MIX[k]
            self._new(cls, max(2.0, self.rng.normal(vd0, 0.1 * vd0 + 0.3)), int(self.rng.integers(self.lanes)))
        s = self.stopper
        held = s is not None and s["hold"]["stopped_at"] is not None and not s["hold"]["released"]
        for lane in range(self.lanes):
            leader = None
            for v in sorted((u for u in self.veh if u["lane"] == lane and not u["filter"]), key=lambda u: -u["y"]):
                self._advance(v, leader, dt, s if held and s is not v else None)
                leader = v
        for v in self.veh:
            if v["filter"]:
                v["v"] = v["vd"]
                v["y"] += v["v"] * dt
        self.veh = [v for v in self.veh if v["y"] <= self.rd.length_m]

    def _advance(self, v, leader, dt, blocker):
        target, h = v["vd"], v["hold"]
        if h is not None and not h["released"]:
            if h["stopped_at"] is None:
                target = min(target, math.sqrt(2 * 2.0 * max(h["y_stop"] - v["y"], 0.0)))      # brake at 2 m/s^2
                if h["y_stop"] - v["y"] <= 0.3 and v["v"] < 0.3:
                    h["stopped_at"] = self.t
            else:
                target = 0.0
                if self.t >= h["stopped_at"] + h["dwell"]:
                    h["released"], target = True, v["vd"]
        elif blocker is not None and abs(v["y"] - blocker["y"]) < 14.0:
            target = min(target, 4.0)                                    # squeezing past the stopped bus
        if leader is not None:
            target = min(target, max(0.0, leader["y"] - 0.5 * (leader["L"] + v["L"]) - v["y"] - 1.0))   # 1 s headway
        v["v"] = min(max(target, v["v"] - 5.0 * dt), v["v"] + 1.5 * dt)
        v["y"] += v["v"] * dt
        if leader is not None:
            limit = leader["y"] - 0.5 * (leader["L"] + v["L"]) - 0.3
            if v["y"] > limit:
                v["y"], v["v"] = limit, min(v["v"], leader["v"])

    def observations(self):
        out = []
        for v in self.veh:
            y = v["y"]
            if not 0.0 <= y <= self.rd.length_m:
                continue
            x = (v["lane"] + 0.5) * self.rd.width_m / self.lanes + self.noise.normal(0, 0.2)
            far = y < 12.0
            out.append(Observation(
                tid=v["tid"], frame_idx=int(self.t * 30), time_s=self.t, cls=v["cls"], purity=1.0, n_frames=30,
                bbox=(0, 0, 10, 10), point=(0.0, 0.0), road=self.rd,
                xy=None if far else (x, y + self.noise.normal(0, 0.2)),
                kph=None if far else abs(v["v"] * 3.6 + self.noise.normal(0, 0.8)), heading=1,
                xy_coarse=(x, y + self.noise.normal(0, 1.0 if far else 0.2))))
        return out

    def run(self, seconds, sink, dt=0.1):
        """Advance `seconds`, calling sink(t, observations) once per second of video (all the collector samples)."""
        per = int(round(1 / dt))
        for i in range(int(round(seconds / dt))):
            self.step(dt)
            if (i + 1) % per == 0:
                sink(self.t, self.observations())


BUSY, QUIET = 1.2, 0.03                                  # vehicles per second onto the approach
SEEDS = (1, 2, 3, 4, 5)
MONDAY_8AM = datetime(2026, 3, 2, 8, 0).timestamp()


class Run:
    """One simulated approach and the collector that watched it."""

    def __init__(self, rate, seed, bus=False, pre=300.0, post=80.0, hold=45.0, wall0=MONDAY_8AM, baseline=None,
                 fires_after=15.0):
        self.rd = road()
        self.col = TrafficStateCollector(1, [self.rd], baseline=baseline)
        self.sim = ApproachSim(self.rd, rate, seed)
        sink = lambda t, o: self.col.observe(o, t, wall0 + t)       # noqa: E731
        self.sim.run(pre, sink)
        if bus:
            self.sim.add_stopper(1, 55.0, hold)
        self.sim.run(post, sink)
        self.tid, self.win = None, None
        if bus:
            stopped = self.sim.stopper["hold"]["stopped_at"]
            assert stopped is not None, "the bus never stopped inside the simulated time"
            self.tid = self.sim.stopper["tid"]
            self.win = event_window(stopped, stopped + fires_after)     # V1 fires after the bus has stood 15 s

    def impact(self, exclude=True, baseline=None, code="V1", weights=None, win=None):
        return compute_event_impact(self.col, baseline, 1, self.rd, *(win or self.win), code, weights,
                                    exclude_tids=(self.tid,) if exclude and self.tid else ())


class ScenarioTests(unittest.TestCase):
    """The behaviour the report asks of the score, on traffic whose ground truth is known."""

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for seed in SEEDS:
            busy = Run(BUSY, seed, bus=True)
            free = Run(BUSY, seed)                       # the same road and traffic, but nobody stops
            free.win = busy.win
            cls.runs[seed] = (busy, free, Run(QUIET, seed, bus=True))

    def test_free_flow_scores_near_zero(self):
        scores = []
        for seed, (_, free, _) in self.runs.items():
            r = free.impact(exclude=False)
            self.assertTrue(r["meaningful"])
            self.assertEqual(r["baseline_source"], "pre-event")
            self.assertLess(r["score"], 1.0, f"seed {seed}")
            scores.append(r["score"])
        self.assertLess(float(np.mean(scores)), 0.5)

    def test_a_bus_on_a_busy_approach_far_outscores_free_flow_and_the_same_stop_on_a_quiet_road(self):
        rows = []
        for seed, (busy, free, quiet) in self.runs.items():
            b, f, q = busy.impact(), free.impact(exclude=False), quiet.impact()
            rows.append((seed, b["score"], f["score"], q["score"], b["excess_delay_veh_s"], f["excess_delay_veh_s"],
                         q["excess_delay_veh_s"], b["excess_delay_signed_veh_s"] - f["excess_delay_signed_veh_s"]))
            self.assertGreater(b["score"], f["score"], f"seed {seed}: busy vs free flow")
            self.assertGreater(b["score"], q["score"], f"seed {seed}: busy vs quiet road")
            # busy and free flow carry identical traffic until the bus enters, and share their baseline error, so the
            # difference of their signed delays isolates what the bus did
            self.assertGreater(rows[-1][7], 150.0, f"seed {seed}: delay the bus added")
            self.assertLess(q["excess_delay_veh_s"], 100.0, f"seed {seed}: quiet road")
        t = np.array(rows)
        self.assertGreater(t[:, 1].mean(), 3 * t[:, 2].mean(), "mean score, busy vs free flow")
        self.assertGreater(t[:, 1].mean(), 4 * t[:, 3].mean(), "mean score, busy vs quiet road")
        self.assertGreater(t[:, 4].mean(), 4 * t[:, 5].mean(), "mean excess delay, busy vs free flow")
        self.assertGreater(t[:, 4].mean(), 8 * t[:, 6].mean(), "mean excess delay, busy vs quiet road")
        # the delay of a single event is noisy (see impact.py): most, not all, busy events read above 200 veh*s
        self.assertGreaterEqual(int((t[:, 4] > 200).sum()), 4)
        print("\n  seed | score: busy  free quiet | excess delay, veh*s: busy  free quiet | bus effect (signed)")
        for r in rows:
            print("  %4d |        %5.2f %5.2f %5.2f |                     %5.0f %5.0f %5.0f | %5.0f" % r)
        print("  mean |        %5.2f %5.2f %5.2f |                     %5.0f %5.0f %5.0f | %5.0f"
              % tuple(t.mean(axis=0)[1:]))

    def test_the_stopped_vehicle_is_left_out_of_its_own_impact(self):
        for seed, (busy, _, quiet) in self.runs.items():
            for run in (busy, quiet):
                with_bus, without = run.impact(exclude=False), run.impact(exclude=True)
                self.assertLess(without["score"], with_bus["score"], f"seed {seed}")
                self.assertLess(without["excess_delay_veh_s"], with_bus["excess_delay_veh_s"])
            q_in, q_ex = quiet.impact(exclude=False), quiet.impact(exclude=True)
            self.assertGreater(q_in["excess_delay_veh_s"], 25, "a bus that counts as its own victim looks like a jam")
            self.assertIn(quiet.tid, q_ex["excluded_tids"])

    def test_the_queue_builds_upstream_of_the_bus_and_grows_while_it_stands(self):
        for seed, (busy, free, _) in self.runs.items():
            t0, t1 = busy.win
            r = busy.impact()
            self.assertGreater(r["during"]["queue_m"], 12.0, f"seed {seed}")
            self.assertEqual(r["baseline"]["queue_m"], 0.0)
            self.assertEqual(free.impact(exclude=False)["during"]["queue_m"], 0.0)
            early = busy.col.state(busy.rd, t0, t0 + 10, exclude=(busy.tid,))["queue_m"]
            late = busy.col.state(busy.rd, t1 - 10, t1, exclude=(busy.tid,))["queue_m"]
            self.assertGreater(late, early + 5.0, f"seed {seed}: {early:.0f} m -> {late:.0f} m")

    def test_vehicles_that_were_held_up_are_counted(self):
        for seed, (busy, free, quiet) in self.runs.items():
            self.assertGreaterEqual(busy.impact()["vehicles_affected"], 4, f"seed {seed}")
            self.assertEqual(free.impact(exclude=False)["vehicles_affected"], 0)
            self.assertLessEqual(quiet.impact()["vehicles_affected"], 2)

    def test_the_result_is_plain_json_and_has_every_documented_field(self):
        busy = self.runs[SEEDS[0]][0]
        r = busy.impact()
        text = json.dumps(r, allow_nan=False)                  # what the server will send and the table will store
        for key in ("score", "calibrated", "weights_source", "deltas", "during", "baseline", "baseline_source",
                    "excess_delay_veh_s", "window", "vehicles_affected", "meaningful", "notes"):
            self.assertIn(key, r)
        self.assertEqual(set(r["deltas"]), set(QUANTITIES))
        self.assertTrue(all(d >= 0 for d in r["deltas"].values()), "deltas measure departure toward worse only")
        self.assertEqual(json.loads(text)["window"]["seconds"], round(busy.win[1] - busy.win[0], 2))

    def test_equal_weights_are_flagged_as_uncalibrated_in_every_result(self):
        r = self.runs[SEEDS[0]][0].impact()
        self.assertFalse(r["calibrated"])
        self.assertEqual(r["weights_source"], "equal")
        self.assertAlmostEqual(r["score"], sum(r["deltas"].values()) / 5, places=2)


class BaselineSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event = Run(BUSY, 2, bus=True, wall0=MONDAY_8AM + 3 * 86400)    # the fourth weekday morning

    def history(self, days=3):
        base = Baseline(min_blocks=12, min_days=2)
        for day in range(days):                                   # ordinary 08:00-08:07 traffic on earlier days
            rd = road()
            col = TrafficStateCollector(1, [rd], baseline=base)
            ApproachSim(rd, BUSY, 50 + day).run(420, lambda t, o, d=day: col.observe(o, t, MONDAY_8AM + d * 86400 + t))
        return base

    def test_the_time_of_day_history_is_used_when_there_is_enough_of_it(self):
        r = self.event.impact(baseline=self.history())
        self.assertEqual(r["baseline_source"], "history")
        self.assertEqual(r["baseline_detail"]["days"], 3)
        self.assertTrue(r["meaningful"])
        self.assertGreater(r["score"], 0.8)
        self.assertGreater(r["excess_delay_veh_s"], 100)

    def test_without_history_it_falls_back_to_the_minutes_before_the_event(self):
        for base in (None, Baseline(min_blocks=12, min_days=2), self.history(days=1)):   # nothing / empty / one day
            r = self.event.impact(baseline=base)
            self.assertEqual(r["baseline_source"], "pre-event")
            self.assertTrue(r["meaningful"])
            self.assertGreater(r["score"], 0.8)

    def test_with_neither_the_score_is_flagged_not_meaningful_and_not_a_number(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        ApproachSim(rd, BUSY, 1).run(40, lambda t, o: col.observe(o, t, MONDAY_8AM + t))
        r = compute_event_impact(col, None, 1, rd, 20.0, 40.0, "V1")
        self.assertEqual(r["baseline_source"], "none")
        self.assertFalse(r["meaningful"])
        self.assertIsNone(r["score"])
        self.assertIn("no baseline", " ".join(r["notes"]))
        json.dumps(r, allow_nan=False)

    def test_a_window_nobody_sampled_is_not_meaningful_either(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        r = compute_event_impact(col, None, 1, rd, 500.0, 530.0, "V1")
        self.assertFalse(r["meaningful"])
        self.assertIsNone(r["score"])
        self.assertIsNone(r["during"])

    def test_the_history_slot_is_the_one_the_event_happened_in(self):
        base = self.history()
        evening = Run(BUSY, 2, bus=True, wall0=datetime(2026, 3, 5, 19, 0).timestamp())      # nothing learned at 19:00
        self.assertEqual(evening.impact(baseline=base)["baseline_source"], "pre-event")
        saturday = Run(BUSY, 2, bus=True, wall0=datetime(2026, 3, 7, 8, 0).timestamp())      # weekend: other history
        self.assertEqual(saturday.impact(baseline=base)["baseline_source"], "pre-event")


class EventWindowTests(unittest.TestCase):
    def test_the_window_runs_from_when_it_began_to_thirty_seconds_after_the_rule_fired(self):
        self.assertEqual(event_window(100.0, 115.0), (100.0, 145.0))
        self.assertEqual(event_window(100.0, 115.0, post_sec=60), (100.0, 175.0))

    def test_a_rule_that_does_not_say_when_it_began_starts_the_window_when_it_fired(self):
        self.assertEqual(event_window(None, 115.0), (115.0, 145.0))
        self.assertEqual(event_window(120.0, 115.0), (115.0, 145.0))          # never starts after it fired


# ---------------------------------------------------------------- non-negative least squares

def brute_force_nnls(A, b):
    """The optimum is the least-squares solution on some support, with every coefficient >= 0: try all supports."""
    n = A.shape[1]
    best = (np.linalg.norm(b), np.zeros(n))
    for mask in range(1, 2 ** n):
        cols = [j for j in range(n) if mask >> j & 1]
        x = np.zeros(n)
        x[cols] = np.linalg.lstsq(A[:, cols], b, rcond=None)[0]
        if (x >= -1e-12).all():
            r = np.linalg.norm(A @ x - b)
            if r < best[0] - 1e-12:
                best = (r, x)
    return best


class NNLSTests(unittest.TestCase):
    def problem(self, rng, m=None, collinear=False):
        m = m or int(rng.integers(6, 40))
        A = rng.normal(size=(m, 5))
        if collinear:
            A[:, 3] = A[:, 1] + rng.normal(0, 0.05, m)
        return A, A @ rng.normal(size=5) + rng.normal(0, 0.5, m)

    def test_it_recovers_known_nonnegative_weights_from_noisy_data(self):
        rng = np.random.default_rng(3)
        f = rng.uniform(0, 4, (300, 1))
        D = 0.5 * f + rng.uniform(0, 3, (300, 5))                 # five departures that rise together, all >= 0
        true = np.array([4.0, 0.0, 9.0, 2.0, 0.0])
        x, _ = nnls(D, D @ true + rng.normal(0, 1.0, 300))
        np.testing.assert_allclose(x, true, atol=0.3)
        self.assertTrue((x >= 0).all())

    def test_it_matches_brute_force_over_every_support(self):
        rng = np.random.default_rng(0)
        for k in range(150):
            A, b = self.problem(rng, collinear=k % 3 == 0)
            x, r = nnls(A, b)
            r0, x0 = brute_force_nnls(A, b)
            self.assertAlmostEqual(r, r0, places=8, msg=f"problem {k}")
            np.testing.assert_allclose(x, x0, atol=1e-6, err_msg=f"problem {k}")

    def test_it_satisfies_the_optimality_conditions(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            A, b = self.problem(rng, collinear=True)
            x, _ = nnls(A, b)
            g = A.T @ (b - A @ x)                              # may not improve any zero weight, and is flat elsewhere
            self.assertTrue((x >= 0).all())
            self.assertLessEqual(g[x == 0].max(initial=-1e9), 1e-7)
            self.assertLessEqual(np.abs(g[x > 0]).max(initial=0.0), 1e-7)

    def test_the_constraint_binds_where_ordinary_least_squares_would_go_negative(self):
        rng = np.random.default_rng(0)
        A = rng.uniform(0, 1, (40, 5))
        A[:, 4] = A[:, 0] + rng.normal(0, 0.01, 40)            # two columns that say the same thing
        b = A @ np.array([1.0, 2.0, 3.0, 4.0, 0.0]) + rng.normal(0, 0.5, 40)
        ols = np.linalg.lstsq(A, b, rcond=None)[0]
        self.assertLess(ols.min(), -5.0, "precondition: the unconstrained fit has a wildly negative weight")
        x, r = nnls(A, b)
        self.assertGreaterEqual(x.min(), 0.0)
        self.assertGreaterEqual(r, np.linalg.norm(A @ ols - b) - 1e-9)       # the price of the constraint

    def test_degenerate_inputs_return_zero_weights_not_errors(self):
        x, r = nnls(np.zeros((0, 5)), np.zeros(0))
        self.assertEqual((x.tolist(), r), ([0.0] * 5, 0.0))
        x, _ = nnls(np.random.default_rng(0).uniform(size=(10, 5)), np.zeros(10))
        self.assertEqual(x.tolist(), [0.0] * 5)
        a = np.random.default_rng(0).uniform(size=(10, 1))
        x, _ = nnls(np.hstack([a, a]), 3 * a.ravel())              # identical columns share the weight
        self.assertAlmostEqual(float(x.sum()), 3.0, places=6)
        self.assertGreaterEqual(x.min(), 0.0)


# ---------------------------------------------------------------- calibration

W1, W2 = np.array([10.0, 0.0, 5.0, 0.0, 20.0]), np.array([0.0, 30.0, 0.0, 10.0, 0.0])


def fake_rows(n, w, code, seed, noise=2.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.uniform(0, 5, 5).round(3)
        out.append({"code": code, "meaningful": True, "deltas": dict(zip(QUANTITIES, d.tolist())),
                    "excess_delay_veh_s": max(0.0, float(d @ w + rng.normal(0, noise)))})
    return out


class CalibrationTests(unittest.TestCase):
    def test_below_fifteen_events_the_weights_are_equal_and_say_so(self):
        w = calibrate(fake_rows(14, W1, "V1", 0))
        ws = w.for_code("V1")
        self.assertFalse(ws.calibrated)
        self.assertEqual(ws.source, "equal")
        self.assertEqual(list(ws.values.values()), [0.2] * 5)
        self.assertEqual(w.for_code("never seen"), ws)
        self.assertFalse(calibrate([]).for_code("V1").calibrated)

    def test_a_code_with_enough_events_gets_its_own_weights_and_a_thin_one_gets_the_pooled_fit(self):
        rows = fake_rows(20, W1, "V1", 1) + fake_rows(20, W2, "V7", 2) + fake_rows(6, W1, "V3", 3)
        w = calibrate(rows)
        v1, v7, v3 = w.for_code("V1"), w.for_code("V7"), w.for_code("V3")
        self.assertEqual((v1.source, v7.source, v3.source), ("code:V1", "code:V7", "pooled"))
        self.assertTrue(v1.calibrated and v7.calibrated and v3.calibrated)
        np.testing.assert_allclose([v1.values[q] for q in QUANTITIES], W1, atol=1.5)
        np.testing.assert_allclose([v7.values[q] for q in QUANTITIES], W2, atol=1.5)
        self.assertEqual(w.for_code("V99").source, "pooled")
        self.assertEqual((v1.n_events, v3.n_events), (20, 46))

    def test_codes_that_are_each_too_thin_are_pooled_when_together_they_are_enough(self):
        w = calibrate(fake_rows(9, W1, "V1", 1) + fake_rows(9, W1, "V7", 2))
        self.assertEqual({w.for_code(c).source for c in ("V1", "V7")}, {"pooled"})
        self.assertTrue(w.for_code("V1").calibrated)

    def test_fitted_weights_are_never_negative(self):
        rng = np.random.default_rng(9)
        rows = fake_rows(40, np.array([10.0, 0.0, 0.0, 5.0, 0.0]), "V1", 5, noise=6.0)
        for r in rows:                                      # make two departures nearly the same signal
            r["deltas"]["occupancy"] = r["deltas"]["queue_m"] + float(rng.normal(0, 0.1))
        for ws in [calibrate(rows).for_code("V1")]:
            self.assertGreaterEqual(min(ws.values.values()), 0.0)
            self.assertTrue(ws.calibrated)

    def test_events_with_nothing_measured_are_skipped_and_counted(self):
        rows = fake_rows(20, W1, "V1", 1) + [
            {"code": "V1", "meaningful": False, "deltas": None, "excess_delay_veh_s": None, "score": None},
            {"code": "V1", "meaningful": True, "deltas": {q: 1.0 for q in QUANTITIES}, "excess_delay_veh_s": None},
            {"code": "V1", "deltas": {q: float("nan") for q in QUANTITIES}, "excess_delay_veh_s": 5.0}]
        w = calibrate(rows)
        self.assertEqual((w.n_rows, w.skipped), (20, 3))

    def test_events_that_caused_no_delay_cannot_calibrate_anything(self):
        rows = fake_rows(30, np.zeros(5), "V1", 1, noise=0.0)
        w = calibrate(rows)
        self.assertFalse(w.for_code("V1").calibrated)
        self.assertEqual(w.for_code("V1").source, "equal")

    def test_weights_survive_json(self):
        w = calibrate(fake_rows(20, W1, "V1", 1) + fake_rows(20, W2, "V7", 2))
        back = Weights.from_dict(json.loads(json.dumps(w.to_dict(), allow_nan=False)))
        for code in ("V1", "V7", "V3"):
            self.assertEqual(back.for_code(code).values, w.for_code(code).values)
            self.assertEqual(back.for_code(code).source, w.for_code(code).source)

    def test_a_score_made_with_fitted_weights_says_it_is_calibrated_and_with_which(self):
        busy = Run(BUSY, 1, bus=True)
        w = calibrate(fake_rows(20, W1, "V1", 1))                  # enough for V1 itself, and so for the pooled fit
        r = busy.impact(weights=w)
        self.assertTrue(r["calibrated"])
        self.assertEqual(r["weights_source"], "code:V1")
        self.assertEqual(busy.impact(weights=w, code="V7")["weights_source"], "pooled")
        self.assertAlmostEqual(r["score"], sum(w.for_code("V1").values[q] * r["deltas"][q] for q in QUANTITIES),
                               delta=0.01)
        thin = calibrate(fake_rows(5, W1, "V1", 1))
        self.assertFalse(busy.impact(weights=thin)["calibrated"])

    def test_weights_fitted_to_simulated_events_predict_their_delay_in_vehicle_seconds(self):
        rows = []
        for seed in (1, 2, 3):
            for rate in (0.6, 0.9, 1.3):
                for hold in (15, 30, 60):
                    run = Run(rate, seed, bus=True, pre=200.0, post=95.0, hold=hold)
                    r = run.impact()
                    r["seed"] = seed
                    rows.append(r)
        train, test = [r for r in rows if r["seed"] != 3], [r for r in rows if r["seed"] == 3]
        w = calibrate(train)
        ws = w.for_code("V1")
        self.assertTrue(ws.calibrated)
        self.assertGreaterEqual(min(ws.values.values()), 0.0)
        self.assertGreater(ws.fit["r2"], 0.6)
        y = np.array([r["excess_delay_signed_veh_s"] for r in test])
        cal = np.array([ws.score(r["deltas"]) for r in test])
        eq = np.array([r["score"] for r in test])
        mae = float(np.abs(cal - y).mean())
        self.assertLess(mae, 0.5 * y.mean(), f"held-out error {mae:.0f} veh*s on a mean delay of {y.mean():.0f}")
        self.assertGreater(spearman(cal, y), 0.75)
        self.assertGreater(spearman(eq, y), 0.75)     # the uncalibrated score ranks about as well: the fit adds units
        print(f"\n  27 simulated events, fitted on 18, scored on 9 unseen: weights "
              f"{ {q: round(v, 1) for q, v in ws.values.items()} }, in-sample R2 {ws.fit['r2']}, held-out MAE "
              f"{mae:.0f} veh*s (mean delay {y.mean():.0f}), rho(calibrated, delay) {spearman(cal, y):.2f}, "
              f"rho(equal weights, delay) {spearman(eq, y):.2f}")


# ---------------------------------------------------------------- rank correlation

class RankTests(unittest.TestCase):
    def test_tied_values_share_the_average_rank(self):
        np.testing.assert_allclose(ranks([10, 20, 20, 30]), [1, 2.5, 2.5, 4])
        np.testing.assert_allclose(ranks([5, 5, 5]), [2, 2, 2])
        np.testing.assert_allclose(ranks([3, 1, 2, 2, 2, 9]), [5, 1, 3, 3, 3, 6])

    def test_spearman_with_ties_matches_the_hand_computed_value(self):
        # ranks a = 1, 2.5, 2.5, 4 ; b = 1, 3, 2, 4 -> centred products 2.25 + 0 + 0 + 2.25 = 4.5, sums of squares
        # 4.5 and 5.0 -> rho = 4.5 / sqrt(4.5 * 5) = 0.94868
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 3, 2, 4]), 4.5 / math.sqrt(22.5), places=10)

    def test_spearman_matches_the_textbook_tie_corrected_formula(self):
        from collections import Counter
        rng = np.random.default_rng(2)
        for _ in range(30):
            a, b = rng.integers(0, 6, 25).astype(float), rng.integers(0, 6, 25).astype(float)
            n = a.size
            ra, rb = ranks(a), ranks(b)
            # sum of the squared centred ranks, less the tie correction
            sx = (n ** 3 - n) / 12 - sum(t ** 3 - t for t in Counter(a).values()) / 12
            sy = (n ** 3 - n) / 12 - sum(t ** 3 - t for t in Counter(b).values()) / 12
            if sx > 0 and sy > 0:
                want = (sx + sy - float(((ra - rb) ** 2).sum())) / (2 * math.sqrt(sx * sy))
                self.assertAlmostEqual(spearman(a, b), want, places=9)

    def test_perfect_agreement_perfect_disagreement_and_undefined(self):
        self.assertEqual(spearman([1, 5, 9, 20], [0.1, 0.2, 7, 8]), 1.0)
        self.assertAlmostEqual(spearman([1, 5, 9, 20], [8, 7, 0.2, 0.1]), -1.0)
        self.assertTrue(math.isnan(spearman([1, 1, 1], [1, 2, 3])))      # one side constant: no ordering to agree with
        self.assertTrue(math.isnan(spearman([1.0], [2.0])))
        self.assertTrue(math.isnan(spearman([1, 2], [1, 2, 3])))

    def test_validate_reports_rho_an_interval_and_how_much_it_left_out(self):
        rng = np.random.default_rng(5)
        score = rng.uniform(0, 10, 40)
        human = np.round(score / 2 + rng.normal(0, 1.0, 40))                    # a 0-5 grade with ties and disagreement
        v = validate(list(score) + [None, float("nan")], list(human) + [3.0, 2.0])
        self.assertEqual((v["n"], v["n_excluded"]), (40, 2))
        self.assertGreater(v["rho"], 0.7)
        lo, hi = v["ci95"]
        self.assertLessEqual(lo, v["rho"])
        self.assertGreaterEqual(hi, v["rho"])
        self.assertLess(v["p_value"], 0.01)
        self.assertGreater(v["ties_human"], 10)
        json.dumps(v, allow_nan=False)

    def test_validate_finds_nothing_in_unrelated_orderings_and_declines_tiny_samples(self):
        rng = np.random.default_rng(6)
        v = validate(rng.uniform(size=30), rng.uniform(size=30))
        self.assertLess(abs(v["rho"]), 0.45)
        self.assertGreater(v["p_value"], 0.01)
        self.assertLess(v["ci95"][0], 0.0 + 0.2)
        tiny = validate([1.0, 2.0], [1.0, 2.0])
        self.assertIsNone(tiny["rho"])
        self.assertEqual(tiny["n"], 2)


# ---------------------------------------------------------------- priority

class PriorityTests(unittest.TestCase):
    CAL_HIGH, CAL_MED = 200.0, 100.0                       # veh*s, calibrated: bands are (30, 150)

    def p(self, score, count=0, days=None, code="V1", calibrated=True):
        return priority(code, score, count, days, calibrated=calibrated)

    def test_history_never_changes_the_band_and_adds_at_most_its_cap(self):
        for score in (0.0, 10.0, 29.0, 31.0, 100.0, 149.0, 151.0, 400.0):
            first = self.p(score)
            for count in (1, 2, 5, 20, 1000):
                for days in (0, 1, 30, 365, None):
                    got = self.p(score, count, days)
                    self.assertEqual(got["severity"], first["severity"], (score, count, days))
                    self.assertGreaterEqual(got["priority"], first["priority"])
                    self.assertLessEqual(got["priority"] - first["priority"], im.HISTORY_MAX_POINTS + 0.1)

    def test_history_can_never_lift_a_case_above_a_high_impact_first_offence(self):
        high_first = self.p(self.CAL_HIGH)
        worst_history = self.p(149.99, count=10_000, days=0)             # top of the medium band, repeat offender
        self.assertEqual(worst_history["severity"], "medium")
        self.assertLess(worst_history["priority"], high_first["priority"])
        # ... and not even a high-impact first offence JUST over the line is beaten by a medium case with history
        edge = self.p(150.0)
        self.assertGreater(edge["priority"], self.p(149.99, 10_000, 0)["priority"])
        for score in np.linspace(0, 149.99, 60):
            for count in (0, 3, 50):
                self.assertLess(self.p(score, count, 0)["priority"], edge["priority"])

    def test_priority_rises_with_impact_and_falls_with_the_age_of_the_last_event(self):
        scores = [self.p(s, 2, 10)["priority"] for s in np.linspace(0, 400, 80)]
        self.assertEqual(scores, sorted(scores))
        ages = [self.p(100.0, 3, d)["priority"] for d in (0, 5, 30, 90, 365)]
        self.assertEqual(ages, sorted(ages, reverse=True))
        counts = [self.p(100.0, c, 10)["priority"] for c in (0, 1, 2, 3, 6)]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(self.p(100.0, 3, 0)["priority"], self.p(100.0, 3, 400)["priority"])

    def test_history_is_modest(self):
        base = self.p(100.0)["priority"]
        self.assertLess(self.p(100.0, 50, 0)["priority"] - base, 10.0)
        self.assertEqual(self.p(100.0, 0, 0)["priority"], base)
        self.assertAlmostEqual(self.p(100.0, 3, 20000)["priority"], base, delta=0.01)    # long ago: nearly nothing

    def test_the_reason_says_what_drove_it_and_that_history_is_only_context(self):
        r = self.p(180.0, 2, 12)
        self.assertEqual(r["severity"], "high")
        self.assertIn("180", r["reason"])
        self.assertIn("calibrated", r["reason"])
        self.assertIn("never into a higher one", r["reason"])
        self.assertTrue(r["advisory"])
        self.assertIn("uncalibrated", self.p(2.0, calibrated=False)["reason"])
        json.dumps(r, allow_nan=False)

    def test_the_two_kinds_of_score_use_their_own_bands(self):
        self.assertEqual(self.p(2.0, calibrated=False)["severity"], "medium")        # 2 on the mean-departure scale
        self.assertEqual(self.p(2.0, calibrated=True)["severity"], "low")            # 2 vehicle-seconds is nothing
        self.assertEqual(self.p(4.0, calibrated=False)["severity"], "high")

    def test_an_unknown_impact_gets_its_severity_from_the_rule_alone(self):
        r = self.p(None, 5, 3)
        self.assertEqual(r["severity"], "low")
        self.assertIn("impact unknown", r["reason"])
        self.assertLess(r["priority"], 33.0)
        self.assertEqual(self.p(float("nan"))["severity"], "low")
        for code in ("V4", "V10", "V11"):                                      # dangerous driving is never "low"
            floor = self.p(None, 0, None, code=code)
            self.assertEqual(floor["severity"], "medium")
            self.assertEqual(self.p(0.0, 0, None, code=code)["severity"], "medium")
            self.assertLess(self.p(None, 50, 0, code=code)["priority"], self.p(self.CAL_HIGH)["priority"])
        self.assertEqual(self.p(self.CAL_HIGH, code="V4")["severity"], "high")      # the floor never lowers a band

    def test_it_accepts_the_impact_result_itself(self):
        impact = {"score": 80.0, "calibrated": True, "meaningful": True}
        self.assertEqual(priority("V1", impact, 0, None), priority("V1", 80.0, 0, None, calibrated=True))
        unmeaningful = {"score": None, "calibrated": True, "meaningful": False}
        self.assertIn("unknown", priority("V1", unmeaningful, 0, None)["reason"])
        self.assertEqual(priority("V1", {"score": 2.0, "calibrated": False, "meaningful": True}, 0, None)["severity"],
                         "medium")


# ---------------------------------------------------------------- storage and glue

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "s.db")
        im.ensure_schema(self.store)
        self.cid = self.store.add_camera("cam", "file", "x.mp4")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def violation(self, plate, wall, status="pending", code="V1"):
        rec = {"code": code, "violation": "Bus stopping", "track_id": 1, "class": "Bus", "class_confidence": 1.0,
               "video_time_s": 10.0, "started_s": 0.0, "wall_time": wall, "zone": "Test Rd", "areas": "", "kph": 0.0,
               "details": {}}
        vid = self.store.add_violation(self.cid, None, None, rec, None, None, None)
        self.store.set_plate(vid, "read", "", plate)
        if status != "pending":
            self.store.review(vid, status)
        return vid

    def impact(self, score=40.0, delay=120.0, calibrated=True, meaningful=True):
        return {"score": score, "meaningful": meaningful, "calibrated": calibrated, "camera_id": self.cid,
                "zone": "Test Rd", "code": "V1", "baseline_source": "pre-event", "excess_delay_veh_s": delay,
                "deltas": {q: 1.0 for q in QUANTITIES}}

    def test_the_schema_is_idempotent_and_does_not_touch_other_tables(self):
        before = {r["name"] for r in self.store.q("SELECT name FROM sqlite_master WHERE type='table'")}
        im.ensure_schema(self.store)
        im.ensure_schema(self.store)
        after = {r["name"] for r in self.store.q("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(before, after)
        self.assertTrue({"traffic_baseline", "event_impact", "impact_weights"} <= after)

    def test_an_impact_round_trips_and_keeps_the_reviewers_grade_when_rescored(self):
        vid = self.violation("DHAKA METRO-GA 12-3456", "2026-03-02 08:05:00")
        im.save_impact(self.store, vid, self.impact())
        back = im.load_impact(self.store, vid)
        self.assertEqual(back["score"], 40.0)
        self.assertIsNone(back["human_severity"])
        im.set_human_severity(self.store, vid, 4)
        im.save_impact(self.store, vid, self.impact(score=55.0))               # the window was re-scored later
        again = im.load_impact(self.store, vid)
        self.assertEqual((again["score"], again["human_severity"]), (55.0, 4.0))
        self.assertIsNone(im.load_impact(self.store, vid + 100))

    def test_an_impact_goes_when_its_case_goes(self):
        vid = self.violation("A 1", "2026-03-02 08:05:00")
        im.save_impact(self.store, vid, self.impact())
        self.store.x("DELETE FROM violations WHERE id=?", (vid,))
        self.assertIsNone(im.load_impact(self.store, vid))

    def test_calibration_and_validation_read_back_only_what_they_can_use(self):
        ids = [self.violation(f"P {i}", "2026-03-02 08:05:00") for i in range(4)]
        im.save_impact(self.store, ids[0], self.impact(10.0, 30.0))
        im.save_impact(self.store, ids[1], self.impact(50.0, 90.0))
        im.save_impact(self.store, ids[2], self.impact(None, None, meaningful=False))
        im.save_impact(self.store, ids[3], self.impact(70.0, None))
        self.assertEqual(len(im.calibration_rows(self.store)), 2)
        im.set_human_severity(self.store, ids[0], 1)
        im.set_human_severity(self.store, ids[1], 3)
        im.set_human_severity(self.store, ids[2], 5)
        self.assertEqual(im.validation_pairs(self.store), ([10.0, 50.0], [1.0, 3.0]))

    def test_weights_are_persisted_and_the_default_is_equal_and_uncalibrated(self):
        self.assertFalse(im.load_weights(self.store).for_code("V1").calibrated)
        im.save_weights(self.store, calibrate(fake_rows(20, W1, "V1", 1)))
        w = im.load_weights(self.store)
        self.assertEqual(w.for_code("V1").source, "code:V1")
        im.save_weights(self.store, Weights())
        self.assertFalse(im.load_weights(self.store).for_code("V1").calibrated, "the latest fit wins")

    def test_plate_history_counts_only_earlier_confirmed_events_of_the_same_plate(self):
        plate = "DHAKA METRO-GA 12-3456"
        self.violation(plate, "2026-02-20 08:00:00", "confirmed")
        self.violation("dhaka metro ga 12 3456", "2026-01-01 09:00:00", "confirmed")      # same plate, read differently
        self.violation(plate, "2026-02-25 08:00:00", "pending")
        self.violation(plate, "2026-02-26 08:00:00", "rejected")
        self.violation("DHAKA METRO-GA 99-9999", "2026-02-28 08:00:00", "confirmed")      # someone else
        self.violation(plate, "2026-03-09 08:00:00", "confirmed")                          # AFTER the case in question
        now = self.violation(plate, "2026-03-02 08:05:00")
        count, days = im.plate_history(self.store, plate, exclude_id=now, before=datetime(2026, 3, 2, 8, 5))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(days, 10 + 5 / 1440, places=3)
        self.assertEqual(im.plate_history(self.store, "", exclude_id=now), (0, None))
        self.assertEqual(im.plate_history(self.store, "NO SUCH PLATE 1"), (0, None))

    def test_the_priority_of_a_case_combines_its_stored_impact_with_its_plates_history(self):
        plate = "DHAKA METRO-GA 12-3456"
        self.violation(plate, "2026-02-20 08:00:00", "confirmed")
        self.violation(plate, "2026-02-27 08:00:00", "confirmed")
        vid = self.violation(plate, "2026-03-02 08:05:00")
        first = self.violation("OTHER 1-1", "2026-03-02 08:06:00")
        im.save_impact(self.store, vid, self.impact(score=100.0))
        im.save_impact(self.store, first, self.impact(score=100.0))
        with_history = im.priority_for(self.store, self.store.violation(vid))
        alone = im.priority_for(self.store, self.store.violation(first))
        self.assertEqual((with_history["prior_confirmed"], alone["prior_confirmed"]), (2, 0))
        self.assertEqual(with_history["severity"], alone["severity"])
        self.assertGreater(with_history["priority"], alone["priority"])
        self.assertAlmostEqual(with_history["days_since_last"], 3.0, places=1)
        unscored = self.violation("NO IMPACT 1", "2026-03-02 08:07:00")
        self.assertIn("unknown", im.priority_for(self.store, self.store.violation(unscored))["reason"])


class TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.event = Run(BUSY, 1, bus=True)

    def test_an_event_is_scored_once_its_window_has_closed_and_not_before(self):
        run = self.event
        t0, t1 = run.win
        stopped = run.sim.stopper["hold"]["stopped_at"]
        tracker = ImpactTracker(run.col, None, 1)
        self.assertTrue(tracker.schedule(7, run.rd, "V1", stopped, stopped + 15, run.tid))
        self.assertEqual(tracker.poll(t1 - 1.0), [])
        done = tracker.poll(t1)
        self.assertEqual([vid for vid, _ in done], [7])
        impact = done[0][1]
        self.assertTrue(impact["meaningful"])
        self.assertEqual(impact["excluded_tids"], [run.tid])
        self.assertEqual(impact["window"]["t0"], round(t0, 2))
        self.assertEqual(tracker.poll(t1 + 100), [], "scored once")

    def test_events_off_the_road_zones_cannot_be_scored(self):
        tracker = ImpactTracker(self.event.col, None, 1)
        self.assertFalse(tracker.schedule(1, None, "V3", 0.0, 10.0))
        self.assertEqual(tracker.flush(), [])

    def test_flush_scores_what_is_left_at_the_end_of_the_run(self):
        tracker = ImpactTracker(self.event.col, None, 1)
        tracker.schedule(1, self.event.rd, "V1", 300.0, 315.0)
        tracker.schedule(2, self.event.rd, "V1", 305.0, 320.0)
        self.assertEqual([vid for vid, _ in tracker.flush()], [1, 2])
        self.assertEqual(tracker.pending, [])

    def test_a_scoring_failure_is_recorded_in_the_result_and_never_raised(self):
        tracker = ImpactTracker(None, None, 1)                     # a broken collector
        tracker.schedule(3, self.event.rd, "V1", 0.0, 10.0)
        (vid, impact), = tracker.poll(1000.0)
        self.assertEqual(vid, 3)
        self.assertFalse(impact["meaningful"])
        self.assertIsNone(impact["score"])
        self.assertIn("AttributeError", impact["error"])
        json.dumps(impact, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
