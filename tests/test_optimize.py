"""The OPTIMIZE stage: network, routing, signal control and the simulator that validates them.

Everything here is a statement about software behaviour on synthetic networks and canned responses. Nothing
here has driven a real signal, and nothing calls the network: the Google provider is given a canned
transport. A simulator that favours max pressure over fixed time proves the controller is sound under THIS
simulator's assumptions (FIFO stop-line queues, saturation-flow discharge, a fixed free-flow time), not that
it would behave so on a Dhaka junction - these tests pin down behaviour that must not regress, they are not
evidence about the street.
"""

import inspect
import io
import json
import math
import os
import re
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.optimize import google_traffic as G  # noqa: E402
from surveillance.optimize import network as N  # noqa: E402
from surveillance.optimize import routing as R  # noqa: E402
from surveillance.optimize import signals as S  # noqa: E402
from surveillance.optimize import sim as M  # noqa: E402

NOW = 1_000_000.0


def snap(queue_m=0.0, speed_kph=40.0, occupancy=0.1, density=10.0, wait_s=0.0, **kw):
    """What TrafficStateCollector.snapshot() reports for one zone."""
    return {"queue_m": queue_m, "speed_kph": speed_kph, "occupancy": occupancy, "density": density,
            "wait_s": wait_s, "samples": 30, "vehicles": 3, "t": 12.0, "lanes": 2, "flow": 0, **kw}


def line_net():
    """A -> B -> C and a slower direct A -> C, no coordinates: a graph to check routing by hand.

    A-B 1000 m and B-C 1000 m at 36 km/h are 100 s each (200 s together); A-C is 1500 m at 18 km/h, 300 s."""
    net = N.Network("line")
    for n in "ABC":
        net.add_node(n, N.UNCONTROLLED)
    net.add_link("AB", "A", "B", 1000, free_flow_kph=36)
    net.add_link("BC", "B", "C", 1000, free_flow_kph=36)
    net.add_link("AC", "A", "C", 1500, free_flow_kph=18)
    return net


class GeometryTests(unittest.TestCase):
    def test_one_degree_of_latitude_is_about_111_km(self):
        self.assertAlmostEqual(N.haversine_m(0, 0, 1, 0), 111_195, delta=10)

    def test_a_thousandth_of_a_degree_in_dhaka_is_111_m_north_and_102_m_east(self):
        self.assertAlmostEqual(N.haversine_m(23.8, 90.36, 23.801, 90.36), 111.2, delta=0.2)
        self.assertAlmostEqual(N.haversine_m(23.8, 90.36, 23.8, 90.361), 101.7, delta=0.2)

    def test_haversine_is_symmetric_and_zero_for_the_same_point(self):
        self.assertEqual(N.haversine_m(23.8, 90.4, 23.8, 90.4), 0.0)
        self.assertAlmostEqual(N.haversine_m(23.8, 90.4, 23.9, 90.5), N.haversine_m(23.9, 90.5, 23.8, 90.4))

    def test_projection_round_trips_and_agrees_with_haversine(self):
        origin = (23.8, 90.36)
        x, y = N.project(23.803, 90.364, origin)
        lat, lon = N.unproject(x, y, origin)
        self.assertAlmostEqual(lat, 23.803, places=9)
        self.assertAlmostEqual(lon, 90.364, places=9)
        self.assertAlmostEqual(math.hypot(x, y), N.haversine_m(23.8, 90.36, 23.803, 90.364), delta=0.5)


class BuildingTests(unittest.TestCase):
    def test_link_length_is_computed_from_lat_lon_when_not_given(self):
        net = N.Network()
        net.add_node("A", lat=23.8, lon=90.36)
        net.add_node("B", lat=23.801, lon=90.36)
        link = net.add_link("AB", "A", "B")
        self.assertAlmostEqual(link.length_m, N.haversine_m(23.8, 90.36, 23.801, 90.36))
        self.assertAlmostEqual(link.length_m, 111.2, delta=0.2)

    def test_an_explicit_length_wins_over_the_coordinates(self):
        net = N.Network()
        net.add_node("A", lat=23.8, lon=90.36)
        net.add_node("B", lat=23.801, lon=90.36)
        self.assertEqual(net.add_link("AB", "A", "B", 500).length_m, 500)

    def test_a_link_with_neither_length_nor_coordinates_is_refused(self):
        net = N.Network()
        net.add_node("A")
        net.add_node("B")
        with self.assertRaises(ValueError):
            net.add_link("AB", "A", "B")

    def test_half_a_coordinate_pair_is_refused(self):
        with self.assertRaises(ValueError):
            N.Network().add_node("A", lat=23.8)
        with self.assertRaises(ValueError):
            N.Network().add_node("A", lat=123.0, lon=90.0)

    def test_the_default_saturation_flow_is_below_the_textbook_figure_and_documented(self):
        self.assertLess(N.DHAKA_SAT_FLOW_VPH_PER_LANE, N.TEXTBOOK_SAT_FLOW_VPH_PER_LANE)
        self.assertEqual(N.TEXTBOOK_SAT_FLOW_VPH_PER_LANE, 1800.0)
        net = line_net()
        self.assertEqual(net.links["AB"].sat_flow_vph_per_lane, N.DHAKA_SAT_FLOW_VPH_PER_LANE)
        self.assertIn("rickshaw", N.__doc__)

    def test_turns_are_classified_from_the_change_of_heading(self):
        net = N.Network()
        net.add_node("S", N.BOUNDARY, x=0, y=-100)
        net.add_node("J", x=0, y=0)
        for n, (x, y) in {"N": (0, 100), "W": (-100, 0), "E": (100, 0)}.items():
            net.add_node(n, N.BOUNDARY, x=x, y=y)
        net.add_link("SJ", "S", "J", 100)
        for n in "NWE":
            net.add_link(f"J{n}", "J", n, 100)
        self.assertEqual(net.turn("SJ", "JN"), "through")
        self.assertEqual(net.turn("SJ", "JW"), "left")       # heading north, turning west is to the left
        self.assertEqual(net.turn("SJ", "JE"), "right")
        self.assertEqual(net.arm("SJ"), "S")

    def test_a_grid_has_the_expected_structure(self):
        g = N.grid(3, 3)
        self.assertEqual(len(g.signals()), 9)
        self.assertEqual(len(g.nodes), 9 + 12)
        self.assertEqual(len(g.links), 12 + 12 + 24)
        self.assertEqual(len(g.movements), 9 * 4 * 3)                 # four approaches, three non-U exits
        self.assertEqual([p.name for p in g.phases["J11"]], ["N", "E", "S", "W"])
        for lid in g.in_links("J11"):
            self.assertAlmostEqual(sum(g.movements[m].ratio for m in g.movements_from(lid)), 1.0)
        g.validate()

    def test_a_grid_can_be_placed_on_the_map(self):
        g = N.grid(2, 2, block_m=300, lat0=23.80, lon0=90.36)
        a, b = g.nodes["J00"], g.nodes["J01"]
        self.assertTrue(a.geolocated and b.geolocated)
        self.assertAlmostEqual(N.haversine_m(a.lat, a.lon, b.lat, b.lon), 300, delta=0.5)

    def test_from_points_builds_two_way_links_with_haversine_lengths(self):
        net = N.from_points(
            [("A", 23.800, 90.360), ("B", 23.800, 90.364), ("C", 23.803, 90.364), ("D", 23.800, 90.368)],
            [("A", "B"), ("B", "C"), {"from": "B", "to": "D", "lanes": 3, "name": "Mirpur Road"}])
        self.assertEqual(sorted(net.links), ["A-B", "B-A", "B-C", "B-D", "C-B", "D-B"])
        self.assertAlmostEqual(net.links["A-B"].length_m, N.haversine_m(23.8, 90.36, 23.8, 90.364))
        self.assertEqual(net.links["B-D"].lanes, 3)
        self.assertEqual(net.links["D-B"].name, "Mirpur Road")
        self.assertEqual({k: v.kind for k, v in net.nodes.items()},
                         {"A": N.BOUNDARY, "B": N.SIGNAL, "C": N.BOUNDARY, "D": N.BOUNDARY})
        self.assertEqual(len(net.phases["B"]), 3)               # one per approach; B is a T-junction
        # driving north from the west arm, B-C is a left turn and B-D (east) a right one
        self.assertEqual(net.movements["A-B>B-C"].turn, "left")
        self.assertEqual(net.movements["A-B>B-D"].turn, "through")

    def test_from_points_maps_each_direction_to_its_own_camera_zone(self):
        net = N.from_points(
            [("A", 23.800, 90.360), ("B", 23.800, 90.364), ("C", 23.800, 90.368)],
            [{"from": "A", "to": "B", "camera_zone": "east", "camera_id": 1,
              "back_camera_zone": "west", "back_camera_id": 1}, ("B", "C")])
        self.assertEqual((net.links["A-B"].camera_zone, net.links["A-B"].camera_id), ("east", "1"))
        self.assertEqual(net.links["B-A"].camera_zone, "west")
        self.assertIsNone(net.links["B-C"].camera_zone)

    def test_from_points_rejects_a_road_to_an_unknown_point(self):
        with self.assertRaises(ValueError):
            N.from_points([("A", 23.8, 90.36)], [("A", "B")])


class PersistenceTests(unittest.TestCase):
    def test_a_network_survives_a_json_round_trip(self):
        net = N.from_points(
            [("A", 23.800, 90.360), ("B", 23.800, 90.364), ("C", 23.803, 90.364)],
            [{"from": "A", "to": "B", "camera_zone": "z1", "camera_id": "7"}, ("B", "C")])
        net.set_turn_ratios({"A-B>B-C": 3, "A-B>B-A": 1})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "net.json"
            net.save(path)
            back = N.Network.load(path)
        self.assertEqual(sorted(back.links), sorted(net.links))
        self.assertEqual(back.links["A-B"].camera_zone, "z1")
        self.assertEqual(back.links["A-B"].camera_id, "7")
        self.assertAlmostEqual(back.nodes["B"].lat, 23.8)
        self.assertAlmostEqual(back.links["A-B"].length_m, net.links["A-B"].length_m, places=1)
        self.assertEqual({k: [p.movements for p in v] for k, v in back.phases.items()},
                         {k: [p.movements for p in v] for k, v in net.phases.items()})
        for mid, m in net.movements.items():
            self.assertAlmostEqual(back.movements[mid].ratio, m.ratio, places=5)

    def test_saving_never_writes_live_state_so_google_numbers_cannot_reach_disk(self):
        net = N.grid(2, 2)
        net.set_link_state("J00-J01", N.GOOGLE, 99.0, NOW, ratio=3.3)
        net.set_link_state("J00-J10", N.CAMERA, 88.0, NOW)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "net.json"
            net.save(path)
            text = path.read_text()
        for word in ('"state"', "google", "travel_time_s", "updated_at", "99.0"):
            self.assertNotIn(word, text)

    def test_the_display_dict_carries_each_links_source_and_time(self):
        net = N.grid(2, 2)
        net.set_link_state("J00-J01", N.GOOGLE, 99.0, NOW, ratio=3.3)
        row = {l["id"]: l for l in net.to_dict(include_state=True)["links"]}
        self.assertEqual(row["J00-J01"]["source"], "google")
        self.assertEqual(row["J00-J01"]["updated_at"], NOW)
        self.assertEqual(row["J00-J10"]["source"], "free-flow")
        self.assertIsNone(row["J00-J10"]["updated_at"])
        json.dumps(net.to_dict(include_state=True), allow_nan=False)

    def test_a_hand_written_spec_may_leave_out_lengths_movements_and_phases(self):
        spec = {"nodes": [{"id": "A", "kind": "boundary", "lat": 23.8, "lon": 90.36},
                          {"id": "J", "lat": 23.8, "lon": 90.364},
                          {"id": "B", "kind": "boundary", "lat": 23.8, "lon": 90.368}],
                "links": [{"id": "AJ", "from": "A", "to": "J"}, {"id": "JB", "from": "J", "to": "B"},
                          {"id": "BJ", "from": "B", "to": "J"}, {"id": "JA", "from": "J", "to": "A"}],
                "zones": {"cam1 east": "AJ"}}
        net = N.Network.from_dict(spec)
        self.assertGreater(net.links["AJ"].length_m, 100)
        self.assertIn("AJ>JB", net.movements)
        self.assertEqual(len(net.phases["J"]), 2)
        self.assertEqual(net.links["AJ"].camera_zone, "cam1 east")

    def test_validation_lists_every_problem_at_once(self):
        spec = {"nodes": [{"id": "A", "kind": "boundary", "x": 0, "y": 0}, {"id": "J", "x": 100, "y": 0},
                          {"id": "B", "kind": "boundary", "x": 200, "y": 0}],
                "links": [{"id": "AJ", "from": "A", "to": "J", "length_m": -5},
                          {"id": "JB", "from": "J", "to": "B", "length_m": 100}],
                "movements": [{"from": "AJ", "to": "JB"}],
                "phases": {"J": [{"name": "p", "movements": ["nope"]}]}}
        with self.assertRaises(ValueError) as e:
            N.Network.from_dict(spec)
        text = str(e.exception)
        self.assertIn("must be positive", text)
        self.assertIn("unknown movement nope", text)
        self.assertIn("never green", text)

    def test_a_camera_zone_mapped_to_two_links_is_refused_unless_cameras_differ(self):
        net = N.grid(2, 2)
        net.links["J00-J01"].camera_zone = "z"
        net.links["J00-J10"].camera_zone = "z"
        with self.assertRaises(ValueError):
            net.validate()
        net.links["J00-J01"].camera_id = "1"
        net.links["J00-J10"].camera_id = "2"
        net.validate()


class CameraStateTests(unittest.TestCase):
    def setUp(self):
        self.net = N.grid(3, 3)              # J00-J01 is 300 m, 2 lanes, 40 km/h: 27 s free flow
        self.link = self.net.links["J00-J01"]

    def measure(self, **kw):
        return N.link_measurement(self.link, snap(**kw))

    def test_a_free_flowing_approach_costs_its_free_flow_time(self):
        self.assertAlmostEqual(self.link.free_flow_s, 27.0)
        self.assertAlmostEqual(self.measure()["travel_time_s"], 27.0)

    def test_a_queue_adds_the_time_to_discharge_it(self):
        # 70 m back of queue at 7 m per vehicle is 10 per lane; 1200 vph per lane clears one every 3 s.
        m = self.measure(queue_m=70)
        self.assertAlmostEqual(m["queue_veh"], 20.0)
        self.assertAlmostEqual(m["travel_time_s"], (300 - 70) / (40 / 3.6) + 30.0, places=6)

    def test_a_longer_queue_never_makes_the_link_faster(self):
        times = [self.measure(queue_m=q)["travel_time_s"] for q in (0, 14, 35, 70, 140, 300)]
        self.assertEqual(times, sorted(times))
        self.assertGreater(times[-1], 4 * times[0])

    def test_a_queue_is_charged_once_not_again_through_its_own_standstill_in_the_mean_speed(self):
        crawling = self.measure(queue_m=70, speed_kph=1.0)["travel_time_s"]
        flowing = self.measure(queue_m=70, speed_kph=30.0)["travel_time_s"]
        self.assertAlmostEqual(crawling, flowing)

    def test_with_no_queue_the_measured_mean_speed_applies_to_the_whole_link(self):
        self.assertAlmostEqual(self.measure(speed_kph=20.0)["travel_time_s"], 54.0)

    def test_a_stopped_vehicle_gives_a_finite_time_through_the_speed_floor(self):
        t = self.measure(speed_kph=0.0)["travel_time_s"]
        self.assertTrue(math.isfinite(t))
        self.assertAlmostEqual(t, 300 / (N.SPEED_FLOOR_KPH / 3.6))

    def test_a_measured_speed_above_free_flow_does_not_speed_the_link_up(self):
        self.assertAlmostEqual(self.measure(speed_kph=90.0)["travel_time_s"], 27.0)

    def test_standing_time_floors_the_queue_delay_when_the_queue_is_not_moving(self):
        # 14 m is 2 per lane: 6 s to discharge if it moved, but these vehicles have stood for two minutes
        m = self.measure(queue_m=14, wait_s=120.0)
        self.assertGreaterEqual(m["travel_time_s"], 120.0)

    def test_with_no_speed_the_density_per_lane_gives_one(self):
        # Greenshields: free-flow speed falls to zero at 1000 / 7 = 143 vehicles per km per lane
        m = self.measure(speed_kph=None, density=100.0)
        self.assertEqual(m["speed_source"], "from density")
        self.assertAlmostEqual(m["speed_kph"], 40 * (1 - 100 / (1000 / 7)), places=6)
        self.assertEqual(self.measure(speed_kph=None, density=None)["speed_source"], "free flow (no speed)")

    def test_a_queue_summed_over_lanes_is_divided_by_the_lane_count(self):
        total = N.link_measurement(self.link, snap(queue_m=140), queue_m_is_total=True)
        self.assertAlmostEqual(total["queue_m"], 70.0)
        self.assertAlmostEqual(total["travel_time_s"], self.measure(queue_m=70)["travel_time_s"])

    def test_a_queue_longer_than_the_link_is_cut_at_the_link(self):
        self.assertAlmostEqual(self.measure(queue_m=5000)["queue_m"], 300.0)

    def test_the_lower_saturation_flow_makes_the_same_queue_cost_more(self):
        slow = self.net.copy()
        slow.links["J00-J01"].sat_flow_vph_per_lane = 600.0
        a = self.measure(queue_m=70)["travel_time_s"]
        b = N.link_measurement(slow.links["J00-J01"], snap(queue_m=70))["travel_time_s"]
        self.assertAlmostEqual(b - a, 30.0)

    def test_updating_records_the_source_and_when(self):
        out = N.update_from_state(self.net, {"J00-J01": snap(queue_m=70)}, now=NOW)
        link = out.links["J00-J01"]
        self.assertEqual((link.source, link.updated_at), ("camera", NOW))
        self.assertGreater(link.travel_time_s, link.free_flow_s)
        self.assertEqual(out.links["J00-J10"].source, "free-flow")
        self.assertIsNone(out.links["J00-J10"].updated_at)

    def test_the_sample_time_the_collector_gives_is_used_when_present(self):
        out = N.update_from_state(self.net, {"J00-J01": snap(wall=NOW - 30)}, now=NOW)
        self.assertEqual(out.links["J00-J01"].updated_at, NOW - 30)

    def test_updating_returns_a_copy_and_leaves_the_input_untouched(self):
        out = N.update_from_state(self.net, {"J00-J01": snap(queue_m=70)}, now=NOW)
        self.assertIsNot(out, self.net)
        self.assertEqual(self.net.links["J00-J01"].source, "free-flow")
        same = N.update_from_state(self.net, {"J00-J01": snap(queue_m=70)}, now=NOW, inplace=True)
        self.assertIs(same, self.net)
        self.assertEqual(self.net.links["J00-J01"].source, "camera")

    def test_a_snapshot_with_no_samples_says_nothing_rather_than_free_flow(self):
        empty = {"queue_m": None, "speed_kph": None, "occupancy": None, "density": None, "wait_s": None,
                 "samples": 0, "vehicles": 0, "t": None, "lanes": 2, "flow": 0}
        out = N.update_from_state(self.net, {"J00-J01": empty, "nope": snap()}, now=NOW)
        self.assertEqual(out.links["J00-J01"].source, "free-flow")
        self.assertEqual(len(out.notes), 2)

    def test_an_empty_road_that_was_observed_is_free_flow_from_a_camera(self):
        quiet = snap(queue_m=0.0, speed_kph=None, occupancy=0.0, density=0.0, vehicles=0)
        out = N.update_from_state(self.net, {"J00-J01": quiet}, now=NOW)
        self.assertEqual(out.links["J00-J01"].source, "camera")
        self.assertAlmostEqual(out.links["J00-J01"].travel_time_s, 27.0)

    def test_a_camera_measurement_is_fresh_only_for_a_couple_of_minutes(self):
        out = N.update_from_state(self.net, {"J00-J01": snap()}, now=NOW)
        link = out.links["J00-J01"]
        self.assertTrue(link.fresh_camera(NOW + 60))
        self.assertFalse(link.fresh_camera(NOW + 600))
        self.assertFalse(out.links["J00-J10"].fresh_camera(NOW))


class ZoneMappingTests(unittest.TestCase):
    def setUp(self):
        self.net = N.grid(3, 3)
        self.net.links["J00-J01"].camera_zone = "eastbound"
        self.net.links["J01-J00"].camera_zone = "westbound"

    def test_links_take_the_zones_they_are_mapped_to(self):
        out = N.from_traffic_state(self.net, {"eastbound": snap(queue_m=70), "westbound": snap()}, now=NOW)
        self.assertEqual(out.links["J00-J01"].source, "camera")
        self.assertGreater(out.links["J00-J01"].travel_time_s, out.links["J01-J00"].travel_time_s)
        self.assertEqual(out.notes, [])

    def test_snapshots_keyed_by_camera_work_too_and_camera_id_tells_equal_zone_names_apart(self):
        self.net.links["J00-J01"].camera_id = "1"
        self.net.links["J01-J00"].camera_zone = "eastbound"          # same zone name, other camera
        self.net.links["J01-J00"].camera_id = "2"
        out = N.from_traffic_state(
            self.net, {1: {"eastbound": snap(queue_m=70)}, 2: {"eastbound": snap()}}, now=NOW)
        self.assertGreater(out.links["J00-J01"].travel_time_s, 40)
        self.assertAlmostEqual(out.links["J01-J00"].travel_time_s, out.links["J01-J00"].free_flow_s)

    def test_zones_that_map_to_no_link_are_listed_not_dropped(self):
        out = N.from_traffic_state(self.net, {"eastbound": snap(), "ghost": snap()}, now=NOW)
        self.assertIn("zone 'ghost' is not mapped to a link", out.notes)
        self.assertTrue(any("westbound" in n for n in out.notes))      # mapped, but no data arrived

    def test_two_cameras_on_one_approach_give_the_worst_queue_and_the_mean_speed(self):
        merged = N.merge_states([snap(queue_m=20, speed_kph=10, wait_s=4), snap(queue_m=50, speed_kph=30, wait_s=1)])
        self.assertEqual((merged["queue_m"], merged["speed_kph"], merged["wait_s"]), (50, 20, 4))


# ---------------------------------------------------------------- Google traffic (canned responses only)

class Canned:
    """A transport that returns canned responses (or raises them) and remembers every request.

    The real transport is never constructed in these tests, so nothing here can reach the network."""

    def __init__(self, *responses, fn=None):
        self.calls, self.responses, self.fn = [], list(responses), fn

    def __call__(self, url, headers, body):
        self.calls.append((url, dict(headers), body))
        if self.fn:
            return self.fn(url, headers, body)
        r = self.responses[min(len(self.calls), len(self.responses)) - 1]
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def route_response(traffic="180s", static="120s", dist=400, intervals=None):
    r = {"duration": traffic, "staticDuration": static, "distanceMeters": dist}
    if intervals is not None:
        r["travelAdvisory"] = {"speedReadingIntervals": intervals}
    return {"routes": [r]}


JAMMED = [{"endPolylinePointIndex": 4, "speed": "NORMAL"},
          {"startPolylinePointIndex": 4, "endPolylinePointIndex": 8, "speed": "TRAFFIC_JAM"},
          {"startPolylinePointIndex": 8, "endPolylinePointIndex": 10, "speed": "SLOW"}]


def geo_pair():
    """Two geolocated junctions about 397 m apart with a link each way, 36 km/h so free flow is ~39.7 s."""
    net = N.Network("pair")
    net.add_node("A", N.BOUNDARY, lat=23.8000, lon=90.3600)
    net.add_node("B", N.BOUNDARY, lat=23.8000, lon=90.3639)
    net.add_link("AB", "A", "B", free_flow_kph=36)
    net.add_link("BA", "B", "A", free_flow_kph=36)
    return net


def geo_chain(n):
    """n+1 geolocated nodes in a line about 100 m apart, one link between each pair."""
    net = N.Network("chain")
    for i in range(n + 1):
        net.add_node(f"N{i}", N.SIGNAL, lat=23.8, lon=90.36 + 0.001 * i)
    for i in range(n):
        net.add_link(f"L{i}", f"N{i}", f"N{i+1}", free_flow_kph=36)
    return net


def provider(*responses, **kw):
    transport = kw.pop("transport", None) or Canned(*responses)
    kw.setdefault("clock", Clock())
    return G.GoogleTraffic(api_key=kw.pop("api_key", "TEST-KEY"), transport=transport, **kw), transport


class GoogleParsingTests(unittest.TestCase):
    def test_durations_arrive_as_strings_of_seconds_and_become_numbers(self):
        self.assertEqual(G.parse_duration("123s"), 123.0)
        self.assertEqual(G.parse_duration("0.5s"), 0.5)
        self.assertEqual(G.parse_duration(" 7s "), 7.0)
        self.assertEqual(G.parse_duration(12), 12.0)
        for bad in (None, "", "abc", "12", "12m", True, float("nan"), [], {}):
            self.assertIsNone(G.parse_duration(bad))

    def test_speed_intervals_become_jam_and_slow_shares_and_a_missing_start_is_zero(self):
        jam, slow = G.speed_shares(JAMMED)           # 4 normal, 4 jam, 2 slow polyline points
        self.assertAlmostEqual(jam, 0.4)
        self.assertAlmostEqual(slow, 0.2)
        self.assertEqual(G.speed_shares([]), (None, None))
        self.assertEqual(G.speed_shares(None), (None, None))
        self.assertEqual(G.speed_shares([{"speed": "TRAFFIC_JAM"}]), (None, None))   # zero-length span

    def test_a_route_response_becomes_a_ratio(self):
        r = G.parse_route(route_response("180s", "120s", 400, JAMMED), NOW)
        self.assertEqual((r["traffic_s"], r["free_s"], r["ratio"], r["distance_m"]), (180.0, 120.0, 1.5, 400.0))
        self.assertAlmostEqual(r["jam_fraction"], 0.4)
        self.assertEqual(r["fetched_at"], NOW)

    def test_a_response_with_no_route_or_no_static_duration_is_no_answer(self):
        self.assertIsNone(G.parse_route({}, NOW))
        self.assertIsNone(G.parse_route({"routes": []}, NOW))
        self.assertIsNone(G.parse_route(route_response("180s", "0s"), NOW))
        self.assertIsNone(G.parse_route({"routes": [{"duration": "180s"}]}, NOW))

    def test_a_matrix_response_defaults_omitted_zero_indices_and_skips_missing_routes(self):
        resp = [{"duration": "90s", "staticDuration": "60s", "distanceMeters": 500, "condition": "ROUTE_EXISTS"},
                {"originIndex": 0, "destinationIndex": 1, "condition": "ROUTE_NOT_FOUND"},
                {"destinationIndex": 2, "duration": "30s", "staticDuration": "30s", "distanceMeters": 90,
                 "condition": "ROUTE_EXISTS"},
                {"destinationIndex": 3, "duration": "30s", "staticDuration": "30s", "status": {"code": 3}}]
        got = G.parse_matrix(resp, NOW)
        self.assertEqual([(e["origin"], e["destination"], e["ratio"]) for e in got], [(0, 0, 1.5), (0, 2, 1.0)])


class GoogleProviderTests(unittest.TestCase):
    def test_the_computeRoutes_request_follows_the_documented_shape(self):
        g, t = provider(route_response())
        g.link_travel_times(geo_pair(), ["AB"])
        (url, headers, body), = t.calls
        self.assertEqual(url, "https://routes.googleapis.com/directions/v2:computeRoutes")
        self.assertEqual(headers["X-Goog-Api-Key"], "TEST-KEY")
        self.assertEqual(headers["X-Goog-FieldMask"],
                         "routes.duration,routes.staticDuration,routes.distanceMeters,"
                         "routes.travelAdvisory.speedReadingIntervals")
        self.assertEqual(body["origin"]["location"]["latLng"], {"latitude": 23.8, "longitude": 90.36})
        self.assertEqual(body["destination"]["location"]["latLng"], {"latitude": 23.8, "longitude": 90.3639})
        self.assertEqual((body["travelMode"], body["routingPreference"], body["extraComputations"]),
                         ("DRIVE", "TRAFFIC_AWARE", ["TRAFFIC_ON_POLYLINE"]))
        json.dumps(body)

    def test_the_key_goes_in_a_header_and_never_in_the_url(self):
        g, t = provider(route_response())
        g.link_travel_times(geo_pair(), ["AB"])
        self.assertNotIn("TEST-KEY", t.calls[0][0])
        self.assertNotIn("TEST-KEY", json.dumps(t.calls[0][2]))

    def test_link_travel_times_parses_durations_ratio_and_jam_fraction(self):
        g, _ = provider(route_response("180s", "120s", 400, JAMMED))
        out = g.link_travel_times(geo_pair(), ["AB"])
        self.assertEqual(set(out), {"AB"})
        r = out["AB"]
        self.assertEqual((r["traffic_s"], r["free_s"], r["ratio"]), (180.0, 120.0, 1.5))
        self.assertAlmostEqual(r["jam_fraction"], 0.4)
        self.assertEqual(r["fetched_at"], NOW)

    def test_only_links_with_both_end_points_located_are_asked(self):
        net = geo_pair()
        net.add_node("C", N.BOUNDARY)                       # no lat/lon
        net.add_link("BC", "B", "C", 300)
        g, t = provider(route_response())
        out = g.link_travel_times(net)                      # all links
        self.assertEqual(set(out), {"AB", "BA"})
        self.assertEqual(len(t.calls), 2)

    def test_results_are_cached_within_the_ttl_and_fetched_again_after_it(self):
        clock = Clock()
        g, t = provider(route_response(), clock=clock, ttl_s=600)
        net = geo_pair()
        first = g.link_travel_times(net, ["AB"])
        clock.t += 599
        again = g.link_travel_times(net, ["AB"])
        self.assertEqual(len(t.calls), 1)
        self.assertEqual(again["AB"]["fetched_at"], first["AB"]["fetched_at"])    # the age stays honest
        self.assertEqual(g.stats["cache_hits"], 1)
        clock.t += 2
        g.link_travel_times(net, ["AB"])
        self.assertEqual(len(t.calls), 2)

    def test_a_pair_with_no_route_is_remembered_so_it_is_not_paid_for_again(self):
        g, t = provider({})
        net = geo_pair()
        self.assertEqual(g.link_travel_times(net, ["AB"]), {})
        self.assertEqual(g.link_travel_times(net, ["AB"]), {})
        self.assertEqual(len(t.calls), 1)

    def test_the_call_budget_is_respected_and_replenishes(self):
        clock = Clock()
        g, t = provider(route_response(), clock=clock, max_calls_per_min=2)
        net = geo_chain(4)
        with self.assertLogs("surveillance.optimize.google", level="WARNING"):
            out = g.link_travel_times(net)
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(len(out), 2)
        self.assertEqual(g.stats["budget_skips"], 1)
        clock.t += 61
        out = g.link_travel_times(net)                      # the two cached answers, then two new calls
        self.assertEqual(len(t.calls), 4)
        self.assertEqual(len(out), 4)

    def test_the_element_budget_limits_a_large_matrix(self):
        g, t = provider([], max_elements_per_min=5)
        pts = [(23.8, 90.36 + 0.001 * i) for i in range(3)]
        with self.assertLogs("surveillance.optimize.google", level="WARNING"):
            self.assertEqual(g.route_matrix(pts, pts), [])           # 9 elements > 5
        self.assertEqual(t.calls, [])

    def test_without_a_key_it_returns_nothing_and_never_calls_the_transport(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(G.ENV_KEY, None)
            t = Canned(route_response())
            g = G.GoogleTraffic(transport=t)
            self.assertFalse(g.configured)
            with self.assertLogs("surveillance.optimize.google", level="WARNING") as cm:
                self.assertEqual(g.link_travel_times(geo_pair()), {})
            self.assertIsNone(g.route((23.8, 90.36), (23.8, 90.364)))
            self.assertEqual(g.route_matrix([(23.8, 90.36)], [(23.8, 90.364)]), [])
            self.assertEqual(t.calls, [])
            self.assertIn(G.ENV_KEY, "\n".join(cm.output))

    def test_the_key_is_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {G.ENV_KEY: "ENV-KEY"}):
            t = Canned(route_response())
            g = G.GoogleTraffic(transport=t, clock=Clock())
            g.link_travel_times(geo_pair(), ["AB"])
            self.assertEqual(t.calls[0][1]["X-Goog-Api-Key"], "ENV-KEY")

    def test_an_http_error_returns_empty_and_logs_why_without_the_key(self):
        err = urllib.error.HTTPError(G.ROUTES_URL, 403, "Forbidden", {},
                                     io.BytesIO(b'{"error": {"message": "API key not valid"}}'))
        g, _ = provider(err)
        with self.assertLogs("surveillance.optimize.google", level="WARNING") as cm:
            self.assertEqual(g.link_travel_times(geo_pair()), {})
        text = "\n".join(cm.output)
        self.assertIn("403", text)
        self.assertIn("API key not valid", text)
        self.assertNotIn("TEST-KEY", text)
        self.assertEqual(g.stats["errors"], 1)
        self.assertNotIn("TEST-KEY", repr(g.last_error))

    def test_a_network_failure_or_a_timeout_is_also_just_an_empty_answer(self):
        for exc in (urllib.error.URLError("no route to host"), TimeoutError("timed out"), ValueError("bad json")):
            g, _ = provider(exc)
            with self.assertLogs("surveillance.optimize.google", level="WARNING"):
                self.assertEqual(g.link_travel_times(geo_pair()), {})

    def test_an_error_object_inside_a_200_response_is_a_failure(self):
        g, _ = provider({"error": {"code": 429, "message": "quota exceeded"}})
        with self.assertLogs("surveillance.optimize.google", level="WARNING") as cm:
            self.assertEqual(g.link_travel_times(geo_pair()), {})
        self.assertIn("quota exceeded", "\n".join(cm.output))

    def test_the_first_failure_stops_the_pass_so_a_bad_key_is_not_retried_for_every_link(self):
        g, t = provider(urllib.error.URLError("down"))
        with self.assertLogs("surveillance.optimize.google", level="WARNING"):
            g.link_travel_times(geo_chain(5))
        self.assertEqual(len(t.calls), 1)

    def test_what_was_fetched_before_a_failure_is_kept(self):
        g, t = provider(route_response(), urllib.error.URLError("down"))
        with self.assertLogs("surveillance.optimize.google", level="WARNING"):
            out = g.link_travel_times(geo_chain(3))
        self.assertEqual(list(out), ["L0"])

    def test_the_matrix_method_asks_once_per_origin_junction_and_has_no_jam_share(self):
        net = geo_pair()
        resp = [{"duration": "90s", "staticDuration": "60s", "distanceMeters": 400, "condition": "ROUTE_EXISTS"}]
        g, t = provider(resp)
        out = g.link_travel_times(net, method="matrix")
        self.assertEqual(set(out), {"AB", "BA"})
        self.assertEqual(len(t.calls), 2)                               # A->[B] and B->[A]
        url, headers, body = t.calls[0]
        self.assertEqual(url, "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix")
        self.assertEqual(headers["X-Goog-FieldMask"],
                         "originIndex,destinationIndex,duration,staticDuration,distanceMeters,condition")
        self.assertEqual(len(body["origins"]), 1)
        self.assertEqual(len(body["destinations"]), 1)
        self.assertEqual(body["routingPreference"], "TRAFFIC_AWARE")
        self.assertEqual(out["AB"]["ratio"], 1.5)
        self.assertIsNone(out["AB"]["jam_fraction"])

    def test_status_reports_use_and_limits_but_never_the_key(self):
        g, _ = provider(route_response())
        g.link_travel_times(geo_pair(), ["AB"])
        st = g.status()
        self.assertTrue(st["configured"])
        self.assertEqual((st["calls"], st["elements"], st["cached"]), (1, 1, 1))
        self.assertNotIn("TEST-KEY", json.dumps(st))
        self.assertNotIn("TEST-KEY", repr(g))

    def test_nothing_google_returns_is_written_to_disk(self):
        src = inspect.getsource(G)
        for token in (r"(?<![A-Za-z_.])open\(", r"\.write_text", r"\.write_bytes", r"\.write\(", "pickle",
                      "sqlite3", "shelve", r"json\.dump\("):
            self.assertIsNone(re.search(token, src), token)


class ApplyGoogleTests(unittest.TestCase):
    def test_google_fills_links_with_no_camera_and_marks_the_source(self):
        net = geo_pair()
        g, _ = provider(route_response("180s", "120s", 400, JAMMED))
        out = G.apply_google(net, g, now=NOW)
        link = out.links["AB"]
        self.assertEqual((link.source, link.updated_at), ("google", NOW))
        self.assertAlmostEqual(link.travel_time_s, link.free_flow_s * 1.5)
        self.assertAlmostEqual(link.state["jam_fraction"], 0.4)
        self.assertEqual(net.links["AB"].source, "free-flow")           # the input is untouched

    def test_google_scales_the_links_own_free_flow_time_not_googles_absolute_duration(self):
        g, _ = provider(route_response("1800s", "1200s", 400))          # Google's own numbers are far larger
        link = G.apply_google(geo_pair(), g, now=NOW).links["AB"]
        self.assertAlmostEqual(link.travel_time_s, link.free_flow_s * 1.5)
        self.assertLess(link.travel_time_s, 100)

    def test_fresh_camera_data_is_never_overridden_and_google_is_not_even_asked_about_it(self):
        net = N.update_from_state(geo_pair(), {"AB": snap(queue_m=0, speed_kph=36.0)}, now=NOW)
        g, t = provider(route_response("1000s", "100s", 400))           # a ratio of 10 if it were believed
        out = G.apply_google(net, g, now=NOW + 30)
        ab = out.links["AB"]
        self.assertEqual(ab.source, "camera")
        self.assertEqual(ab.updated_at, NOW)
        self.assertAlmostEqual(ab.travel_time_s, ab.free_flow_s)
        self.assertEqual(out.links["BA"].source, "google")              # the gap is filled
        self.assertEqual(len(t.calls), 1)                               # and only the gap was paid for

    def test_stale_camera_data_is_replaced_when_google_has_an_answer(self):
        net = N.update_from_state(geo_pair(), {"AB": snap(queue_m=0, speed_kph=36.0)}, now=NOW)
        g, _ = provider(route_response("150s", "100s", 400))
        out = G.apply_google(net, g, now=NOW + 600)
        self.assertEqual(out.links["AB"].source, "google")

    def test_stale_camera_data_is_kept_when_google_has_nothing(self):
        net = N.update_from_state(geo_pair(), {"AB": snap(queue_m=70)}, now=NOW)
        g, _ = provider(urllib.error.URLError("down"))
        with self.assertLogs("surveillance.optimize.google", level="WARNING"):
            out = G.apply_google(net, g, now=NOW + 600)
        self.assertEqual(out.links["AB"].source, "camera")
        self.assertAlmostEqual(out.links["AB"].travel_time_s, net.links["AB"].travel_time_s)

    def test_a_ratio_below_one_never_makes_a_link_faster_than_free_flow(self):
        g, _ = provider(route_response("80s", "100s", 400))
        link = G.apply_google(geo_pair(), g, now=NOW).links["AB"]
        self.assertAlmostEqual(link.travel_time_s, link.free_flow_s)
        self.assertEqual(link.state["source"], "google")

    def test_an_absurd_ratio_is_cut(self):
        g, _ = provider(route_response("100000s", "100s", 400))
        link = G.apply_google(geo_pair(), g, now=NOW, max_ratio=8).links["AB"]
        self.assertAlmostEqual(link.travel_time_s, link.free_flow_s * 8)

    def test_a_route_that_is_another_road_is_not_used(self):
        g, _ = provider(route_response("180s", "120s", 1500))           # 1.5 km for a 397 m link
        out = G.apply_google(geo_pair(), g, now=NOW)
        self.assertEqual(out.links["AB"].source, "free-flow")
        self.assertTrue(any("another road" in n for n in out.notes))

    def test_without_a_key_the_network_comes_back_as_it_was(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(G.ENV_KEY, None)
            t = Canned(route_response())
            with self.assertLogs("surveillance.optimize.google", level="WARNING"):
                out = G.apply_google(geo_pair(), G.GoogleTraffic(transport=t), now=NOW)
            self.assertEqual({l.source for l in out.links.values()}, {"free-flow"})
            self.assertEqual(t.calls, [])
        self.assertEqual({l.source for l in G.apply_google(geo_pair(), None).links.values()}, {"free-flow"})

    def test_a_network_with_google_numbers_still_saves_without_them(self):
        g, _ = provider(route_response("180s", "120s", 400))
        out = G.apply_google(geo_pair(), g, now=NOW)
        with tempfile.TemporaryDirectory() as d:
            out.save(Path(d) / "n.json")
            text = (Path(d) / "n.json").read_text()
        self.assertNotIn("google", text)
        self.assertNotIn('"ratio"', text)

    def test_the_planner_sees_the_slower_link_after_google_fills_it(self):
        g, _ = provider(route_response("240s", "120s", 400))            # ratio 2: the link takes twice as long
        out = G.apply_google(geo_pair(), g, now=NOW)
        self.assertAlmostEqual(out.travel_times()["AB"], 2 * geo_pair().travel_times()["AB"])


# ---------------------------------------------------------------- routing

def textbook_graph():
    """S -> T with a greedy trap. 1 m/s links, so a link's length in metres is its time in seconds.

    S-D 1 and D-T 20 are the bait (the cheapest first step leads to the dearest finish); S-A 7, S-B 2, B-A 3,
    A-C 1, B-C 8, C-T 4, A-T 12. Hand-checked: S-B-A-C-T costs 2+3+1+4 = 10; S-A-C-T 12; S-B-C-T 14;
    S-B-A-T 17; S-A-T 19; S-D-T 21."""
    net = N.Network("textbook")
    for n in "SABCDT":
        net.add_node(n, N.UNCONTROLLED)
    for lid, a, b, t in [("SD", "S", "D", 1), ("DT", "D", "T", 20), ("SA", "S", "A", 7), ("SB", "S", "B", 2),
                         ("BA", "B", "A", 3), ("AC", "A", "C", 1), ("BC", "B", "C", 8), ("CT", "C", "T", 4),
                         ("AT", "A", "T", 12)]:
        net.add_link(lid, a, b, t, free_flow_kph=3.6)
    return net


class ShortestPathTests(unittest.TestCase):
    def test_dijkstra_finds_the_cheapest_path_on_a_hand_checked_graph_and_is_not_greedy(self):
        r = R.shortest_path(textbook_graph(), "S", "T")
        self.assertEqual(r.nodes, list("SBACT"))
        self.assertEqual(r.links, ["SB", "BA", "AC", "CT"])
        self.assertAlmostEqual(r.time_s, 10.0)
        self.assertAlmostEqual(r.length_m, 10.0)
        self.assertAlmostEqual(r.free_flow_s, 10.0)

    def test_the_line_graph_prefers_two_fast_links_to_one_slow_one(self):
        r = R.shortest_path(line_net(), "A", "C")
        self.assertEqual(r.links, ["AB", "BC"])
        self.assertAlmostEqual(r.time_s, 200.0)

    def test_an_unreachable_destination_is_none_and_unknown_nodes_are_refused(self):
        self.assertIsNone(R.shortest_path(textbook_graph(), "T", "S"))          # links are directed
        with self.assertRaises(ValueError):
            R.shortest_path(textbook_graph(), "S", "nowhere")

    def test_going_nowhere_is_an_empty_route(self):
        r = R.shortest_path(textbook_graph(), "S", "S")
        self.assertEqual((r.nodes, r.links, r.time_s), (["S"], [], 0.0))

    def test_the_same_network_always_gives_the_same_route(self):
        net = N.grid(3, 3)
        routes = {tuple(R.shortest_path(net, "W0", "E2").links) for _ in range(5)}
        self.assertEqual(len(routes), 1)

    def test_a_closed_link_is_never_used_and_closing_every_way_in_means_no_route(self):
        net = textbook_graph()
        r = R.shortest_path(net, "S", "T", extra_delay={"SB": math.inf})
        self.assertNotIn("SB", r.links)
        self.assertAlmostEqual(r.time_s, 12.0)                                  # S-A-C-T
        self.assertIsNone(R.shortest_path(net, "S", "T", avoid={"CT", "AT", "DT"}))

    def test_avoid_keeps_links_out_of_the_route(self):
        r = R.shortest_path(textbook_graph(), "S", "T", avoid={"BA"})
        self.assertNotIn("BA", r.links)
        self.assertAlmostEqual(r.time_s, 12.0)

    def test_boundary_nodes_are_terminals_and_never_a_shortcut_through_the_area(self):
        net = N.Network()
        net.add_node("A", N.SIGNAL)
        net.add_node("X", N.BOUNDARY)                       # the edge of the map on a through road
        net.add_node("B", N.SIGNAL)
        net.add_link("AX", "A", "X", 100)
        net.add_link("XB", "X", "B", 100)
        net.add_link("AB", "A", "B", 1000)
        self.assertEqual(R.shortest_path(net, "A", "B").links, ["AB"])
        self.assertEqual(R.shortest_path(net, "A", "X").links, ["AX"])          # ending there is fine
        self.assertEqual(R.shortest_path(net, "X", "B").links, ["XB"])          # and so is starting there

    def test_weights_replace_the_networks_own_times_for_those_links(self):
        r = R.shortest_path(textbook_graph(), "S", "T", weights={"SB": 100})
        self.assertAlmostEqual(r.time_s, 12.0)
        self.assertEqual(r.breakdown[0]["link"], "SA")
        r2 = R.shortest_path(textbook_graph(), "S", "T", weights={"SB": 100, "SA": 100, "SD": 100})
        self.assertEqual({b["source"] for b in r2.breakdown if b["link"] in ("SB", "SA", "SD")}, {"override"})

    def test_the_reverse_tree_agrees_with_a_forward_search_from_every_origin(self):
        import random
        rng = random.Random(7)
        net = N.grid(3, 3)
        weights = {lid: rng.uniform(10, 100) for lid in net.links}
        for dest in ("E1", "S0", "J11"):
            cost, nxt = R.tree_to(net, dest, weights=weights)
            for origin in net.nodes:
                fwd = R.shortest_path(net, origin, dest, weights=weights)
                if fwd is None:
                    self.assertNotIn(origin, cost)
                    continue
                self.assertAlmostEqual(cost[origin], fwd.time_s, places=6)
                path = R.follow_tree(net, nxt, origin, dest)
                self.assertAlmostEqual(R.route_from_links(net, path, weights=weights).time_s, fwd.time_s, places=6)

    def test_following_a_tree_from_a_node_that_cannot_reach_the_destination_gives_none(self):
        net = textbook_graph()
        cost, nxt = R.tree_to(net, "S")
        self.assertIsNone(R.follow_tree(net, nxt, "T", "S"))

    def test_links_that_are_not_consecutive_are_refused(self):
        with self.assertRaises(ValueError):
            R.route_from_links(textbook_graph(), ["SB", "CT"])


class RouteBreakdownTests(unittest.TestCase):
    def test_each_link_reports_its_time_and_where_that_time_came_from(self):
        net = N.grid(3, 3)
        net = N.update_from_state(net, {"J10-J11": snap(queue_m=70)}, now=NOW)
        net.set_link_state("J11-J12", N.GOOGLE, 40.0, NOW - 60)
        r = R.shortest_path(net, "W1", "E1")
        by = {b["link"]: b for b in r.breakdown}
        self.assertEqual(r.links, ["W1-J10", "J10-J11", "J11-J12", "J12-E1"])
        self.assertEqual(by["J10-J11"]["source"], "camera")
        self.assertEqual(by["J11-J12"]["source"], "google")
        self.assertEqual(by["J11-J12"]["updated_at"], NOW - 60)
        self.assertEqual(by["W1-J10"]["source"], "free-flow")
        self.assertIsNone(by["W1-J10"]["updated_at"])
        self.assertEqual(r.source_mix(), {"free-flow": 2, "camera": 1, "google": 1})
        self.assertAlmostEqual(by["J12-E1"]["end_s"], r.time_s, places=0)
        self.assertGreater(r.delay_s, 0)
        json.dumps(r.to_dict(), allow_nan=False)

    def test_a_congested_camera_state_raises_link_time_and_changes_the_route(self):
        net = N.grid(3, 3)
        free = R.shortest_path(net, "W1", "E1")
        self.assertIn("J10-J11", free.links)                       # straight along the middle street
        # a queue the whole length of J10-J11: 300 m of 2 lanes, ~43 vehicles per lane at 3 s each
        net.links["J10-J11"].camera_zone = "middle street"
        jammed = N.from_traffic_state(net, {"middle street": snap(queue_m=300.0, speed_kph=2.0, wait_s=60.0)},
                                      now=NOW)
        self.assertGreater(jammed.links["J10-J11"].travel_time_s, 4 * net.links["J10-J11"].free_flow_s)
        detour = R.shortest_path(jammed, "W1", "E1")
        self.assertNotIn("J10-J11", detour.links)
        stuck = R.route_from_links(jammed, free.links)
        self.assertLess(detour.time_s, stuck.time_s)               # going round really is faster
        self.assertGreater(detour.time_s, free.time_s)             # but not as fast as the road was
        self.assertEqual(detour.breakdown[0]["source"], "free-flow")


class AlternativesTests(unittest.TestCase):
    def test_alternatives_start_with_the_shortest_and_are_distinct_and_never_faster(self):
        alts = R.k_alternatives(textbook_graph(), "S", "T", k=3, max_stretch=3.0, max_overlap=0.95)
        self.assertGreaterEqual(len(alts), 2)
        self.assertEqual(alts[0].links, ["SB", "BA", "AC", "CT"])
        self.assertEqual(len({tuple(a.links) for a in alts}), len(alts))
        times = [a.time_s for a in alts]
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(times[0], 10.0)
        self.assertTrue(all(t >= 10.0 for t in times))

    def test_alternatives_report_real_times_not_the_penalised_ones(self):
        net = textbook_graph()
        for a in R.k_alternatives(net, "S", "T", k=4, penalty=3.0, max_stretch=3.0, max_overlap=0.95):
            self.assertAlmostEqual(a.time_s, sum(net.links[l].travel_time_s for l in a.links))

    def test_a_route_much_slower_than_the_best_is_not_offered_as_an_alternative(self):
        alts = R.k_alternatives(textbook_graph(), "S", "T", k=5, max_stretch=1.3, max_overlap=1.0)
        self.assertTrue(all(a.time_s <= 13.0 + 1e-9 for a in alts))

    def test_a_corridor_with_no_genuine_alternative_returns_just_one_route(self):
        net = N.Network()
        for n in "ABC":
            net.add_node(n, N.UNCONTROLLED)
        net.add_link("AB", "A", "B", 100)
        net.add_link("BC", "B", "C", 100)
        self.assertEqual(len(R.k_alternatives(net, "A", "C", k=3)), 1)

    def test_on_a_grid_there_are_several_different_ways_across(self):
        alts = R.k_alternatives(N.grid(3, 3), "W0", "E2", k=3, max_overlap=0.9)
        self.assertGreaterEqual(len(alts), 2)
        self.assertEqual(len({tuple(a.links) for a in alts}), len(alts))


class IncidentRoutingTests(unittest.TestCase):
    def test_an_incident_penalty_reroutes_and_reports_the_saving(self):
        out = R.reroute_around(line_net(), "A", "C", {"AB": 150.0})     # via B would be 350 s; direct is 300
        self.assertEqual(out["route"].links, ["AC"])
        self.assertAlmostEqual(out["route"].time_s, 300.0)
        self.assertEqual(out["unaware"].links, ["AB", "BC"])
        self.assertAlmostEqual(out["unaware"].time_s, 350.0)             # timed WITH the incident
        self.assertAlmostEqual(out["saving_s"], 50.0)
        self.assertEqual(out["avoided"], ["AB", "BC"])

    def test_an_incident_too_small_to_matter_leaves_the_route_alone(self):
        out = R.reroute_around(line_net(), "A", "C", {"AB": 50.0})       # 250 s via B still beats 300 s direct
        self.assertEqual(out["route"].links, ["AB", "BC"])
        self.assertAlmostEqual(out["route"].time_s, 250.0)
        self.assertAlmostEqual(out["route"].extra_delay_s, 50.0)
        self.assertAlmostEqual(out["saving_s"], 0.0)
        self.assertEqual(out["avoided"], [])

    def test_a_closed_road_reroutes_and_has_no_finite_saving(self):
        out = R.reroute_around(line_net(), "A", "C", {"AB": math.inf})
        self.assertEqual(out["route"].links, ["AC"])
        self.assertTrue(out["unaware_closed"])
        self.assertIsNone(out["saving_s"])
        self.assertIsNone(R.reroute_around(line_net(), "A", "C", {"AB": math.inf, "AC": math.inf}))

    def test_an_incident_on_an_unknown_link_is_refused(self):
        with self.assertRaises(ValueError):
            R.reroute_around(line_net(), "A", "C", {"ZZ": 10.0})

    def test_incident_delay_is_part_of_the_routes_breakdown(self):
        out = R.reroute_around(line_net(), "A", "C", {"AB": 50.0})
        row = out["route"].breakdown[0]
        self.assertEqual((row["link"], row["extra_delay_s"]), ("AB", 50.0))

    def test_impact_excess_delay_becomes_seconds_per_vehicle_in_either_shape(self):
        self.assertAlmostEqual(R.extra_delay_from_impact({"veh_s": 600.0, "vehicles": 12, "affected": 9}), 50.0)
        self.assertAlmostEqual(R.extra_delay_from_impact(
            {"excess_delay_veh_s": 600.0, "vehicles_seen": 20, "vehicles_affected": 9}), 30.0)
        self.assertAlmostEqual(R.extra_delay_from_impact({"veh_s": 90.0, "vehicles": 0, "affected": 3}), 30.0)

    def test_impact_with_no_delay_or_no_figure_adds_nothing_and_a_tiny_sample_cannot_close_a_road(self):
        self.assertEqual(R.extra_delay_from_impact({"veh_s": 0.0, "vehicles": 5}), 0.0)
        self.assertEqual(R.extra_delay_from_impact({}), 0.0)
        self.assertEqual(R.extra_delay_from_impact({"veh_s": None}), 0.0)
        self.assertEqual(R.extra_delay_from_impact({"veh_s": 5000.0, "vehicles": 1}, cap_s=300), 300.0)

    def test_an_impact_estimate_can_drive_the_reroute(self):
        delay = R.extra_delay_from_impact({"veh_s": 1800.0, "vehicles": 9})        # 200 s per vehicle
        out = R.reroute_around(line_net(), "A", "C", {"AB": delay})
        self.assertEqual(out["route"].links, ["AC"])


# ---------------------------------------------------------------- signal control

def junction():
    """One signalised junction J00, approaches N0 E0 S0 W0; phases N, E, S, W in that order (one approach each)."""
    return N.grid(1, 1)


class Hold(S.Controller):
    """Always shows one phase (or all red when None)."""

    def __init__(self, phase):
        self.phase = phase

    def decide(self, node, state, t):
        return self.phase


class MaxPressureTests(unittest.TestCase):
    def setUp(self):
        self.net = junction()
        self.w = self.net.phases["J00"].index(next(p for p in self.net.phases["J00"] if p.name == "W"))

    def queues(self, **q):
        return S.QueueState(self.net, {f"{k}0-J00": v for k, v in q.items()})

    def test_it_picks_the_phase_with_the_greater_pressure_after_a_clearance_interval(self):
        c = S.MaxPressureController(self.net, min_green=10, clearance_s=4)
        st = self.queues(N=2, E=2, S=2, W=20)
        self.assertEqual(c.decide("J00", st, 0), 0)                       # starts on the first phase
        self.assertEqual(c.decide("J00", st, 10), None)                   # min green over: clear, then switch
        self.assertEqual([c.decide("J00", st, t) for t in (11, 13)], [None, None])
        self.assertEqual(c.decide("J00", st, 14), self.w)
        self.assertEqual(c.decide("J00", st, 40), self.w)                 # and stays while it is still the worst

    def test_it_respects_the_minimum_green_however_big_the_other_queue(self):
        c = S.MaxPressureController(self.net, min_green=15)
        st = self.queues(W=500)
        self.assertEqual([c.decide("J00", st, t) for t in range(0, 15)], [0] * 15)
        self.assertIsNone(c.decide("J00", st, 15))

    def test_equal_pressures_do_not_make_it_flicker(self):
        c = S.MaxPressureController(self.net, min_green=1)
        st = self.queues(N=5, E=5, S=5, W=5)
        self.assertEqual({c.decide("J00", st, t) for t in range(60)}, {0})

    def test_a_maximum_green_hands_the_junction_on_even_if_the_current_phase_is_still_busiest(self):
        c = S.MaxPressureController(self.net, min_green=5, max_green=20, clearance_s=0)
        st = self.queues(N=100, E=3)
        seen = [c.decide("J00", st, t) for t in range(60)]
        self.assertEqual(seen[19], 0)
        self.assertNotEqual(seen[20], 0)

    def test_a_full_downstream_link_lowers_a_movements_pressure(self):
        net = N.grid(1, 2)                                                # J00 -> J01 along the street
        c = S.MaxPressureController(net)
        mid = "W0-J00>J00-J01"
        free = S.QueueState(net, {"W0-J00": 10})
        blocked = S.QueueState(net, {"W0-J00": 10, "J00-J01": 30})
        self.assertGreater(c.movement_weight(mid, free, 0), c.movement_weight(mid, blocked, 0))
        self.assertAlmostEqual(c.movement_weight(mid, free, 0), 10 * net.movements[mid].ratio)

    def test_pressure_is_weighted_by_the_saturation_flow(self):
        net = junction()
        net.links["W0-J00"].lanes = 4                                      # a wider approach discharges faster
        c = S.MaxPressureController(net)
        st = S.QueueState(net, {"W0-J00": 5, "N0-J00": 5})
        p = dict(zip([ph.name for ph in net.phases["J00"]], c.pressures("J00", st)))
        self.assertAlmostEqual(p["W"], 2 * p["N"])

    def test_smoothing_keeps_one_noisy_reading_from_switching_the_phase(self):
        calm = S.QueueState(self.net, {"N0-J00": 10, "W0-J00": 2})
        spike = S.QueueState(self.net, {"N0-J00": 10, "W0-J00": 14})
        raw = S.MaxPressureController(self.net, min_green=1, clearance_s=0)
        smooth = S.MaxPressureController(self.net, min_green=1, clearance_s=0, smoothing=0.9)
        for c in (raw, smooth):
            for t in range(5):
                c.decide("J00", calm, t)
        self.assertEqual(raw.decide("J00", spike, 5), self.w)
        self.assertEqual(smooth.decide("J00", spike, 5), 0)

    def test_the_phases_it_recommends_follow_the_queues_and_fill_the_cycle(self):
        rec = S.recommend_timings(self.net, self.queues(N=1, E=1, S=1, W=40), cycle_s=90, clearance_s=4, min_green=7)["J00"]
        greens = {p["name"]: p["green_s"] for p in rec["phases"]}
        self.assertGreater(greens["W"], 3 * greens["N"])
        self.assertGreaterEqual(min(greens.values()), 7 - 1e-6)
        self.assertAlmostEqual(sum(greens.values()) + rec["lost_s"], rec["cycle_s"], places=0)
        self.assertEqual(rec["demand"], "high")

    def test_with_no_queues_the_recommendation_is_equal_splits(self):
        rec = S.recommend_timings(self.net, S.QueueState(self.net))["J00"]
        self.assertEqual(len({p["green_s"] for p in rec["phases"]}), 1)
        self.assertEqual(rec["demand"], "none")

    def test_the_recommendation_can_be_made_from_camera_measurements(self):
        net = N.update_from_state(junction(), {"W0-J00": snap(queue_m=70)}, now=NOW)
        rec = S.recommend_timings(net)["J00"]
        self.assertEqual(max(rec["phases"], key=lambda p: p["green_s"])["name"], "W")
        json.dumps(rec, allow_nan=False)


class FixedTimeTests(unittest.TestCase):
    def test_it_cycles_through_the_phases_with_a_clearance_between_each(self):
        net = junction()
        c = S.FixedTimeController(net, cycle_s=80, clearance_s=4)
        seq = [c.decide("J00", None, t) for t in range(80)]
        self.assertEqual([k for k, _ in __import__("itertools").groupby(seq)], [0, None, 1, None, 2, None, 3, None])
        self.assertEqual(seq.count(None), 16)
        self.assertEqual([seq.count(i) for i in range(4)], [16] * 4)
        self.assertEqual(c.decide("J00", None, 80), seq[0])                # and around again

    def test_splits_follow_the_flows_it_is_tuned_to(self):
        net = junction()
        c = S.FixedTimeController.from_flows(net, {"W0-J00": 1200, "N0-J00": 150, "E0-J00": 150, "S0-J00": 150})
        seq = [c.decide("J00", None, t) for t in range(int(c._plan["J00"][1]))]
        names = [p.name for p in net.phases["J00"]]
        self.assertEqual(max(range(4), key=seq.count), names.index("W"))
        self.assertGreater(seq.count(names.index("W")), 3 * seq.count(names.index("N")))

    def test_bad_splits_are_refused(self):
        with self.assertRaises(ValueError):
            S.FixedTimeController(junction(), splits={"J00": [1, 1]})


class EmergencyPriorityTests(unittest.TestCase):
    def setUp(self):
        self.net = N.grid(3, 3)
        self.route = ["W1-J10", "J10-J11", "J11-J12", "J12-E1"]

    def ev(self, idx=0, pos=250.0, speed=10.0):
        return {"route": self.route, "idx": idx, "pos_m": pos, "speed_mps": speed}

    def test_arrival_times_are_predicted_from_position_speed_and_free_flow(self):
        arr = S.ev_arrivals(self.net, self.ev(0, 100.0, 10.0))          # 100 m left at 10 m/s
        self.assertEqual([(n, a, b) for n, a, b, _ in arr],
                         [("J10", "W1-J10", "J10-J11"), ("J11", "J10-J11", "J11-J12"), ("J12", "J11-J12", "J12-E1")])
        self.assertAlmostEqual(arr[0][3], 10.0)
        self.assertAlmostEqual(arr[1][3], 10.0 + 27.0)                  # then 300 m at 40 km/h
        self.assertEqual(S.ev_arrivals(self.net, self.ev(3, 10.0, 10.0)), [])   # last link: nothing to cross

    def test_the_hard_corridor_serves_the_evs_movement_before_it_arrives_then_hands_back(self):
        base = S.FixedTimeController(self.net)
        c = S.EmergencyCorridor(base, self.net, clearance_s=4, margin_s=2)
        ph = self.net.phases_with("W1-J10>J10-J11")[0]
        far = S.QueueState(self.net, evs=[self.ev(0, 0.0, 10.0)])         # 20 s away: no need yet (lead is 6 s)
        near = S.QueueState(self.net, evs=[self.ev(0, 170.0, 10.0)])      # 3 s away
        gone = S.QueueState(self.net, evs=[self.ev(1, 50.0, 10.0)])       # through J10, heading for J11
        shown_far = c.decide("J10", far, 0)
        self.assertEqual(shown_far, base.decide("J10", far, 0))
        out = [c.decide("J10", near, t) for t in range(1, 8)]
        self.assertIn(None, out[:5])                                       # clearance before the switch
        self.assertEqual(out[-1], ph)
        self.assertEqual([c.decide("J10", near, t) for t in (8, 9)], [ph, ph])
        back = c.decide("J10", gone, 10)                                   # handed back to the base plan
        self.assertIn(back, (None, base.decide("J10", gone, 10)))

    def test_a_queue_ahead_of_the_ev_makes_the_corridor_start_earlier(self):
        c = S.EmergencyCorridor(S.FixedTimeController(self.net), self.net)
        ev = self.ev(0, 0.0, 10.0)                                         # 20 s from the stop line
        self.assertIsNone(c._demand("J10", S.QueueState(self.net, evs=[ev])))
        self.assertIsNotNone(c._demand("J10", S.QueueState(self.net, {"W1-J10": 30}, evs=[ev])))

    def test_a_junction_is_not_held_for_ever_for_a_vehicle_that_never_clears(self):
        c = S.EmergencyCorridor(S.FixedTimeController(self.net), self.net, max_override_s=20)
        stuck = S.QueueState(self.net, evs=[self.ev(0, 199.0, 0.0)])
        ph = self.net.phases_with("W1-J10>J10-J11")[0]
        shown = [c.decide("J10", stuck, t) for t in range(60)]
        self.assertEqual(shown[10], ph)
        self.assertNotEqual(set(shown[25:45]), {ph})

    def test_soft_priority_adds_to_the_pressure_but_a_big_enough_conflicting_queue_still_wins(self):
        net = junction()
        route = ["W0-J00", "J00-E0"]
        ev = {"route": route, "idx": 0, "pos_m": 150.0, "speed_mps": 10.0}
        plain = S.MaxPressureController(net)
        soft = S.SoftPriority(net, weight=25)
        wi = next(i for i, p in enumerate(net.phases["J00"]) if p.name == "W")
        st = S.QueueState(net, {"N0-J00": 8, "W0-J00": 2}, evs=[ev])
        self.assertGreater(soft.pressures("J00", st)[wi], plain.pressures("J00", st)[wi])
        self.assertGreater(soft.pressures("J00", st)[wi], soft.pressures("J00", st)[0])      # helps the EV here
        st_big = S.QueueState(net, {"N0-J00": 80, "W0-J00": 2}, evs=[ev])
        pr = soft.pressures("J00", st_big)
        self.assertGreater(pr[0], pr[wi])                                                      # but is no override

    def test_soft_priority_fades_in_with_the_vehicles_approach(self):
        net = junction()
        soft = S.SoftPriority(net, weight=20, lead_s=15, horizon_s=60)
        mid = "W0-J00>J00-E0"
        st = lambda pos: S.QueueState(net, evs=[{"route": ["W0-J00", "J00-E0"], "idx": 0, "pos_m": pos, "speed_mps": 5.0}])
        bonus = [soft.extra_queue(mid, st(p), 0) for p in (0.0, 100.0, 190.0)]       # 40 s, 20 s, 2 s away
        self.assertEqual(bonus, sorted(bonus))
        self.assertAlmostEqual(bonus[-1], 20.0)
        self.assertEqual(soft.extra_queue("N0-J00>J00-S0", st(190.0), 0), 0.0)


# ---------------------------------------------------------------- the simulator

def asymmetric_flows():
    return [M.Flow.constant("W0", "E0", 900), M.Flow.constant("E0", "W0", 150),
            M.Flow.constant("N0", "S0", 100), M.Flow.constant("S0", "N0", 100)]


class SimulatorTests(unittest.TestCase):
    def test_vehicles_are_conserved_at_every_step_and_links_never_overfill(self):
        net = junction()
        sim = M.Simulator(net, asymmetric_flows(), S.MaxPressureController(net), horizon_s=400, seed=3)
        while sim.t < 700:
            sim.step()
            self.assertTrue(sim.conserved(), f"t={sim.t}")
            for lid, n in sim._count.items():
                self.assertLessEqual(n, sim._store[lid] + 1e-9, lid)
        self.assertGreater(sim.generated, 100)
        self.assertEqual(sim.generated, sim.finished + sim.in_network())

    def test_everything_generated_finishes_when_there_is_time_to_drain(self):
        net = junction()
        r = M.simulate(net, asymmetric_flows(), S.MaxPressureController(net), horizon_s=300, drain_s=900, seed=1)
        self.assertEqual((r["generated"], r["finished"], r["in_network"], r["unfinished"]), (r["generated"], r["generated"], 0, 0))

    def test_a_run_is_reproducible_from_its_seed_and_differs_between_seeds(self):
        net = junction()
        run = lambda seed: M.simulate(net, asymmetric_flows(), S.MaxPressureController(net), horizon_s=300, seed=seed)
        a, b, c = run(5), run(5), run(6)
        for k in ("generated", "finished", "mean_delay_s", "p95_delay_s", "max_queue_veh", "queue_series"):
            self.assertEqual(a[k], b[k], k)
        self.assertNotEqual((a["generated"], a["mean_delay_s"]), (c["generated"], c["mean_delay_s"]))

    def test_each_flows_arrivals_do_not_change_when_another_flow_is_added(self):
        f1 = M.Flow.constant("W0", "E0", 600)
        alone = [t for t, k in M.generate_arrivals([f1], 600, seed=9) if k == 0]
        with_other = [t for t, k in M.generate_arrivals([f1, M.Flow.constant("N0", "S0", 600)], 600, seed=9) if k == 0]
        self.assertEqual(alone, with_other)

    def test_arrivals_follow_the_requested_rate_and_its_changes(self):
        f = M.Flow("W0", "E0", [(0, 0), (600, 1200), (1200, 0)])           # nothing, then 1200/h for 10 minutes
        ts = [t for t, _ in M.generate_arrivals([f], 1800, seed=2)]
        self.assertTrue(all(600 <= t < 1200 for t in ts))
        self.assertAlmostEqual(len(ts), 200, delta=40)                       # Poisson: sd ~ 14

    def test_nothing_crosses_a_junction_on_red(self):
        net = junction()
        r = M.simulate(net, [M.Flow.constant("W0", "E0", 600)], Hold(None), horizon_s=300, drain_s=100, seed=1)
        self.assertEqual(r["finished"], 0)
        self.assertEqual(r["in_network"], r["generated"])

    def test_a_green_discharges_no_faster_than_the_saturation_flow(self):
        net = junction()
        w = next(i for i, p in enumerate(net.phases["J00"]) if p.name == "W")
        r = M.simulate(net, [M.Flow.constant("W0", "E0", 3600)], Hold(w), horizon_s=300, drain_s=0, seed=1)
        sat = net.links["W0-J00"].sat_flow_vph                                # 2 lanes x 1200
        self.assertLessEqual(r["finished"], sat * 300 / 3600 + 5)
        self.assertGreater(r["finished"], 0.7 * sat * 300 / 3600)            # and close to it when saturated

    def test_delay_is_measured_against_the_free_flow_time_and_unfinished_vehicles_count(self):
        net = junction()
        light = M.simulate(net, [M.Flow.constant("W0", "E0", 60)], Hold(next(i for i, p in enumerate(net.phases["J00"]) if p.name == "W")),
                           horizon_s=300, seed=1)
        self.assertLess(light["mean_delay_s"], 3.0)                          # a green all the way: no delay
        red = M.simulate(net, [M.Flow.constant("W0", "E0", 600)], Hold(None), horizon_s=300, drain_s=100, seed=1)
        self.assertEqual(red["unfinished"], red["generated"])
        self.assertGreater(red["mean_delay_s"], 100)                         # stuck vehicles show their delay so far

    def test_max_pressure_beats_fixed_time_under_strongly_asymmetric_demand(self):
        net = junction()
        mp, ft = [], []
        for seed in (1, 2):
            mp.append(M.simulate(net, asymmetric_flows(), S.MaxPressureController(net, max_green=60), horizon_s=900, seed=seed))
            ft.append(M.simulate(net, asymmetric_flows(), S.FixedTimeController(net), horizon_s=900, seed=seed))
        for a, b in zip(mp, ft):
            self.assertLess(a["mean_delay_s"], 0.5 * b["mean_delay_s"])
            self.assertLess(a["p95_delay_s"], b["p95_delay_s"])
            self.assertLess(a["max_queue_veh"], b["max_queue_veh"])
            self.assertGreaterEqual(a["throughput_vph"], b["throughput_vph"])

    def test_both_controllers_face_identical_arrivals(self):
        net = junction()
        a = M.simulate(net, asymmetric_flows(), S.MaxPressureController(net), horizon_s=300, seed=4)
        b = M.simulate(net, asymmetric_flows(), S.FixedTimeController(net), horizon_s=300, seed=4)
        self.assertEqual(a["generated"], b["generated"])

    def test_frames_record_link_occupancy_and_signal_phases_for_a_3d_view(self):
        net = junction()
        sim = M.Simulator(net, asymmetric_flows(), S.MaxPressureController(net), horizon_s=60, seed=1, record_frames=True)
        r = sim.run()
        self.assertGreaterEqual(len(r["frames"]), 60)
        f = r["frames"][30]
        self.assertEqual(set(f), {"t", "links", "evs", "signals"})
        self.assertEqual(set(f["signals"]), {"J00"})
        self.assertTrue(all(len(v) == 2 for v in f["links"].values()))
        self.assertIn(-1, {fr["signals"]["J00"] for fr in r["frames"]})       # clearance shows up
        json.dumps({k: r[k] for k in ("frames", "queue_series", "ev")}, allow_nan=False)

    def test_routing_at_departure_goes_round_a_queue(self):
        net = N.grid(3, 3)
        sim = M.Simulator(net, [], S.MaxPressureController(net), horizon_s=10)
        before = sim._route("W1", "E1")
        self.assertIn("J10-J11", before)
        sim._qcount["J10-J11"] = 300                                          # a standing jam on the straight road
        sim._next_at = -1e9
        self.assertNotIn("J10-J11", sim._route("W1", "E1"))


class EmergencyVehicleSimTests(unittest.TestCase):
    def scenario(self, make, seed=1):
        net = N.grid(3, 3)
        flows = [M.Flow.constant("W1", "E1", 700), M.Flow.constant("E1", "W1", 300)] + \
            [M.Flow.constant(f"N{c}", f"S{c}", 250) for c in range(3)] + [M.Flow.constant(f"S{c}", f"N{c}", 150) for c in range(3)]
        route = R.shortest_path(net, "W1", "E1").links
        sim = M.Simulator(net, flows, make(net), horizon_s=700, drain_s=300, seed=seed, evs=[M.EV(300, "W1", "E1", route)])
        r = sim.run()
        self.assertTrue(sim.conserved())
        return r

    def test_the_hard_corridor_reduces_the_emergency_vehicles_travel_time(self):
        for seed in (1, 2):
            none = self.scenario(lambda net: S.FixedTimeController(net), seed)
            hard = self.scenario(lambda net: S.EmergencyCorridor(S.FixedTimeController(net), net), seed)
            self.assertTrue(none["ev"][0]["finished"] and hard["ev"][0]["finished"])
            self.assertLess(hard["ev"][0]["travel_time_s"], 0.8 * none["ev"][0]["travel_time_s"])

    def test_soft_priority_does_not_slow_the_emergency_vehicle(self):
        none = self.scenario(lambda net: S.MaxPressureController(net))
        soft = self.scenario(lambda net: S.SoftPriority(net))
        self.assertLessEqual(soft["ev"][0]["travel_time_s"], none["ev"][0]["travel_time_s"] + 1e-9)

    def test_the_ev_is_excluded_from_the_ordinary_traffic_figures(self):
        r = self.scenario(lambda net: S.MaxPressureController(net))
        self.assertEqual(r["vehicles"], r["generated"] - 1)
        self.assertEqual(len(r["ev"]), 1)
        self.assertGreaterEqual(r["ev"][0]["travel_time_s"], r["ev"][0]["free_flow_s"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
