"""The OPTIMIZE stage: network, routing, signal control and the simulator that validates them.

Everything here is a statement about software behaviour on synthetic networks and canned responses. Nothing
here has driven a real signal, and nothing calls the network: the Google provider is given a canned
transport. A simulator that favours max pressure over fixed time proves the controller is sound under THIS
simulator's assumptions (FIFO stop-line queues, saturation-flow discharge, a fixed free-flow time), not that
it would behave so on a Dhaka junction - these tests pin down behaviour that must not regress, they are not
evidence about the street.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.optimize import network as N  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
