"""The 3D city / optimiser API: roles, editor validation, live network from fake snapshots, freshness,
recommendation, bounded deterministic simulation, routing with an incident, and the Google rule.

The router is mounted on a fresh FastAPI app with fake authentication over a temp Store; no real Google call is
possible (the provider's transport is a fake that counts calls)."""

import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from datetime import datetime
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="city-tests-")
os.environ.setdefault("TRAFFIC_DATA_DIR", _SCRATCH)
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from surveillance import api_map, api_optimize as ao  # noqa: E402
from surveillance import geo as geo_mod  # noqa: E402
from surveillance.impact import save_impact  # noqa: E402
from surveillance.optimize.google_traffic import GoogleTraffic  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402
from tests.test_map import FakeAuth  # noqa: E402

LAT, LON = 23.7500, 90.3900
SERVER = "AIzaSyServerKeySECRET99998888ABCD1234"
NOW = 1_800_000_000.0


def at(north_m=0.0, east_m=0.0):
    return {"lat": round(LAT + north_m / 111195.0, 7), "lon": round(LON + east_m / 102000.0, 7)}


def doc(cam1=None, cam2=None):
    """B1 -- J1 -- J2 -- B2 along an avenue, spurs B3 (off J1) and B4 (off J2), plus a bypass J1 -- J3 -- J2."""
    nodes = [("B1", "boundary", 0, -300), ("J1", "signal", 0, 0), ("J2", "signal", 0, 600), ("B2", "boundary", 0, 900),
             ("B3", "boundary", 300, 0), ("B4", "boundary", -300, 600), ("J3", "signal", 300, 300)]  # (id, kind, east m, north m)
    roads = [("B1", "J1"), ("J1", "J2"), ("J2", "B2"), ("J1", "B3"), ("J2", "B4"), ("J1", "J3"), ("J3", "J2")]
    d = {"name": "Test avenue", "synthetic": False,
         "nodes": [{"id": i, "kind": k, **at(north, east)} for i, k, east, north in nodes],
         "roads": [{"from": a, "to": b, "lanes": 2} for a, b in roads]}
    if cam1:
        d["roads"][1]["cameras"] = [{"camera_id": cam1, "zone": "avenue", "dir": "fwd"}]
    if cam2:
        d["roads"][0]["cameras"] = [{"camera_id": cam2, "zone": "west", "dir": "fwd"}]
    return d


def zone(density=40.0, queue_m=40.0, speed=8.0, wall=NOW, t=10.0):
    return {"queue_m": queue_m, "speed_kph": speed, "occupancy": 0.5, "density": density, "wait_s": 20.0,
            "samples": 30, "vehicles": 12, "t": t, "lanes": 2, "flow": 100, "wall": wall}


class CityTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = Store(Path(self._dir.name) / "s.db")
        self.addCleanup(self.store.db.close)
        self.c1 = self.store.add_camera("Farmgate", "file", "/secret/a.mp4", "Farmgate")
        self.c2 = self.store.add_camera("Karwan Bazar", "file", "/secret/b.mp4", "Karwan Bazar")
        ao.ensure_schema(self.store)
        geo_mod.save(self.store, self.c1, {**at(0, 300), "heading_deg": 90})
        geo_mod.save(self.store, self.c2, {**at(0, -150), "heading_deg": 90})
        self.snapshots = {}
        self.calls = []
        self.clock_now = [NOW]

        def transport(url, headers, body):
            self.calls.append(url)
            return {"routes": [{"duration": "300s", "staticDuration": "100s", "distanceMeters": 600}]}
        self.provider = GoogleTraffic(api_key=SERVER, transport=transport)
        self.google = [self.provider]
        fa = FakeAuth()
        web = Path(self._dir.name) / "web"
        web.mkdir()
        (web / "city.html").write_text("<html>city</html>")
        app = FastAPI()
        app.include_router(ao.make_router(self.store, fa.current_user, fa.require, "controller", "admin",
                                          snapshots_fn=lambda: self.snapshots, google_fn=lambda: self.google[0],
                                          status_fn=lambda c: {"state": "running"}, web_dir=web,
                                          clock=lambda: self.clock_now[0], sim_timeout_s=60.0))
        self.app = app

    def call(self, method, path, role=None, body=None):
        headers = {"x-role": role} if role else {}
        data = b""
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        status, _, out = asgi_call(self.app, method, path, headers, data)
        try:
            return status, json.loads(out)
        except ValueError:
            return status, out.decode()

    def save(self, d=None):
        s, out = self.call("PUT", "/api/optimize/network", "admin", d or doc(self.c1, self.c2))
        self.assertEqual(s, 200, out)
        return out

    def enable_google(self):
        self.assertEqual(self.call("PUT", "/api/optimize/google", "admin", {"enabled": True})[0], 200)

    # ---- roles
    def test_roles_on_every_endpoint(self):
        d = doc(self.c1, self.c2)
        ep = [("GET", "/api/optimize/network", None, "staff"), ("PUT", "/api/optimize/network", d, "admin"),
              ("POST", "/api/optimize/network/validate", d, "admin"), ("POST", "/api/optimize/network/demo", None, "admin"),
              ("GET", "/api/optimize/live", None, "staff"), ("GET", "/api/optimize/recommendation", None, "staff"),
              ("POST", "/api/optimize/simulate", {"horizon_s": 60}, "staff"),
              ("POST", "/api/optimize/route", {"origin": "B1", "dest": "B2"}, "staff"),
              ("GET", "/api/optimize/google", None, "admin"), ("PUT", "/api/optimize/google", {"enabled": False}, "admin"),
              ("DELETE", "/api/optimize/network", None, "admin")]
        self.save()
        for method, path, body, need in ep:
            self.assertEqual(self.call(method, path, None, body)[0], 401, path)
            self.assertEqual(self.call(method, path, "driver", body)[0], 403, path)
            c = self.call(method, path, "controller", body)[0]
            self.assertEqual(c, 200 if need == "staff" else 403, f"{method} {path}")
        for method, path, body, need in ep[:-1]:
            self.assertEqual(self.call(method, path, "admin", body)[0], 200, path)

    def test_page_redirects(self):
        self.assertEqual(asgi_call(self.app, "GET", "/city")[0], 302)
        s, h, _ = asgi_call(self.app, "GET", "/city", {"x-role": "driver"})
        self.assertEqual((s, h["location"]), (302, "/portal"))
        self.assertEqual(asgi_call(self.app, "GET", "/city", {"x-role": "controller"})[0], 200)

    # ---- editor
    def test_save_load_and_audit(self):
        out = self.save()
        self.assertEqual(len(out["doc"]["nodes"]), 7)
        s, got = self.call("GET", "/api/optimize/network", "controller")
        self.assertEqual((got["source"], got["can_edit"], got["synthetic"]), ("saved", False, False))
        self.assertEqual(len(got["cameras"]), 2)
        self.assertTrue(got["cameras"][0]["placed"])
        self.assertEqual(self.store.audit_trail("network", "default")[0]["action"], "optimize.network")
        self.assertEqual(self.call("DELETE", "/api/optimize/network", "admin")[0], 200)
        self.assertEqual(self.call("DELETE", "/api/optimize/network", "admin")[0], 404)

    def test_validation_lists_every_problem(self):
        bad = doc()
        bad["nodes"].append({"id": "J1", "kind": "signal", **at(5, 5)})                 # duplicate id
        bad["nodes"].append({"id": "X-1", "kind": "signal", **at(5, 5)})                # illegal id
        bad["nodes"].append({"id": "Z", "kind": "roundabout", **at(5, 5)})              # unknown kind
        bad["roads"].append({"from": "J1", "to": "J1"})                                 # self loop
        bad["roads"].append({"from": "J2", "to": "J1"})                                 # duplicate pair
        bad["roads"].append({"from": "J1", "to": "NOPE"})
        bad["roads"][0]["lanes"] = 99
        bad["roads"][2]["speed_kph"] = 0
        bad["roads"][3]["cameras"] = [{"camera_id": 9999, "zone": "z", "dir": "fwd"}]   # no such camera
        bad["roads"][4]["cameras"] = [{"camera_id": self.c1, "zone": "", "dir": "fwd"}]
        s, out = self.call("PUT", "/api/optimize/network", "admin", bad)
        self.assertEqual(s, 400)
        text = " | ".join(out["detail"]["problems"])
        for frag in ("id used twice", "id must be", "kind must be", "same intersection", "already a road",
                     "intersections of the network", "lanes", "speed_kph", "camera 9999", "road zone"):
            self.assertIn(frag, text)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM optimize_networks")[0]["n"], 0)

    def test_more_validation(self):
        for mutate, frag in ((lambda d: d["nodes"][1].update(lat=95), "lat"),
                             (lambda d: d["nodes"][1].update(lat=None), "lat"),
                             (lambda d: d["roads"][0].update(two_way="yes"), "two_way"),
                             (lambda d: d["roads"][1].update(cameras=[{"camera_id": self.c1, "zone": "a", "dir": "sideways"}]), "dir"),
                             (lambda d: d["roads"][1].update(cameras=[{"camera_id": self.c1, "zone": "a", "dir": "fwd"},
                                                                       {"camera_id": self.c2, "zone": "a", "dir": "fwd"}]), "each direction"),
                             (lambda d: d["roads"][1].update(length_m=1), "length_m")):
            d = doc()
            mutate(d)
            s, out = self.call("PUT", "/api/optimize/network", "admin", d)
            self.assertEqual(s, 400, frag)
            self.assertIn(frag, " ".join(out["detail"]["problems"]))
        self.assertEqual(self.call("PUT", "/api/optimize/network", "admin", [1])[0], 400)
        self.assertEqual(self.call("PUT", "/api/optimize/network?site=BAD SITE", "admin", doc())[0], 400)

    def test_same_zone_on_two_roads_is_refused_and_unplaced_camera_warns(self):
        d = doc()
        d["roads"][0]["cameras"] = [{"camera_id": self.c1, "zone": "avenue", "dir": "fwd"}]
        d["roads"][1]["cameras"] = [{"camera_id": self.c1, "zone": "avenue", "dir": "fwd"}]
        self.assertEqual(self.call("PUT", "/api/optimize/network", "admin", d)[0], 400)
        geo_mod.remove(self.store, self.c2)
        s, out = self.call("PUT", "/api/optimize/network", "admin", doc(self.c1, self.c2))
        self.assertEqual(s, 200)
        self.assertTrue(any("not placed" in w for w in out["warnings"]))

    def test_demo_is_labelled_synthetic_and_not_saved(self):
        self.snapshots = {self.c1: {"avenue": zone()}}
        s, out = self.call("POST", "/api/optimize/network/demo", "admin")
        self.assertEqual(s, 200)
        self.assertTrue(out["synthetic"] and out["doc"]["synthetic"])
        self.assertIn("synthetic", out["doc"]["name"].lower())
        self.assertTrue(any("Synthetic layout" in n for n in out["notes"]))
        kinds = [n["kind"] for n in out["doc"]["nodes"]]
        self.assertEqual(kinds.count("signal"), 2)
        self.assertGreaterEqual(kinds.count("boundary"), 4)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM optimize_networks")[0]["n"], 0)
        # with nothing drawn, reads fall back to it, labelled
        s, got = self.call("GET", "/api/optimize/network", "controller")
        self.assertEqual((got["source"], got["synthetic"]), ("demo", True))
        attached = [c for r in got["doc"]["roads"] for c in r["cameras"]]
        self.assertEqual([c["camera_id"] for c in attached], [self.c1])      # c2 has no zone, so is not attached
        s, live = self.call("GET", "/api/optimize/live", "controller")
        self.assertEqual((live["source"], live["synthetic"]), ("demo", True))
        # and the demo can be saved as the editable network
        self.assertEqual(self.call("PUT", "/api/optimize/network", "admin", out["doc"])[0], 200)

    def test_no_cameras_no_network(self):
        geo_mod.remove(self.store, self.c1)
        geo_mod.remove(self.store, self.c2)
        s, out = self.call("GET", "/api/optimize/live", "controller")   # an empty state, not an error
        self.assertEqual((s, out["network"], out["links"]), (200, False, []))
        self.assertIn("no network", out["reason"].lower())
        self.assertEqual(self.call("GET", "/api/optimize/recommendation", "controller")[0], 404)
        self.assertEqual(self.call("POST", "/api/optimize/network/demo", "admin")[0], 400)

    # ---- live
    def test_live_network_from_fake_snapshots_with_freshness(self):
        self.save()
        self.snapshots = {self.c1: {"avenue": zone(wall=NOW - 10)},          # fresh
                          self.c2: {"west": zone(density=5, queue_m=0, speed=35, wall=NOW - 600)}}   # stalled 10 min
        s, live = self.call("GET", "/api/optimize/live", "controller")
        self.assertEqual(s, 200)
        links = {l["id"]: l for l in live["links"]}
        fresh, stale, none = links["J1-J2"], links["B1-J1"], links["J1-B1"]
        self.assertEqual((fresh["source"], fresh["status"]), ("camera", "live"))
        self.assertAlmostEqual(fresh["age_s"], 10, delta=1)
        self.assertGreater(fresh["delay_s"], 10)                                  # a 40 m queue costs time
        self.assertEqual(fresh["measured"]["density"], 40.0)
        self.assertEqual((stale["source"], stale["status"]), ("camera", "stale"))
        self.assertAlmostEqual(stale["age_s"], 600, delta=1)
        self.assertEqual((none["source"], none["status"]), ("free_flow", "none"))
        self.assertNotIn("measured", none)
        cams = {c["id"]: c for c in live["cameras"]}
        self.assertTrue(cams[self.c1]["fresh"] and not cams[self.c1]["stale"])
        self.assertTrue(cams[self.c2]["stale"])
        self.assertEqual(cams[self.c1]["density"], 40.0)
        self.assertEqual(len(cams[self.c1]["geo"]["wedge"]), 14)
        self.assertFalse(live["contains_google"])
        self.assertEqual(sum(1 for n in live["nodes"] if n["kind"] == "signal"), 3)

    def test_zone_with_no_samples_is_not_free_flow(self):
        self.save()
        self.snapshots = {self.c1: {"avenue": {**zone(), "samples": 0}}}
        links = {l["id"]: l for l in self.call("GET", "/api/optimize/live", "controller")[1]["links"]}
        self.assertEqual(links["J1-J2"]["status"], "none")

    def test_snapshots_from_workers_marks_a_stalled_camera(self):
        class Col:
            def __init__(self, wall): self.wall = wall
            def snapshot(self): return {"avenue": {"density": 3.0, "queue_m": 0.0, "t": 100.0, "samples": 5, "speed_kph": 30.0}}
            def wall_at(self, t): return self.wall

        class W:
            def __init__(self, c): self.traffic = c
        out = ao.snapshots_from_workers({1: W(Col(NOW - 500)), 2: W(None), 3: W(Col(None))}, now=NOW)
        self.assertEqual(set(out), {1, 3})
        self.assertEqual(out[1]["avenue"]["wall"], NOW - 500)          # the collector's own clock
        self.assertEqual(out[3]["avenue"]["wall"], NOW)                # no clock: first seen now
        later = ao.snapshots_from_workers({3: W(Col(None))}, now=NOW + 300)
        self.assertEqual(later[3]["avenue"]["wall"], NOW)              # t did not advance: it is 300 s old

    # ---- recommendation
    def test_recommendation_shape_and_basis(self):
        self.save()
        self.snapshots = {self.c1: {"avenue": zone(wall=NOW - 5)}}
        s, rec = self.call("GET", "/api/optimize/recommendation", "controller")
        self.assertEqual(s, 200)
        self.assertIn("Nothing is applied", rec["disclaimer"])
        by = {x["node"]: x for x in rec["signals"]}
        self.assertEqual(set(by), {"J1", "J2", "J3"})
        j2 = by["J2"]
        self.assertEqual(j2["basis"], "camera")                          # J1-J2 is an approach of J2
        self.assertGreater(j2["approaches_measured"], 0)
        for sig in rec["signals"]:
            total = sum(p["green_s"] for p in sig["phases"]) + sig["lost_s"]
            self.assertAlmostEqual(total, sig["cycle_s"], delta=0.6)
            for p in sig["phases"]:
                self.assertGreaterEqual(p["green_s"], 7.0 - 1e-6)
                self.assertTrue(set(p) >= {"name", "green_s", "share", "pressure", "queue_veh", "links"})
        self.assertEqual(by["J3"]["basis"], "none")
        self.assertIsNone(by["J3"]["favoured_phase"])
        greens = {p["green_s"] for p in by["J3"]["phases"]}
        self.assertEqual(len(greens), 1)                                  # no data: equal greens
        self.assertEqual(self.call("GET", "/api/optimize/recommendation?cycle_s=5", "controller")[0], 422)

    def test_stale_camera_is_no_basis_for_a_recommendation(self):
        self.save()
        self.snapshots = {self.c1: {"avenue": zone(wall=NOW - 900)}}
        rec = self.call("GET", "/api/optimize/recommendation", "controller")[1]
        self.assertTrue(all(s["basis"] == "none" for s in rec["signals"]))

    # ---- simulation
    SIM = {"horizon_s": 120, "load": 0.5, "seed": 3}

    def test_simulate_is_bounded_and_deterministic(self):
        self.save()
        a = self.call("POST", "/api/optimize/simulate", "controller", self.SIM)
        b = self.call("POST", "/api/optimize/simulate", "controller", self.SIM)
        self.assertEqual(a[0], 200, a[1])

        def core(x):
            x = dict(x[1])
            x.pop("elapsed_s")
            return x
        self.assertEqual(core(a), core(b))
        out = a[1]
        self.assertEqual([r["id"] for r in out["runs"]], ["fixed", "max_pressure"])
        self.assertEqual(out["demand"], "synthetic")
        self.assertGreater(out["runs"][0]["summary"]["generated"], 0)
        self.assertEqual(out["runs"][0]["summary"]["generated"], out["runs"][1]["summary"]["generated"])   # same arrivals
        self.assertNotIn("frames", out["runs"][0])
        self.assertTrue(any("synthetic" in n.lower() for n in out["notes"]))
        self.assertTrue(any("not a trained model" in n for n in out["notes"]))
        self.assertTrue(any("1200" in n for n in out["notes"]))
        c = self.call("POST", "/api/optimize/simulate", "controller", {**self.SIM, "seed": 4})
        self.assertNotEqual(core(a)["runs"][0]["summary"], core(c)["runs"][0]["summary"])
        for bad in ({"horizon_s": 100000}, {"horizon_s": 5}, {"load": 50}, {"seed": -1}, {"controllers": ["sumo"]},
                    {"flows": [{"origin": "B1", "dest": "B1", "vph": 5}]}, {"flows": [{"origin": "B1", "dest": "B2", "vph": 1e6}]},
                    {"horizon_s": "x"}):
            self.assertEqual(self.call("POST", "/api/optimize/simulate", "controller", bad)[0], 400, bad)

    def test_simulate_frames_only_on_request_and_ev_corridor(self):
        self.save()
        body = {**self.SIM, "frames": True, "ev": {"origin": "B1", "dest": "B2", "depart_s": 20}}
        s, out = self.call("POST", "/api/optimize/simulate", "controller", body)
        self.assertEqual(s, 200, out)
        self.assertEqual([r["id"] for r in out["runs"]], ["fixed", "max_pressure", "max_pressure_corridor"])
        fr = out["runs"][1]["frames"]
        self.assertTrue(0 < len(fr) <= ao.SIM_FRAMES_MAX)
        self.assertTrue({"t", "links", "signals", "evs"} <= set(fr[0]))
        self.assertEqual(len(out["runs"][2]["ev"]), 1)
        self.assertIn("corridor_ev_time_pct", out["comparison"])
        self.assertEqual(self.call("POST", "/api/optimize/simulate", "controller", {**body, "ev": {"origin": "B1", "dest": "ZZ"}})[0], 400)

    def test_simulation_that_overruns_is_stopped(self):
        """A simulator that never finishes on its own: the request must answer 504 at the timeout and the run must
        see the cancel. (Timing a real run against a tiny timeout passed or failed with the machine's load.)"""
        self.save()
        fa = FakeAuth()
        app = FastAPI()
        app.include_router(ao.make_router(self.store, fa.current_user, fa.require, sim_timeout_s=0.05))
        stopped = threading.Event()

        def endless(sim_self, stop=None):
            deadline = time.time() + 5
            while time.time() < deadline:
                if stop is not None and stop():
                    stopped.set()
                    break
                time.sleep(0.005)
            raise RuntimeError("cancelled")
        with mock.patch.object(ao.M.Simulator, "run", endless):
            s, _, body = asgi_call(app, "POST", "/api/optimize/simulate", {"x-role": "admin", "content-type": "application/json"},
                                   json.dumps({"horizon_s": 900, "load": 2.0}).encode())
            self.assertEqual(s, 504)
            self.assertTrue(stopped.wait(2), "the running simulation never saw the cancel")

    # ---- routing
    def test_route_alternatives_and_incident(self):
        self.save()
        s, plain = self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2", "k": 3})
        self.assertEqual(s, 200, plain)
        r0 = plain["routes"][0]
        self.assertEqual(r0["nodes"], ["B1", "J1", "J2", "B2"])
        self.assertEqual(len(plain["routes"]), 2)                          # via the bypass J3 is the alternative
        self.assertIn("J3", plain["routes"][1]["nodes"])
        self.assertEqual(plain["incidents"], [])
        self.assertTrue(all(leg["data_source"].startswith("no measurement") for leg in r0["breakdown"]))
        # a recorded violation on the avenue with a scored impact -> a delay that sends the route round
        run = None
        rec = {"code": "V1", "violation": "Bus blocking", "track_id": 1, "class": "bus", "class_confidence": 0.9,
               "video_time_s": 5.0, "started_s": 1.0, "wall_time": "", "zone": "avenue", "areas": "[]", "kph": 0.0, "details": "{}"}
        try:
            vid = self.store.add_violation(self.c1, run, None, rec, None, None, None)
        except Exception:
            vid = None
        if vid is None:
            self.skipTest("could not create a violation row")
        impact = {"excess_delay_veh_s": 9000.0, "vehicles_seen": 10, "camera_id": self.c1, "zone": "avenue", "code": "V1",
                  "score": 80}
        save_impact(self.store, vid, impact)
        self.store.x("UPDATE event_impact SET created_at=?", (datetime.fromtimestamp(NOW - 60).isoformat(timespec="seconds"),))
        s, inc = self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2", "k": 2})
        self.assertEqual([(i["link"], i["extra_delay_s"]) for i in inc["incidents"]], [("J1-J2", 900.0)])
        self.assertEqual(inc["routes"][0]["nodes"], ["B1", "J1", "J3", "J2", "B2"])
        self.assertGreater(inc["incident_comparison"]["saving_s"], 0)
        self.assertIn("J1-J2", inc["incident_comparison"]["avoided"])
        off = self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2", "use_incidents": False})[1]
        self.assertEqual(off["routes"][0]["nodes"], ["B1", "J1", "J2", "B2"])
        self.store.x("UPDATE violations SET status='rejected'")
        self.assertEqual(self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2"})[1]["incidents"], [])

    def test_route_snaps_coordinates_and_rejects_nonsense(self):
        self.save()
        s, out = self.call("POST", "/api/optimize/route", "controller", {"origin": {**at(-290, 10)}, "dest": "B2", "k": 1})
        self.assertEqual((s, out["origin"]["node"]), (200, "B1"))
        self.assertGreater(out["origin"]["snapped_m"], 1)
        for bad in ({"origin": "NOPE", "dest": "B2"}, {"origin": "B1"}, {"origin": "B1", "dest": "B2", "k": 9},
                    {"origin": 5, "dest": "B2"}):
            self.assertEqual(self.call("POST", "/api/optimize/route", "controller", bad)[0], 400, bad)

    # ---- the Google rule
    def test_google_never_called_without_key_and_enable(self):
        self.save()
        self.snapshots = {}
        # enabled but no key
        self.google[0] = None
        self.enable_google()
        live = self.call("GET", "/api/optimize/live?use_google=true", "controller")[1]
        self.assertFalse(live["contains_google"])
        self.assertEqual(self.calls, [])
        # key but not enabled
        self.google[0] = self.provider
        self.call("PUT", "/api/optimize/google", "admin", {"enabled": False})
        live = self.call("GET", "/api/optimize/live?use_google=true", "controller")[1]
        self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2", "use_google": True})
        self.assertFalse(live["contains_google"])
        self.assertEqual(self.calls, [])
        # a provider without a key
        self.google[0] = GoogleTraffic(api_key=None, transport=lambda *a: self.calls.append("x"))
        os.environ.pop("GOOGLE_MAPS_SERVER_KEY", None)
        self.enable_google()
        self.call("GET", "/api/optimize/live?use_google=true", "controller")
        self.assertEqual(self.calls, [])

    def test_google_used_only_when_asked_and_allowed_and_3d_endpoints_never_carry_it(self):
        self.save()
        self.enable_google()
        # the 3D view's calls: use_google absent or false -> Google is not even asked
        for path in ("/api/optimize/live", "/api/optimize/live?use_google=false", "/api/optimize/recommendation"):
            out = self.call("GET", path, "controller")[1]
            self.assertFalse(out.get("contains_google", False))
            self.assertNotIn('"source": "google"', json.dumps(out))
            self.assertNotIn('"status": "google"', json.dumps(out))
        r = self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2"})[1]
        self.assertFalse(r["contains_google"])
        self.assertNotIn("google", {l["source"] for x in r["routes"] for l in x["breakdown"]})
        self.call("POST", "/api/optimize/simulate", "controller", self.SIM)
        self.assertEqual(self.calls, [])
        # explicitly asked (text / Google-map consumers): used and labelled
        live = self.call("GET", "/api/optimize/live?use_google=true", "controller")[1]
        self.assertTrue(self.calls)
        self.assertTrue(live["contains_google"])
        self.assertIn("google", {l["source"] for l in live["links"]})
        self.assertEqual({l["status"] for l in live["links"] if l["source"] == "google"}, {"google"})
        rt = self.call("POST", "/api/optimize/route", "controller", {"origin": "B1", "dest": "B2", "use_google": True})[1]
        self.assertTrue(rt["contains_google"])
        # asking again for the 3D view afterwards is clean again (nothing leaks through a shared cache)
        again = self.call("GET", "/api/optimize/live", "controller")[1]
        self.assertNotIn('"source": "google"', json.dumps(again))

    def test_camera_data_wins_over_google_and_budget_applies(self):
        self.save()
        self.enable_google()
        self.snapshots = {self.c1: {"avenue": zone(wall=NOW - 2)}}
        live = self.call("GET", "/api/optimize/live?use_google=true", "controller")[1]
        by = {l["id"]: l for l in live["links"]}
        self.assertEqual(by["J1-J2"]["source"], "camera")
        self.assertNotIn("J1-J2", [u for u in self.calls if "J1-J2" in u])
        n = len(self.calls)
        self.call("GET", "/api/optimize/live?use_google=true", "controller")
        self.assertEqual(len(self.calls), n)                              # the provider's cache answered

    def test_google_status_never_shows_the_key(self):
        self.enable_google()
        s, out = self.call("GET", "/api/optimize/google", "admin")
        self.assertEqual(s, 200)
        self.assertTrue(out["enabled"])
        self.assertNotIn(SERVER, json.dumps(out))
        api_map.ensure_schema(self.store)
        self.store.x("INSERT INTO map_settings (key, value, updated_at) VALUES ('server_key', ?, 1)", (SERVER,))
        out = self.call("GET", "/api/optimize/google", "admin")[1]
        self.assertTrue(out["server_key_set"] and out["active"])
        self.assertNotIn(SERVER, json.dumps(out))
        self.assertEqual(self.call("PUT", "/api/optimize/google", "admin", {"enabled": "yes"})[0], 400)
        self.assertNotIn(SERVER, json.dumps(self.store.q("SELECT * FROM audit_log")))


if __name__ == "__main__":
    unittest.main()
