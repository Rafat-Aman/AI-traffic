"""The camera map: placement validation, section (point-to-point) speed, and the HTTP surface.

The router is mounted on a fresh FastAPI app with fake authentication, over a temp Store; nothing here opens
data/surveillance.db or imports the dashboard server.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from fastapi import Depends, FastAPI, HTTPException, Request
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="map-tests-")
os.environ.setdefault("TRAFFIC_DATA_DIR", _SCRATCH)
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import api_map, auth  # noqa: E402
from surveillance import geo as geo_mod  # noqa: E402
from surveillance import section_speed as ss  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402

BROWSER = "AIzaSyBrowserKeyForTestsOnly000111"
SERVER = "AIzaSyServerKeySECRET99998888ABCD1234"
LAT, LON = 23.7500, 90.3900
M_PER_DEG_LAT = 111194.9          # haversine on the 6371 km sphere: one degree of latitude


def lat_for(metres):
    """Latitude `metres` north of LAT."""
    return LAT + metres / M_PER_DEG_LAT


def tk(cam, gid, t, cls="car"):
    return {"camera_id": cam, "global_id": gid, "t_cross": t, "cls": cls}


def geos_at(*metres_north, offsets=None):
    """Cameras 1..n on a north-south line, each `metres` north of the first."""
    g = {}
    for i, m in enumerate(metres_north, 1):
        g[i] = {"camera_id": i, "lat": lat_for(m), "lon": LON, "heading_deg": 0, "fov_deg": 60, "range_m": 80,
                "clock_offset_s": (offsets or {}).get(i, 0.0)}
    return g


class GeoValidationTests(unittest.TestCase):
    def test_heading_is_normalised(self):
        for given, want in ((-90, 270.0), (360, 0.0), (725.25, 5.2), (0, 0.0), (359.96, 0.0)):
            self.assertEqual(geo_mod.validate({"lat": LAT, "lon": LON, "heading_deg": given})[0]["heading_deg"], want)

    def test_impossible_positions_are_rejected(self):
        for bad in ({"lat": 91, "lon": LON}, {"lat": LAT, "lon": 181}, {"lat": "x", "lon": LON},
                    {"lat": float("nan"), "lon": LON}, {"lat": True, "lon": LON}, {"lon": LON}):
            with self.assertRaises(geo_mod.GeoError, msg=str(bad)):
                geo_mod.validate(bad)

    def test_fov_range_and_clock_bounds(self):
        for bad in ({"fov_deg": 0}, {"fov_deg": 200}, {"range_m": 1}, {"range_m": 5000}, {"clock_offset_s": 1e6}):
            with self.assertRaises(geo_mod.GeoError, msg=str(bad)):
                geo_mod.validate({"lat": LAT, "lon": LON, **bad})

    def test_outside_dhaka_warns_but_is_accepted(self):
        clean, warn = geo_mod.validate({"lat": 51.5, "lon": -0.12})
        self.assertEqual(clean["lat"], 51.5)
        self.assertEqual(len(warn), 1)
        self.assertEqual(geo_mod.validate({"lat": LAT, "lon": LON})[1], [])

    def test_swapped_pair_is_rejected_and_called_out(self):
        with self.assertRaises(geo_mod.GeoError) as cm:
            geo_mod.validate({"lat": LON, "lon": LAT})
        self.assertIn("swapped", str(cm.exception))

    def test_partial_update_keeps_the_stored_values(self):
        clean, _ = geo_mod.validate({"heading_deg": 45}, {"lat": LAT, "lon": LON, "fov_deg": 90})
        self.assertEqual((clean["lat"], clean["fov_deg"], clean["heading_deg"]), (LAT, 90.0, 45.0))

    def test_wedge_shape(self):
        w = geo_mod.fov_wedge(LAT, LON, 90, 60, 100, arc_points=6)
        self.assertEqual(len(w), 8)
        self.assertEqual(w[0], [LAT, LON])
        for p in w[1:]:
            self.assertAlmostEqual(geo_mod.haversine_m(LAT, LON, p[0], p[1]), 100, delta=0.5)
        mid = w[4]       # the middle of the arc lies on the heading: due east
        self.assertAlmostEqual(geo_mod.bearing_deg(LAT, LON, mid[0], mid[1]), 90, delta=0.2)
        self.assertAlmostEqual(geo_mod.bearing_deg(LAT, LON, w[1][0], w[1][1]), 60, delta=0.5)

    def test_network_inputs_feed_from_points(self):
        g = geos_at(0, 500, 1000)
        pts, roads, skipped = geo_mod.network_inputs(g, [(1, 2), (2, 3, 520.0), (3, 9)])
        self.assertEqual([p["id"] for p in pts], ["cam1", "cam2", "cam3"])
        self.assertEqual(skipped, [(3, 9)])
        net = geo_mod.build_network(g, [(1, 2), (2, 3, 520.0)], circuity=1.0)
        self.assertAlmostEqual(net.links["cam1-cam2"].length_m, 500, delta=1)
        self.assertEqual(net.links["cam2-cam3"].length_m, 520.0)


class SectionSpeedTests(unittest.TestCase):
    def test_known_distance_and_time_give_the_known_speed(self):
        g = geos_at(0, 1000)
        # 1000 m x 1.4 circuity = 1400 m in 100 s = 14 m/s = 50.4 km/h
        segs = ss.segments_of_vehicle([tk(1, 7, 1000.0), tk(2, 7, 1100.0)], g)
        self.assertEqual(len(segs), 1)
        s = segs[0]
        self.assertEqual(s["distance_basis"], "straight_line")
        self.assertAlmostEqual(s["distance_m"], 1400, delta=2)
        self.assertEqual(s["travel_s"], 100.0)
        self.assertAlmostEqual(s["speed_kph"], 50.4, delta=0.2)
        self.assertIsNone(s["rejected"])

    def test_road_distance_wins_and_is_labelled(self):
        g = geos_at(0, 1000)
        net = geo_mod.build_network(g, [(1, 2, 1200.0)])
        fn = ss.network_road_fn(net)
        s = ss.segments_of_vehicle([tk(1, 7, 0.0), tk(2, 7, 100.0)], g, fn)[0]
        self.assertEqual(s["distance_basis"], "road")
        self.assertEqual(s["distance_m"], 1200.0)
        self.assertAlmostEqual(s["speed_kph"], 43.2, delta=0.1)

    def test_no_route_falls_back_to_straight_line(self):
        g = geos_at(0, 1000, 2000)
        fn = ss.network_road_fn(geo_mod.build_network(g, [(1, 2, 1200.0)]))     # camera 3 is not connected
        self.assertEqual(ss.pair_distance(1, 2, g, fn)[1], "road")
        self.assertEqual(ss.pair_distance(2, 3, g, fn)[1], "straight_line")

    def test_clock_offset_is_applied(self):
        g = geos_at(0, 1000, offsets={2: 40.0})          # camera 2's clock runs 40 s slow
        s = ss.segments_of_vehicle([tk(1, 1, 0.0), tk(2, 1, 60.0)], g)[0]
        self.assertEqual(s["travel_s"], 100.0)

    def test_implausible_segments_are_rejected_with_a_reason(self):
        g = geos_at(0, 1000, 1010)
        cases = {"clock_inconsistent": (g, [tk(1, 1, 100.0), tk(2, 1, 100.0)]),
                 "too_fast": (g, [tk(1, 1, 0.0), tk(2, 1, 5.0)]),                  # 1400 m in 5 s = 1008 km/h
                 "too_slow": (g, [tk(1, 1, 0.0), tk(2, 1, 1700.0)]),               # 2.96 km/h is allowed...
                 "gap_too_long": (g, [tk(1, 1, 0.0), tk(2, 1, 3600.0)]),
                 "cameras_too_close": (g, [tk(2, 1, 0.0), tk(3, 1, 10.0)])}        # 10 m apart
        cases["too_slow"] = (g, [tk(1, 1, 0.0), tk(2, 1, 1790.0)])                 # 2.8 km/h: still above 2
        for reason, (gg, rows) in cases.items():
            s = ss.segments_of_vehicle(rows, gg)[0]
            if reason == "too_slow":
                self.assertIsNone(s["rejected"])        # 2.8 km/h is inside the bound
                s = ss.segments_of_vehicle([tk(1, 1, 0.0), tk(2, 1, 1799.0)], geos_at(0, 100))[0]   # 0.28 km/h
            self.assertEqual(s["rejected"], reason)
            self.assertIsNone(s["speed_kph"])

    def test_aggregate_counts_rejections_and_reports_stats(self):
        g = geos_at(0, 1000)          # 1400 m
        rows = []
        for gid, dt in enumerate((100, 110, 120, 130, 140, 150), 1):          # 6 plausible vehicles
            rows += [tk(1, gid, 1000.0 + gid), tk(2, gid, 1000.0 + gid + dt)]
        rows += [tk(1, 50, 1000.0), tk(2, 50, 1003.0)]                        # too fast
        rows += [tk(1, 51, 1000.0), tk(2, 51, 1000.0)]                        # same instant
        out = ss.compute(rows, g)
        (p,) = out["pairs"]
        self.assertEqual((p["from"], p["to"], p["n"], p["n_vehicles"]), (1, 2, 6, 6))
        self.assertEqual(p["rejected"]["too_fast"], 1)
        self.assertEqual(p["rejected"]["clock_inconsistent"], 1)
        self.assertEqual(p["n_rejected"], 2)
        self.assertEqual(out["rejected"]["too_fast"], 1)
        speeds = sorted(1400.0 / dt * 3.6 for dt in (100, 110, 120, 130, 140, 150))
        self.assertAlmostEqual(p["speed"]["median_kph"], (speeds[2] + speeds[3]) / 2, delta=0.1)
        self.assertAlmostEqual(p["speed"]["mean_kph"], sum(speeds) / 6, delta=0.1)
        self.assertAlmostEqual(p["speed"]["space_mean_kph"], 1400.0 / 125.0 * 3.6, delta=0.1)
        self.assertLess(p["speed"]["p15_kph"], p["speed"]["median_kph"])
        self.assertGreater(p["speed"]["p85_kph"], p["speed"]["median_kph"])
        self.assertEqual(out["bounds"]["max_speed_kph"], 120.0)

    def test_too_few_vehicles_give_no_number(self):
        g = geos_at(0, 1000)
        rows = []
        for gid in range(1, 5):                         # four vehicles; five are needed
            rows += [tk(1, gid, 0.0), tk(2, gid, 100.0)]
        (p,) = ss.compute(rows, g)["pairs"]
        self.assertFalse(p["enough"])
        self.assertIsNone(p["speed"])
        self.assertNotIn("speeds_kph", p)
        self.assertEqual(p["n_vehicles"], 4)
        self.assertIn("at least 5", p["reason"])

    def test_one_vehicle_looping_does_not_count_as_many(self):
        g = geos_at(0, 1000)
        rows = []
        for k in range(6):                              # the same car, six trips
            rows += [tk(1, 1, 100.0 * 10 * k), tk(2, 1, 100.0 * 10 * k + 100)]
        p = [q for q in ss.compute(rows, g)["pairs"] if (q["from"], q["to"]) == (1, 2)][0]
        self.assertEqual((p["n"], p["n_vehicles"], p["enough"]), (6, 1, False))

    def test_window_uses_arrival_time_and_direction_is_separate(self):
        g = geos_at(0, 1000)
        rows = [tk(1, 1, 0.0), tk(2, 1, 100.0), tk(1, 1, 200.0)]       # A->B then B->A
        out = ss.compute(rows, g)
        self.assertEqual({(p["from"], p["to"]) for p in out["pairs"]}, {(1, 2), (2, 1)})
        out = ss.compute(rows, g, since=150.0)
        self.assertEqual([(p["from"], p["to"]) for p in out["pairs"]], [(2, 1)])

    def test_unplaced_cameras_and_unlinked_tracklets_are_ignored(self):
        g = geos_at(0, 1000)
        rows = [tk(1, 1, 0.0), tk(9, 1, 50.0), tk(2, 1, 100.0), {"camera_id": 1, "global_id": None, "t_cross": 5.0}]
        segs = ss.segments_of_vehicle([r for r in rows if r["global_id"]], g)
        self.assertEqual([(s["from"], s["to"]) for s in segs], [(1, 2)])       # 9 is not on the map

    def test_percentile(self):
        self.assertEqual(ss.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(ss.percentile([5], 85), 5)


class FakeAuth:
    """Role from the x-role header: driver / controller / admin / none."""

    def __init__(self):
        def current_user(request: Request):
            role = request.headers.get("x-role")
            return None if not role else {"user_id": 1, "role": role, "name": role.title(), "username": role,
                                          "address": "10.0.0.7"}

        def require(*roles):
            def guard(user=Depends(current_user)):
                if user is None:
                    raise HTTPException(401, "Sign in to continue")
                if not any(auth.can(user["role"], r) for r in roles):
                    raise HTTPException(403, "Your account does not have access to this")
                return user
            return guard
        self.current_user, self.require = current_user, require


class ApiTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = Store(Path(self._dir.name) / "s.db")
        self.addCleanup(self.store.db.close)
        self.c1 = self.store.add_camera("Farmgate", "file", "/secret/path/a.mp4", "Farmgate")
        self.c2 = self.store.add_camera("Karwan Bazar", "file", "rtsp://user:pw@host/b", "Karwan Bazar")
        fa = FakeAuth()
        self.env = {}
        app = FastAPI()
        web = Path(self._dir.name) / "web"
        web.mkdir()
        (web / "map.html").write_text("<html>map</html>")
        app.include_router(api_map.make_router(self.store, fa.current_user, fa.require, "controller", "admin",
                                               status_fn=lambda cid: {"state": "running" if cid == self.c1 else "idle"},
                                               web_dir=web, env=self.env))
        self.app = app

    def call(self, method, path, role=None, body=None):
        headers = {"x-role": role} if role else {}
        data = b""
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        status, hdrs, out = asgi_call(self.app, method, path, headers, data)
        try:
            return status, json.loads(out)
        except ValueError:
            return status, out.decode()

    def place(self, cid, **kw):
        return self.call("PUT", f"/api/map/cameras/{cid}", "admin", {"lat": LAT, "lon": LON, **kw})

    def link(self, rows):
        for gid, (a, ta, b, tb) in enumerate(rows, 1):
            pass

    def add_pass(self, cam, gid, t):
        class T:
            track_id, cls, class_conf, zone, lateral, speed, width = 1, "car", 0.9, "z", 0.0, 5.0, 1.8
            direction, extrap_m, hist, lab, embed, link_source, video_t = "fwd", 0.0, None, None, None, "batch", 0.0
        T.t_cross, T.global_id = t, gid
        self.store.add_tracklet(cam, None, T)

    # ---- roles
    def test_roles_on_every_endpoint(self):
        endpoints = [("GET", "/api/map/cameras", "staff"), ("GET", "/api/map/section-speeds", "staff"),
                     ("GET", "/api/map/vehicle/1/segments", "staff"), ("GET", "/api/map/config", "staff"),
                     ("PUT", f"/api/map/cameras/{self.c1}", "admin"), ("DELETE", f"/api/map/cameras/{self.c1}", "admin"),
                     ("GET", "/api/settings/maps", "admin"), ("PUT", "/api/settings/maps", "admin")]
        for method, path, need in endpoints:
            body = {"lat": LAT, "lon": LON} if method == "PUT" else None
            self.assertEqual(self.call(method, path, None, body)[0], 401, path)
            self.assertEqual(self.call(method, path, "driver", body)[0], 403, path)
            ctrl = self.call(method, path, "controller", body)[0]
            self.assertEqual(ctrl, 200 if need == "staff" else 403, f"{method} {path}")
            self.assertIn(self.call(method, path, "admin", body)[0], (200, 404), f"{method} {path}")

    def test_page_redirects(self):
        status, _, _ = asgi_call(self.app, "GET", "/map")
        self.assertEqual(status, 302)
        for role, ok in (("driver", False), ("controller", True), ("admin", True)):
            s, h, body = asgi_call(self.app, "GET", "/map", {"x-role": role})
            self.assertEqual(s, 200 if ok else 302, role)
            if not ok:
                self.assertEqual(h["location"], "/portal")

    # ---- cameras
    def test_place_list_and_audit(self):
        status, out = self.place(self.c1, heading_deg=-90, fov_deg=70, range_m=120)
        self.assertEqual(status, 200)
        self.assertEqual(out["geo"]["heading_deg"], 270.0)
        self.assertEqual(len(out["geo"]["wedge"]), 14)
        s, cams = self.call("GET", "/api/map/cameras", "controller")
        by = {c["id"]: c for c in cams["cameras"]}
        self.assertTrue(by[self.c1]["placed"])
        self.assertFalse(by[self.c2]["placed"])
        self.assertEqual((by[self.c1]["state"], by[self.c2]["state"]), ("running", "idle"))
        self.assertFalse(cams["can_edit"])
        self.assertNotIn("source", by[self.c2])          # RTSP credentials never leave the server
        self.assertNotIn("rtsp", json.dumps(cams))
        rows = self.store.audit_trail("camera", str(self.c1))
        self.assertEqual(rows[0]["action"], "camera.geo")
        self.assertEqual(rows[0]["role"], "admin")
        self.assertEqual(rows[0]["address"], "10.0.0.7")

    def test_bad_placements_are_400_and_unknown_camera_404(self):
        self.assertEqual(self.call("PUT", f"/api/map/cameras/{self.c1}", "admin", {"lat": 95, "lon": LON})[0], 400)
        self.assertEqual(self.place(self.c1, fov_deg=999)[0], 400)
        self.assertEqual(self.call("PUT", f"/api/map/cameras/{self.c1}", "admin", [1, 2])[0], 400)
        self.assertEqual(self.place(9999)[0], 404)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM camera_geo")[0]["n"], 0)

    def test_outside_dhaka_returns_a_warning(self):
        s, out = self.call("PUT", f"/api/map/cameras/{self.c1}", "admin", {"lat": 40.7, "lon": -74.0})
        self.assertEqual(s, 200)
        self.assertEqual(len(out["warnings"]), 1)

    def test_remove_and_cascade(self):
        self.place(self.c1)
        self.assertEqual(self.call("DELETE", f"/api/map/cameras/{self.c1}", "admin")[0], 200)
        self.assertEqual(self.call("DELETE", f"/api/map/cameras/{self.c1}", "admin")[0], 404)
        self.place(self.c2)
        self.store.x("DELETE FROM cameras WHERE id=?", (self.c2,))
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM camera_geo")[0]["n"], 0)

    # ---- speeds over HTTP
    def test_section_speeds_and_vehicle_segments_over_http(self):
        self.place(self.c1)
        self.call("PUT", f"/api/map/cameras/{self.c2}", "admin", {"lat": lat_for(1000), "lon": LON})
        for gid in range(1, 7):
            self.add_pass(self.c1, gid, 1000.0 + gid)
            self.add_pass(self.c2, gid, 1100.0 + gid)
        s, out = self.call("GET", "/api/map/section-speeds", "controller")
        self.assertEqual(s, 200)
        (p,) = out["pairs"]
        self.assertEqual((p["n_vehicles"], p["distance_basis"], p["from_name"]), (6, "straight_line", "Farmgate"))
        self.assertAlmostEqual(p["speed"]["median_kph"], 50.4, delta=0.3)
        s, seg = self.call("GET", "/api/map/vehicle/3/segments", "controller")
        self.assertEqual(len(seg["segments"]), 1)
        self.assertAlmostEqual(seg["segments"][0]["speed_kph"], 50.4, delta=0.3)
        s, out = self.call("GET", "/api/map/section-speeds?since=2000&until=3000", "controller")
        self.assertEqual(out["pairs"], [])
        self.assertEqual(self.call("GET", "/api/map/section-speeds?since=banana", "controller")[0], 400)

    # ---- keys
    def test_no_key_means_openstreetmap(self):
        s, cfg = self.call("GET", "/api/map/config", "controller")
        self.assertEqual((cfg["provider"], cfg["google_browser_key"], cfg["traffic_layer"]), ("osm", None, False))
        self.assertIn("openstreetmap", cfg["osm"]["attribution"].lower())

    def test_server_key_never_leaves_the_server(self):
        s, out = self.call("PUT", "/api/settings/maps", "admin", {"browser_key": BROWSER, "server_key": SERVER})
        self.assertEqual(s, 200)
        self.assertEqual(out["server_key"]["hint"], "set, ends …1234")
        responses = [out, self.call("GET", "/api/settings/maps", "admin")[1],
                     self.call("GET", "/api/map/config", "controller")[1],
                     self.call("GET", "/api/map/config", "admin")[1],
                     self.call("GET", "/api/map/cameras", "admin")[1],
                     self.call("PUT", "/api/settings/maps", "admin", {"server_key": "bad key!"})[1],
                     self.call("PUT", "/api/settings/maps", "admin", {"server_key": 5})[1],
                     self.call("PUT", "/api/settings/maps", "controller", {"server_key": SERVER})[1],
                     self.call("PUT", "/api/settings/maps", "driver", {"server_key": SERVER})[1]]
        for r in responses:
            self.assertNotIn(SERVER, json.dumps(r))
        audit = json.dumps(self.store.q("SELECT * FROM audit_log"))
        self.assertNotIn(SERVER, audit)
        self.assertNotIn(BROWSER, audit)
        self.assertNotIn("1234", audit)
        self.assertIn("settings.maps", audit)
        self.assertEqual(api_map.server_key(self.store, {}), SERVER)         # server code can still read it
        cfg = self.call("GET", "/api/map/config", "controller")[1]
        self.assertEqual((cfg["provider"], cfg["google_browser_key"], cfg["traffic_layer"]), ("google", BROWSER, True))

    def test_environment_wins_and_clear_works(self):
        self.call("PUT", "/api/settings/maps", "admin", {"browser_key": BROWSER})
        self.env["GOOGLE_MAPS_BROWSER_KEY"] = "AIzaSyFromTheEnvironment0000000"
        cfg = self.call("GET", "/api/map/config", "controller")[1]
        self.assertEqual(cfg["google_browser_key"], "AIzaSyFromTheEnvironment0000000")
        self.assertEqual(self.call("GET", "/api/settings/maps", "admin")[1]["browser_key"]["source"], "env")
        self.env.clear()
        out = self.call("PUT", "/api/settings/maps", "admin", {"browser_key": ""})[1]
        self.assertEqual(out["browser_key"], {"set": False, "source": "none", "hint": None})
        self.assertEqual(self.call("GET", "/api/map/config", "controller")[1]["provider"], "osm")

    def test_network_fn_gives_road_distance(self):
        self.place(self.c1)
        self.call("PUT", f"/api/map/cameras/{self.c2}", "admin", {"lat": lat_for(1000), "lon": LON})
        for gid in range(1, 7):
            self.add_pass(self.c1, gid, 1000.0 + gid)
            self.add_pass(self.c2, gid, 1100.0 + gid)
        fa = FakeAuth()
        app = FastAPI()
        net = geo_mod.build_network(geo_mod.all_geo(self.store), [(self.c1, self.c2, 1200.0)])
        app.include_router(api_map.make_router(self.store, fa.current_user, fa.require, network_fn=lambda: net))
        status, _, body = asgi_call(app, "GET", "/api/map/section-speeds", {"x-role": "admin"})
        (p,) = json.loads(body)["pairs"]
        self.assertEqual((p["distance_basis"], p["distance_m"]), ("road", 1200.0))
        self.assertAlmostEqual(p["speed"]["median_kph"], 43.2, delta=0.3)


if __name__ == "__main__":
    unittest.main()
