"""Who may do what: police (controller) work cases and start/stop analysis; only admins create, edit, delete or
re-learn cameras, zones, settings and results; drivers reach nothing staff-only."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_citation as tc  # noqa: E402

ADMIN_ONLY = [
    ("POST", "/api/cameras", {"name": "x", "source": "x.mp4"}),
    ("PATCH", "/api/cameras/{cam}", {"name": "y"}),
    ("DELETE", "/api/cameras/{cam}", None),
    ("POST", "/api/cameras/{cam}/auto-zones", {}),
    ("DELETE", "/api/cameras/{cam}/auto-zones", None),
    ("DELETE", "/api/cameras/{cam}/violations", None),
    ("PUT", "/api/cameras/{cam}/zones", {"zones": []}),
    ("POST", "/api/cameras/{cam}/zones/validate", {"zones": []}),
    ("GET", "/api/vlm/settings", None),
    ("PUT", "/api/vlm/settings", {"settings": {}}),
    ("GET", "/api/users", None),
    ("GET", "/api/audit", None),
    ("GET", "/api/registry", None),
    ("PUT", "/api/map/cameras/{cam}", {}),
    ("PUT", "/api/storage/policy", {}),
    ("PUT", "/api/optimize/network", {}),
    ("POST", "/api/optimize/network/demo", None),
    ("DELETE", "/api/optimize/network", None),
    ("PUT", "/api/settings/maps", {}),
    ("POST", "/api/query/index/rebuild", {}),
]


class PermissionTests(tc.ServerScratch):
    def setUp(self):
        super().setUp()
        self.cam = self.store.add_camera("Cam", "file", "x.mp4")
        mgr = mock.patch.object(self.srv, "manager")
        self.mgr = mgr.start()
        self.addCleanup(mgr.stop)
        self.mgr.status.return_value = {"state": "running"}

    def path(self, p):
        return p.format(cam=self.cam)

    def test_police_get_403_on_every_admin_route(self):
        for m, p, b in ADMIN_ONLY:
            with self.subTest(route=f"{m} {p}"):
                self.assertEqual(self.call(m, self.path(p), self.ctrl, b)[0], 403)

    def test_drivers_get_403_on_every_staff_route(self):
        for m, p, b in ADMIN_ONLY + [("POST", "/api/cameras/{cam}/start", None), ("POST", "/api/cameras/{cam}/stop", None),
                                     ("GET", "/api/cameras", None), ("GET", "/api/violations", None)]:
            with self.subTest(route=f"{m} {p}"):
                self.assertEqual(self.call(m, self.path(p), self.drv, b)[0], 403)

    def test_anonymous_is_401(self):
        for m, p, b in ADMIN_ONLY[:4]:
            self.assertEqual(self.call(m, self.path(p), None, b)[0], 401)

    def test_police_may_stop_and_start_an_existing_camera(self):
        self.assertEqual(self.call("POST", f"/api/cameras/{self.cam}/stop", self.ctrl)[0], 200)
        with mock.patch.object(self.store, "zones", return_value=[{"id": 1}]):
            self.assertEqual(self.call("POST", f"/api/cameras/{self.cam}/start", self.ctrl)[0], 200)

    def test_admin_passes_the_gate_on_auto_zones(self):
        self.assertEqual(self.call("DELETE", f"/api/cameras/{self.cam}/auto-zones", self.boss)[0], 200)


class ExplainStageEventTests(tc.ServerScratch):
    def test_stage_events_reach_the_event_stream_with_the_camera(self):
        vid = self.case()
        sent = []
        with mock.patch.object(self.srv, "notify", sent.append):
            self.srv.vlm_stage({"violation_id": vid, "stage": "reasoning", "message": "Reasoning", "eta_s": 150})
        self.assertEqual(len(sent), 1)
        self.assertEqual((sent[0]["type"], sent[0]["stage"], sent[0]["violation_id"]),
                         ("explain_stage", "reasoning", vid))
        self.assertEqual(sent[0]["camera_id"], self.store.violation(vid)["camera_id"])

    def test_the_server_cascade_reports_through_that_function(self):
        self.assertIs(self.srv.cascade.on_stage, self.srv.vlm_stage)


if __name__ == "__main__":
    unittest.main()
