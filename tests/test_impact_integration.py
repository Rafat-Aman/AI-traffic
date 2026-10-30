"""The impact work wired into the running app: pipeline -> collector, worker -> stored impacts, server -> staff views.

tests/test_impact.py shows the measuring and scoring behave on synthetic traffic; these pin the seams: the collector
sees the rule engine's own observations, a fault in impact code never stops the analysis, a recorded violation ends
up with an impact row, staff (and only staff) see impact and priority, and fitted weights are only adopted when the
reviewers' grades say they beat equal weights.

Importing the dashboard server opens its database, so tests.test_citation (imported below) points it at a scratch
folder first; nothing here touches data/surveillance.db.
"""

import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests import test_camera_motion as cm
from tests import test_citation as tc
from tests.test_impact import BUSY, Run, road
from surveillance import impact as im
from surveillance import worker as W
from surveillance.pipeline import Pipeline
from surveillance.store import Store
from surveillance.traffic_state import Baseline, TrafficStateCollector
from violations.base import Observation, ViolationEvent
from violations.zones import camera_config_from_dict


class Recorder:
    """Stands in for the collector and keeps what the pipeline hands it."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def observe(self, observations, t, wall=None, valid=True):
        self.calls.append((list(observations), t, wall, valid))
        if self.fail:
            raise ValueError("boom")


class PipelineFeedsTheCollector(unittest.TestCase):
    def make(self, traffic):
        ref = cm.scene()
        cfg = {"frame_size": [cm.W, cm.H], "zones": [{"type": "road", "name": "Road", "points": cm.ROAD,
                                                       "width_m": 10, "length_m": 60,
                                                       "allowed_direction": "toward_camera"}]}
        det = cm.FakeDetector()
        det.boxes = [((560.0, 330.0, 640.0, 390.0), 1, "Car")]
        cam = camera_config_from_dict(cfg, cm.W, cm.H)
        p = Pipeline(det, cam, (cm.W, cm.H), reference=ref, auto_roi=False, stabilize=True, traffic=traffic)
        engine_saw = []
        real = p.engine.process
        p.engine.process = lambda obs, *a, **k: (engine_saw.append(obs), real(obs, *a, **k))[1]
        return p, ref, engine_saw

    def test_the_collector_gets_the_very_list_the_rule_engine_gets(self):
        rec = Recorder()
        p, ref, engine_saw = self.make(rec)
        for i in range(6):
            p.process(cm.view(ref, cm.similarity(0, 0), seed=i), i * 0.1)
        p.close()
        self.assertEqual(len(rec.calls), 6)
        for (obs, _, _, valid), seen in zip(rec.calls, engine_saw):
            self.assertIs(obs[0] if obs else None, seen[0] if seen else None)
            self.assertEqual(len(obs), len(seen))
            self.assertTrue(valid)
        self.assertTrue(any(c[0] for c in rec.calls), "the car was never an observation: the test checks nothing")

    def test_a_lost_view_is_passed_on_as_not_valid(self):
        rec = Recorder()
        p, ref, _ = self.make(rec)
        p.process(cm.view(ref, cm.similarity(0, 0)), 0.0)
        for i in range(1, 6):          # the picture leaves the zones altogether
            p.process(cm.view(ref, cm.similarity(-850, 0), seed=i), i * 0.1)
        p.close()
        self.assertEqual(p.motion.state, "lost")
        self.assertTrue(rec.calls[0][3])
        self.assertFalse(rec.calls[-1][3], "an empty zone seen through a lost view is not an empty road")

    def test_a_fault_in_the_collector_does_not_stop_the_rules(self):
        p, ref, engine_saw = self.make(Recorder(fail=True))
        with mock.patch("builtins.print") as said:
            for i in range(4):
                p.process(cm.view(ref, cm.similarity(0, 0), seed=i), i * 0.1)
        p.close()
        self.assertEqual(len(engine_saw), 4)
        self.assertEqual(sum("traffic state not measured" in str(c) for c in said.call_args_list), 1)


class CollectorSnapshotsAreThreadSafe(unittest.TestCase):
    def test_reading_while_the_worker_thread_feeds_it(self):
        rd = road()
        col = TrafficStateCollector(1, [rd])
        errors, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    json.dumps(col.snapshot(), allow_nan=False)
                except Exception as e:      # noqa: BLE001
                    errors.append(e)
                    return

        th = threading.Thread(target=reader)
        th.start()
        try:
            for t in range(400):
                col.observe([Observation(tid=k, frame_idx=t, time_s=float(t), cls="Car", purity=1.0, n_frames=9,
                                         bbox=(0, 0, 1, 1), point=(0, 0), road=rd, xy=(5.0, 10.0 + 5 * k + t % 7),
                                         kph=float(t % 40), heading=1) for k in range(12)], float(t))
        finally:
            stop.set()
            th.join()
        self.assertEqual(errors, [])


class WorkerScratch(unittest.TestCase):
    """A CameraWorker that is not running, over a scratch store, with the collector of a simulated busy approach."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = Run(BUSY, 1, bus=True)

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.store = Store(self.root / "s.db")
        self.addCleanup(self.store.db.close)
        self.cid = self.store.add_camera("Farmgate", "file", "v.mp4")        # id 1, as the collector was made for
        self.sent = []
        self.worker = W.CameraWorker(self.store, self.store.camera(self.cid), self.root, self.sent.append,
                                     baseline=Baseline(self.store))
        self.worker.pipeline = SimpleNamespace(start_time=None, is_live=False)
        self.worker.traffic = self.scenario.col
        self.worker.impacts = im.ImpactTracker(self.scenario.col, None, self.cid)

    def event(self):
        stopped = self.scenario.sim.stopper["hold"]["stopped_at"]
        obs = Observation(tid=self.scenario.tid, frame_idx=1, time_s=stopped + 15, cls="Bus", purity=1.0, n_frames=30,
                          bbox=(0, 0, 20, 20), point=(10.0, 10.0), road=self.scenario.rd, xy=(5.0, 55.0), kph=0.0)
        return ViolationEvent("V1", "Bus stopped outside a bus stop", obs, {"dwell_s": 15.0}, started_s=stopped)

    def record(self):
        frame = np.zeros((60, 80, 3), np.uint8)
        clips = SimpleNamespace(trigger=lambda *a: None)
        self.worker._record(self.event(), frame, None, self.root / "evidence", clips, None, None, self.event().obs.time_s)
        return self.store.q("SELECT id FROM violations")[0]["id"]


class WorkerStoresImpacts(WorkerScratch):
    def test_a_recorded_violation_is_scored_once_its_window_has_closed(self):
        vid = self.record()
        self.assertEqual(len(self.worker.impacts.pending), 1)
        t1 = self.scenario.win[1]
        self.worker._poll_impacts(t1 - 1)
        self.assertIsNone(im.load_impact(self.store, vid), "not scored before its window closed")
        self.worker._poll_impacts(t1)
        got = im.load_impact(self.store, vid)
        self.assertTrue(got["meaningful"])
        self.assertEqual(got["code"], "V1")
        self.assertEqual(got["excluded_tids"], [self.scenario.tid])
        msg = [m for m in self.sent if m["type"] == "impact"]
        self.assertEqual([(m["violation_id"], m["camera_id"]) for m in msg], [(vid, self.cid)])
        self.assertEqual(msg[0]["impact"]["score"], got["score"])
        self.assertIn("priority", msg[0]["priority"])
        json.dumps(msg[0], allow_nan=False)

    def test_what_is_still_waiting_when_the_run_ends_is_scored_on_the_part_that_exists(self):
        vid = self.record()
        with mock.patch.object(self.scenario.col, "close") as closed:
            self.worker._finish_impact()
        self.assertIsNotNone(im.load_impact(self.store, vid))
        self.assertEqual(self.worker.impacts.pending, [])
        closed.assert_called_once()

    def test_storing_never_raises_into_the_video_loop(self):
        vid = self.record()
        with mock.patch.object(W, "save_impact", side_effect=sqlite3.OperationalError("disk I/O error")), \
                mock.patch("builtins.print") as said:
            self.worker._poll_impacts(self.scenario.win[1])        # must not raise
        self.assertTrue(any("not stored" in str(c) for c in said.call_args_list))
        self.assertEqual([m for m in self.sent if m["type"] == "impact"], [])
        self.assertIsNone(im.load_impact(self.store, vid))

    def test_a_case_cleared_while_its_window_was_open_is_a_logged_miss_not_a_crash(self):
        self.record()
        self.store.x("DELETE FROM violations")
        with mock.patch("builtins.print") as said:
            self.worker._poll_impacts(self.scenario.win[1])
        self.assertTrue(any("not stored" in str(c) for c in said.call_args_list))

    def test_a_broken_notifier_does_not_raise_either(self):
        self.record()
        self.worker.notify = mock.Mock(side_effect=RuntimeError("loop closed"))
        with mock.patch("builtins.print"):
            self.worker._poll_impacts(self.scenario.win[1])

    def test_an_event_off_the_road_zones_is_not_scheduled(self):
        ev = self.event()
        ev.obs.road = None
        self.worker._schedule_impact(1, ev)
        self.assertEqual(self.worker.impacts.pending, [])

    def test_impact_switched_off_costs_nothing(self):
        self.worker.impacts = self.worker.traffic = None
        self.record()
        self.worker._poll_impacts(1e9)
        self.worker._finish_impact()
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM event_impact")[0]["n"], 0)

    def test_a_store_without_the_tables_cannot_be_started_by_mistake(self):
        """Store creates the impact tables itself, so every Store a tool or test opens can serve the staff views."""
        names = {r["name"] for r in self.store.q("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"event_impact", "impact_weights", "traffic_baseline"} <= names)


class Api(tc.ServerScratch):
    """The staff and portal views over a scratch store."""

    def case_with_impact(self, score=40.0, calibrated=False, code="V1", delay=120.0, grade=None, deltas=None):
        vid = self.case(code=code)
        deltas = deltas or {q: 1.0 for q in im.QUANTITIES}
        im.save_impact(self.store, vid, {"score": score, "meaningful": True, "calibrated": calibrated,
                                         "camera_id": self.cam, "zone": "z", "code": code,
                                         "baseline_source": "pre-event", "excess_delay_veh_s": delay,
                                         "deltas": deltas})
        if grade is not None:
            im.set_human_severity(self.store, vid, grade)
        return vid


class StaffViewsCarryImpact(Api):
    def test_list_get_and_review_carry_impact_and_priority(self):
        vid = self.case_with_impact(score=40.0, calibrated=True)
        lst = self.get_json("/api/violations", self.ctrl)[1]["items"]
        one = self.get_json(f"/api/violations/{vid}", self.boss)[1]
        reviewed = self.call("PATCH", f"/api/violations/{vid}", self.ctrl, {"status": "confirmed", "note": ""})
        for v in (lst[0], one, json.loads(reviewed[2])):
            self.assertEqual(v["impact"]["score"], 40.0)
            self.assertEqual(v["priority"]["severity"], "medium")
            self.assertIn("reason", v["priority"])

    def test_a_case_not_yet_scored_says_so_and_still_gets_a_priority(self):
        vid = self.case(code="V4")
        one = self.get_json(f"/api/violations/{vid}", self.ctrl)[1]
        self.assertIsNone(one["impact"])
        self.assertEqual(one["priority"]["severity"], "medium")           # V4 is never shown below medium
        self.assertIn("unknown", one["priority"]["reason"])

    def test_the_driver_portal_never_carries_either(self):
        self.issue(self.case_with_impact(), delivery=tc.fake_delivery)     # the portal lists cited cases only
        listing = self.get_json("/api/portal/violations", self.drv)[1]["items"]
        self.assertTrue(listing)
        one = self.get_json(f"/api/portal/violation/{listing[0]['id']}", self.drv)[1]
        for item in (*listing, one):
            self.assertNotIn("impact", item)
            self.assertNotIn("priority", item)
            self.assertNotIn("human_severity", json.dumps(item))


class SeverityRatings(Api):
    def rate(self, vid, who, grade):
        return self.post_json(f"/api/violations/{vid}/severity", who, {"grade": grade})

    def test_a_controller_rates_and_it_is_audited_and_shown(self):
        vid = self.case_with_impact()
        status, out = self.rate(vid, self.ctrl, 4)
        self.assertEqual((status, out["impact"]["human_severity"]), (200, 4.0))
        self.assertEqual([(a["action"], a["detail"]) for a in self.store.audit_trail("violation", vid)],
                         [("violation.severity", "4/5")])
        status, out = self.rate(vid, self.boss, None)
        self.assertEqual((status, out["impact"]["human_severity"]), (200, None))

    def test_a_driver_and_a_stranger_cannot(self):
        vid = self.case_with_impact()
        self.assertEqual(self.call("POST", f"/api/violations/{vid}/severity", None, {"grade": 3})[0], 401)
        self.assertEqual(self.rate(vid, self.drv, 3)[0], 403)
        self.assertIsNone(im.load_impact(self.store, vid)["human_severity"])

    def test_bad_grades_unknown_cases_and_unscored_cases_are_refused(self):
        vid = self.case_with_impact()
        for bad in (0, 6, -1):
            self.assertEqual(self.rate(vid, self.ctrl, bad)[0], 400, bad)
        self.assertEqual(self.rate(999, self.ctrl, 3)[0], 404)
        self.assertEqual(self.rate(self.case(), self.ctrl, 3)[0], 409)


class CalibrationEndpoints(Api):
    """Fitted weights are adopted only when the reviewers' grades say they beat equal weights."""

    def fill(self, grade_by, n=30, seed=3):
        """n scored cases whose delay depends on the queue alone; `grade_by(deltas)` is the reviewer's grade."""
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            d = dict(zip(im.QUANTITIES, rng.uniform(0, 5, 5).round(3).tolist()))
            rows.append(d)
        order = np.argsort(np.argsort([grade_by(d) for d in rows]))
        for d, rank in zip(rows, order):
            self.case_with_impact(score=float(np.mean(list(d.values()))), delay=20.0 * d["queue_m"], deltas=d,
                                  grade=1 + int(5 * rank / n))

    def test_only_an_administrator_can_calibrate_or_validate(self):
        for method, path in (("POST", "/api/impact/calibrate"), ("GET", "/api/impact/validation")):
            self.assertEqual(self.call(method, path)[0], 401, path)
            self.assertEqual(self.call(method, path, self.drv)[0], 403, path)
            self.assertEqual(self.call(method, path, self.ctrl)[0], 403, path)
            self.assertEqual(self.call(method, path, self.boss)[0], 200, path)

    def test_with_nothing_to_fit_equal_weights_stay_and_it_says_so(self):
        status, out = self.post_json("/api/impact/calibrate", self.boss)
        self.assertEqual((status, out["adopted"], out["n_events"]), (200, False, 0))
        self.assertFalse(out["weights_in_use"]["calibrated"])
        self.assertEqual(out["weights_in_use"]["label"], "equal weights (uncalibrated)")
        self.assertIn("equal weights stay", out["reason"])

    def test_equal_weights_are_kept_when_they_validate_better(self):
        self.fill(lambda d: sum(d.values()))                 # reviewers weigh all five quantities alike
        status, out = self.post_json("/api/impact/calibrate", self.boss)
        self.assertEqual(status, 200)
        self.assertTrue(out["candidate"]["calibrated"], "the fit itself succeeded")
        cmp = out["comparison"]
        self.assertGreater(cmp["equal"]["rho"], cmp["candidate"]["rho"])
        self.assertFalse(out["adopted"])
        self.assertFalse(out["weights_in_use"]["calibrated"])
        self.assertFalse(im.load_weights(self.store).any_calibrated())
        self.assertEqual(out["n_events"], 30)
        audit = self.store.audit_trail("impact", "weights")[0]
        self.assertEqual((audit["action"], audit["role"]), ("impact.calibrate", "admin"))
        self.assertIn("kept equal weights", audit["detail"])

    def test_calibrated_weights_are_adopted_when_they_validate_better_and_revoked_when_they_stop_to(self):
        self.fill(lambda d: d["queue_m"])                    # reviewers care about the queue, as the delay does
        out = self.post_json("/api/impact/calibrate", self.boss)[1]
        self.assertTrue(out["adopted"], out["reason"])
        self.assertGreater(out["comparison"]["candidate"]["rho"], out["comparison"]["equal"]["rho"])
        self.assertTrue(out["weights_in_use"]["calibrated"])
        self.assertTrue(im.load_weights(self.store).any_calibrated())
        # the reviewers now grade the other way round: the standing fit no longer beats equal weights
        self.store.x("UPDATE event_impact SET human_severity = 6 - human_severity")
        again = self.post_json("/api/impact/calibrate", self.boss)[1]
        self.assertFalse(again["adopted"])
        self.assertFalse(again["weights_in_use"]["calibrated"])
        self.assertFalse(im.load_weights(self.store).any_calibrated())

    def test_too_few_graded_cases_never_adopt(self):
        self.fill(lambda d: d["queue_m"], n=16)
        self.store.x("UPDATE event_impact SET human_severity = NULL WHERE violation_id > 5")
        out = self.post_json("/api/impact/calibrate", self.boss)[1]
        self.assertFalse(out["adopted"])
        self.assertEqual(out["comparison"]["n_graded"], 5)
        self.assertIn("at least 10", out["reason"])

    def test_validation_reports_both_weightings_on_the_same_cases(self):
        self.fill(lambda d: sum(d.values()), n=12)
        out = self.get_json("/api/impact/validation", self.boss)[1]
        self.assertEqual((out["n_impacts"], out["n_scored"], out["n_graded"]), (12, 12, 12))
        self.assertGreater(out["equal"]["rho"], 0.8)
        self.assertEqual(out["equal"]["rho"], out["in_use"]["rho"], "equal weights are what is in use")


class TrafficEndpoint(Api):
    def test_staff_only_and_a_clear_answer_when_the_camera_is_not_running(self):
        path = f"/api/cameras/{self.cam}/traffic"
        self.assertEqual(self.call("GET", path)[0], 401)
        self.assertEqual(self.call("GET", path, self.drv)[0], 403)
        status, out = self.get_json(path, self.ctrl)
        self.assertEqual((status, out["running"], out["zones"]), (200, False, {}))
        self.assertIn("not running", out["message"])
        self.assertEqual(self.call("GET", "/api/cameras/999/traffic", self.ctrl)[0], 404)

    def test_a_running_camera_reports_each_zone(self):
        scenario = Run(BUSY, 1, bus=True)
        fake = SimpleNamespace(traffic=scenario.col, baseline=Baseline(self.store), is_alive=lambda: True,
                               status={"video_time": 12.5}, cam={"settings": {"impact": True}})
        with mock.patch.object(self.srv.manager, "workers", {self.cam: fake}):
            status, out = self.get_json(f"/api/cameras/{self.cam}/traffic", self.ctrl)
        self.assertEqual((status, out["running"]), (200, True))
        zone = out["zones"]["Test Rd"]
        for q in im.QUANTITIES:
            self.assertIn(q, zone)
        self.assertEqual(zone["baseline"], {"source": "none"})
        self.assertEqual(out["video_time"], 12.5)

    def test_measuring_switched_off_is_said_not_hidden(self):
        fake = SimpleNamespace(traffic=None, baseline=None, is_alive=lambda: True, status={},
                               cam={"settings": {"impact": False}})
        with mock.patch.object(self.srv.manager, "workers", {self.cam: fake}):
            out = self.get_json(f"/api/cameras/{self.cam}/traffic", self.ctrl)[1]
        self.assertFalse(out["running"])
        self.assertIn("switched off", out["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
