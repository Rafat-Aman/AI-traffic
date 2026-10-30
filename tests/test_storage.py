"""Storage management and the ingest queue (WP13). Everything runs on temp folders and a temp Store: nothing here
opens data/ or the dashboard server, and every deletion is on files this test created."""

import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

_SCRATCH = tempfile.mkdtemp(prefix="storage-tests-")
os.environ.setdefault("TRAFFIC_DATA_DIR", _SCRATCH)
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from surveillance import api_storage, citation, ingest  # noqa: E402
from surveillance import storage_mgr as sm  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402
from tests.test_map import FakeAuth  # noqa: E402

DAY = 86400.0
NOW = 1_800_000_000.0           # a fixed "today" for every test
Usage = namedtuple("Usage", "total used free")


def iso(ts):
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        for d in ("videos", "evidence", "autozone", "zone_refs"):
            (self.data / d).mkdir(parents=True)
        self.store = Store(self.root / "s.db")
        self.addCleanup(self.store.db.close)
        self.used = 50
        self.busy = set()
        self.mgr = sm.StorageManager(self.store, self.data, now_fn=lambda: NOW,
                                     usage_fn=lambda p: Usage(1000, self.used * 10, 1000 - self.used * 10),
                                     busy_cameras_fn=lambda: self.busy)
        self.admin = {"user_id": 1, "role": "admin", "name": "Admin", "username": "admin", "address": "10.0.0.7"}

    def file(self, rel, size=100, age_days=0.0):
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        t = NOW - age_days * DAY
        os.utime(p, (t, t))
        return p

    def camera(self, video_rel=None, age_days=0.0):
        src = str(self.file(video_rel, 1000, age_days)) if video_rel else "rtsp://x"
        return self.store.add_camera("cam", "file" if video_rel else "live", src)

    def case(self, cid, name, status="pending", age_days=0.0, cited=False, withdrawn=False, with_delivery=False):
        """A case with snapshot, crop and clip on disk, `age_days` since it was decided."""
        base = f"evidence/cam{cid}/run1/{name}"
        for ext, size in ((".jpg", 10), ("_crop.jpg", 5), (".mp4", 50)):
            self.file(base + ext, size, age_days)
        rec = {"code": "V1", "violation": "x", "track_id": 1, "class": "car", "class_confidence": 0.9,
               "video_time_s": 1.0, "started_s": 0.0, "wall_time": "", "zone": "z", "areas": "", "kph": None,
               "details": {}}
        vid = self.store.add_violation(cid, None, None, rec, base + ".jpg", base + "_crop.jpg", base + ".mp4")
        if status != "pending":
            self.store.x("UPDATE violations SET status=?, reviewed_at=? WHERE id=?", (status, iso(NOW - age_days * DAY), vid))
        delivery = None
        if with_delivery:
            delivery = citation.delivery_path(base + ".mp4")
            self.file(delivery, 20, age_days)
        if cited:
            _, made = self.store.create_citation(vid, self.admin, None, "0" * 64, "{}", delivery)
            if withdrawn:
                self.store.x("UPDATE citations SET status='withdrawn' WHERE violation_id=?", (vid,))
        return vid, base

    def planned(self):
        return {i["path"]: i for i in self.mgr.plan(self.admin)["items"]}

    def exists(self, rel):
        return (self.data / rel).exists()


class PolicyTests(Base):
    def test_defaults_and_bounds(self):
        p = self.mgr.policy()
        self.assertEqual((p["raw_video_days"], p["warn_pct"], p["pause_pct"]), (7, 80, 90))
        self.assertFalse(p["legal_retention_confirmed"])
        for bad in ({"raw_video_days": 0}, {"warn_pct": 95}, {"nope": 1}, {"embed_days": "x"},
                    {"legal_retention_days": 30}, {"legal_retention_confirmed": 1}):
            with self.assertRaises(sm.PolicyError, msg=str(bad)):
                self.mgr.set_policy(bad, self.admin)
        self.assertEqual(self.mgr.policy()["raw_video_days"], 7)

    def test_policy_change_persists_and_is_audited(self):
        self.mgr.set_policy({"raw_video_days": 3}, self.admin)
        self.assertEqual(self.mgr.policy()["raw_video_days"], 3)
        self.assertTrue(self.store.audit_trail(object_type="settings", object_id="storage"))


class DecisionTests(Base):
    def test_raw_video_deleted_after_n_days_when_nothing_needs_it(self):
        self.camera("videos/old.mp4", age_days=8)
        self.camera("videos/new.mp4", age_days=2)
        plan = self.planned()
        self.assertIn("videos/old.mp4", plan)
        self.assertEqual(plan["videos/old.mp4"]["rule"], "raw_video")
        self.assertNotIn("videos/new.mp4", plan)

    def test_pending_case_keeps_raw_video(self):
        cid = self.camera("videos/a.mp4", age_days=30)
        self.case(cid, "c1", "pending")
        plan = self.planned()
        self.assertNotIn("videos/a.mp4", plan)
        kept = self.mgr.plan(self.admin)["kept"]
        self.assertIn("raw_video_held", kept)
        self.assertIn("pending", kept["raw_video_held"]["example"])

    def test_standing_citation_keeps_raw_video_but_withdrawn_does_not(self):
        cid = self.camera("videos/a.mp4", age_days=30)
        self.case(cid, "c1", "confirmed", cited=True)
        self.assertNotIn("videos/a.mp4", self.planned())
        self.store.x("UPDATE citations SET status='withdrawn'")
        self.assertIn("videos/a.mp4", self.planned())

    def test_running_or_queued_camera_keeps_its_video(self):
        cid = self.camera("videos/a.mp4", age_days=30)
        self.busy = {cid}
        self.assertNotIn("videos/a.mp4", self.planned())
        self.busy = set()
        self.assertIn("videos/a.mp4", self.planned())

    def test_unreferenced_old_video_is_removable(self):
        self.file("videos/stray.mp4", 10, 9)
        self.assertIn("videos/stray.mp4", self.planned())

    def test_cited_evidence_is_never_deleted_whatever_the_policy(self):
        cid = self.camera()
        _, base = self.case(cid, "c1", "confirmed", age_days=4000, cited=True, with_delivery=True)
        self.mgr.set_policy({"legal_retention_confirmed": True, "legal_retention_days": 365,
                             "delivery_clip_days": 1, "dismissed_days": 1}, self.admin)
        plan = self.planned()
        for rel in (base + ".jpg", base + "_crop.jpg", base + ".mp4", citation.delivery_path(base + ".mp4")):
            self.assertNotIn(rel, plan, rel)
        self.assertIn("sealed", self.mgr.plan(self.admin)["kept"])

    def test_pending_case_evidence_is_kept(self):
        cid = self.camera()
        _, base = self.case(cid, "c1", "pending", age_days=900)
        self.assertEqual([p for p in self.planned() if p.startswith("evidence/")], [])

    def test_dismissed_case_media_deleted_after_m_days(self):
        cid = self.camera()
        _, old = self.case(cid, "old", "rejected", age_days=31)
        _, new = self.case(cid, "new", "rejected", age_days=5)
        plan = self.planned()
        self.assertIn(old + ".mp4", plan)
        self.assertEqual(plan[old + ".mp4"]["rule"], "dismissed")
        self.assertNotIn(new + ".mp4", plan)

    def test_confirmed_uncited_waits_for_the_authority_then_for_the_period(self):
        cid = self.camera()
        _, base = self.case(cid, "c1", "confirmed", age_days=2000)
        self.assertNotIn(base + ".mp4", self.planned())            # period not confirmed by the authority
        self.mgr.set_policy({"legal_retention_confirmed": True}, self.admin)
        self.assertIn(base + ".mp4", self.planned())               # 2000 days > 1095
        self.mgr.set_policy({"legal_retention_days": 3000}, self.admin)
        self.assertNotIn(base + ".mp4", self.planned())

    def test_file_shared_by_two_cases_is_kept_if_either_needs_it(self):
        cid = self.camera()
        _, base = self.case(cid, "c1", "rejected", age_days=60)
        vid2, _ = self.case(cid, "c2", "pending", age_days=60)
        self.store.x("UPDATE violations SET snapshot=? WHERE id=?", (base + ".jpg", vid2))
        self.assertNotIn(base + ".jpg", self.planned())

    def test_delivery_clip_without_citation_is_regenerable_garbage(self):
        cid = self.camera()
        _, base = self.case(cid, "c1", "confirmed", age_days=40, with_delivery=True)
        d = citation.delivery_path(base + ".mp4")
        self.assertEqual(self.planned()[d]["rule"], "delivery_clip")
        self.assertNotIn(base + ".mp4", self.planned())

    def test_stale_part_cleaned_fresh_part_kept(self):
        self.file("evidence/cam1/run1/a.part.mp4", 10, age_days=2 / 24)
        self.file("evidence/cam1/run1/b.part.mp4", 10, age_days=5 / 1440)
        plan = self.planned()
        self.assertIn("evidence/cam1/run1/a.part.mp4", plan)
        self.assertEqual(plan["evidence/cam1/run1/a.part.mp4"]["rule"], "stale_part")
        self.assertNotIn("evidence/cam1/run1/b.part.mp4", plan)

    def test_orphans_reported_not_deleted_by_default(self):
        self.file("evidence/demo/snap.jpg", 10, 900)
        self.assertNotIn("evidence/demo/snap.jpg", self.planned())
        self.mgr.set_policy({"orphan_days": 30}, self.admin)
        self.assertIn("evidence/demo/snap.jpg", self.planned())

    def test_autozone_and_zone_refs_are_never_managed(self):
        self.file("autozone/p.jpg", 10, 900)
        self.file("zone_refs/r.jpg", 10, 900)
        self.assertEqual(self.planned(), {})


class CompactionTests(Base):
    def tracklet(self, age_days, embed=b"e" * 2304, reid=b"r" * 8192):
        cid = self.store.add_camera("t", "live", "rtsp://x")
        return self.store.x("INSERT INTO tracklets (camera_id, track_id, cls, hist, lab, embed, embed_reid, global_id,"
                            " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                            (cid, 1, "car", b"h" * 16, b"l" * 12, embed, reid, 7, iso(NOW - age_days * DAY)))

    def test_old_vectors_dropped_rows_and_summaries_kept(self):
        old, new = self.tracklet(45), self.tracklet(2)
        plan = self.mgr.plan(self.admin)
        item = next(i for i in plan["items"] if i["action"] == "compact_tracklets")
        self.assertEqual((item["rows"], item["bytes"]), (1, 2304 + 8192))
        self.mgr.apply(plan["id"], self.admin)
        r = self.store.q("SELECT * FROM tracklets WHERE id=?", (old,))[0]
        self.assertIsNone(r["embed"])
        self.assertIsNone(r["embed_reid"])
        self.assertEqual((r["global_id"], r["cls"], r["lab"], r["hist"]), (7, "car", b"l" * 12, b"h" * 16))
        self.assertIsNotNone(self.store.q("SELECT embed FROM tracklets WHERE id=?", (new,))[0]["embed"])
        self.assertTrue(self.store.audit_trail(object_type="tracklets"))
        self.assertFalse([i for i in self.mgr.plan(self.admin)["items"] if i["action"] == "compact_tracklets"])


class PlanApplyTests(Base):
    def setUp(self):
        super().setUp()
        self.camera("videos/old.mp4", age_days=10)      # its own camera: the one below has a standing citation
        cid = self.camera()
        self.cid = cid
        _, self.dead = self.case(cid, "dead", "rejected", age_days=60)
        _, self.live = self.case(cid, "live", "rejected", age_days=1)
        _, self.sealed = self.case(cid, "sealed", "confirmed", age_days=3000, cited=True)
        self.file("evidence/cam1/run1/x.part.mp4", 7, age_days=1)

    def snapshot(self):
        return sorted(str(p.relative_to(self.data)) for p in self.data.rglob("*") if p.is_file())

    def test_dry_run_deletes_nothing(self):
        before = self.snapshot()
        plan = self.mgr.plan(self.admin)
        self.assertTrue(plan["dry_run"])
        self.assertGreater(plan["totals"]["files"], 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.store.q("SELECT status FROM storage_plans")[0]["status"], "planned")

    def test_apply_deletes_exactly_the_plan(self):
        before = set(self.snapshot())
        plan = self.mgr.plan(self.admin)
        listed = {i["path"] for i in plan["items"] if i["action"] == "delete_file"}
        res = self.mgr.apply(plan["id"], self.admin)
        after = set(self.snapshot())
        self.assertEqual(before - after, listed)
        self.assertEqual(res["deleted"], len(listed))
        self.assertEqual(res["freed_bytes"], sum(i["bytes"] for i in plan["items"] if i["action"] == "delete_file"))
        self.assertIn("videos/old.mp4", listed)
        self.assertTrue(self.exists(self.live + ".mp4"))
        self.assertTrue(self.exists(self.sealed + ".mp4"))
        self.assertFalse(self.exists("evidence/cam1/run1/x.part.mp4"))
        self.assertEqual(self.store.q("SELECT status FROM storage_plans")[0]["status"], "applied")

    def test_every_deletion_is_audited_and_remembered(self):
        plan = self.mgr.plan(self.admin)
        self.mgr.apply(plan["id"], self.admin)
        deleted = {r["object_id"] for r in self.store.audit_trail(object_type="file")}
        self.assertEqual(deleted, {i["path"] for i in plan["items"] if i["action"] == "delete_file"})
        a = self.store.q("SELECT * FROM audit_log WHERE action='storage.delete' LIMIT 1")[0]
        self.assertEqual(a["who"], "Admin")
        self.assertIn(f"plan {plan['id']}", a["detail"])
        self.assertTrue(self.store.audit_trail(object_type="plan", object_id=plan["id"]))
        self.assertEqual(self.store.q("SELECT COUNT(*) n FROM storage_purged")[0]["n"], len(deleted))

    def test_citation_issued_after_the_plan_protects_the_file(self):
        plan = self.mgr.plan(self.admin)
        vid = self.store.q("SELECT id FROM violations WHERE snapshot=?", (self.dead + ".jpg",))[0]["id"]
        self.store.x("UPDATE violations SET status='confirmed' WHERE id=?", (vid,))
        self.store.create_citation(vid, self.admin, None, "0" * 64, "{}")
        res = self.mgr.apply(plan["id"], self.admin)
        self.assertTrue(self.exists(self.dead + ".jpg"))
        self.assertTrue(any(s["path"] == self.dead + ".jpg" for s in res["skipped"]))

    def test_a_plan_applies_once_and_expires(self):
        plan = self.mgr.plan(self.admin)
        self.mgr.apply(plan["id"], self.admin)
        with self.assertRaises(sm.PolicyError):
            self.mgr.apply(plan["id"], self.admin)
        stale = self.mgr.plan(self.admin)
        self.mgr.now = lambda: NOW + 25 * 3600
        with self.assertRaises(sm.PolicyError):
            self.mgr.apply(stale["id"], self.admin)

    def test_tampered_plan_cannot_reach_outside(self):
        outside = self.root / "outside.txt"
        outside.write_text("keep")
        plan = self.mgr.plan(self.admin)
        items = plan["items"] + [{"action": "delete_file", "path": "../outside.txt", "bytes": 4, "rule": "x", "reason": "x"},
                                 {"action": "delete_file", "path": "s.db", "bytes": 4, "rule": "x", "reason": "x"},
                                 {"action": "delete_file", "path": "autozone/p.jpg", "bytes": 4, "rule": "x", "reason": "x"}]
        self.file("autozone/p.jpg", 4, 900)
        self.store.x("UPDATE storage_plans SET items=? WHERE id=?", (json.dumps(items), plan["id"]))
        res = self.mgr.apply(plan["id"], self.admin)
        self.assertTrue(outside.exists())
        self.assertTrue(self.exists("autozone/p.jpg"))
        self.assertGreaterEqual(len(res["skipped"]), 3)

    def test_symlink_is_not_followed(self):
        target = self.root / "precious.txt"
        target.write_text("keep")
        link = self.data / "videos" / "link.mp4"
        link.symlink_to(target)
        os.utime(link, (NOW - 30 * DAY, NOW - 30 * DAY), follow_symlinks=False)
        self.mgr.apply(self.mgr.plan(self.admin)["id"], self.admin)
        self.assertTrue(target.exists())

    def test_sealed_citation_still_verifies_after_a_full_apply(self):
        cid = self.camera()
        vid, base = self.case(cid, "real", "confirmed", age_days=5)
        v = self.store.violation(vid)
        text, bundle = citation.build_manifest(v, self.data)
        self.store.create_citation(vid, self.admin, None, bundle, text)
        self.mgr.set_policy({"legal_retention_confirmed": True, "legal_retention_days": 365, "dismissed_days": 1,
                             "raw_video_days": 1}, self.admin)
        self.mgr.apply(self.mgr.plan(self.admin)["id"], self.admin)
        c = self.store.citation_by_violation(vid)
        result = citation.verify_seal(c, self.store.violation(vid), self.data)
        self.assertTrue(result["ok"], result)

    def test_empty_evidence_folders_are_removed_but_evidence_root_stays(self):
        self.mgr.set_policy({"dismissed_days": 1}, self.admin)
        cid = self.camera()
        _, base = self.case(cid, "gone", "rejected", age_days=60)
        self.mgr.apply(self.mgr.plan(self.admin)["id"], self.admin)
        self.assertFalse(self.exists(base + ".jpg"))
        self.assertTrue((self.data / "evidence").is_dir())


class DiskMonitorTests(Base):
    def test_thresholds(self):
        self.used = 79
        self.assertEqual(self.mgr.disk()["state"], "ok")
        self.used = 80
        self.assertEqual(self.mgr.disk()["state"], "warn")
        self.used = 90
        d = self.mgr.disk()
        self.assertEqual(d["state"], "pause")
        self.assertTrue(self.mgr.ingest_paused())
        self.assertIn("paused", d["message"])

    def test_thresholds_are_configurable(self):
        self.mgr.set_policy({"warn_pct": 40, "pause_pct": 60}, self.admin)
        self.used = 45
        self.assertEqual(self.mgr.disk()["state"], "warn")
        self.used = 61
        self.assertTrue(self.mgr.ingest_paused())

    def test_inventory_growth_and_days_to_full(self):
        cid = self.camera("videos/a.mp4")
        self.case(cid, "c1", "confirmed", cited=True)
        self.case(cid, "c2", "pending")
        self.case(cid, "c3", "rejected")
        inv = self.mgr.inventory(fresh=True)
        cats = inv["categories"]
        self.assertEqual(cats["raw_video"]["bytes"], 1000)
        self.assertEqual(cats["evidence_cited"]["files"], 3)
        self.assertEqual(cats["evidence_pending"]["files"], 3)
        self.assertEqual(cats["evidence_dismissed"]["files"], 3)
        self.assertGreater(inv["db_bytes"], 0)
        g = self.mgr.growth()
        self.assertGreater(g["bytes_per_day"], 0)
        self.assertGreater(g["days_to_full"], 0)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Store(Path(self._tmp.name) / "s.db")
        self.addCleanup(self.store.db.close)
        self.cams = [self.store.add_camera(f"c{i}", "live", "rtsp://x") for i in range(5)]
        self.t = 1000.0
        self.alive, self.started, self.final, self.fail_next = set(), [], {}, {}
        self.paused = False
        self.q = self.make()

    def start_fn(self, cid):
        if self.fail_next.get(cid):
            self.fail_next[cid] -= 1
            raise RuntimeError("boom")
        self.started.append(cid)
        self.alive.add(cid)

    def make(self, max_parallel=2):
        return ingest.IngestQueue(self.store, self.start_fn, lambda c: c in self.alive, max_parallel,
                                  status_fn=lambda c: {"state": self.final.get(c, "running"), "error": "bad video"},
                                  paused_fn=lambda: self.paused, now_fn=lambda: self.t, backoff_base=30,
                                  stop_fn=lambda c: self.alive.discard(c))

    def finish(self, cid, state="done"):
        self.alive.discard(cid)
        self.final[cid] = state

    def test_max_parallel_and_order(self):
        for c in self.cams[:4]:
            self.q.enqueue(c)
        self.q.enqueue(self.cams[4], priority=5)
        self.assertEqual(self.q.tick(), [self.cams[4], self.cams[0]])
        self.assertEqual(self.q.tick(), [])                      # both slots busy
        self.finish(self.cams[4])
        self.assertEqual(self.q.tick(), [self.cams[1]])
        self.assertEqual(self.q.stats()["counts"]["done"], 1)

    def test_same_camera_is_never_queued_or_started_twice(self):
        a = self.q.enqueue(self.cams[0])
        b = self.q.enqueue(self.cams[0], priority=3)
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(self.store.q("SELECT COUNT(*) n FROM ingest_jobs")[0]["n"], 1)
        self.assertEqual(self.q.job(a["id"])["priority"], 3)
        self.q.tick()
        self.q.tick()
        self.assertEqual(self.started, [self.cams[0]])

    def test_camera_already_running_by_hand_is_not_started(self):
        self.alive.add(self.cams[0])
        self.q.enqueue(self.cams[0])
        self.assertEqual(self.q.tick(), [])
        self.assertEqual(self.started, [])
        self.alive.discard(self.cams[0])
        self.assertEqual(self.q.tick(), [self.cams[0]])

    def test_failure_retries_with_backoff_then_fails_for_good(self):
        c = self.cams[0]
        j = self.q.enqueue(c)
        self.fail_next[c] = 5
        self.q.tick()
        row = self.q.job(j["id"])
        self.assertEqual((row["state"], row["attempts"]), ("queued", 1))
        self.assertIn("boom", row["error"])
        self.assertEqual(row["next_try_at"], self.t + 30)
        self.q.tick()                                           # too early: nothing
        self.assertEqual(self.q.job(j["id"])["attempts"], 1)
        self.t += 31
        self.q.tick()
        row = self.q.job(j["id"])
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(row["next_try_at"], self.t + 60)       # doubled
        self.t += 61
        self.q.tick()
        row = self.q.job(j["id"])
        self.assertEqual((row["state"], row["attempts"]), ("failed", 3))

    def test_run_that_ends_in_error_is_retried(self):
        c = self.cams[0]
        j = self.q.enqueue(c)
        self.q.tick()
        self.finish(c, "error")
        self.q.tick()
        row = self.q.job(j["id"])
        self.assertEqual((row["state"], row["attempts"], row["error"]), ("queued", 1, "bad video"))

    def test_not_ready_waits_without_counting_an_attempt(self):
        def start(cid):
            raise ingest.NotReady("learning")
        q = ingest.IngestQueue(self.store, start, lambda c: False, 1, now_fn=lambda: self.t)
        j = q.enqueue(self.cams[0])
        q.tick()
        row = q.job(j["id"])
        self.assertEqual((row["state"], row["attempts"]), ("queued", 0))
        self.assertGreater(row["next_try_at"], self.t)

    def test_restart_puts_running_jobs_back_in_the_queue(self):
        j = self.q.enqueue(self.cams[0])
        self.q.tick()
        self.assertEqual(self.q.job(j["id"])["state"], "running")
        self.alive.clear()
        q2 = self.make()                                        # a new process
        row = q2.job(j["id"])
        self.assertEqual(row["state"], "queued")
        self.assertIn("restart", row["error"])
        self.assertEqual(q2.tick(), [self.cams[0]])

    def test_disk_pause_starts_nothing_and_resumes(self):
        self.q.enqueue(self.cams[0])
        self.paused = True
        self.assertEqual(self.q.tick(), [])
        self.assertTrue(self.q.stats()["paused"])
        self.paused = False
        self.assertEqual(self.q.tick(), [self.cams[0]])

    def test_pause_follows_the_disk_monitor(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        used = {"pct": 95}
        mgr = sm.StorageManager(self.store, tmp.name, usage_fn=lambda p: Usage(100, used["pct"], 100 - used["pct"]))
        q = ingest.IngestQueue(self.store, self.start_fn, lambda c: c in self.alive, 1, paused_fn=mgr.ingest_paused)
        q.enqueue(self.cams[0])
        self.assertEqual(q.tick(), [])
        used["pct"] = 50
        self.assertEqual(q.tick(), [self.cams[0]])

    def test_cancel_and_retry(self):
        a, b = self.q.enqueue(self.cams[0]), self.q.enqueue(self.cams[1])
        self.q.tick()
        self.assertEqual(self.q.cancel(a["id"])["state"], "cancelled")
        self.assertNotIn(self.cams[0], self.alive)              # stop_fn was called
        self.assertIsNone(self.q.cancel(a["id"]))
        self.assertIsNone(self.q.retry(b["id"]))                # running jobs cannot be retried
        back = self.q.retry(a["id"])
        self.assertEqual((back["state"], back["attempts"]), ("queued", 0))

    def test_scheduler_thread_runs_and_stops(self):
        q = ingest.IngestQueue(self.store, self.start_fn, lambda c: c in self.alive, 1, tick_sec=0.05)
        j = q.enqueue(self.cams[0])
        q.start()
        import time
        for _ in range(60):
            if q.job(j["id"])["state"] == "running":
                break
            time.sleep(0.05)
        q.stop()
        self.assertEqual(q.job(j["id"])["state"], "running")


class RouterTests(Base):
    def setUp(self):
        super().setUp()
        fa = FakeAuth()
        self.alive = set()
        self.q = ingest.IngestQueue(self.store, lambda c: self.alive.add(c), lambda c: c in self.alive, 1,
                                    now_fn=lambda: NOW)
        web = self.root / "web"
        web.mkdir()
        (web / "storage.html").write_text("<html>storage</html>")
        app = FastAPI()
        app.include_router(api_storage.make_router(self.store, self.mgr, self.q, fa.current_user, fa.require,
                                                   "controller", "admin", web_dir=web))
        self.app = app
        self.cid = self.camera("videos/old.mp4", age_days=10)

    def call(self, method, path, role=None, body=None):
        headers = {"x-role": role} if role else {}
        data = b""
        if body is not None:
            data, headers["content-type"] = json.dumps(body).encode(), "application/json"
        status, _, out = asgi_call(self.app, method, path, headers, data)
        try:
            return status, json.loads(out)
        except ValueError:
            return status, out

    def test_roles(self):
        for method, path in (("GET", "/api/storage"), ("GET", "/api/ingest")):
            self.assertEqual(self.call(method, path)[0], 401)
            self.assertEqual(self.call(method, path, "driver")[0], 403)
            self.assertEqual(self.call(method, path, "controller")[0], 200)
        for method, path, body in (("GET", "/api/storage/policy", None), ("PUT", "/api/storage/policy", {"embed_days": 5}),
                                   ("POST", "/api/storage/plan", None), ("POST", "/api/storage/apply", {"plan_id": 1}),
                                   ("POST", "/api/ingest/1/retry", None), ("POST", "/api/ingest/1/cancel", None)):
            self.assertEqual(self.call(method, path, "controller", body)[0], 403, path)
            self.assertEqual(self.call(method, path, "driver", body)[0], 403, path)
            self.assertEqual(self.call(method, path, None, body)[0], 401, path)

    def test_summary_shape(self):
        st, body = self.call("GET", "/api/storage", "controller")
        self.assertEqual(st, 200)
        self.assertFalse(body["can_edit"])
        self.assertEqual(body["inventory"]["categories"]["raw_video"]["files"], 1)
        self.assertEqual(body["disk"]["state"], "ok")
        self.assertIn("days_to_full", body["growth"])

    def test_policy_roundtrip(self):
        self.assertEqual(self.call("PUT", "/api/storage/policy", "admin", {"embed_days": 5})[0], 200)
        self.assertEqual(self.call("GET", "/api/storage/policy", "admin")[1]["policy"]["embed_days"], 5)
        self.assertEqual(self.call("PUT", "/api/storage/policy", "admin", {"embed_days": 0})[0], 400)

    def test_plan_then_apply_over_http(self):
        st, plan = self.call("POST", "/api/storage/plan", "admin")
        self.assertEqual(st, 200)
        self.assertTrue(plan["dry_run"])
        self.assertTrue(self.exists("videos/old.mp4"))
        st, res = self.call("POST", "/api/storage/apply", "admin", {"plan_id": plan["id"]})
        self.assertEqual((st, res["deleted"]), (200, 1))
        self.assertFalse(self.exists("videos/old.mp4"))
        self.assertEqual(self.call("POST", "/api/storage/apply", "admin", {"plan_id": plan["id"]})[0], 409)
        self.assertEqual(self.call("POST", "/api/storage/apply", "admin", {"plan_id": 999})[0], 404)
        self.assertEqual(self.call("POST", "/api/storage/apply", "admin", {})[0], 400)

    def test_ingest_endpoints(self):
        job = self.q.enqueue(self.cid)
        st, body = self.call("GET", "/api/ingest", "controller")
        self.assertEqual(body["jobs"][0]["camera_name"], "cam")
        self.assertEqual(self.call("POST", f"/api/ingest/{job['id']}/cancel", "admin")[1]["state"], "cancelled")
        self.assertEqual(self.call("POST", f"/api/ingest/{job['id']}/cancel", "admin")[0], 409)
        self.assertEqual(self.call("POST", f"/api/ingest/{job['id']}/retry", "admin")[1]["state"], "queued")
        self.assertTrue(self.store.audit_trail(object_type="ingest"))

    def test_page_redirects(self):
        self.assertEqual(self.call("GET", "/storage")[0], 302)
        self.assertEqual(self.call("GET", "/storage", "driver")[0], 302)
        self.assertEqual(self.call("GET", "/storage", "controller")[0], 200)


if __name__ == "__main__":
    unittest.main()
