"""Passed vehicles (WP17): the recorder, the plate worker that must never block, colour naming, search, CSV, roles,
retention. Synthetic frames and crops only; no models, no data/ folder, no server (every file goes to a temp dir)."""

import csv
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import api_vehicles, passed as pv  # noqa: E402
from surveillance import storage_mgr as sm  # noqa: E402
from surveillance.embed import MOBILENET_TAG  # noqa: E402
from surveillance.anpr import PlateResult, parse_plate, read_crops, vote  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_anpr import plate_dets  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402
from tests.test_map import FakeAuth  # noqa: E402

NOW = 1_800_000_000.0
DAY = 86400.0


def solid(bgr, h=120, w=160):
    return np.full((h, w, 3), bgr, np.uint8)


class ColourTests(unittest.TestCase):
    def test_plain_colours(self):
        cases = {"white": (245, 245, 245), "silver": (175, 175, 175), "grey": (110, 110, 110), "black": (20, 20, 20),
                 "red": (30, 30, 200), "orange": (20, 140, 240), "yellow": (30, 220, 230), "green": (60, 160, 40),
                 "blue": (200, 90, 30), "purple": (160, 50, 130), "brown": (30, 70, 110)}
        for want, bgr in cases.items():
            self.assertEqual(pv.name_colour(solid(bgr))[0], want, want)

    def test_only_the_middle_counts(self):
        img = solid((20, 20, 20))                                   # black road around ...
        img[30:90, 40:120] = (30, 30, 200)                          # ... a red body
        self.assertEqual(pv.name_colour(img)[0], "red")

    def test_coloured_body_with_dark_windows_is_the_body_colour(self):
        img = solid((60, 160, 40))                                  # a green CNG
        img[20:60, 40:120] = (25, 25, 25)                           # dark windows (35% of the middle is dark)
        self.assertEqual(pv.name_colour(img)[0], "green")

    def test_grey_car_stays_grey_not_a_hue(self):
        rng = np.random.default_rng(1)
        noise = np.clip(110 + rng.normal(0, 6, (120, 160, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
        self.assertEqual(pv.name_colour(noise)[0], "grey")

    def test_no_clear_winner_is_mixed(self):
        img = solid((200, 90, 30))                                  # the middle half (columns 40-120) in thirds
        img[:, 40:67] = (30, 30, 200)
        img[:, 67:93] = (60, 160, 40)
        self.assertEqual(pv.name_colour(img)[0], pv.MIXED)

    def test_unreadable_input(self):
        self.assertEqual(pv.name_colour(None), ("", 0.0))
        self.assertEqual(pv.name_colour(np.zeros((3, 3, 3), np.uint8)), ("", 0.0))


class PlateTextTests(unittest.TestCase):
    def test_bangla_and_latin_digits_meet(self):
        self.assertEqual(pv.normalise_plate("ঢাকা মেট্রো-গ ১২-৩৪৫৬"), pv.normalise_plate("ঢাকা মেট্রো-গ 12-3456"))
        self.assertEqual(pv.normalise_plate("dhaka metro-ga 12 3456"), "DHAKAMETROGA123456")
        self.assertEqual(pv.normalise_plate(None), "")


class ConfirmWidthTests(unittest.TestCase):
    """'Confirmed' needs the agreeing reads to come from a plate at least 50 px wide."""

    def reads(self, px, n=4):
        return [parse_plate(plate_dets(), 300, 160, plate_px=px) for _ in range(n)]

    def test_narrow_plates_never_confirm_even_if_the_floor_is_lowered(self):
        for px in (25, 41, 44, 49):
            self.assertEqual(vote(self.reads(px), min_plate_px=20).status, "unreadable", px)

    def test_wide_enough_plates_confirm_when_the_floor_is_lowered(self):
        self.assertEqual(vote(self.reads(60), min_plate_px=50).status, "read")

    def test_default_behaviour_unchanged(self):
        self.assertEqual(vote(self.reads(150)).status, "read")
        self.assertEqual(vote(self.reads(100)).status, "unreadable")      # under MIN_PLATE_PX = 110

    def test_mixed_widths_only_wide_ones_count_as_agreeing(self):
        r = self.reads(30, 2) + self.reads(80, 2)                         # 4 agree, but only 2 are wide enough
        self.assertEqual(vote(r, min_plate_px=20).status, "unreadable")


class CropReadingTests(unittest.TestCase):
    class Reader:
        def __init__(self, shows_plate):
            self.calls, self.shows = 0, shows_plate

        def read(self, img):
            self.calls += 1
            if not self.shows:
                return None, None
            return parse_plate(plate_dets(), 300, 160, plate_px=150), np.zeros((10, 20, 3), np.uint8)

    jpg = cv2.imencode(".jpg", solid((90, 90, 90)))[1].tobytes()

    def test_gives_up_at_eight_when_no_plate_shows(self):
        rd = self.Reader(False)
        res = read_crops(rd, [self.jpg] * 12, max_crops=10, give_up_after=8)
        self.assertEqual(rd.calls, 8)
        self.assertEqual((res.status, res.plates_seen), ("unreadable", 0))
        self.assertEqual(pv.status_of(res), pv.NO_PLATE)

    def test_reads_ten_when_a_plate_shows(self):
        rd = self.Reader(True)
        res = read_crops(rd, [self.jpg] * 12, max_crops=10, give_up_after=8)
        self.assertEqual(rd.calls, 10)
        self.assertEqual(pv.status_of(res), pv.READ)

    def test_pause_hook_runs_before_each_crop(self):
        n = []
        read_crops(self.Reader(False), [self.jpg] * 3, pause=lambda: n.append(1))
        self.assertEqual(len(n), 3)


class WorkerNeverBlocks(unittest.TestCase):
    def test_full_queue_returns_false_at_once(self):
        gate = threading.Event()

        class Stuck:
            def read(self, img):
                gate.wait(5)
                return None, None

        jpg = CropReadingTests.jpg
        got = []
        w = pv.PassedPlateWorker(Stuck(), lambda k, r: got.append(k), maxsize=2)
        self.addCleanup(gate.set)
        t0 = time.perf_counter()
        answers = [w.submit(i, [jpg]) for i in range(20)]
        elapsed = time.perf_counter() - t0
        gate.set()
        self.assertLess(elapsed, 0.5)                       # 20 submits while the reader is stuck
        self.assertTrue(answers[0])
        self.assertIn(False, answers)
        self.assertGreaterEqual(w.stats()["skipped_busy"], 10)
        self.assertEqual(w.close(5), [])

    def test_waits_for_violators_before_each_crop(self):
        busy = {"n": 3}

        def busy_fn():
            busy["n"] -= 1
            return busy["n"] > 0

        order = []

        class Rd:
            def read(self, img):
                order.append("read")
                return None, None

        w = pv.PassedPlateWorker(Rd(), lambda k, r: order.append("done"), busy_fn=busy_fn)
        w.submit(1, [CropReadingTests.jpg])
        w.close(5)
        self.assertLessEqual(busy["n"], 0)                  # it asked, and waited until the violators' work ended
        self.assertEqual(order, ["read", "done"])


class FakeWorker:
    def __init__(self, accept=True):
        self.accept, self.jobs, self.closed = accept, [], False

    def submit(self, key, jpgs):
        if self.accept:
            self.jobs.append((key, len(jpgs)))
        return self.accept

    def close(self, timeout=30):
        self.closed = True
        return []

    def stats(self):
        return {}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.store = Store(self.root / "s.db")
        self.addCleanup(self.store.db.close)
        self.cam = self.store.add_camera("Farmgate", "file", "/secret/a.mp4", "Farmgate")
        self.cam2 = self.store.add_camera("Karwan Bazar", "file", "/secret/b.mp4", "Karwan Bazar")
        pv.ensure_schema(self.store)
        for _ in range(7):                                  # runs 1-7 exist, as tracklets point at them
            self.store.start_run(self.cam, None, "full")

    def rec(self, **kw):
        base = {"camera_id": self.cam, "run_id": 1, "track_id": 1, "cls": "Car", "colour": "white",
                "first_video_t": 1.0, "last_video_t": 3.0, "first_wall": "2026-10-29 16:30:01",
                "last_wall": "2026-10-29 16:30:03", "first_epoch": NOW, "last_epoch": NOW + 2, "frames": 20,
                "width_px": 200, "thumb": "", "plate_status": pv.NOT_READ, "created_at": NOW}
        base.update(kw)
        return pv.add_record(self.store, base)

    def add_tracklet(self, camera_id, run_id, track_id, gid=None, t_cross=NOW, cls="Car", **vec):
        from surveillance.reid import Tracklet
        t = Tracklet(camera_id, track_id, cls, t_cross, 1.0, 8.0, "outgoing", 1.7, run_id=run_id)
        t.global_id = gid
        for k, v in vec.items():
            setattr(t, k, v)
        return self.store.add_tracklet(camera_id, run_id, t)


def frame(w=640, h=480):
    return np.full((h, w, 3), 100, np.uint8)


class RecorderTests(Base):
    def recorder(self, worker=None, **kw):
        r = pv.PassedRecorder(self.store, self.cam, self.data, worker=worker, **kw)
        r.run_id = 7
        return r

    def drive(self, rec, tid, cls="Car", frames=8, width=200, on_road=True, bgr=(30, 30, 200), t0=0.0):
        f = frame()
        f[:] = 100
        for i in range(frames):
            t = t0 + i * 0.25
            box = (100, 100, 100 + width, 100 + int(width * 0.7))
            f[box[1]:box[3], box[0]:box[2]] = bgr
            rec.observe(tid, f, box, cls, on_road, t, NOW + t)
        return t0 + (frames - 1) * 0.25

    def finish(self, rec, t):
        rec.tick(t + 5, set())

    def rows(self):
        return self.store.q("SELECT * FROM passed_vehicles ORDER BY id")

    def test_record_has_colour_class_times_thumbnail(self):
        rec = self.recorder(FakeWorker())
        t = self.drive(rec, 5)
        self.assertEqual(self.rows(), [])                                   # still in view: nothing yet
        self.finish(rec, t)
        (r,) = self.rows()
        self.assertEqual((r["camera_id"], r["run_id"], r["track_id"], r["cls"], r["colour"]), (self.cam, 7, 5, "Car", "red"))
        self.assertEqual((r["first_video_t"], r["last_video_t"]), (0.0, 1.75))
        self.assertEqual(r["first_epoch"], NOW)
        self.assertTrue(r["first_wall"].startswith("20"))
        thumb = self.data / "passed" / r["thumb"]
        self.assertTrue(thumb.is_file())
        img = cv2.imread(str(thumb))
        self.assertLessEqual(max(img.shape[:2]), 160)
        self.assertEqual(r["plate_status"], pv.PENDING)

    def test_flicker_and_off_road_vehicles_are_not_recorded(self):
        rec = self.recorder(FakeWorker())
        self.finish(rec, self.drive(rec, 1, frames=3))                     # fewer than min_frames
        self.finish(rec, self.drive(rec, 2, on_road=False))
        self.assertEqual(self.rows(), [])

    def test_pedal_and_small_vehicles_are_marked_not_read(self):
        w = FakeWorker()
        rec = self.recorder(w)
        self.finish(rec, self.drive(rec, 1, cls="Rickshaw"))
        self.finish(rec, self.drive(rec, 2, width=60))
        self.assertEqual([r["plate_status"] for r in self.rows()], [pv.SKIP_PEDAL, pv.SKIP_SMALL])
        self.assertEqual(w.jobs, [])

    def test_without_a_plate_worker_the_record_says_not_read(self):
        rec = self.recorder(None)
        self.finish(rec, self.drive(rec, 1))
        self.assertEqual(self.rows()[0]["plate_status"], pv.NOT_READ)

    def test_busy_queue_marks_skipped_busy_and_still_records(self):
        rec = self.recorder(FakeWorker(accept=False))
        self.finish(rec, self.drive(rec, 1))
        (r,) = self.rows()
        self.assertEqual(r["plate_status"], pv.SKIP_BUSY)
        self.assertEqual(rec.skipped_busy, 1)

    def test_jobs_carry_the_largest_views_up_to_ten(self):
        w = FakeWorker()
        rec = self.recorder(w)
        self.finish(rec, self.drive(rec, 1, frames=40))
        self.assertEqual(len(w.jobs), 1)
        self.assertLessEqual(w.jobs[0][1], 10)
        self.assertGreaterEqual(w.jobs[0][1], 2)

    def test_plate_result_updates_the_record(self):
        rec = self.recorder(FakeWorker())
        self.finish(rec, self.drive(rec, 1))
        pid = self.rows()[0]["id"]
        res = vote([parse_plate(plate_dets(), 300, 160, plate_px=150)] * 4)
        res.plates_seen = 4
        rec.plate_done(pid, res)
        r = self.rows()[0]
        self.assertEqual(r["plate_status"], "read")
        self.assertEqual(r["plate_en"], "DHAKA METRO-GA 12-3456")
        self.assertIn("123456", r["plate_norm"])

    def test_violator_plate_arriving_before_or_after_the_record_is_reused_not_read_twice(self):
        res = vote([parse_plate(plate_dets(), 300, 160, plate_px=150)] * 4)
        res.plates_seen = 4
        w = FakeWorker()
        rec = self.recorder(w)
        t = self.drive(rec, 1)
        rec.violator_plate(1, res)                                          # vehicle still in view
        self.finish(rec, t)
        self.assertEqual(self.rows()[0]["plate_status"], "read")
        self.assertEqual(w.jobs, [])                                        # no second reading
        t = self.drive(rec, 2)
        self.finish(rec, t)
        rec.violator_plate(2, res)                                          # vehicle already recorded
        self.assertEqual(self.rows()[1]["plate_status"], "read")
        rec.violator_plate(3, PlateResult("unreadable"))                    # a violator's "nothing" says nothing
        self.assertNotIn(3, rec.external)

    def test_close_records_the_vehicles_still_in_view_and_marks_undrained_ones(self):
        w = FakeWorker()
        w.close = lambda timeout=30: [self.rows()[0]["id"]]
        rec = self.recorder(w)
        self.drive(rec, 1)
        rec.close()
        self.assertEqual(self.rows()[0]["plate_status"], pv.SKIP_BUSY)


class HttpBase(Base):
    def setUp(self):
        super().setUp()
        fa = FakeAuth()
        web = self.root / "web"
        web.mkdir()
        (web / "vehicles.html").write_text("<html>vehicles</html>")
        app = FastAPI()
        self.vehicles = api_vehicles.build(self.store, self.data, fa.current_user, fa.require, "controller", "admin",
                                           web_dir=web)
        app.include_router(self.vehicles.router)
        self.app = app

    def call(self, path, role="controller"):
        status, hdrs, out = asgi_call(self.app, "GET", path, {"x-role": role} if role else {})
        if "json" in hdrs.get("content-type", ""):
            return status, json.loads(out)
        return status, out

    def get(self, path, role="controller"):
        status, body = self.call(path, role)
        self.assertEqual(status, 200, body)
        return body


class RolesTests(HttpBase):
    def test_staff_only_on_every_endpoint(self):
        pid = self.rec()
        paths = ["/api/passed", "/api/passed/cameras", f"/api/passed/runs?camera_id={self.cam}", "/api/passed/export.csv",
                 "/api/passed/find?q=car", f"/api/passed/{pid}", f"/api/passed/{pid}/thumb", f"/api/passed/{pid}/similar",
                 "/api/passed/lookalike/status"]
        for p in paths:
            self.assertEqual(self.call(p, None)[0], 401, p)
            self.assertEqual(self.call(p, "driver")[0], 403, p)
            self.assertIn(self.call(p, "controller")[0], (200, 404), p)
            self.assertIn(self.call(p, "admin")[0], (200, 404), p)

    def test_page_redirects_drivers_away(self):
        for role, want in ((None, 302), ("driver", 302), ("controller", 200), ("admin", 200)):
            self.assertEqual(self.call("/passed", role)[0], want, role)

    def test_server_driver_list_does_not_include_these(self):
        src = (Path(__file__).resolve().parents[1] / "surveillance" / "server.py").read_text()
        self.assertNotIn("/api/passed", src.split("DRIVER_API = {")[1].split("}")[0])


class SearchTests(HttpBase):
    def setUp(self):
        super().setUp()
        t = lambda n: NOW + n * 60
        self.a = self.rec(track_id=1, cls="Car", colour="white", plate_status="read", first_epoch=t(0), last_epoch=t(0) + 5)
        pv.set_plate(self.store, self.a, "read", "ঢাকা মেট্রো-গ ১২-৩৪৫৬", "DHAKA METRO-GA 12-3456", 0.9, 160)
        self.b = self.rec(track_id=2, cls="Bus", colour="red", first_epoch=t(10), last_epoch=t(10) + 5)
        self.c = self.rec(track_id=3, cls="Motorbike", colour="black", first_epoch=t(20), last_epoch=t(20) + 5)
        pv.set_plate(self.store, self.c, "partial", "", "DHAKA-HA 99-0000", 0.5, 120)
        self.old = self.rec(track_id=9, run_id=1, camera_id=self.cam2, cls="Car", colour="white", first_epoch=t(-100), last_epoch=t(-100) + 5)
        self.new_run = self.rec(track_id=1, run_id=2, camera_id=self.cam, cls="Car", colour="blue", first_epoch=t(30), last_epoch=t(30) + 5)
        self.add_tracklet(self.cam, 1, 1, gid=42, t_cross=NOW)
        self.add_tracklet(self.cam2, 1, 9, gid=42, t_cross=NOW + 600, cls="Car")

    def ids(self, query):
        return sorted(r["id"] for r in self.get("/api/passed?" + query)["items"])

    def test_newest_run_is_the_default_for_a_camera(self):
        body = self.get(f"/api/passed?camera_id={self.cam}")
        self.assertEqual(body["run_id"], 2)
        self.assertEqual([r["id"] for r in body["items"]], [self.new_run])
        self.assertEqual(self.ids(f"camera_id={self.cam}&run_id=1"), sorted([self.a, self.b, self.c]))
        self.assertEqual(self.ids(f"camera_id={self.cam}&run_id=all"), sorted([self.a, self.b, self.c, self.new_run]))

    def test_plate_search_is_partial_and_script_blind(self):
        for q in ("12-3456", "123456", "১২৩৪৫৬", "345", "dhaka metro", "ga 12", "ঢাকা"):
            self.assertEqual(self.ids("run_id=all&plate=" + quote(q)), [self.a], q)
        self.assertEqual(self.ids("run_id=all&plate=DHAKA"), sorted([self.a, self.c]))
        self.assertEqual(self.ids("run_id=all&plate=%25"), [])               # a literal percent sign matches nothing

    def test_class_colour_status_and_free_text(self):
        self.assertEqual(self.ids("run_id=all&cls=bus"), [self.b])
        self.assertEqual(self.ids("run_id=all&cls=bus,motorbike"), sorted([self.b, self.c]))
        self.assertEqual(self.ids("run_id=all&colour=white"), sorted([self.a, self.old]))
        self.assertEqual(self.ids("run_id=all&status=partial"), [self.c])
        self.assertEqual(self.ids("run_id=all&q=white car"), sorted([self.a, self.old]))
        self.assertEqual(self.ids("run_id=all&q=red"), [self.b])
        self.assertEqual(self.ids("run_id=all&q=%232"), sorted([self.b]))      # "#2": track id 2 (run 1 of cam 1)
        self.assertEqual(self.ids("run_id=all&q=99-0000"), [self.c])

    def test_global_id_filter_reads_the_linker_tables(self):
        self.assertEqual(self.ids("run_id=all&gid=42"), sorted([self.a, self.old]))
        self.assertEqual(self.ids("run_id=all&q=g42"), sorted([self.a, self.old]))
        row = next(r for r in self.get("/api/passed?run_id=all&gid=42")["items"] if r["id"] == self.a)
        self.assertEqual(row["global_id"], 42)

    def test_time_range(self):
        self.assertEqual(self.ids(f"run_id=all&since={NOW + 9 * 60}&until={NOW + 11 * 60}"), [self.b])
        self.assertEqual(self.ids(f"run_id=all&since={NOW + 21 * 60}"), [self.new_run])
        self.assertEqual(self.get("/api/passed?run_id=all&until=1900-01-01T00:00")["items"], [])
        self.assertEqual(self.call("/api/passed?since=yesterday-ish")[0], 400)

    def test_paging_counts_and_status_chips(self):
        body = self.get("/api/passed?run_id=all&limit=2")
        self.assertEqual((len(body["items"]), body["total"]), (2, 5))
        self.assertEqual(self.get("/api/passed?run_id=all&limit=2&offset=4")["items"].__len__(), 1)
        self.assertEqual(self.get("/api/passed?run_id=all&status=read")["plate_status_counts"]["partial"], 1)

    def test_nothing_server_side_leaks(self):
        item = self.get("/api/passed?run_id=all")["items"][0]
        self.assertNotIn("thumb", item)
        self.assertNotIn("plate_norm", item)
        self.assertNotIn("/secret", json.dumps(self.get("/api/passed?run_id=all")))

    def test_detail_and_journey(self):
        d = self.get(f"/api/passed/{self.a}")
        self.assertEqual(d["global_id"], 42)
        self.assertEqual([j["camera_id"] for j in d["journey"]], [self.cam, self.cam2])
        self.assertEqual(self.call("/api/passed/99999")[0], 404)

    def test_find_across_the_network_says_where_last_seen(self):
        body = self.get("/api/passed/find?plate=12-3456")
        (item,) = body["items"]
        self.assertEqual(item["global_id"], 42)
        self.assertEqual(item["last_seen"]["camera_id"], self.cam2)        # the later sighting, on the other camera
        self.assertEqual(item["last_seen"]["camera_name"], "Karwan Bazar")
        self.assertEqual(item["cameras"], 2)
        by_gid = self.get("/api/passed/find?gid=42")["items"]
        self.assertEqual(by_gid[0]["last_seen"]["camera_id"], self.cam2)
        look = self.get("/api/passed/find?colour=red&cls=bus")["items"]
        self.assertEqual((look[0]["passed_id"], look[0]["global_id"]), (self.b, None))
        self.assertEqual(self.call("/api/passed/find")[0], 400)

    def test_find_by_global_id_survives_expired_records(self):
        self.add_tracklet(self.cam, 5, 5, gid=77, t_cross=NOW + 5)
        (item,) = self.get("/api/passed/find?gid=77")["items"]
        self.assertEqual((item["global_id"], item["passed_id"], item["last_seen"]["source"]), (77, None, "tracklet"))


class SimilarHttpTests(HttpBase):
    def test_suggestions_for_an_open_vehicle_and_no_links_written(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=576).astype(np.float32)
        q /= np.linalg.norm(q)
        twin = q + 0.03 * rng.normal(size=576).astype(np.float32)
        me = self.rec(track_id=1)
        near = self.rec(track_id=2, camera_id=self.cam2, colour="blue")
        lone = self.rec(track_id=3)                                          # no tracklet: nothing to compare
        self.add_tracklet(self.cam, 1, 1, embed=q, embed_model=MOBILENET_TAG)
        self.add_tracklet(self.cam2, 1, 2, embed=twin, embed_model=MOBILENET_TAG)
        for i in range(30):
            v = rng.normal(size=576).astype(np.float32)
            self.add_tracklet(self.cam, 1, 100 + i, embed=v, embed_model=MOBILENET_TAG)
        before = self.store.q("SELECT id, global_id FROM tracklets")
        out = self.get(f"/api/passed/{me}/similar?k=3")
        self.assertEqual(out["items"][0]["passed_id"], near)
        self.assertEqual(out["items"][0]["colour"], "blue")
        self.assertTrue(out["suggestions_only"])
        self.assertEqual(self.store.q("SELECT id, global_id FROM tracklets"), before)
        self.assertEqual(self.get(f"/api/passed/{lone}/similar")["items"], [])
        self.assertEqual(self.get("/api/passed/lookalike/status")["spaces"][0]["vectors"], 32)


class CsvTests(HttpBase):
    def test_streams_a_full_csv_with_bangla_and_a_bom_and_is_audited(self):
        ids = []
        for i in range(2500):                                             # more than one 1000-row batch
            ids.append(self.rec(track_id=i, run_id=3, first_epoch=NOW + i))
        pv.set_plate(self.store, ids[0], "read", "ঢাকা মেট্রো-গ ১২-৩৪৫৬", "DHAKA METRO-GA 12-3456", 0.9, 160)
        pv.set_plate(self.store, ids[1], "read", "", "=HYPERLINK(\"x\")", 0.9, 160)
        status, hdrs, body = asgi_call(self.app, "GET", f"/api/passed/export.csv?camera_id={self.cam}", {"x-role": "controller"})
        self.assertEqual(status, 200)
        self.assertIn("text/csv", hdrs["content-type"])
        self.assertIn("attachment", hdrs["content-disposition"])
        text = body.decode("utf-8")
        self.assertTrue(text.startswith("﻿"))
        rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
        self.assertEqual(rows[0], list(api_vehicles.CSV_COLUMNS))
        self.assertEqual(len(rows) - 1, 2500)
        first = rows[1]
        self.assertEqual(first[rows[0].index("plate_bn")], "ঢাকা মেট্রো-গ ১২-৩৪৫৬")
        self.assertEqual(rows[2][rows[0].index("plate_en")], "'=HYPERLINK(\"x\")")     # not a live formula
        self.assertEqual(self.store.audit_trail("passed_vehicles")[0]["action"], "passed.export")

    def test_csv_uses_the_same_filter(self):
        self.rec(track_id=1, cls="Bus")
        self.rec(track_id=2, cls="Car")
        _, _, body = asgi_call(self.app, "GET", "/api/passed/export.csv?run_id=all&cls=bus", {"x-role": "admin"})
        self.assertEqual(len(list(csv.reader(io.StringIO(body.decode().lstrip("﻿"))))), 2)


class ThumbTests(HttpBase):
    def test_thumb_served_and_path_escape_refused(self):
        d = self.data / "passed" / "cam1"
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "run1_t1.jpg"), solid((1, 2, 3)))
        (self.data / "secret.txt").write_text("nope")
        ok = self.rec(thumb="cam1/run1_t1.jpg")
        bad = self.rec(thumb="../secret.txt")
        status, hdrs, body = asgi_call(self.app, "GET", f"/api/passed/{ok}/thumb", {"x-role": "controller"})
        self.assertEqual((status, hdrs["content-type"]), (200, "image/jpeg"))
        self.assertEqual(self.call(f"/api/passed/{bad}/thumb")[0], 404)


class RetentionTests(Base):
    def setUp(self):
        super().setUp()
        self.d = self.data / "passed" / f"cam{self.cam}"
        self.d.mkdir(parents=True)

    def make(self, name, age_days, thumb=True):
        if thumb:
            p = self.d / f"{name}.jpg"
            p.write_bytes(b"x" * 100)
        return self.rec(track_id=abs(hash(name)) % 10**6, thumb=f"cam{self.cam}/{name}.jpg" if thumb else "",
                        created_at=NOW - age_days * DAY)

    def test_old_records_and_thumbnails_go_young_ones_stay(self):
        old, young = self.make("old", 31), self.make("young", 29)
        out = pv.purge(self.store, self.data, days=30, now=NOW)
        self.assertEqual((out["rows"], out["files"]), (1, 1))
        left = [r["id"] for r in self.store.q("SELECT id FROM passed_vehicles")]
        self.assertEqual(left, [young])
        self.assertFalse((self.d / "old.jpg").exists())
        self.assertTrue((self.d / "young.jpg").exists())

    def test_dry_run_deletes_nothing(self):
        self.make("old", 40)
        out = pv.purge(self.store, self.data, days=30, now=NOW, dry_run=True)
        self.assertEqual(out["rows"], 1)
        self.assertEqual(len(self.store.q("SELECT id FROM passed_vehicles")), 1)
        self.assertTrue((self.d / "old.jpg").exists())

    def test_orphan_thumbnails_go_once_a_day_old(self):
        import os
        p = self.d / "orphan.jpg"
        p.write_bytes(b"x")
        os.utime(p, (NOW - 2 * DAY, NOW - 2 * DAY))
        fresh = self.d / "fresh.jpg"
        fresh.write_bytes(b"x")
        os.utime(fresh, (NOW - 60, NOW - 60))
        pv.purge(self.store, self.data, days=30, now=NOW)
        self.assertFalse(p.exists())
        self.assertTrue(fresh.exists())                                   # may be a record being written right now

    def test_retention_days_default_and_env(self):
        import os
        from unittest import mock
        self.assertEqual(pv.retention_days(), 30)
        with mock.patch.dict(os.environ, {"PASSED_RETENTION_DAYS": "7"}):
            self.assertEqual(pv.retention_days(), 7)
        with mock.patch.dict(os.environ, {"PASSED_RETENTION_DAYS": "junk"}):
            self.assertEqual(pv.retention_days(), 30)
        with mock.patch.dict(os.environ, {"PASSED_RETENTION_DAYS": "-3"}):
            self.assertEqual(pv.retention_days(), 30)

    def test_purge_loop_runs_once_and_audits(self):
        self.rec(track_id=77, created_at=time.time() - 99 * DAY)           # the loop uses the real clock
        self.rec(track_id=78, created_at=time.time())
        out = pv.PurgeLoop(self.store, self.data).run_once()
        self.assertEqual(out["rows"], 1)
        self.assertEqual(self.store.q("SELECT COUNT(*) n FROM passed_vehicles")[0]["n"], 1)
        self.assertEqual(self.store.audit_trail("passed_vehicles")[0]["action"], "passed.purge")

    def test_purge_loop_thread_starts_and_stops(self):
        loop = pv.PurgeLoop(self.store, self.data, interval_s=3600).start()
        for _ in range(100):
            if loop.last is not None:
                break
            time.sleep(0.02)
        loop.stop()
        self.assertIsNotNone(loop.last)

    def test_storage_plan_lists_it_and_apply_deletes_it(self):
        self.make("old", 45)
        self.make("young", 1)
        for sub in ("videos", "evidence"):
            (self.data / sub).mkdir(exist_ok=True)
        mgr = sm.StorageManager(self.store, self.data, now_fn=lambda: NOW,
                                usage_fn=lambda p: type("U", (), {"total": 1000, "used": 100, "free": 900})(),
                                busy_cameras_fn=lambda: set())
        admin = {"user_id": 1, "role": "admin", "name": "A", "username": "a", "address": "x"}
        plan = mgr.plan(admin)
        (item,) = [i for i in plan["items"] if i["action"] == "purge_passed"]
        self.assertEqual(item["rows"], 1)
        self.assertEqual(len(self.store.q("SELECT id FROM passed_vehicles")), 2)    # a dry run deletes nothing
        res = mgr.apply(plan["id"], admin)
        self.assertEqual(len(self.store.q("SELECT id FROM passed_vehicles")), 1)
        self.assertFalse((self.d / "old.jpg").exists())
        self.assertTrue(any("passed_vehicles" in p for p in res["paths"]))
        self.assertTrue(any(a["action"] == "storage.purge_passed" for a in self.store.audit_trail("passed_vehicles")))

    def test_a_deleted_camera_takes_its_records_along(self):
        self.make("x", 1)
        self.store.x("DELETE FROM cameras WHERE id=?", (self.cam,))
        self.assertEqual(self.store.q("SELECT COUNT(*) n FROM passed_vehicles")[0]["n"], 0)


class PipelineSeamTests(Base):
    """The pipeline hands the recorder every tracked vehicle, and a fault in the recorder never stops the rules."""

    def make(self, recorder):
        from tests import test_camera_motion as cm
        from surveillance.pipeline import Pipeline
        from violations.zones import camera_config_from_dict
        self.cm = cm
        ref = cm.scene()
        cfg = {"frame_size": [cm.W, cm.H], "zones": [{"type": "road", "name": "Road", "points": cm.ROAD, "width_m": 10,
                                                       "length_m": 60, "allowed_direction": "toward_camera"}]}
        self.det = cm.FakeDetector()
        self.det.boxes = [((480.0, 300.0, 640.0, 390.0), 1, "Car")]
        p = Pipeline(self.det, camera_config_from_dict(cfg, cm.W, cm.H), (cm.W, cm.H), reference=ref, auto_roi=False,
                     stabilize=False, passed=recorder)
        self.addCleanup(p.close)
        return p, ref

    def test_a_vehicle_that_passes_becomes_a_record(self):
        rec = pv.PassedRecorder(self.store, self.cam, self.data, worker=None)
        rec.run_id = 3
        p, ref = self.make(rec)
        for i in range(8):
            p.process(self.cm.view(ref, self.cm.similarity(0, 0), seed=i), i * 0.1)
        self.assertEqual(self.store.q("SELECT * FROM passed_vehicles"), [])         # still in view
        self.det.boxes = []
        for i in range(8, 30):
            p.process(self.cm.view(ref, self.cm.similarity(0, 0), seed=i), i * 0.1)
        (r,) = self.store.q("SELECT * FROM passed_vehicles")
        self.assertEqual((r["camera_id"], r["run_id"], r["track_id"], r["cls"]), (self.cam, 3, 1, "Car"))
        self.assertEqual(r["plate_status"], pv.NOT_READ)

    def test_a_fault_in_the_recorder_does_not_stop_the_rules(self):
        from unittest import mock

        class Broken:
            def observe(self, *a):
                raise ValueError("boom")

            def tick(self, *a):
                raise ValueError("boom")

        p, ref = self.make(Broken())
        with mock.patch("builtins.print") as said:
            for i in range(4):
                res = p.process(self.cm.view(ref, self.cm.similarity(0, 0), seed=i), i * 0.1)
        self.assertEqual(res.tracks, 1)
        self.assertEqual(sum("passed vehicles not recorded" in str(c) for c in said.call_args_list), 1)   # said once


if __name__ == "__main__":
    unittest.main()
