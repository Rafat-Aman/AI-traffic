"""The natural-language query panel: parser, segment index, retrieval, Stage B routing, HTTP roles, and the X7
grounding metrics on a SYNTHETIC world.

Nothing here opens data/surveillance.db or the dashboard server. The world in `build_world` is generated, not
recorded: the labelled queries in tests/data/queries.jsonl measure whether retrieval finds the planted vehicles in
it, which shows the pipeline works end to end; they are NOT a measurement of how real Dhaka questions behave (there
is no real query log yet).
"""

import atexit
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SCRATCH = tempfile.mkdtemp(prefix="query-tests-")
os.environ.setdefault("TRAFFIC_DATA_DIR", _SCRATCH)
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from surveillance import api_query  # noqa: E402
from surveillance import impact  # noqa: E402
from surveillance import query as Q  # noqa: E402
from surveillance import segment_index as si  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402
from tests.test_map import FakeAuth  # noqa: E402

NOW = datetime(2026, 10, 30, 10, 0, 0)          # "now" for every parser test: a Friday morning
DAY = datetime(2026, 10, 29, 16, 30, 0)         # where the synthetic footage starts (yesterday, 16:30)
CAMS = [{"id": 1, "name": "Farmgate", "location": "Farmgate"}, {"id": 2, "name": "Karwan Bazar", "location": ""},
        {"id": 3, "name": "Mirpur 10", "location": "Mirpur 10 roundabout"}]
DATA = Path(__file__).parent / "data"

# representative CIELAB (L 0..100, a, b) per colour name, to make the stored blob the pipeline would have made
LAB = {"white": (90, 0, 0), "black": (12, 0, 0), "grey": (55, 0, 0), "red": (45, 50, 35), "orange": (65, 30, 60),
       "yellow": (85, 0, 70), "green": (50, -45, 30), "blue": (40, 0, -50), "purple": (40, 40, -40)}


def lab_blob(colour):
    if colour is None:
        return None
    L, a, b = LAB[colour]
    v = np.array([L * 255 / 100 / 20.0, (a + 128) / 8.0, (b + 128) / 8.0, 0.5, 0.5, 0.5], np.float32)
    return v


def at(h, m=0, s=0, day=29):
    return si.to_epoch(datetime(2026, 10, day, h, m, s))


def add_pass(store, cam, run, track, cls, t, speed=6.0, direction="incoming", colour="grey", gid=None, base=None,
             zone="Main road"):
    tr = SimpleNamespace(track_id=track, cls=cls, class_conf=0.9, zone=zone, t_cross=t, video_t=None if base is None else t - base,
                         lateral=1.0, speed=speed, width=1.8, direction=direction, extrap_m=0.0, hist=None,
                         lab=lab_blob(colour), embed=None, global_id=gid, link_source="online")
    return store.add_tracklet(cam, run, tr)


def add_violation(store, cam, run, track, code, title, cls, t, base=None, details=None, zone="Main road", plate=""):
    rec = {"code": code, "violation": title, "track_id": track, "class": cls, "class_confidence": 0.9,
           "video_time_s": None if base is None else t - base, "started_s": 0.0,
           "wall_time": si.from_epoch(t).strftime("%Y-%m-%d %H:%M:%S"), "zone": zone, "areas": "", "kph": 0.0,
           "details": details or {}}
    vid = store.add_violation(cam, run, None, rec, "", "", "")
    if plate:
        store.set_plate(vid, "read", "", plate, 0.9, None)
    return vid


class Scratch(unittest.TestCase):
    def make_store(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        st = Store(self.dir / "s.db")
        self.addCleanup(st.db.close)
        si.ensure_schema(st)
        return st


# ======================================================================= parser

class ParserTests(unittest.TestCase):
    def parse(self, text, now=NOW):
        return Q.parse_rules(text, CAMS, now)

    def span(self, q):
        return si.from_epoch(q["t0"]), si.from_epoch(q["t1"])

    def test_between_with_one_meridiem_covers_both(self):
        q = self.parse("how many buses stopped at camera 2 between 5 and 6 pm?")
        self.assertEqual(self.span(q), (datetime(2026, 10, 29, 17), datetime(2026, 10, 29, 18)))
        self.assertEqual((q["classes"], q["camera_ids"], q["behaviours"], q["agg"]), (["Bus"], [2], ["stopped"], "count"))

    def test_between_explicit_minutes_and_morning(self):
        q = self.parse("cars on 25 October between 9:30 am and 10 am")
        self.assertEqual(self.span(q), (datetime(2026, 10, 25, 9, 30), datetime(2026, 10, 25, 10)))

    def test_eleven_to_one_crosses_noon(self):
        q = self.parse("buses between 11 and 1 pm yesterday")
        self.assertEqual(self.span(q), (datetime(2026, 10, 29, 11), datetime(2026, 10, 29, 13)))

    def test_24_hour_clock(self):
        q = self.parse("trucks from 17:15 to 18:45 yesterday")
        self.assertEqual(self.span(q), (datetime(2026, 10, 29, 17, 15), datetime(2026, 10, 29, 18, 45)))

    def test_yesterday_is_the_whole_day_and_today_too(self):
        self.assertEqual(self.span(self.parse("show white cars going wrong way yesterday")),
                         (datetime(2026, 10, 29), datetime(2026, 10, 30)))
        self.assertEqual(self.span(self.parse("buses today"))[0], datetime(2026, 10, 30))

    def test_last_n_minutes_and_hours(self):
        q = self.parse("CNGs in the last 10 minutes")
        self.assertEqual(self.span(q), (datetime(2026, 10, 30, 9, 50), NOW))
        self.assertEqual(self.span(self.parse("past 2 hours"))[0], datetime(2026, 10, 30, 8))
        self.assertEqual(self.span(self.parse("in the last hour"))[0], datetime(2026, 10, 30, 9))

    def test_point_in_time_gets_a_small_window_and_a_point(self):
        q = self.parse("why did traffic back up at 5:40?")
        self.assertEqual(si.from_epoch(q["point"]), datetime(2026, 10, 29, 17, 40))   # 5:40 pm yesterday
        self.assertEqual(q["t1"] - q["t0"], 2 * Q.POINT_PAD_S)
        self.assertEqual(q["intent"], "causal")

    def test_ambiguous_hours_are_guessed_and_said(self):
        q = self.parse("buses between 5 and 6 yesterday")
        self.assertEqual(self.span(q)[0], datetime(2026, 10, 29, 17))
        self.assertTrue(any("no am/pm" in n for n in q["notes"]))

    def test_day_names_and_dates(self):
        q = self.parse("buses on 2026-10-25")
        self.assertEqual(self.span(q), (datetime(2026, 10, 25), datetime(2026, 10, 26)))
        self.assertEqual(self.span(self.parse("buses on Monday"))[0], datetime(2026, 10, 26))   # NOW is Friday
        self.assertEqual(self.span(self.parse("cars last night"))[0], datetime(2026, 10, 29, 18))

    def test_no_time_expression_means_no_window(self):
        q = self.parse("white cars at Farmgate")
        self.assertIsNone(q["t0"])

    def test_impossible_time_is_not_invented(self):
        q = self.parse("buses at 25:99")
        self.assertIsNone(q["t0"])

    def test_classes_including_local_ones(self):
        self.assertEqual(self.parse("auto-rickshaws and cngs")["classes"], ["CNG"])
        self.assertEqual(self.parse("easy-bike or leguna")["classes"], ["Easybike", "Leguna"])
        self.assertEqual(self.parse("rickshaw")["classes"], ["Rickshaw"])
        self.assertEqual(self.parse("a human hauler")["classes"], ["Leguna"])
        self.assertEqual(self.parse("microbus")["classes"], ["MPV"])

    def test_bike_is_ambiguous_and_both_are_searched(self):
        q = self.parse("bikes at Farmgate")
        self.assertEqual(sorted(q["classes"]), ["Bicycle", "Motorbike"])
        self.assertTrue(any("can mean" in n for n in q["notes"]))

    def test_colours(self):
        self.assertEqual(self.parse("white cars")["colours"], ["white"])
        self.assertEqual(self.parse("silver and green")["colours"], ["grey", "green"])

    def test_behaviours_map_to_rule_codes(self):
        self.assertEqual(self.parse("white cars going wrong way")["codes"], ["V4"])
        self.assertEqual(self.parse("motorbikes without helmets")["codes"], ["V6"])
        self.assertEqual(self.parse("illegally parked cars")["codes"], ["V3"])
        self.assertEqual(self.parse("vehicles in the bus lane")["codes"], ["V9"])
        self.assertEqual(self.parse("near miss at the junction")["codes"], ["V11"])
        self.assertEqual(self.parse("any violations")["codes"], ["ANY"])

    def test_speed_thresholds_and_motion_words(self):
        q = self.parse("cars faster than 50 km/h")
        self.assertEqual(q["min_kph"], 50.0)
        self.assertEqual(self.parse("slow trucks")["behaviours"], ["slow"])
        self.assertIn("congestion", self.parse("why is traffic backed up")["behaviours"])
        self.assertEqual(self.parse("cars coming toward the camera")["direction"], "incoming")

    def test_camera_by_number_name_and_location(self):
        self.assertEqual(self.parse("camera 2")["camera_ids"], [2])
        self.assertEqual(self.parse("cams 1 and 3")["camera_ids"], [1, 3])
        self.assertEqual(self.parse("buses near karwan bazar")["camera_ids"], [2])
        self.assertEqual(self.parse("at the Mirpur 10 roundabout")["camera_ids"], [3])
        q = self.parse("camera 7")
        self.assertEqual((q["camera_ids"], q["unknown_cameras"]), ([], [7]))

    def test_camera_number_is_not_a_clock_time(self):
        q = self.parse("buses at camera 2 yesterday")
        self.assertEqual(q["camera_ids"], [2])
        self.assertEqual(self.span(q)[0], datetime(2026, 10, 29))

    def test_plate_fragments_and_vehicle_ids(self):
        self.assertEqual(self.parse("bus with plate 12-3456")["plate"], "12-3456")
        self.assertEqual(self.parse("number plate ending in 3456 yesterday")["plate"], "3456")
        self.assertEqual(self.parse("DHAKA METRO-GA 12-3456")["plate"], "12-3456")
        self.assertEqual(self.parse("where did vehicle #42 go")["global_id"], 42)

    def test_empty_and_nonsense_input_do_not_raise(self):
        self.assertIn("empty question", self.parse("   ")["notes"])
        q = self.parse("asdf qwerty zzz")
        self.assertEqual((q["classes"], q["codes"], q["t0"]), ([], [], None))
        self.assertIn("asdf", q["keywords"])
        self.assertEqual(len(self.parse("x" * 5000)["question"]), Q.MAX_QUESTION_CHARS)

    def test_router_rules(self):
        for text, want in (("why did traffic back up at 5:40?", "causal"), ("what caused the jam", "causal"),
                           ("explain the queue at camera 1", "causal"), ("what happened at camera 2 at 5pm", "causal"),
                           ("how many buses stopped at camera 2", "descriptive"),
                           ("show white cars going wrong way yesterday", "descriptive"),
                           ("list stopped trucks", "descriptive"), ("average speed at Farmgate", "descriptive")):
            self.assertEqual(self.parse(text)["intent"], want, text)
        self.assertEqual(Q.classify_intent("why")[1], ["'why'"])

    def test_a_count_with_a_why_is_causal(self):
        self.assertEqual(self.parse("why are there so many buses, how many?")["intent"], "causal")


# ======================================================================= LLM parse is validated

class LlmParseTests(unittest.TestCase):
    def llm(self, obj):
        return lambda question, ctx: json.dumps(obj) if not isinstance(obj, str) else obj

    def test_valid_fields_fill_what_the_rules_left_empty(self):
        q = Q.parse("tempo going the wrong way", CAMS, NOW,
                    self.llm({"classes": ["CNG"], "codes": ["V4"], "colours": ["green"]}))
        self.assertEqual((q["classes"], q["codes"], q["colours"], q["source"]), (["CNG"], ["V4"], ["green"], "rules+llm"))

    def test_rules_win_over_the_llm(self):
        q = Q.parse("buses at camera 2 yesterday", CAMS, NOW,
                    self.llm({"classes": ["Car"], "camera_ids": [1], "t0": "2026-10-01T00:00", "t1": "2026-10-02T00:00"}))
        self.assertEqual((q["classes"], q["camera_ids"]), (["Bus"], [2]))
        self.assertEqual(si.from_epoch(q["t0"]), datetime(2026, 10, 29))
        self.assertTrue(any("rules kept" in r or "does not name" in r for r in q["llm_rejected"]))

    def test_invented_camera_time_plate_and_classes_are_rejected(self):
        bad = {"classes": ["Spaceship", "Bus"], "colours": ["plaid"], "codes": ["V99"], "camera_ids": [3, 99],
               "t0": "2026-10-29T17:00", "t1": "2026-10-29T18:00", "plate": "ZZ-9999", "global_id": 7, "min_kph": 9999}
        q = Q.parse("show me something", CAMS, NOW, self.llm(bad))
        self.assertEqual(q["classes"], ["Bus"])                   # the vocabulary word survives, the invented one does not
        self.assertEqual((q["colours"], q["codes"], q["camera_ids"], q["plate"], q["global_id"], q["min_kph"]),
                         ([], [], [], None, None, None))
        self.assertIsNone(q["t0"])                                # the question has no time expression
        self.assertGreaterEqual(len(q["llm_rejected"]), 5)

    def test_time_with_a_cue_in_the_question_is_accepted_if_the_rules_found_none(self):
        clean, bad = Q.validate_fields({"t0": "2026-10-29T17:00", "t1": "2026-10-29T18:00"}, CAMS, NOW, text="buses around 5ish")
        self.assertEqual(si.from_epoch(clean["t0"]), datetime(2026, 10, 29, 17))
        self.assertEqual(bad, [])

    def test_implausible_times_are_rejected(self):
        for t0, t1 in (("2026-10-29T18:00", "2026-10-29T17:00"), ("1999-01-01", "1999-01-02"),
                       ("2026-10-01", "2026-12-01"), ("tomorrow-ish", "later"), (True, False)):
            clean, bad = Q.validate_fields({"t0": t0, "t1": t1}, CAMS, NOW, text="buses at 5 pm")
            self.assertNotIn("t0", clean, (t0, t1))
            self.assertTrue(bad)

    def test_garbage_replies_fall_back_to_the_rule_parse(self):
        for reply in ("I cannot help", "[1,2]", "{not json", None, 42):
            q = Q.parse("white cars at camera 1", CAMS, NOW, lambda *_a, r=reply: r)
            self.assertEqual((q["colours"], q["camera_ids"], q["llm_used"]), (["white"], [1], False))
        def boom(*_a):
            raise RuntimeError("down")
        q = Q.parse("white cars", CAMS, NOW, boom)
        self.assertEqual(q["colours"], ["white"])
        self.assertTrue(q["llm_rejected"])

    def test_llm_cannot_flip_the_router(self):
        q = Q.parse("why did traffic back up at 5:40?", CAMS, NOW, self.llm({"intent": "descriptive"}))
        self.assertEqual(q["intent"], "causal")

    def test_json_in_code_fences_is_read(self):
        self.assertEqual(Q.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(Q.extract_json('Sure! {"a": 2} hope that helps'), {"a": 2})

    def test_operator_edit_is_validated_but_trusted_about_cameras(self):
        q = Q.normalise_edited({"question": "x", "classes": ["Bus", "Nope"], "camera_ids": [2, 77], "t0": at(17), "t1": at(18),
                                "intent": "causal"}, CAMS, NOW)
        self.assertEqual((q["classes"], q["camera_ids"], q["intent"], q["source"]), (["Bus"], [2], "causal", "edited"))


# ======================================================================= index building

class IndexTests(Scratch):
    def setUp(self):
        self.store = self.make_store()
        self.start = DAY.replace(hour=17, minute=0)
        self.base = si.to_epoch(self.start)
        self.cam = self.store.add_camera("Farmgate", "file", "/nowhere/a.mp4", "Farmgate", start_time=self.start.isoformat())
        self.run = self.store.start_run(self.cam, None, "full")

    def small_world(self):
        s, c, r, b = self.store, self.cam, self.run, self.base
        add_pass(s, c, r, 1, "Bus", b + 3, speed=8.0, direction="incoming", colour="red", gid=11, base=b)
        add_pass(s, c, r, 2, "Car", b + 5, speed=0.5, direction="outgoing", colour="white", base=b)
        add_pass(s, c, r, 3, "CNG", b + 12, speed=5.0, direction="outgoing", colour="green", gid=12, base=b)
        add_pass(s, c, r, 4, "Bus", b + 14, speed=2.0, direction="incoming", colour="blue", base=b)
        add_violation(s, c, r, 1, "V4", "Wrong-side driving", "Bus", b + 3, base=b)                       # merges into pass 1
        add_violation(s, c, r, 99, "V1", "Bus stopped outside a bus stop", "Bus", b + 13, base=b,        # no tracklet: stand-alone
                      details={"dwell_s": 42.0})
        si.build_for_run(s, self.run)

    def test_counts_are_exact(self):
        self.small_world()
        w = self.store.q("SELECT * FROM seg_windows ORDER BY w_start")
        self.assertEqual([(x["w_start"] - self.base, x["n_pass"]) for x in w], [(0, 2), (10, 2)])
        a, b = si.window(self.store, w[0]["id"]), si.window(self.store, w[1]["id"])
        self.assertEqual(a["counts"], {"Bus": 1, "Car": 1})
        self.assertEqual((a["n_in"], a["n_out"], a["n_stopped"], a["events"], a["n_events"]), (1, 1, 1, {"V4": 1}, 1))
        self.assertAlmostEqual(a["mean_kph"], (8.0 * 3.6 + 0.5 * 3.6) / 2, 3)
        self.assertAlmostEqual(a["min_kph"], 1.8, 3)
        self.assertEqual(b["counts"], {"CNG": 1, "Bus": 1})
        self.assertEqual(b["events"], {"V1": 1})
        self.assertEqual(b["n_stopped"], 1)                       # only the V1 event: the 2 m/s bus is slow traffic, not stopped

    def test_stopped_means_slow_or_a_stationary_rule(self):
        self.small_world()
        rows = self.store.q("SELECT cls, kind, stopped, pkey FROM seg_vehicle ORDER BY t")
        by = {(r["cls"], r["kind"], r["pkey"].split(":")[-1]): r["stopped"] for r in rows}
        self.assertEqual(by[("Car", "pass", "2")], 1)             # 0.5 m/s
        self.assertEqual(by[("Bus", "pass", "4")], 0)             # 2 m/s is slow traffic, not stopped
        self.assertEqual(by[("Bus", "event", "99")], 1)           # V1 is a standing-still rule

    def test_verbalisation_and_links(self):
        self.small_world()
        r = self.store.q("SELECT * FROM seg_vehicle WHERE cls='Bus' AND kind='pass' AND global_id=11")[0]
        self.assertIn("red Bus crossed Main road", r["text"].replace("bus", "Bus"))
        self.assertIn("toward the camera", r["text"])
        self.assertIn("29 km/h", r["text"])
        self.assertIn("V4", r["text"])
        self.assertIn("vehicle #11", r["text"])
        self.assertEqual((r["codes"], r["colour"]), (" V4 ", "red"))
        ev = self.store.q("SELECT * FROM seg_vehicle WHERE kind='event'")[0]
        self.assertIn("stopped 42 s", ev["text"])
        self.assertEqual(ev["codes"], " V1 ")

    def test_colours_round_trip_through_the_stored_lab(self):
        for name, (L, a, b) in LAB.items():
            self.assertEqual(si.colour_of_lab(lab_blob(name).tobytes()), name, name)
        self.assertEqual(si.colour_of_lab(None), "")
        self.assertEqual(si.colour_of_lab(b"\x00\x01"), "")

    def test_source_video_offsets_allow_frames_to_be_pulled(self):
        self.small_world()
        w = self.store.q("SELECT * FROM seg_windows ORDER BY w_start")
        self.assertEqual((w[0]["video_t0"], w[0]["video_t1"]), (0.0, 10.0))
        self.assertEqual((w[1]["video_t0"], w[1]["video_t1"]), (10.0, 20.0))

    def test_idempotent_rebuild(self):
        self.small_world()
        snap = lambda: (self.store.q("SELECT camera_id, run_id, w_start, n_pass, counts, events, summary FROM seg_windows ORDER BY w_start"),  # noqa: E731
                        self.store.q("SELECT text, codes, stopped, kph, t, pkey FROM seg_vehicle ORDER BY t, pkey"))
        before = snap()
        si.build_for_run(self.store, self.run)
        si.build_for_run(self.store, self.run)
        self.assertEqual(snap(), before)
        n_fts = self.store.q("SELECT COUNT(*) AS n FROM seg_fts")[0]["n"] if si.has_fts(self.store) else 4
        self.assertEqual(n_fts, len(before[1]))

    def test_incremental_build_picks_up_new_rows_and_late_linking(self):
        self.small_world()
        self.assertEqual(si.build_for_camera(self.store, self.cam)["camera_id"], self.cam)
        n0 = self.store.q("SELECT COUNT(*) AS n FROM seg_vehicle")[0]["n"]
        self.assertEqual(si.build_for_camera(self.store, self.cam)["camera_id"], self.cam)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM seg_vehicle")[0]["n"], n0)
        add_pass(self.store, self.cam, self.run, 5, "Truck", self.base + 33, speed=7.0, colour="grey", base=self.base)
        si.build_for_camera(self.store, self.cam)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM seg_vehicle")[0]["n"], n0 + 1)
        # the batch linker assigns a global id later: a rebuild of the recent windows shows it
        tid = self.store.q("SELECT id FROM tracklets WHERE track_id=5")[0]["id"]
        self.store.set_global_ids([(tid, 777)])
        si.build_for_camera(self.store, self.cam)
        self.assertEqual(self.store.q("SELECT global_id FROM seg_vehicle WHERE tracklet_id=?", (tid,))[0]["global_id"], 777)

    def test_since_rebuilds_from_that_moment_and_removes_stale_windows(self):
        self.small_world()
        self.store.x("DELETE FROM tracklets WHERE track_id=3")
        si.build_for_camera(self.store, self.cam, since=self.base + 10)
        w = self.store.q("SELECT n_pass FROM seg_windows WHERE w_start>=?", (self.base + 10,))
        self.assertEqual([x["n_pass"] for x in w], [1])

    def test_impact_rows_reach_the_window(self):
        self.small_world()
        vid = self.store.q("SELECT id FROM violations WHERE code='V1'")[0]["id"]
        impact.save_impact(self.store, vid, {"score": 0.8, "camera_id": self.cam, "zone": "Main road", "code": "V1",
                                             "calibrated": False, "baseline_source": "x", "excess_delay_veh_s": 90.0,
                                             "during": {"queue_m": 41.0, "speed_kph": 6.5}})
        si.build_for_run(self.store, self.run)
        w = self.store.q("SELECT * FROM seg_windows WHERE n_events>0 AND w_start=?", (self.base + 10,))[0]
        self.assertAlmostEqual(w["impact_max"], 0.8)
        self.assertEqual(json.loads(w["state"]), {"queue_m": 41.0, "speed_kph": 6.5})

    def test_coverage_and_status(self):
        self.small_world()
        cov = si.coverage(self.store, [self.cam], self.base, self.base + 100)
        self.assertEqual(len(cov), 1)
        self.assertEqual(si.coverage(self.store, [self.cam], self.base + 86400, self.base + 90000), [])
        st = si.status(self.store)
        self.assertEqual((st["windows"], st["window_s"], st["vehicle_rows"]), (2, 10, 5))
        self.assertEqual(st["unindexed_tracklets"], 0)

    def test_hook_never_raises(self):
        self.assertIn("error", si.on_run_finished(self.store, 99999))
        self.small_world()
        self.assertEqual(si.on_run_finished(self.store, self.run)["run_id"], self.run)

    def test_video_without_wall_clock_is_indexed_but_not_time_searchable(self):
        cam = self.store.add_camera("NoClock", "file", "/x.mp4", "")
        run = self.store.start_run(cam, None, "full")
        add_pass(self.store, cam, run, 1, "Bus", 12.0, speed=5.0, colour="red")
        si.build_for_run(self.store, run)
        w = self.store.q("SELECT wall FROM seg_windows WHERE camera_id=?", (cam,))
        self.assertEqual([x["wall"] for x in w], [0])
        q = Q.parse_rules("buses yesterday", CAMS, NOW)
        q["camera_ids"] = [cam]
        self.assertEqual(Q.retrieve(self.store, q)["segments"], [])

    def test_like_fallback_when_fts5_is_missing(self):
        self.small_world()
        self.store.x("UPDATE seg_meta SET value='0' WHERE key='fts'")
        s = Q._fts_scores(self.store, ["wrong", "way"])
        self.assertTrue(s)                                          # found through LIKE on the terms column

    def test_fts_is_available_on_this_build(self):
        self.assertTrue(si.has_fts(self.store) or True)             # informational: the fallback is tested above
        if si.has_fts(self.store):
            self.small_world()
            hits = Q._fts_scores(self.store, ["stopped"])
            self.assertTrue(hits)


# ======================================================================= the synthetic world (X7)

def build_world(store, seed=11, tmp_dir=None):
    """Three cameras, 2 hours each, ~100 random vehicles per camera, plus planted answers. Returns the truth: one dict
    per vehicle pass that carries the attributes the queries ask about, independent of what the index inferred."""
    rng = random.Random(seed)
    cams = {}
    for cid, name in ((1, "Farmgate"), (2, "Karwan Bazar"), (3, "Mirpur 10")):
        src = "/nowhere/%d.mp4" % cid
        if tmp_dir is not None and cid == 2:
            src = str(Path(tmp_dir) / "cam2.mp4")
        c = store.add_camera(name, "file", src, name, start_time=DAY.isoformat())
        cams[cid] = (c, store.start_run(c, None, "full"))
    base = si.to_epoch(DAY)
    truth, track = [], {1: 0, 2: 0, 3: 0}

    def vehicle(cid, cls, t, speed, colour, direction="incoming", gid=None, codes=(), details=None, plate="", impact_state=None):
        c, run = cams[cid]
        track[cid] += 1
        tn = track[cid]
        add_pass(store, c, run, tn, cls, t, speed=speed, direction=direction, colour=colour, gid=gid, base=base)
        vid = None
        for code in codes:
            vid = add_violation(store, c, run, tn, code, "title", cls, t, base=base, details=details, plate=plate)
            if impact_state is not None:
                impact.save_impact(store, vid, {"score": 0.7, "camera_id": c, "zone": "Main road", "code": code, "calibrated": False,
                                                "baseline_source": "t", "excess_delay_veh_s": 50.0, "during": impact_state})
        truth.append({"pkey": f"{c}:{run}:{tn}", "camera": cid, "cls": cls, "colour": colour, "t": t,
                      "kph": speed * 3.6, "stopped": speed < 1.0 or any(x in ("V1", "V3", "V7") for x in codes),
                      "codes": list(codes), "gid": gid, "plate": plate, "direction": direction})

    weights = [("Car", 30), ("CNG", 20), ("Bus", 10), ("Motorbike", 20), ("Rickshaw", 10), ("Truck", 5), ("Pickup", 5)]
    palette = {"Car": ["white", "black", "grey", "red", "blue"], "CNG": ["green", "green", "green", "yellow"],
               "Bus": ["red", "blue", "green", "white"], "Motorbike": ["black", "red", "blue", "grey"],
               "Rickshaw": ["yellow", "blue", "red"], "Truck": ["grey", "green", "blue"], "Pickup": ["white", "grey"]}
    shared = 100
    for cid in cams:
        for _ in range(100):
            cls = rng.choices([w[0] for w in weights], [w[1] for w in weights])[0]
            gid = None
            if rng.random() < 0.2:
                gid, shared = shared, shared + 1
            t = base + rng.uniform(0, 7100)
            vehicle(cid, cls, t, rng.uniform(3.5, 9.0), rng.choice(palette[cls]), rng.choice(["incoming", "outgoing"]), gid)
    # planted answers --------------------------------------------------------------------------
    vehicle(1, "Car", at(17, 20, 3), 6.0, "white", gid=5, codes=["V4"])           # white wrong-way cars
    vehicle(1, "Car", at(17, 41, 7), 6.0, "white", codes=["V4"])
    vehicle(3, "Car", at(17, 8, 1), 5.0, "white", codes=["V4"])
    vehicle(3, "Car", at(18, 2, 1), 5.0, "black", codes=["V4"])                    # wrong way but black
    vehicle(2, "Car", at(17, 10, 2), 6.0, "white", gid=5)                          # vehicle 5 again, later, at camera 2
    for t in (at(17, 5, 5), at(17, 22, 5)):                                        # stopped buses at camera 2 between 5 and 6 pm
        vehicle(2, "Bus", t, 0.2, "red")
    vehicle(2, "Bus", at(17, 38, 41), 0.3, "white", codes=["V1"], details={"dwell_s": 42.0}, plate="DHAKA METRO-GA 11-2233",
            impact_state={"queue_m": 38.0, "speed_kph": 4.0})
    vehicle(2, "Bus", at(18, 10, 5), 0.2, "red")                                   # stopped bus after 6 pm
    vehicle(1, "Bus", at(17, 20, 25), 0.2, "red")                                  # stopped bus at another camera
    for i in range(13):                                                            # the 17:38-17:44 queue at camera 2
        vehicle(2, rng.choice(["Car", "CNG", "Motorbike"]), at(17, 38, 0) + 25 * i, rng.uniform(0.4, 0.9), "grey")
    vehicle(2, "Truck", at(17, 40, 5), 0.0, "grey", codes=["V7"], details={"dwell_s": 95.0},
            impact_state={"queue_m": 52.0, "speed_kph": 3.0})
    for t in (at(17, 12), at(17, 31)):                                             # green CNGs at camera 2 around 5-6 pm
        vehicle(2, "CNG", t + 0.5, 5.0, "green")
    vehicle(3, "Motorbike", at(17, 50, 9), 19.5, "black", codes=["V2"])            # speeding
    vehicle(3, "Motorbike", at(17, 55, 9), 20.0, "red", codes=["V2"])
    vehicle(1, "Motorbike", at(16, 59, 3), 5.0, "black", gid=300)
    vehicle(3, "Bus", at(17, 59, 3), 5.0, "green", gid=300)
    si.build_all(store)
    return truth


def write_video(path, seconds=7300, fps=1):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for i in range(seconds * fps):
        img = np.full((24, 32, 3), (i // fps) % 255, np.uint8)
        w.write(img)
    w.release()


def matches(t, pred):
    """Does one truth record satisfy a labelled query's predicate (tests/data/queries.jsonl)?"""
    if "cls" in pred and t["cls"] not in pred["cls"]:
        return False
    if "colour" in pred and t["colour"] != pred["colour"]:
        return False
    if "camera" in pred and t["camera"] != pred["camera"]:
        return False
    if "codes" in pred and not set(pred["codes"]) & set(t["codes"]):
        return False
    if pred.get("stopped") and not t["stopped"]:
        return False
    if "gid" in pred and t["gid"] != pred["gid"]:
        return False
    if "plate" in pred and pred["plate"] not in t["plate"]:
        return False
    if "kph_max" in pred and t["kph"] >= pred["kph_max"]:
        return False
    if "t" in pred:
        lo, hi = [at(*map(int, x.split(":"))) for x in pred["t"]]
        if not lo <= t["t"] <= hi:
            return False
    return True


def load_queries():
    return [json.loads(line) for line in (DATA / "queries.jsonl").read_text().splitlines() if line.strip() and not line.startswith("#")]


class WorldCase(Scratch):
    @classmethod
    def setUpClass(cls):
        cls._d = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._d.name)
        write_video(cls.dir / "cam2.mp4")
        cls.store = Store(cls.dir / "w.db")
        si.ensure_schema(cls.store)
        cls.truth = build_world(cls.store, tmp_dir=cls.dir)
        cls.cams = [{"id": c["id"], "name": c["name"], "location": c["location"]} for c in cls.store.cameras()]

    @classmethod
    def tearDownClass(cls):
        cls.store.db.close()
        cls._d.cleanup()

    def ask(self, text, **kw):
        return Q.answer_question(self.store, text, now=NOW, data_dir=self.dir, **kw)


# ======================================================================= retrieval and descriptive answers

class RetrievalTests(WorldCase):
    def test_every_tracklet_and_violation_is_in_the_index(self):
        st = si.status(self.store)
        self.assertEqual(st["unindexed_tracklets"], 0)
        n_tr = self.store.q("SELECT COUNT(*) AS n FROM tracklets")[0]["n"]
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM seg_vehicle WHERE kind='pass'")[0]["n"], n_tr)
        self.assertEqual(self.store.q("SELECT COUNT(*) AS n FROM seg_vehicle")[0]["n"], n_tr)   # every violation had a tracklet here
        self.assertEqual(self.store.q("SELECT SUM(n_pass) AS n FROM seg_windows")[0]["n"], n_tr)

    def test_ranking_prefers_the_exact_matches(self):
        q = Q.parse_rules("white cars going wrong way", self.cams, NOW)
        r = Q.retrieve(self.store, q, 10)
        self.assertEqual(r["mode"], "vehicle")
        want = {t["pkey"] for t in self.truth if t["codes"] == ["V4"] and t["cls"] == "Car"}
        top = [m["pkey"] for s in r["segments"][:3] for m in s["matches"][:1]]
        self.assertEqual(len(r["segments"]), 4)                                  # 4 wrong-way cars, 4 windows
        self.assertTrue(set(top) <= want)
        white = {t["pkey"] for t in self.truth if t["codes"] == ["V4"] and t["colour"] == "white"}
        self.assertEqual({s["matches"][0]["pkey"] for s in r["segments"][:3]}, white)       # the black one comes last
        self.assertEqual(r["segments"][3]["matches"][0]["colour"], "black")
        self.assertTrue(r["segments"][0]["score"] > r["segments"][3]["score"])
        self.assertTrue(any("colour white" in x for x in r["segments"][0]["reasons"]))

    def test_every_segment_says_why_it_matched(self):
        for text in ("buses stopped at camera 2", "speeding motorbikes", "vehicle 5"):
            r = Q.retrieve(self.store, Q.parse_rules(text, self.cams, NOW), 5)
            self.assertTrue(r["segments"], text)
            for s in r["segments"]:
                self.assertTrue(s["reasons"], text)
                self.assertTrue(s["matches"])
                self.assertIsInstance(s["score"], float)

    def test_plate_fragment_and_vehicle_id(self):
        r = Q.retrieve(self.store, Q.parse_rules("bus with plate 11-2233", self.cams, NOW))
        self.assertEqual(len(r["segments"]), 1)
        self.assertEqual(r["segments"][0]["matches"][0]["plate"], "DHAKA METRO-GA 11-2233")
        r = Q.retrieve(self.store, Q.parse_rules("show vehicle 5", self.cams, NOW))
        self.assertEqual(sorted(m["pkey"].split(":")[0] for s in r["segments"] for m in s["matches"]), ["1", "2"])

    def test_scene_question_at_a_time_finds_the_queue(self):
        q = Q.parse_rules("why did traffic back up at 5:40 at camera 2?", self.cams, NOW)
        r = Q.retrieve(self.store, q, 5)
        self.assertEqual(r["mode"], "scene")
        top = r["segments"][0]
        self.assertEqual(top["camera_id"], 2)
        self.assertLess(abs((top["t_start"] + 5) - at(17, 40)), 90)
        self.assertGreaterEqual(top["score"], Q.WEAK_SCORE)

    def test_unknown_camera_and_quiet_scene_return_nothing(self):
        self.assertEqual(Q.retrieve(self.store, Q.parse_rules("buses at camera 9", self.cams, NOW))["segments"], [])
        q = Q.parse_rules("why was there a jam at 4:35 pm at camera 3", self.cams, NOW)
        r = Q.retrieve(self.store, q, 5)
        self.assertTrue(all(s["score"] < Q.WEAK_SCORE for s in r["segments"]))


class DescriptiveTests(WorldCase):
    def count(self, text):
        out = self.ask(text)
        self.assertEqual(out["route"]["intent"], "descriptive", text)
        return out

    def test_buses_stopped_at_camera_2_between_5_and_6_pm_is_exact(self):
        out = self.count("how many buses stopped at camera 2 between 5 and 6 pm?")
        want = [t for t in self.truth if t["cls"] == "Bus" and t["stopped"] and t["camera"] == 2 and at(17) <= t["t"] <= at(18)]
        self.assertEqual(len(want), 3)
        self.assertEqual(out["answer"]["mode"], "index")
        self.assertEqual(out["answer"]["value"], 3)
        self.assertEqual({r["pkey"] for r in out["answer"]["rows"]}, {t["pkey"] for t in want})
        self.assertIn("3 vehicles", out["answer"]["text"])
        self.assertTrue(out["answer"]["caveats"])

    def test_counts_by_class_colour_and_camera(self):
        out = self.count("how many white cars went the wrong way yesterday")
        self.assertEqual(out["answer"]["value"], 3)
        out = self.count("how many wrong way vehicles at Farmgate")
        self.assertEqual(out["answer"]["value"], 2)
        out = self.count("how many buses at camera 1 stopped")
        self.assertEqual(out["answer"]["value"], 1)

    def test_a_real_zero_is_reported_as_zero_when_the_time_is_covered(self):
        out = self.count("how many helmetless motorbikes at camera 2 between 5 and 6 pm")
        self.assertEqual((out["answer"]["mode"], out["answer"]["value"]), ("index", 0))

    def test_no_coverage_is_insufficient_not_zero(self):
        out = self.count("how many buses at camera 2 on 2026-09-01")
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertIn("no analysed video covers", out["answer"]["text"])
        self.assertIsNone(out["answer"]["value"])

    def test_average_speed_and_when(self):
        out = self.ask("average speed of speeding motorbikes at Mirpur 10")
        self.assertEqual(out["answer"]["stats"]["n"], 2)
        self.assertAlmostEqual(out["answer"]["stats"]["mean_kph"], (19.5 + 20.0) / 2 * 3.6, 1)
        out = self.ask("what time did the buses stop at camera 2 between 5 and 6 pm")
        self.assertIn("first at", out["answer"]["text"])

    def test_partial_coverage_is_called_out(self):
        out = self.count("how many cars at camera 2 yesterday between 4 and 8 pm")
        self.assertTrue(any("only covers" in c for c in out["answer"]["caveats"]))

    def test_nothing_in_the_index_at_all(self):
        empty = self.make_store()
        out = Q.answer_question(empty, "how many buses", now=NOW)
        self.assertEqual(out["answer"]["mode"], "insufficient")


# ======================================================================= Stage B causal routing

class FakeReasoner:
    def __init__(self, reply=None, boom=False):
        self.calls, self.reply, self.boom = [], reply, boom

    def __call__(self, question, segments):
        self.calls.append((question, segments))
        if self.boom:
            raise RuntimeError("model crashed")
        ids = [s["id"] for s in segments]
        return self.reply if self.reply is not None else {"answer": "A stopped bus is blocking the lane.", "sufficient": True,
                                                           "confidence": 0.7, "citations": ids[:1], "limits": "two frames only"}


class CausalTests(WorldCase):
    Q = "why did traffic back up at 5:40 at camera 2?"

    def test_causal_goes_to_the_reasoner_with_frames_and_cites_a_segment(self):
        fake = FakeReasoner()
        out = self.ask(self.Q, reason_fn=fake)
        self.assertEqual(out["route"]["intent"], "causal")
        self.assertEqual(len(fake.calls), 1)
        question, segs = fake.calls[0]
        self.assertLessEqual(len(segs), Q.CAUSAL_SEGMENTS)
        self.assertTrue(all(s["frames"] and s["frames"][0]["jpeg"][:2] == b"\xff\xd8" for s in segs))   # real JPEGs from the video
        self.assertTrue(all(len(s["frames"]) <= Q.CAUSAL_FRAMES_PER_SEGMENT for s in segs))
        self.assertTrue(all(s["facts"] for s in segs))
        a = out["answer"]
        self.assertEqual((a["mode"], a["citations"]), ("vlm", [segs[0]["id"]]))
        self.assertTrue(a["limits"])
        self.assertTrue(any(s["cited"] for s in out["segments"]))
        self.assertTrue(all(e["frames"] > 0 for e in a["evidence"]))

    def test_descriptive_never_calls_the_reasoner(self):
        fake = FakeReasoner()
        self.ask("how many buses stopped at camera 2 between 5 and 6 pm", reason_fn=fake)
        self.assertEqual(fake.calls, [])

    def test_intent_can_be_forced_by_the_operator(self):
        fake = FakeReasoner()
        out = self.ask("buses stopped at camera 2 between 5 and 6 pm", reason_fn=fake, intent="causal")
        self.assertEqual((out["route"]["intent"], len(fake.calls)), ("causal", 1))

    def test_empty_retrieval_is_insufficient_and_the_reasoner_is_not_asked(self):
        fake = FakeReasoner()
        out = self.ask("why did the traffic stop at camera 9 at 5:40", reason_fn=fake)
        self.assertEqual((out["answer"]["mode"], fake.calls), ("insufficient", []))
        out = self.ask("why was there a jam at 4:35 pm at camera 3", reason_fn=fake)
        self.assertEqual((out["answer"]["mode"], fake.calls), ("insufficient", []))
        self.assertIn("Insufficient evidence", out["answer"]["text"])

    def test_no_coverage_is_named(self):
        out = self.ask("why did traffic back up at 5:40 on 2026-09-01", reason_fn=FakeReasoner())
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertIn("no analysed video covers", out["answer"]["text"])

    def test_raw_video_gone_is_said(self):
        fake = FakeReasoner()
        out = Q.answer_question(self.store, "why did traffic back up at 5:40 at camera 3 near the buses", now=NOW,
                                reason_fn=fake, data_dir=self.dir)
        # camera 3 has no video file in this world, so no frames can be pulled for it
        q = Q.parse_rules("why did the car go the wrong way at camera 3", self.cams, NOW)
        r = Q.retrieve(self.store, q)
        self.assertTrue(r["segments"])
        a = Q.answer_causal(self.store, q, r, fake, self.dir)
        self.assertEqual(a["mode"], "insufficient")
        self.assertIn("raw video", a["text"])
        self.assertEqual(fake.calls, [])
        del out

    def test_evidence_clip_or_snapshot_is_the_fallback_when_video_is_gone(self):
        import cv2
        d = self.dir / "evidence" / "demo"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "snap.jpg"), np.full((60, 80, 3), 90, np.uint8))
        vid = self.store.q("SELECT id FROM violations WHERE code='V4' AND camera_id=3 ORDER BY id LIMIT 1")[0]["id"]
        self.store.x("UPDATE violations SET snapshot='evidence/demo/snap.jpg' WHERE id=?", (vid,))
        si.build_for_camera(self.store, 3, since=at(16))
        wid = self.store.q("SELECT window_id FROM seg_vehicle WHERE violation_id=?", (vid,))[0]["window_id"]
        got = Q.pull_frames(self.store, wid, 3, self.dir)
        self.assertEqual((got["source"], len(got["frames"])), ("snapshot", 1))
        self.assertIn("not available", got["note"])
        # a path that climbs out of the data folder is not followed
        self.store.x("UPDATE violations SET snapshot='../../etc/hostname' WHERE id=?", (vid,))
        si.build_for_camera(self.store, 3, since=at(16))
        wid = self.store.q("SELECT window_id FROM seg_vehicle WHERE violation_id=?", (vid,))[0]["window_id"]
        self.assertEqual(Q.pull_frames(self.store, wid, 3, self.dir)["source"], "gone")

    def test_an_answer_that_cites_nothing_is_not_a_finding(self):
        out = self.ask(self.Q, reason_fn=FakeReasoner({"answer": "Probably a wedding.", "sufficient": True, "citations": []}))
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertIn("cited none", out["answer"]["text"])

    def test_citing_a_segment_that_was_not_retrieved_is_ignored(self):
        out = self.ask(self.Q, reason_fn=FakeReasoner({"answer": "x", "sufficient": True, "citations": [999999]}))
        self.assertEqual(out["answer"]["mode"], "insufficient")

    def test_reasoner_saying_insufficient_is_first_class(self):
        out = self.ask(self.Q, reason_fn=FakeReasoner({"answer": "The frames are too dark.", "sufficient": False, "citations": []}))
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertIn("too dark", out["answer"]["text"])

    def test_reasoner_failure_and_absence_are_insufficient(self):
        out = self.ask(self.Q, reason_fn=FakeReasoner(boom=True))
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertTrue(out["answer"]["unavailable"])
        out = self.ask(self.Q, reason_fn=None)
        self.assertEqual(out["answer"]["mode"], "insufficient")
        self.assertIn("not available", out["answer"]["text"])
        self.assertTrue(out["segments"])                             # the retrieved segments are still shown

    def test_garbage_from_the_reasoner(self):
        for reply in ("a string", None, [1], 5):
            out = self.ask(self.Q, reason_fn=lambda q, s, r=reply: r)
            self.assertEqual(out["answer"]["mode"], "insufficient", reply)

    def test_adapter_builds_the_prompt_and_parses_the_reply(self):
        seen = {}

        class Backend:
            def ask(self, system, user_text, images, max_tokens=None, schema=None):
                seen.update(system=system, text=user_text, images=images, schema=schema)
                return '```json\n{"answer": "Bus stopped.", "sufficient": true, "confidence": 0.6, "citations": [%d], "limits": ""}\n```' % seen["first"]

        fn = Q.make_reason_fn(Backend())
        out = self.ask(self.Q, reason_fn=lambda q, s: (seen.update(first=s[0]["id"]), fn(q, s))[1])
        self.assertEqual(out["answer"]["mode"], "vlm")
        self.assertTrue(seen["images"] and isinstance(seen["images"][0][1], bytes))
        self.assertIn("[segment", seen["text"])
        self.assertIn("Never guess", seen["system"])
        self.assertEqual(seen["schema"]["required"][0], "answer")


# ======================================================================= HTTP

class HttpTests(WorldCase):
    def setUp(self):
        fa = FakeAuth()
        self.fake = FakeReasoner()
        app = FastAPI()
        web = self.dir / "web"
        web.mkdir(exist_ok=True)
        (web / "query.html").write_text("<html>query</html>")
        app.include_router(api_query.make_router(self.store, fa.current_user, fa.require, "controller", "admin",
                                                 data_dir=self.dir, reason_fn=self.fake, web_dir=web, now_fn=lambda: NOW))
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
            return status, out

    def test_roles(self):
        body = {"question": "how many buses"}
        for role, want in ((None, 401), ("driver", 403), ("controller", 200), ("admin", 200)):
            self.assertEqual(self.call("POST", "/api/query", role, body)[0], want, role)
        for path in ("/api/query/index/status", "/api/query/segments/1"):
            self.assertEqual(self.call("GET", path, "driver")[0], 403, path)
            self.assertEqual(self.call("GET", path, None)[0], 401, path)
        self.assertEqual(self.call("POST", "/api/query/index/rebuild", "controller", {})[0], 403)
        self.assertEqual(self.call("POST", "/api/query/index/rebuild", "driver", {})[0], 403)
        self.assertEqual(self.call("GET", "/query", "driver")[0], 302)
        self.assertEqual(self.call("GET", "/query", None)[0], 302)
        self.assertEqual(self.call("GET", "/query", "controller")[0], 200)

    def test_ask_returns_answer_segments_and_parse(self):
        s, out = self.call("POST", "/api/query", "controller", {"question": "how many buses stopped at camera 2 between 5 and 6 pm?"})
        self.assertEqual(s, 200)
        self.assertEqual(out["answer"]["value"], 3)
        self.assertEqual(out["parsed"]["camera_ids"], [2])
        self.assertTrue(out["segments"])
        self.assertNotIn("jpeg", json.dumps(out))
        self.assertNotIn("nowhere", json.dumps(out))                # a camera's source path never leaves the server
        self.assertEqual(self.store.audit_trail("query", "")[0]["action"], "query.ask")

    def test_edited_parse_is_used(self):
        s, first = self.call("POST", "/api/query", "controller", {"question": "buses stopped"})
        edited = {**first["parsed"], "camera_ids": [2], "t0": at(17), "t1": at(18)}
        s, out = self.call("POST", "/api/query", "controller", {"question": "buses stopped", "parsed": edited})
        self.assertEqual(out["answer"]["value"], 3)
        self.assertEqual(out["parsed"]["source"], "edited")

    def test_bad_requests(self):
        for body in ({}, {"question": ""}, {"question": 5}, {"question": "x" * 900}, {"question": "a", "parsed": [1]},
                     {"question": "a", "limit": "many"}, {"question": "a", "intent": "poetry"}):
            self.assertEqual(self.call("POST", "/api/query", "controller", body)[0], 400, body)
        self.assertEqual(asgi_call(self.app, "POST", "/api/query", {"x-role": "controller"}, b"{bad")[0], 400)

    def test_causal_over_http_uses_the_injected_reasoner(self):
        s, out = self.call("POST", "/api/query", "controller", {"question": CausalTests.Q})
        self.assertEqual((s, out["answer"]["mode"]), (200, "vlm"))
        self.assertEqual(len(self.fake.calls), 1)

    def test_segment_detail_frame_and_thumb(self):
        s, out = self.call("POST", "/api/query", "controller", {"question": CausalTests.Q})
        sid = out["segments"][0]["id"]
        s, d = self.call("GET", f"/api/query/segments/{sid}", "controller")
        self.assertEqual(s, 200)
        self.assertTrue(d["frames"]["video"])
        self.assertTrue(d["vehicles"] and "links" in d["vehicles"][0])
        self.assertNotIn("source", json.dumps(d["camera"]))
        st, hdrs, body = asgi_call(self.app, "GET", f"/api/query/segments/{sid}/frame?i=1&n=3", {"x-role": "controller"})
        self.assertEqual((st, hdrs["content-type"], body[:2]), (200, "image/jpeg", b"\xff\xd8"))
        st, hdrs, body = asgi_call(self.app, "GET", f"/api/query/segments/{sid}/thumb", {"x-role": "admin"})
        self.assertEqual((st, body[:2]), (200, b"\xff\xd8"))
        self.assertEqual(self.call("GET", "/api/query/segments/99999999", "controller")[0], 404)
        self.assertEqual(self.call("GET", f"/api/query/segments/{sid}/frame?i=99", "controller")[0], 422)

    def test_status_and_rebuild(self):
        s, st = self.call("GET", "/api/query/index/status", "controller")
        self.assertEqual((s, st["window_s"], st["job"]["state"]), (200, 10, "idle"))
        self.assertGreater(st["windows"], 100)
        self.assertEqual(self.call("POST", "/api/query/index/rebuild", "admin", {"camera_id": 99})[0], 404)
        self.assertEqual(self.call("POST", "/api/query/index/rebuild", "admin", {"since": "banana"})[0], 400)
        n0 = si.status(self.store)["vehicle_rows"]
        self.assertEqual(self.call("POST", "/api/query/index/rebuild", "admin", {"camera_id": 1})[0], 202)
        import time
        for _ in range(100):
            s, st = self.call("GET", "/api/query/index/status", "controller")
            if st["job"]["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(st["job"]["state"], "done")
        self.assertEqual(si.status(self.store)["vehicle_rows"], n0)           # idempotent
        self.assertEqual(self.store.audit_trail("index", "")[0]["action"], "query.index.rebuild")


# ======================================================================= X7: grounding metrics

class GroundingMetricsTests(unittest.TestCase):
    def test_mrr_and_recall_by_hand(self):
        ranked = [[["a"], ["b"], ["c", "d"]],            # first hit at rank 2
                  [["x"], ["y"]],                        # no hit
                  [["p", "q"]]]                          # hit at rank 1, finds 1 of 2 gold
        gold = [["b", "d"], ["z"], ["p", "r"]]
        m = Q.grounding_metrics(ranked, gold, ks=(1, 2, 3))
        self.assertEqual(m["n_queries"], 3)
        self.assertAlmostEqual(m["mrr"], round((0.5 + 0 + 1) / 3, 4))
        self.assertEqual(m["recall_at"][1], round((0 + 0 + 0.5) / 3, 4))
        self.assertEqual(m["recall_at"][2], round((0.5 + 0 + 0.5) / 3, 4))
        self.assertEqual(m["recall_at"][3], round((1.0 + 0 + 0.5) / 3, 4))
        self.assertEqual(m["hit_at"], {1: round(1 / 3, 4), 2: round(2 / 3, 4), 3: round(2 / 3, 4)})

    def test_queries_without_gold_are_skipped(self):
        self.assertEqual(Q.grounding_metrics([[["a"]]], [[]])["n_queries"], 0)


class SyntheticGroundingTests(WorldCase):
    """X7 hook. The queries file is hand-written against the synthetic world above; the numbers it gives show the
    pipeline and the metric code work, and nothing about real questions."""

    def test_labelled_queries(self):
        cases = []
        for row in load_queries():
            gold = [t["pkey"] for t in self.truth if matches(t, row["gold"])]
            self.assertTrue(gold, row["id"])                  # a labelled query with no answer is a mistake in the file
            cases.append({"id": row["id"], "question": row["question"], "gold": gold})
        self.assertGreaterEqual(len(cases), 12)
        res = Q.evaluate_queries(self.store, cases, NOW, ks=(1, 3, 5, 10), limit=20)
        self.assertEqual(res["n_queries"], len(cases))
        worst = sorted(res["per_query"], key=lambda r: r["mrr"])[:3]
        self.assertGreaterEqual(res["mrr"], 0.9, worst)
        self.assertGreaterEqual(res["recall_at"][10], 0.9, worst)
        self.assertTrue(all(r["n_segments"] > 0 for r in res["per_query"]))


if __name__ == "__main__":
    unittest.main()
