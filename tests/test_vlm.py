"""Prompt building, reply parsing and the queue of the vision-language reasoner (no model needed).

The parsing tests matter more than they look: everything the model returns is written onto an enforcement
case, so a malformed or hostile reply must degrade to a decline and never to "supported".
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import vlm  # noqa: E402


def case(**over):
    v = {"id": 7, "code": "V1", "title": "Bus stopped away from a bus stop", "cls": "Bus", "class_conf": 0.93,
         "kph": 0.0, "zone": "Mirpur Road southbound", "areas": "", "wall_time": "2026-03-04 17:42:11",
         "details": {"dwell_s": 42, "vehicles_passed": 19}, "snapshot": None, "crop": None, "evidence": []}
    v.update(over)
    return v


class PromptTests(unittest.TestCase):
    def test_claim_and_innocent_explanation_are_both_given(self):
        p = vlm.build_prompt(case(), [("close-up", b"x")])
        self.assertIn("away from any marked bus stop", p)       # what the engine decided
        self.assertIn("red light, jam", p)                      # what would overturn it
        self.assertIn("close-up", p)

    def test_measurements_are_passed_as_given_facts(self):
        p = vlm.build_prompt(case(), [])
        self.assertIn("dwell s: 42", p)
        self.assertIn("vehicles passed: 19", p)
        self.assertIn("Mirpur Road southbound", p)

    def test_every_rule_code_has_a_claim(self):
        for code in [f"V{n}" for n in range(1, 12)]:
            claim, innocent = vlm.RULE_CLAIM[code]
            self.assertTrue(claim and innocent, code)

    def test_unknown_code_does_not_crash(self):
        self.assertIn("V99", vlm.build_prompt(case(code="V99"), []))

    def test_plate_is_offered_but_not_requested_back(self):
        p = vlm.build_prompt(case(plate_text_en="DHAKA METRO-GA 12-3456"), [])
        self.assertIn("do not repeat this", p)


class ParseTests(unittest.TestCase):
    good = {"pre_event": "Traffic was flowing.", "event": "A bus halted in the near lane.",
            "aftermath": "Vehicles moved around it.", "cause": "Passengers were boarding.",
            "assessment": "supported", "confidence": 0.8, "summary": "A bus loaded in a running lane."}

    def parse(self, text):
        return vlm.to_explanation(vlm.parse_reply(text), "m", "local", 1.0, 1)

    def test_plain_json(self):
        ex = self.parse(json.dumps(self.good))
        self.assertEqual(ex.assessment, "supported")
        self.assertEqual(ex.confidence, 0.8)
        self.assertEqual(ex.cause, "Passengers were boarding.")

    def test_fenced_json(self):
        self.assertEqual(self.parse("```json\n" + json.dumps(self.good) + "\n```").assessment, "supported")

    def test_json_with_chatter_around_it(self):
        ex = self.parse("Sure! Here is the result:\n" + json.dumps(self.good) + "\nHope that helps.")
        self.assertEqual(ex.assessment, "supported")

    def test_unparsable_reply_is_a_failure_not_an_agreement(self):
        ex = self.parse("I cannot help with that.")
        self.assertFalse(ex.ok)
        self.assertEqual(ex.assessment, "insufficient")

    def test_unknown_assessment_degrades_to_decline(self):
        ex = self.parse(json.dumps({**self.good, "assessment": "definitely guilty"}))
        self.assertEqual(ex.assessment, "insufficient")

    def test_confidence_is_clamped_and_percentages_accepted(self):
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": 0.8})).confidence, 0.8)
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": 95})).confidence, 0.95)
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": 150})).confidence, 1.0)
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": -1})).confidence, 0.0)
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": "high"})).confidence, 0.0)

    def test_an_odd_value_above_one_errs_low_not_high(self):
        """4.2 is read as 4.2%, not rounded up to certainty: overstating confidence on a case that may
        support a prosecution is the harmful mistake."""
        self.assertEqual(self.parse(json.dumps({**self.good, "confidence": 4.2})).confidence, 0.04)

    def test_declining_is_a_valid_successful_answer(self):
        ex = self.parse(json.dumps({**self.good, "assessment": "insufficient", "confidence": 0.3,
                                    "cause": "cannot be determined"}))
        self.assertTrue(ex.ok)
        self.assertEqual(ex.assessment, "insufficient")

    def test_contradicted_is_carried_through(self):
        self.assertEqual(self.parse(json.dumps({**self.good, "assessment": "contradicted"})).assessment,
                         "contradicted")

    def test_overlong_text_is_cut(self):
        ex = self.parse(json.dumps({**self.good, "event": "x" * 5000}))
        self.assertLessEqual(len(ex.event), 600)

    def test_missing_summary_falls_back_to_the_event(self):
        d = dict(self.good)
        del d["summary"]
        self.assertEqual(self.parse(json.dumps(d)).summary, self.good["event"])

    def test_non_object_json_is_rejected(self):
        self.assertFalse(self.parse("[1, 2, 3]").ok)


class GuardTests(unittest.TestCase):
    """A small model tends to agree with whatever the prompt proposes. These guards overrule an agreement
    the model's own observation does not back up, and they may only ever weaken a verdict.

    Both were written against measured failures of Qwen2.5-VL-3B on tools/vlm_probe.py: it confirmed a bus
    violation from a photograph with no bus in it, and a helmet violation from a photograph of a bus.
    """

    def judge(self, subject="Bus", **over):
        obj = {"visible": "A wide street with shops.", "present": True, "assessment": "supported",
               "confidence": 0.9, "summary": "s", "event": "e"}
        obj.update(over)
        return vlm.to_explanation(obj, "m", "local", 1.0, 1, subject)

    def test_supported_is_kept_when_the_model_saw_the_vehicle(self):
        ex = self.judge(visible="A blue bus stands in the near lane.")
        self.assertEqual(ex.assessment, "supported")
        self.assertEqual(ex.confidence, 0.9)
        self.assertEqual(ex.downgraded, "")

    def test_supported_is_overruled_when_the_vehicle_was_not_identifiable(self):
        ex = self.judge(present=False, visible="A blue bus stands in the near lane.")
        self.assertEqual(ex.assessment, "insufficient")
        self.assertLessEqual(ex.confidence, 0.3)
        self.assertIn("could not identify", ex.downgraded)

    def test_supported_is_overruled_when_its_own_description_has_no_such_vehicle(self):
        ex = self.judge(subject="Motorbike", visible="A stationary electric bus and some pedestrians.")
        self.assertEqual(ex.assessment, "insufficient")
        self.assertIn("without mentioning", ex.downgraded)

    def test_a_decline_is_never_promoted(self):
        ex = self.judge(assessment="insufficient", visible="A blue bus stands in the near lane.")
        self.assertEqual(ex.assessment, "insufficient")
        self.assertEqual(ex.downgraded, "")

    def test_contradicted_is_left_alone(self):
        self.assertEqual(self.judge(assessment="contradicted", present=False).assessment, "contradicted")

    def test_synonyms_count_as_seeing_the_vehicle(self):
        for word in ("motorcycle", "scooter", "the rider", "two-wheeler"):
            ex = self.judge(subject="Motorbike", visible=f"A {word} in the middle of the road.")
            self.assertEqual(ex.assessment, "supported", word)

    def test_an_unknown_class_is_never_downgraded_on_a_guess(self):
        ex = self.judge(subject="Spaceship", visible="A wide street with shops.")
        self.assertEqual(ex.assessment, "supported")

    def test_an_empty_description_is_not_treated_as_a_contradiction(self):
        ex = self.judge(visible="")
        self.assertEqual(ex.assessment, "supported")

    def test_subject_missing_is_case_insensitive(self):
        self.assertFalse(vlm.subject_missing("BUS", "a Bus waits at the kerb"))
        self.assertTrue(vlm.subject_missing("Bus", "two people on a footpath"))


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "ev").mkdir()
        for name, size in (("snap.jpg", (1440, 2560)), ("crop.jpg", (200, 160))):
            cv2.imwrite(str(self.root / "ev" / name), np.full((*size, 3), 128, np.uint8))

    def tearDown(self):
        self.dir.cleanup()

    def test_large_frame_is_scaled_down_for_the_encoder(self):
        jpg = vlm.encode_image(self.root / "ev" / "snap.jpg", 768)
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(max(img.shape[:2]), 768)

    def test_small_image_is_not_upscaled(self):
        jpg = vlm.encode_image(self.root / "ev" / "crop.jpg", 768)
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (200, 160))

    def test_close_up_comes_first_and_the_limit_is_respected(self):
        got = vlm.gather_images(case(crop="ev/crop.jpg", snapshot="ev/snap.jpg"), self.root, 2, 512)
        self.assertEqual(len(got), 2)
        self.assertIn("close-up", got[0][0])

    def test_missing_file_is_skipped_not_fatal(self):
        got = vlm.gather_images(case(crop="ev/gone.jpg", snapshot="ev/snap.jpg"), self.root, 2, 512)
        self.assertEqual(len(got), 1)


class QueueTests(unittest.TestCase):
    """The Explainer must never block the caller, never run two cases at once, and never die on an error."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.done = threading.Event()
        self.results = []

    def tearDown(self):
        self.dir.cleanup()

    def explainer(self, work):
        ex = vlm.Explainer(self.dir.name, {"vlm_backend": "off"}, self._collect)
        ex.explain_now = work
        return ex

    def _collect(self, vid, explanation):
        self.results.append((vid, explanation))
        self.done.set()

    def test_no_backend_configured_is_reported_not_raised(self):
        ex = vlm.Explainer(self.dir.name, {"vlm_backend": "off"}, self._collect)
        out = ex.explain_now(case())
        self.assertFalse(out.ok)
        self.assertIn("No reasoning model", out.error)

    def test_a_crashing_backend_does_not_kill_the_thread(self):
        def boom(v, settings=None):
            raise RuntimeError("backend exploded")
        ex = self.explainer(boom)
        ex.request(case())
        self.assertTrue(self.done.wait(5))
        self.assertIn("exploded", self.results[0][1].error)
        self.assertEqual(ex.busy(), [])          # and the case is released

    def test_the_same_case_is_not_queued_twice(self):
        hold = threading.Event()
        ex = self.explainer(lambda v, settings=None: (hold.wait(5), vlm.Explanation())[1])
        self.assertTrue(ex.request(case()))
        self.assertFalse(ex.request(case()))     # already pending
        hold.set()

    def test_per_request_settings_override_the_default(self):
        seen = {}
        ex = self.explainer(lambda v, settings=None: (seen.update(settings or {}), vlm.Explanation())[1])
        ex.request(case(), {"vlm_backend": "cloud"})
        self.assertTrue(self.done.wait(5))
        self.assertEqual(seen.get("vlm_backend"), "cloud")


class StoreTests(unittest.TestCase):
    """The explanation must survive a round trip through SQLite, and a database written by an earlier
    version of the system must gain the new columns rather than fail to open."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "s.db"

    def tearDown(self):
        self.dir.cleanup()

    def seed(self, store):
        cid = store.add_camera("cam", "file", "v.mp4")
        rec = {"code": "V1", "violation": "Bus stopped", "track_id": 3, "class": "Bus",
               "class_confidence": 0.9, "video_time_s": 1.0, "started_s": 0.0, "wall_time": "",
               "zone": "z", "areas": "", "kph": 0.0, "details": {}}
        return store.add_violation(cid, None, None, rec, "s.jpg", "c.jpg", "c.mp4")

    def test_round_trip(self):
        from surveillance.store import Store
        store = Store(self.path)
        vid = self.seed(store)
        self.assertEqual(store.violation(vid)["vlm_status"], "")
        self.assertIsNone(store.violation(vid)["vlm"])

        store.set_vlm(vid, "pending")
        self.assertEqual(store.violation(vid)["vlm_status"], "pending")

        ex = vlm.Explanation(assessment="contradicted", confidence=0.7, summary="Traffic was stopped too.")
        store.set_vlm(vid, "done", ex.as_dict())
        got = store.violation(vid)
        self.assertEqual(got["vlm_status"], "done")
        self.assertEqual(got["vlm"]["assessment"], "contradicted")
        self.assertEqual(got["vlm"]["summary"], "Traffic was stopped too.")

    def test_old_database_gains_the_columns(self):
        from surveillance.store import Store
        store = Store(self.path)
        vid = self.seed(store)
        store.db.execute("ALTER TABLE violations DROP COLUMN vlm")
        store.db.execute("ALTER TABLE violations DROP COLUMN vlm_status")
        store.db.commit()
        store.db.close()

        reopened = Store(self.path)                      # the migration runs here
        self.assertEqual(reopened.violation(vid)["vlm_status"], "")
        reopened.set_vlm(vid, "done", vlm.Explanation(summary="ok").as_dict())
        self.assertEqual(reopened.violation(vid)["vlm"]["summary"], "ok")


class BackendTests(unittest.TestCase):
    def test_off_means_off(self):
        self.assertIsNone(vlm.resolve_backend({"vlm_backend": "off"}))
        self.assertIsNone(vlm.resolve_backend({"vlm": False}))

    def test_local_is_chosen_without_probing_when_asked_for(self):
        b = vlm.resolve_backend({"vlm_backend": "local", "vlm_local_url": "http://127.0.0.1:9"})
        self.assertEqual(b.name, "local")

    def test_status_shape(self):
        st = vlm.backend_status({"vlm_backend": "off"})
        self.assertEqual(set(st), {"enabled", "backend", "model", "local_up", "cloud_key", "local_url"})


# ---------------------------------------------------------------- WP16: label hiding, closed pre-question, V2, speed talk

from types import SimpleNamespace  # noqa: E402

from violations import evidence  # noqa: E402


def drawn_scene(folder, frame_wh=(1920, 1080), bbox=(700, 400, 900, 560), code="V3", cls="Motorbike", tid=20):
    """A snapshot written by the real violations.evidence.save_images on a textured frame. -> (path, violation)."""
    w, h = frame_wh
    rng = np.random.default_rng(1)
    frame = cv2.GaussianBlur(rng.integers(60, 200, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    k = h / 1080
    bbox = tuple(round(v * k) for v in bbox)
    ev = SimpleNamespace(code=code, title="No-parking stop", path=[], obs=SimpleNamespace(bbox=bbox, cls=cls, tid=tid))
    snap, _ = evidence.save_images(ev, frame, folder, "x", "2026-03-04 17:42:11")
    v = case(code=code, title="No-parking stop", cls=cls, track_id=tid, kph=2.0)
    return snap, v, bbox, h


class HideLabelTests(unittest.TestCase):
    def check(self, frame_wh):
        with tempfile.TemporaryDirectory() as d:
            snap, v, bbox, h = drawn_scene(d, frame_wh)
            img = cv2.imread(str(snap))
            org, scale = evidence.label_org(bbox, h)
            rect = evidence.label_rect(evidence.label_text("V3", "No-parking stop", "Motorbike", 20), org, scale)
            k = img.shape[1] / frame_wh[0]
            x1, y1, x2, y2 = (round(c * k) for c in rect)
            out = vlm.hide_label(img, v)
            strip = (slice(y1 + 2, y2 - 2), slice(x1 + 2, x2 - 2))
            self.assertGreater(np.abs(out[strip].astype(int) - img[strip]).mean(), 10)      # the strip changed
            # the dark label fill is gone: the strip now looks like its surroundings, not like (20, 20, 20)
            self.assertGreater(out[strip].mean(), img[strip].mean() + 20)
            outside = np.ones(img.shape[:2], bool)
            outside[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8] = False
            self.assertTrue((out[outside] == img[outside]).all())                            # nothing else moved
            self.assertEqual(out.shape, img.shape)

    def test_label_is_painted_out_and_the_rest_is_untouched(self):
        self.check((1920, 1080))

    def test_downsized_snapshot_of_a_4k_frame(self):
        self.check((3840, 2160))

    def test_geometry_round_trip_with_the_drawing_helper(self):
        """Where the drawing helper says the strip is, the picture has the label's dark fill; where the matcher finds it
        is the same box."""
        with tempfile.TemporaryDirectory() as d:
            snap, v, bbox, h = drawn_scene(d)
            img = cv2.imread(str(snap))
            org, scale = evidence.label_org(bbox, h)
            x1, y1, x2, y2 = evidence.label_rect(evidence.label_text("V3", "No-parking stop", "Motorbike", 20),
                                                 org, scale)
            self.assertLess(img[y1 + 1:y1 + 3, x1 + 1:x2 - 1].mean(), 40)
            fx1, fy1, fx2, fy2 = vlm.find_label(img, v)
            self.assertLessEqual(max(abs(fx1 - x1), abs(fy1 - y1), abs(fx2 - x2), abs(fy2 - y2)), 2)
            exact = vlm.find_label(img, {**v, "bbox": bbox, "frame_size": (1920, 1080)})   # a row that keeps its box
            self.assertEqual(exact, (x1, y1, x2, y2))

    def test_old_label_with_a_speed_appended_is_hidden_too(self):
        with tempfile.TemporaryDirectory() as d:
            w, h = 1920, 1080
            frame = cv2.GaussianBlur(np.random.default_rng(2).integers(60, 200, (h, w, 3), dtype=np.uint8), (0, 0), 3)
            text = evidence.label_text("V3", "No-parking stop", "Motorbike", 20) + " 2 km/h"
            org, scale = evidence.label_org((700, 400, 900, 560), h)
            from violations.zones import draw_label
            draw_label(frame, text, org, (0, 0, 255), scale)
            x1, y1, x2, y2 = evidence.label_rect(text, org, scale)
            v = case(code="V3", title="No-parking stop", cls="Motorbike", track_id=20, kph=2.0)
            out = vlm.hide_label(frame, v)
            self.assertGreater(out[y1 + 2:y2 - 2, x2 - 40:x2 - 5].mean(), frame[y1 + 2:y2 - 2, x2 - 40:x2 - 5].mean() + 20)

    def test_unfound_label_leaves_the_image_alone(self):
        img = np.full((540, 960, 3), 100, np.uint8)
        out = vlm.hide_label(img, case(code="V3", cls="Motorbike", track_id=20))
        self.assertTrue((out == img).all())

    def test_gather_images_hides_the_label_only_in_the_scene_photo(self):
        with tempfile.TemporaryDirectory() as d:
            snap, v, bbox, h = drawn_scene(d)
            shutil_crop = Path(d) / "x_crop.jpg"
            cv2.imwrite(str(Path(d) / "c.jpg"), np.full((100, 100, 3), 70, np.uint8))
            v.update(snapshot=snap.name, crop="c.jpg")
            seen = []
            real = vlm.hide_label
            with mock.patch.object(vlm, "hide_label", side_effect=lambda im, vv: seen.append(1) or real(im, vv)):
                out = vlm.gather_images(v, d, 4, 1920)
            self.assertEqual(len(out), 2)
            self.assertEqual(len(seen), 1)


class ClosedQuestionTests(unittest.TestCase):
    class Fake:
        def __init__(self, reply=None, boom=False):
            self.reply, self.boom, self.calls = reply, boom, []

        def ask(self, system, text, images, **kw):
            self.calls.append((text, images, kw))
            if self.boom:
                raise OSError("down")
            return self.reply

    imgs = [("close-up of the vehicle", b"jpg"), ("the whole scene", b"jpg2")]

    def test_each_answer_sets_the_verdict_fact(self):
        want = {"helmet": "contradicted", "hair": "supported", "cap": "supported", "unclear": "insufficient"}
        for word, verdict in want.items():
            for reply in (word, word.upper() + ".", json.dumps({"head": word})):
                f = self.Fake(reply)
                chk = vlm.visual_check(f, case(code="V6", cls="Motorbike"), self.imgs)
                self.assertEqual((chk["answer"], chk["verdict"]), (word, verdict), reply)
                p = vlm.build_prompt(case(code="V6", cls="Motorbike"), self.imgs, chk)
                self.assertIn(f'The verdict is therefore "{verdict}"', p)

    def test_garbage_is_unclear(self):
        for reply in ("", None, "maybe a turban?", "{}", "{not json", "42"):
            chk = vlm.visual_check(self.Fake(reply), case(code="V6"), self.imgs)
            self.assertEqual(chk["answer"], "unclear", reply)
            self.assertEqual(chk["verdict"], "insufficient")

    def test_one_short_call_with_only_the_closeup(self):
        f = self.Fake("hair")
        vlm.visual_check(f, case(code="V6"), self.imgs)
        self.assertEqual(len(f.calls), 1)
        text, images, kw = f.calls[0]
        self.assertEqual(images, self.imgs[:1])
        self.assertLessEqual(kw["max_tokens"], 16)
        self.assertIn("helmet, hair, cap or unclear", text)

    def test_other_rules_and_failures_ask_nothing_or_nothing_is_invented(self):
        f = self.Fake("helmet")
        self.assertIsNone(vlm.visual_check(f, case(code="V1"), self.imgs))
        self.assertEqual(f.calls, [])
        self.assertIsNone(vlm.visual_check(f, case(code="V6"), [("the whole scene", b"x")]))
        self.assertIsNone(vlm.visual_check(self.Fake(boom=True), case(code="V6"), self.imgs))
        self.assertNotIn("close-up check", vlm.build_prompt(case(code="V6"), self.imgs, None))

    def test_explain_now_asks_the_closed_question_first(self):
        with tempfile.TemporaryDirectory() as d:
            cv2.imwrite(str(Path(d) / "c.jpg"), np.full((100, 100, 3), 70, np.uint8))
            f = self.Fake("helmet")
            replies = ["helmet", json.dumps({"assessment": "contradicted", "confidence": 0.8, "summary": "s"})]
            f.ask = lambda system, text, images, **kw: (f.calls.append(text), replies.pop(0))[1]
            f.model, f.name = "m", "local"
            ex = vlm.Explainer.__new__(vlm.Explainer)
            ex.data_dir, ex.cascade, ex.settings = Path(d), None, {**vlm.DEFAULTS}
            with mock.patch.object(vlm, "resolve_backend", return_value=f):
                res = vlm.Explainer.explain_now(ex, case(code="V6", cls="Motorbike", crop="c.jpg"))
            self.assertEqual(res.assessment, "contradicted")
            self.assertIn("helmet, hair, cap or unclear", f.calls[0])
            self.assertIn("close-up check", f.calls[1])


class OverSpeedTests(unittest.TestCase):
    def test_a_still_cannot_confirm_or_contradict_a_speed(self):
        p = vlm.build_prompt(case(code="V2", cls="Car", kph=88.0), [("close-up", b"x")])
        self.assertIn("can neither confirm nor contradict a measured speed", p)
        self.assertIn("not on a motor vehicle", p)
        self.assertIn("a concern, not a contradiction", p)
        self.assertIn("88.0 km/h", p)                          # the measurement is still given as a fact
        self.assertIn("motor vehicle in the red box", p)       # the question is about the box, not the recorded class

    def test_a_different_type_does_not_overrule_a_supported_v2(self):
        obj = {"visible": "A white bus on a wide road.", "present": True, "assessment": "supported", "confidence": 0.8,
               "summary": "s"}
        ex = vlm.to_explanation(obj, "m", "b", 1, 1, vlm.guard_subject(case(code="V2", cls="Car")), "V2")
        self.assertEqual(ex.assessment, "supported")
        ex = vlm.to_explanation(obj, "m", "b", 1, 1, vlm.guard_subject(case(code="V4", cls="Car")), "V4")
        self.assertEqual(ex.assessment, "insufficient")        # other rules keep the class guard


class SpeedTalkTests(unittest.TestCase):
    obj = {"event": "The motorbike crosses the line. It travels at about 40 km/h. It is slow here.",
           "cause": "Avoiding a pothole. The speed looks low.", "summary": "Crossed a line at a low speed.",
           "visible": "A road. The vehicle seems stopped.", "other_cameras": "Seen at 30 km/h earlier.",
           "assessment": "insufficient", "confidence": 0.5}

    def run_code(self, code):
        return vlm.to_explanation(dict(self.obj), "m", "b", 1, 1, "", code)

    def test_speed_sentences_removed_for_non_speed_rules(self):
        for code in ("V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9"):
            ex = self.run_code(code)
            self.assertNotRegex(ex.event + ex.cause + ex.summary + ex.visible, r"km/h|speed", code)
            self.assertIn("crosses the line", ex.event)
        self.assertEqual(self.run_code("V5").cause, "Avoiding a pothole.")
        self.assertEqual(self.run_code("V1").other_cameras, "Seen at 30 km/h earlier.")   # journey facts are given data

    def test_stopped_or_slow_claims_removed_for_moving_rules_only(self):
        for code in ("V2", "V4", "V5", "V9"):
            ex = self.run_code(code)
            self.assertNotRegex(ex.visible, "stopped")
            self.assertNotRegex(ex.event, "slow")
        self.assertIn("stopped", self.run_code("V7").visible)           # standing rules may say it
        self.assertIn("stopped", self.run_code("V1").visible)

    def test_speed_rule_keeps_its_speed_talk(self):
        ex = self.run_code("V2")
        self.assertIn("40 km/h", ex.event)
        self.assertIn("speed looks low", ex.cause)
        self.assertEqual(self.run_code("V10").event.count("km/h"), 1)    # advisory kinematic rules keep their numbers

    def test_no_code_no_stripping(self):
        ex = vlm.to_explanation(dict(self.obj), "m", "b", 1, 1)
        self.assertIn("40 km/h", ex.event)

    def test_speed_left_out_of_the_prompt_facts_of_other_rules(self):
        for code in ("V1", "V3", "V4", "V5", "V6", "V7", "V8", "V9"):
            p = vlm.build_prompt(case(code=code, kph=12.0, details={"kph": 12.0, "max_kph": 15, "dwell_s": 30}), [])
            self.assertNotRegex(p.split("MEASUREMENTS")[1].split("Now answer")[0], r"km/h|kph|speed", code)
            self.assertIn("dwell s: 30", p)
        for code in ("V2", "V10", "V11"):
            self.assertIn("12.0 km/h", vlm.build_prompt(case(code=code, kph=12.0), []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
