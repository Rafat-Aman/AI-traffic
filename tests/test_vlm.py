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


if __name__ == "__main__":
    unittest.main(verbosity=2)
