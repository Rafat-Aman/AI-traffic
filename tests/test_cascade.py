"""The two-tier VLM cascade with fake backends: no network, no model, no llama-server.

What is pinned here: the asymmetric escalation rule, the three ways a case ends (stopped before the reasoner,
reasoned, reasoner unavailable), that every guard of vlm.to_explanation still applies to the reasoner's answer, the
on-demand process manager's lifecycle with a fake launcher, and the choice of evidence images.
"""

import json
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import vlm, vlm_cascade as vc  # noqa: E402


class FakeBackend:
    def __init__(self, reply, model="fake", name="local", raises=None):
        self.reply, self.model, self.name, self.raises = reply, model, name, raises
        self.calls = []

    def available(self):
        return True

    def ask(self, system, user_text, images, **kw):
        self.calls.append({"system": system, "text": user_text, "n_images": len(images), **kw})
        if self.raises:
            raise self.raises
        return self.reply


def screen_reply(verdict, conf=0.9, image=1):
    return json.dumps({"shows_event": verdict, "image": image, "confidence": conf})


def reason_reply(**over):
    o = {"visible": "a red bus on a wide road", "present": True, "pre_event": "traffic flowing", "event": "bus halted",
         "aftermath": "traffic goes round", "cause": "passenger loading", "assessment": "supported",
         "confidence": 0.8, "summary": "The bus stood in the lane."}
    o.update(over)
    return json.dumps(o)


class FakeProc:
    def __init__(self):
        self.code, self.terminated = None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated, self.code = True, -15

    def kill(self):
        self.code = -9

    def wait(self, timeout=None):
        return self.code


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Env:
    """A temp data dir with evidence images, and cases built on it."""

    def setUpEnv(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for n in ("snap.jpg", "crop.jpg", "contact.jpg", "stopped.jpg"):
            cv2.imwrite(str(self.dir / n), np.full((120, 160, 3), 90, np.uint8))

    def case(self, code="V1", details=None, **over):
        v = {"id": 5, "code": code, "title": "t", "cls": "Bus", "class_conf": 0.9, "kph": 0.0, "zone": "z",
             "areas": "", "wall_time": "", "details": details or {}, "snapshot": "snap.jpg", "crop": "crop.jpg",
             "evidence": []}
        v.update(over)
        return v

    def weak_v10(self):
        """A V10 whose weakest criterion sits at ~1.0x its firing level: pre-screen score ~0.5."""
        return self.case("V10", {"lateral_m": 2.0, "peak_lateral_mps": 2.2, "lateral_accel_mps2": 3.5,
                                 "heading_change_deg": 30.0, "fit_rms_m": 0.1, "class_group": "standard"}, cls="Car")


class EscalationRule(unittest.TestCase):
    PRE_RULE = {"score": 1.0, "send_to_vlm": True, "class": "abnormal_road_use"}
    PRE_WEAK = {"score": 0.5, "send_to_vlm": True, "class": "direction_or_manoeuvre"}
    S = {"vlm_escalation": "asymmetric", "vlm_stop_score": 0.6, "vlm_no_confidence": 0.7}

    def sc(self, verdict, conf=0.9):
        return {"verdict": verdict, "confidence": conf, "image": 1}

    def test_yes_and_cannot_tell_escalate(self):
        for v in ("yes", "cannot_tell"):
            self.assertTrue(vc.decide_escalation(self.PRE_WEAK, self.sc(v), self.S)[0], v)

    def test_screen_failure_escalates(self):
        self.assertTrue(vc.decide_escalation(self.PRE_WEAK, None, self.S)[0])

    def test_only_a_confident_no_with_a_weak_prescreen_stops(self):
        self.assertFalse(vc.decide_escalation(self.PRE_WEAK, self.sc("no", 0.9), self.S)[0])

    def test_asymmetry_no_does_not_stop_a_strong_or_rule_decided_case(self):
        self.assertTrue(vc.decide_escalation(self.PRE_RULE, self.sc("no", 0.99), self.S)[0])
        self.assertTrue(vc.decide_escalation({**self.PRE_WEAK, "score": 0.67}, self.sc("no", 0.99), self.S)[0])

    def test_asymmetry_unconfident_no_escalates(self):
        self.assertTrue(vc.decide_escalation(self.PRE_WEAK, self.sc("no", 0.4), self.S)[0])

    def test_boundaries(self):
        self.assertFalse(vc.decide_escalation({**self.PRE_WEAK, "score": 0.599}, self.sc("no", 0.7), self.S)[0])
        self.assertTrue(vc.decide_escalation({**self.PRE_WEAK, "score": 0.6}, self.sc("no", 0.7), self.S)[0])

    def test_policies(self):
        always = {**self.S, "vlm_escalation": "always"}
        self.assertTrue(vc.decide_escalation(self.PRE_WEAK, self.sc("no", 1.0), always)[0])
        bal = {**self.S, "vlm_escalation": "balanced"}
        self.assertFalse(vc.decide_escalation(self.PRE_RULE, self.sc("no", 0.9), bal)[0])
        self.assertTrue(vc.decide_escalation(self.PRE_RULE, self.sc("no", 0.5), bal)[0])

    def test_a_reason_is_always_given(self):
        for v in ("yes", "no", "cannot_tell"):
            self.assertTrue(vc.decide_escalation(self.PRE_WEAK, self.sc(v), self.S)[1])


class ScreenParsing(unittest.TestCase):
    def test_garbage_is_cannot_tell(self):
        for t in ("", "not json", "[]", json.dumps({"shows_event": "maybe"})):
            self.assertEqual(vc.parse_screen(t)["verdict"], "cannot_tell", t)

    def test_confidence_percentage_and_clamp(self):
        self.assertEqual(vc.parse_screen(screen_reply("no", 85))["confidence"], 0.85)
        self.assertEqual(vc.parse_screen(screen_reply("no", -3))["confidence"], 0.0)

    def test_certainty_levels(self):
        for lvl, v in (("low", 0.3), ("medium", 0.6), ("high", 0.9)):
            t = json.dumps({"shows_event": "no", "image": 0, "certainty": lvl})
            self.assertEqual(vc.parse_screen(t)["confidence"], v)

    def test_prompt_is_short_and_carries_the_claim(self):
        p = vc.build_screen_prompt({"id": 1, "code": "V11", "cls": "Car"}, [("a", b"x"), ("b", b"y")],
                                   "near_collision_or_collision")
        self.assertIn("came dangerously close", p)
        self.assertLess(len(p), 1200)
        self.assertLess(len(p), len(vlm.build_prompt({"id": 1, "code": "V11", "cls": "Car"}, [])) / 1.5)


class CascadeFlow(Env, unittest.TestCase):
    def setUp(self):
        self.setUpEnv()
        self.addCleanup(self.tmp.cleanup)

    def cascade(self, screen, reason=None, server=None, **settings):
        return vc.Cascade(self.dir, settings, screen=screen, reason=reason, server=server)

    def test_prescreen_drop_calls_no_model(self):
        v = self.weak_v10()
        v["details"]["lateral_m"] = 0.1                    # far below its firing level: obviously empty
        screen, reason = FakeBackend(screen_reply("yes")), FakeBackend(reason_reply())
        ex = self.cascade(screen, reason).run(v)
        self.assertEqual((ex.decided_by, ex.escalated, ex.assessment), ("prescreen", False, "insufficient"))
        self.assertEqual((len(screen.calls), len(reason.calls)), (0, 0))
        self.assertIn("Not examined by any model", ex.summary)
        self.assertEqual([g["stage"] for g in ex.stages], ["prescreen"])

    def test_screened_out_path(self):
        screen, reason = FakeBackend(screen_reply("no", 0.95), model="qwen3-vl-2b"), FakeBackend(reason_reply())
        ex = self.cascade(screen, reason).run(self.weak_v10())
        self.assertEqual((ex.decided_by, ex.escalated, ex.assessment), ("screen", False, "insufficient"))
        self.assertEqual(len(reason.calls), 0)
        self.assertFalse(ex.reasoner_unavailable)
        self.assertIn("not a reasoned verdict", ex.summary)
        self.assertEqual([g["stage"] for g in ex.stages], ["prescreen", "screen"])
        self.assertEqual(ex.stages[1]["verdict"], "no")
        self.assertEqual(ex.stages[1]["model"], "qwen3-vl-2b")
        self.assertEqual(ex.confidence, 0.0)

    def test_screen_uses_a_schema_and_few_tokens(self):
        screen = FakeBackend(screen_reply("no", 0.95))
        self.cascade(screen, FakeBackend(reason_reply())).run(self.weak_v10())
        c = screen.calls[0]
        self.assertEqual(c["schema"], vc.SCREEN_SCHEMA)
        self.assertLessEqual(c["max_tokens"], 64)
        self.assertEqual(c["system"], vc.SCREEN_SYSTEM)

    def test_rule_decided_case_is_not_stopped_by_a_no(self):
        screen, reason = FakeBackend(screen_reply("no", 0.99)), FakeBackend(reason_reply())
        ex = self.cascade(screen, reason).run(self.case("V1"))
        self.assertEqual(ex.decided_by, "reasoner")
        self.assertEqual(len(reason.calls), 1)

    def test_escalated_and_reasoned_records_both_stages(self):
        screen = FakeBackend(screen_reply("yes"), model="qwen3-vl-2b")
        reason = FakeBackend(reason_reply(), model="qwen3-vl-8b")
        ex = self.cascade(screen, reason).run(self.case("V1"))
        self.assertTrue(ex.ok)
        self.assertEqual((ex.decided_by, ex.escalated, ex.assessment, ex.model), ("reasoner", True, "supported",
                                                                                  "qwen3-vl-8b"))
        self.assertEqual([g["stage"] for g in ex.stages], ["prescreen", "screen", "reason"])
        self.assertEqual(ex.stages[2]["verdict"], "supported")
        self.assertEqual(ex.stages[1]["model"], "qwen3-vl-2b")
        self.assertTrue(ex.escalation_reason)
        # The reasoner got the full phase-aware prompt, not the short one.
        self.assertEqual(reason.calls[0]["system"], vlm.SYSTEM)
        self.assertIn("STEP 1", reason.calls[0]["system"])
        json.dumps(ex.as_dict())                            # storable

    def test_guards_still_apply_to_the_reasoner(self):
        # present=false and a description that never mentions a bus: "supported" must be overruled.
        for over in ({"present": False}, {"visible": "two men standing on a pavement"}):
            reason = FakeBackend(reason_reply(**over))
            ex = self.cascade(FakeBackend(screen_reply("yes")), reason).run(self.case("V1"))
            self.assertEqual(ex.assessment, "insufficient", over)
            self.assertTrue(ex.downgraded)
            self.assertLessEqual(ex.confidence, 0.3)
            self.assertEqual(ex.decided_by, "reasoner")
            self.assertEqual(ex.stages[-1]["verdict"], "insufficient")

    def test_unparsable_reasoner_reply_is_not_a_verdict(self):
        ex = self.cascade(FakeBackend(screen_reply("yes")), FakeBackend("Sure! It is supported.")).run(self.case())
        self.assertTrue(ex.reasoner_unavailable)
        self.assertEqual(ex.assessment, "insufficient")
        self.assertEqual(ex.decided_by, "screen")

    def test_screen_failure_escalates(self):
        screen = FakeBackend("", raises=urllib.error.URLError("refused"))
        ex = self.cascade(screen, FakeBackend(reason_reply())).run(self.weak_v10())
        self.assertEqual(ex.decided_by, "reasoner")
        self.assertEqual(ex.stages[1]["verdict"], "unavailable")
        self.assertIn("refused", ex.stages[1]["error"])

    def test_reasoner_request_failure_keeps_screen_result_unavailable(self):
        reason = FakeBackend("", raises=TimeoutError("slow"))
        ex = self.cascade(FakeBackend(screen_reply("yes")), reason).run(self.case())
        self.assertEqual((ex.decided_by, ex.reasoner_unavailable, ex.assessment), ("screen", True, "insufficient"))
        self.assertIn("Reasoner unavailable", ex.summary)
        self.assertNotEqual(ex.assessment, "supported")
        self.assertEqual(ex.stages[-1]["stage"], "reason")
        self.assertTrue(ex.stages[-1]["error"])

    def test_no_reasoner_configured(self):
        c = vc.Cascade(self.dir, {"vlm_reason_backend": "local"}, screen=FakeBackend(screen_reply("cannot_tell")),
                       model_dir=self.dir / "empty")
        ex = c.run(self.case())
        self.assertTrue(ex.reasoner_unavailable)
        self.assertIn("not installed", ex.stages[-1]["error"])

    def test_no_photos_is_a_failure_not_a_verdict(self):
        ex = self.cascade(FakeBackend(screen_reply("yes")), FakeBackend(reason_reply())).run(
            self.case(snapshot=None, crop=None))
        self.assertFalse(ex.ok)

    def test_screen_only_decision_never_looks_like_the_reasoners(self):
        ex = self.cascade(FakeBackend(screen_reply("no", 0.95)), FakeBackend(reason_reply())).run(self.weak_v10())
        self.assertNotEqual(ex.decided_by, "reasoner")
        self.assertNotIn(ex.assessment, ("supported", "contradicted"))

    def test_explainer_delegates_when_active_and_not_otherwise(self):
        got = []
        c = self.cascade(FakeBackend(screen_reply("yes")), FakeBackend(reason_reply()))
        ex = vlm.Explainer(self.dir, {"vlm_backend": "auto"}, lambda *a: got.append(a), cascade=c)
        self.assertEqual(ex.explain_now(self.case()).decided_by, "reasoner")
        off = vlm.Explainer(self.dir, {"vlm_backend": "off"}, lambda *a: None, cascade=c)
        self.assertFalse(off.explain_now(self.case()).ok)           # off stays off
        single = vlm.Explainer(self.dir, {"vlm_backend": "off", "vlm_mode": "single"}, lambda *a: None, cascade=c)
        self.assertEqual(single.explain_now(self.case()).decided_by, "")


class Fallback(Env, unittest.TestCase):
    def setUp(self):
        self.setUpEnv()
        self.addCleanup(self.tmp.cleanup)

    def test_auto_falls_back_to_single_when_no_8b_and_no_key(self):
        import os
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        c = vc.Cascade(self.dir, {"vlm_mode": "auto"}, model_dir=self.dir / "none")
        self.assertFalse(c.active())

    def test_mode_single_and_cascade(self):
        c = vc.Cascade(self.dir, model_dir=self.dir / "none")
        self.assertFalse(c.active({"vlm_mode": "single"}))
        self.assertTrue(c.active({"vlm_mode": "cascade"}))
        self.assertFalse(c.active({"vlm_mode": "cascade", "vlm_backend": "off"}))


class StageEvents(Env, unittest.TestCase):
    """Progress events a dashboard shows while a case is explained: nothing was emitted before, so an escalated
    case looked frozen for minutes."""

    def setUp(self):
        self.setUpEnv()
        self.addCleanup(self.tmp.cleanup)
        self.events = []

    def run_case(self, screen, reason, v=None, **kw):
        c = vc.Cascade(self.dir, {}, screen=screen, reason=reason)
        ex = c.run(v or self.case("V1"), on_stage=self.events.append, **kw)
        return c, ex

    def names(self):
        return [e["stage"] for e in self.events]

    def test_escalated_case_reports_each_step_in_order(self):
        c, ex = self.run_case(FakeBackend(screen_reply("yes")), FakeBackend(reason_reply(), model="q8"))
        self.assertTrue(ex.ok)
        self.assertEqual(self.names(), ["prescreen_done", "screening", "escalated", "reasoning"])
        reasoning = self.events[-1]
        self.assertEqual((reasoning["model"], reasoning["violation_id"]), ("q8", self.case("V1")["id"]))
        self.assertIn("eta_s", reasoning)
        self.assertEqual(reasoning["eta_basis"], "typical, not yet measured here")

    def test_estimate_becomes_the_measured_average(self):
        c, _ = self.run_case(FakeBackend(screen_reply("yes")), FakeBackend(reason_reply()))
        c._hist["reason"] = [100.0, 200.0]
        self.events.clear()
        c.run(self.case("V1"), on_stage=self.events.append)
        last = [e for e in self.events if e["stage"] == "reasoning"][0]
        self.assertTrue(last["eta_s"] > 0 and last["eta_basis"] == "average of recent runs")

    def test_screened_out_case_never_reaches_the_reasoner(self):
        self.run_case(FakeBackend(screen_reply("no", 0.95)), FakeBackend(reason_reply()), v=self.weak_v10())
        self.assertEqual(self.names(), ["prescreen_done", "screening"])

    def test_prescreen_drop_stops_after_the_first_event(self):
        v = self.weak_v10()
        v["details"]["lateral_m"] = 0.1
        self.run_case(FakeBackend(screen_reply("yes")), FakeBackend(reason_reply()), v=v)
        self.assertEqual(self.names(), ["prescreen_done"])
        self.assertFalse(self.events[0]["send"])

    def test_local_reasoner_reports_loading_before_reasoning(self):
        class Srv:
            def acquire(self):
                return True, {"started": True, "load_seconds": 42.0, "reason": ""}

            def release(self):
                pass

        c = vc.Cascade(self.dir, {"vlm_reason_backend": "local"}, screen=FakeBackend(screen_reply("yes")),
                       reason=None, server=Srv())
        with mock.patch.object(c, "reasoner_kind", return_value="local"), \
                mock.patch.object(vc.vlm, "OpenAIChat", return_value=FakeBackend(reason_reply(), model="q8")):
            c.run(self.case("V1"), on_stage=self.events.append)
        self.assertEqual(self.names(), ["prescreen_done", "screening", "escalated", "loading_reasoner", "reasoning"])
        self.assertEqual(c._hist["load_reasoner"], [42.0])

    def test_a_broken_listener_cannot_break_the_explanation(self):
        def boom(info):
            raise RuntimeError("listener down")

        c = vc.Cascade(self.dir, {}, screen=FakeBackend(screen_reply("yes")), reason=FakeBackend(reason_reply()))
        self.assertTrue(c.run(self.case("V1"), on_stage=boom).ok)

    def test_default_listener_is_used_when_none_is_passed(self):
        c = vc.Cascade(self.dir, {}, screen=FakeBackend(screen_reply("yes")), reason=FakeBackend(reason_reply()))
        c.on_stage = self.events.append
        c.run(self.case("V1"))
        self.assertIn("reasoning", self.names())


class OnDemandLifecycle(unittest.TestCase):
    def make(self, free=10 * 1024 ** 3, need=6 * 1024 ** 3, up_after=3.0, idle=60, start=30):
        self.clock = Clock()
        self.launched = []
        state = {"launch_t": None}

        def launcher(url):
            p = FakeProc()
            self.launched.append(p)
            state["launch_t"] = self.clock.t
            return p

        def health(url):
            return (state["launch_t"] is not None and not self.launched[-1].code
                    and self.clock.t - state["launch_t"] >= up_after)

        self.srv = vc.OnDemandServer("http://127.0.0.1:8082", launcher, idle_sec=idle, start_sec=start,
                                     need_bytes=lambda: need, free_bytes=lambda: free, health=health,
                                     clock=self.clock, sleep=self.clock.sleep, reaper=False)
        return self.srv

    def test_starts_on_demand_serves_and_stops_when_idle(self):
        s = self.make()
        self.assertEqual(s.state(), "off")
        ok, info = s.acquire()
        self.assertTrue(ok)
        self.assertTrue(info["started"])
        self.assertGreaterEqual(info["load_seconds"], 3.0)
        self.assertEqual(s.state(), "up")
        s.release()
        self.clock.t += 59
        self.assertFalse(s.reap())                         # not idle long enough
        self.clock.t += 2
        self.assertTrue(s.reap())
        self.assertTrue(self.launched[0].terminated)
        self.assertEqual(s.state(), "off")

    def test_second_case_reuses_the_running_server(self):
        s = self.make()
        s.acquire(); s.release()
        ok, info = s.acquire()
        self.assertTrue(ok)
        self.assertFalse(info["started"])
        self.assertEqual(len(self.launched), 1)

    def test_not_stopped_while_a_case_is_held(self):
        s = self.make()
        s.acquire()
        self.clock.t += 10_000
        self.assertFalse(s.reap())
        s.release()
        self.assertFalse(s.reap())                         # idle clock restarts at release
        self.clock.t += 61
        self.assertTrue(s.reap())

    def test_refuses_to_start_when_memory_is_short(self):
        s = self.make(free=2 * 1024 ** 3)
        ok, info = s.acquire()
        self.assertFalse(ok)
        self.assertIn("not enough free memory", info["reason"])
        self.assertEqual(self.launched, [])

    def test_start_timeout_kills_the_process(self):
        s = self.make(up_after=1e9, start=10)
        ok, info = s.acquire()
        self.assertFalse(ok)
        self.assertIn("not serving", info["reason"])
        self.assertTrue(self.launched[0].terminated)

    def test_crash_while_loading_is_reported(self):
        s = self.make(up_after=1e9, start=100)
        orig = s.sleep
        s.sleep = lambda x: (setattr(self.launched[-1], "code", 1), orig(x))
        ok, info = s.acquire()
        self.assertFalse(ok)
        self.assertIn("exited", info["reason"])

    def test_launcher_exception_is_a_clean_failure(self):
        s = vc.OnDemandServer("http://x", lambda u: 1 / 0, need_bytes=lambda: 0, free_bytes=lambda: 1,
                              health=lambda u: False, reaper=False)
        ok, info = s.acquire()
        self.assertFalse(ok)
        self.assertIn("could not start", info["reason"])

    def test_external_server_is_used_and_never_stopped(self):
        s = vc.OnDemandServer("http://x", lambda u: self.fail("must not launch"), idle_sec=0, need_bytes=lambda: 0,
                              free_bytes=lambda: 0, health=lambda u: True, reaper=False)
        ok, info = s.acquire()
        self.assertTrue(ok)
        self.assertFalse(info["started"])
        s.release()
        self.assertFalse(s.reap())
        self.assertEqual(s.state(), "external")

    def test_cascade_with_the_manager_unavailable_and_available(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = Env()
        env.dir = Path(tmp.name)
        cv2.imwrite(str(env.dir / "snap.jpg"), np.full((60, 80, 3), 90, np.uint8))
        short = self.make(free=1)
        c = vc.Cascade(env.dir, {"vlm_reason_backend": "local"}, screen=FakeBackend(screen_reply("yes")),
                       server=short)
        ex = c.run(env.case(crop=None))
        self.assertTrue(ex.reasoner_unavailable)
        self.assertIn("not enough free memory", ex.stages[-1]["error"])
        self.assertEqual(short.users, 0)


class ImageChoice(Env, unittest.TestCase):
    def setUp(self):
        self.setUpEnv()
        self.addCleanup(self.tmp.cleanup)

    def labels(self, v, n=2):
        return [lbl for lbl, _ in vlm.gather_images(v, self.dir, n, 160)]

    def test_contact_frame_survives_the_default_of_two_images(self):
        for code in ("V11", "V10"):
            v = self.case(code, evidence=[{"label": "contact", "path": "contact.jpg"}])
            got = self.labels(v, vlm.DEFAULTS["vlm_max_images"])
            self.assertEqual(len(got), 2, code)
            self.assertIn("touched", got[0], code)

    def test_other_codes_keep_their_order_and_extras(self):
        v = self.case("V1", evidence=[{"label": "stopped", "path": "stopped.jpg"}])
        self.assertEqual(self.labels(v, 3), ["close-up of the vehicle", "the whole scene, vehicle outlined in red, "
                                             "its recent path in yellow", "the moment it stopped"])
        v = self.case("V1", evidence=[{"label": "contact", "path": "contact.jpg"}])
        self.assertNotIn("touched", " ".join(self.labels(v, 2)))   # contact is only special for V10/V11

    def test_v11_without_a_contact_frame_is_unchanged(self):
        self.assertEqual(self.labels(self.case("V11"), 2)[0], "close-up of the vehicle")

    def test_missing_file_backfills(self):
        v = self.case("V11", evidence=[{"label": "contact", "path": "gone.jpg"}])
        self.assertEqual(len(self.labels(v, 2)), 2)


if __name__ == "__main__":
    unittest.main()
