"""Two-tier vision-language cascade: a 2B model screens every case, an 8B model reasons on the escalated ones.

This is requirement R-7 and the TAU-R1 design of the 499A report (3.1.4, 5.1.2). Three stages, each of which can
end the case:

    1. kinematic pre-screen   surveillance/screening.py. Free (no model). Drops only obviously empty V10/V11.
    2. 2B screen              Qwen3-VL-2B, always on (port 8081). A SHORT prompt, a tiny JSON schema:
                              "does a photograph show the hypothesised event?  yes / no / cannot_tell, and which".
    3. 8B reasoner            Qwen3-VL-8B (port 8082), started ON DEMAND when a case is escalated and stopped after
                              an idle timeout. It gets the full phase-aware prompt of vlm.py and every guard of
                              vlm.to_explanation (a "supported" its own observation does not back up is overruled).
                              The cloud backends (Anthropic / OpenAI-compatible) can be the reasoner instead.

The escalation rule (asymmetric; same costs as screening.py)
    screening.py prices dropping a real anomaly at COST_MISSED = 10 and sending an empty one on at COST_WASTED = 1.
    Here sending on costs one 8B pass (about a minute of all six cores, plus loading the model if it is cold) and a
    reviewer's glance; dropping costs the case, which no later stage sees again. The same 10:1 applies, and the rule is:

        pre-screen says drop                      -> stop (no model was used; nothing is claimed about the photos)
        2B says "yes"                             -> escalate  (confirmation is what the 8B is for)
        2B says "cannot tell"                     -> escalate  (it is a small model; doubt is not a reason to drop)
        2B unreachable / unparsable               -> escalate  (never drop on a failure)
        2B says "no", not confident               -> escalate  (confidence < vlm_no_confidence, default 0.7)
        2B says "no", confident, but the pre-screen score is >= vlm_stop_score (default 0.6) -> escalate
        2B says "no", confident, AND pre-screen score < vlm_stop_score                       -> STOP, screened out

    So a "no" from the 2B can only stop a case whose kinematic evidence is already weak (weakest criterion under 1.5x
    its firing level). A rule-decided case (V1-V9 score 1.0) is never stopped by the 2B: those carry the heavier
    consequences, and the 2B is the less reliable reader. Cost of the rule: more 8B passes than a symmetric "trust the
    screen" would run (it is a deliberate trade, and the screen's saving shows up mainly on weak V10/V11 candidates);
    the stop score is a policy constant, not a measured quantity - there is no labelled set to calibrate the 2B's miss
    rate (X5). Policies: "asymmetric" (above), "balanced" (any confident "no" stops), "always" (the screen only
    annotates; everything that passes the pre-screen is reasoned).

Honesty about who said what
    A case stopped before the reasoner is stored as assessment "insufficient", `decided_by` "prescreen" or "screen",
    with its summary saying so; it is never shown as the 8B's verdict. If the 8B is needed but cannot run (not
    installed, not enough free memory, did not come up, request failed) the case keeps the screen result with
    `reasoner_unavailable` set. No verdict is invented in any of these paths.

Nothing here is on the per-frame path; it runs on the Explainer's single background thread.
"""

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import screening, vlm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "vlm"
GB = 1024 ** 3

TIER_FILES = {
    "screen": ("Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"),
    "reason": ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
}
# Resident size above the weights once the KV cache and vision buffers exist (ctx 8192, --image-min-tokens 1024).
# Measured on this machine (docs/VLM.md): 2B 3.4 GB resident for 1.9 GB of files, 8B 10.5 GB for 6.2 GB of files
# (llama.cpp repacks the 4-bit weights for AVX-512 into anonymous memory on top of the mapped file); the 3B was 1.6 GB.
TIER_OVERHEAD_GB = {"screen": 1.5, "reason": 4.4}


# ---------------------------------------------------------------- memory

def available_bytes():
    """MemAvailable from /proc/meminfo (reclaimable page cache counts), 0 if unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def tier_paths(tier, model_dir=None):
    d = Path(model_dir) if model_dir else MODEL_DIR
    w, m = TIER_FILES[tier]
    return d / w, d / m


def tier_installed(tier, model_dir=None):
    return all(p.exists() for p in tier_paths(tier, model_dir))


def tier_need_bytes(tier, model_dir=None):
    """What a tier will hold resident once serving: weights + projector + measured overhead."""
    size = 0
    for p in tier_paths(tier, model_dir):
        try:
            size += p.stat().st_size
        except OSError:
            pass
    return int(size + TIER_OVERHEAD_GB[tier] * GB)


# ---------------------------------------------------------------- the 2B screen

SCREEN_SYSTEM = ("You screen traffic-camera photographs for a human reviewer. You do not judge guilt. "
                 "Answer with one JSON object and nothing else.")

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "shows_event": {"type": "string", "enum": ["yes", "no", "cannot_tell"]},
        "image": {"type": "integer", "enum": [0, 1, 2]},
        "certainty": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["shows_event", "image", "certainty"],
    "additionalProperties": False,
}

SCREEN_TEMPLATE = """Case {vid}: a {subject}. A rule engine suspects this ({kind}):
{claim}

Photographs:
{photos}

Does any photograph show this event happening? Reply with JSON:
{{"shows_event": "yes" | "no" | "cannot_tell", "image": <number of the most telling photograph, 0 if none>, "certainty": "low" | "medium" | "high"}}
Use "cannot_tell" if the vehicle is too small, blurred, dark or hidden to judge."""

KIND_TEXT = {"direction_or_manoeuvre": "an abnormal manoeuvre or direction",
             "near_collision_or_collision": "a near-collision or collision",
             "abnormal_road_use": "abnormal use of the road",
             "none": "nothing in particular"}


def build_screen_prompt(violation, images, klass):
    code = (violation.get("code") or "").upper()
    claim = vlm.RULE_CLAIM.get(code, ("The system flagged this vehicle.", ""))[0]
    photos = "\n".join(f"{i + 1}. {label}" for i, (label, _) in enumerate(images)) or "none"
    return SCREEN_TEMPLATE.format(vid=violation.get("id", "?"), subject=(violation.get("cls") or "vehicle").lower(),
                                  kind=KIND_TEXT.get(klass, klass), claim=claim, photos=photos)


def parse_screen(text):
    """-> {"verdict": yes|no|cannot_tell, "image": int, "confidence": 0..1}. Anything unreadable is cannot_tell."""
    obj = vlm.parse_reply(text) or {}
    raw = str(obj.get("shows_event", "")).strip().lower().replace(" ", "_")
    verdict = "yes" if raw in ("yes", "true") else "no" if raw in ("no", "false") else "cannot_tell"
    # A 2B model cannot calibrate a number (it answered 0.0 to everything in testing), so it states a level;
    # a numeric "confidence" is still read if a model volunteers one. low/medium/high = 0.3/0.6/0.9.
    levels = {"low": 0.3, "medium": 0.6, "high": 0.9}
    cert = str(obj.get("certainty", "")).strip().lower()
    if cert in levels:
        conf = levels[cert]
    else:
        try:
            c = float(obj.get("confidence", 0))
            conf = round(min(1.0, max(0.0, c / 100 if c > 1 else c)), 2)
        except (TypeError, ValueError):
            conf = 0.0
    try:
        img = max(0, int(obj.get("image", 0)))
    except (TypeError, ValueError):
        img = 0
    return {"verdict": verdict, "image": img, "confidence": conf, "parsed": bool(obj)}


# ---------------------------------------------------------------- the escalation rule

POLICIES = ("asymmetric", "balanced", "always")


def decide_escalation(pre, screen, settings):
    """(escalate, why). `pre` is screening.screen's dict; `screen` is parse_screen's dict, or None when the 2B could
    not answer. See the module notes for the rule and its costs."""
    policy = settings.get("vlm_escalation", "asymmetric")
    stop_score = float(settings.get("vlm_stop_score", 0.6))
    no_conf = float(settings.get("vlm_no_confidence", 0.7))
    if screen is None:
        return True, "the screening model gave no answer; a failure never drops a case"
    v, c = screen["verdict"], screen["confidence"]
    if policy == "always":
        return True, f"policy 'always': the screen said {v} but only annotates"
    if v == "yes":
        return True, "the screen says a photograph shows the event; the reasoner checks it"
    if v == "cannot_tell":
        return True, "the screen could not tell; doubt is not a reason to drop a case"
    # v == "no"
    if c < no_conf:
        return True, f"the screen said no with confidence {c:.2f} < {no_conf:.2f}: not confident enough to drop"
    if policy == "balanced":
        return False, f"the screen said no with confidence {c:.2f}; policy 'balanced' trusts a confident no"
    if pre["score"] >= stop_score:
        return True, (f"the screen said no (confidence {c:.2f}) but the kinematic score {pre['score']:.2f} >= "
                      f"{stop_score:.2f}: strong motion evidence or a rule-decided case is never dropped by the 2B")
    return False, (f"the screen said no (confidence {c:.2f}) and the kinematic score {pre['score']:.2f} < "
                   f"{stop_score:.2f}: weak on both counts")


# ---------------------------------------------------------------- the on-demand 8B

def http_health(url, timeout=1.5):
    """llama-server answers /health 200 only once the model is loaded (503 while loading)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class ProcessLauncher:
    """Starts tools/vlm_server.sh --tier <tier> on the port of `url` as its own process group."""

    def __init__(self, tier="reason", log_path=None):
        self.tier = tier
        self.log_path = log_path or str(Path(tempfile.gettempdir()) / f"vlm_{tier}.log")

    def __call__(self, url):
        port = urlparse(url).port or 8082
        log = open(self.log_path, "ab")
        env = {**os.environ, "VLM_PORT": str(port)}
        return subprocess.Popen([str(ROOT / "tools" / "vlm_server.sh"), "--tier", self.tier],
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


class OnDemandServer:
    """Owns a llama-server for the reason tier: starts it when a case needs it and RAM allows, one case at a time,
    stops it after `idle_sec` without a case. A server somebody else already started on that URL is used but
    never stopped.

    Everything environmental is injectable (launcher, health, free memory, clock, sleep) so the lifecycle is tested
    without a model. `reap()` is what the background thread calls; tests call it directly."""

    def __init__(self, url, launcher, idle_sec=300, start_sec=240, need_bytes=lambda: 0,
                 free_bytes=available_bytes, health=http_health, clock=time.monotonic, sleep=time.sleep,
                 poll_sec=1.0, reaper=True):
        self.url, self.launcher = url, launcher
        self.idle_sec, self.start_sec = idle_sec, start_sec
        self.need_bytes, self.free_bytes, self.health = need_bytes, free_bytes, health
        self.clock, self.sleep, self.poll_sec = clock, sleep, poll_sec
        self.proc = None
        self.users = 0
        self.last_used = clock()
        self.op = threading.RLock()
        self.last_error = ""
        self._reaper = None
        if reaper:
            self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="vlm-reaper")
            self._reaper.start()

    # -- state
    def owned(self):
        return self.proc is not None and self.proc.poll() is None

    def state(self):
        if self.health(self.url):
            return "up" if self.owned() else "external"
        return "starting" if self.owned() else "off"

    def status(self):
        idle_in = None
        if self.owned() and self.users == 0:
            idle_in = max(0.0, round(self.idle_sec - (self.clock() - self.last_used), 1))
        return {"state": self.state(), "idle_stop_in": idle_in, "error": self.last_error}

    # -- use
    def acquire(self):
        """-> (ok, info). On ok the caller holds the server until release(). info: started, load_seconds, reason."""
        with self.op:
            info = {"started": False, "load_seconds": 0.0, "reason": ""}
            if self.health(self.url):
                self.users += 1
                return True, info
            if not self.owned():
                free, need = self.free_bytes(), self.need_bytes()
                if need and free < need:
                    info["reason"] = (f"not enough free memory to load the reasoning model: needs about "
                                      f"{need / GB:.1f} GB, {free / GB:.1f} GB available")
                    self.last_error = info["reason"]
                    return False, info
                try:
                    self.proc = self.launcher(self.url)
                except Exception as e:
                    info["reason"] = f"could not start the reasoning model: {e}"
                    self.last_error = info["reason"]
                    return False, info
                info["started"] = True
            t0 = self.clock()
            while True:
                if self.health(self.url):
                    info["load_seconds"] = round(self.clock() - t0, 1)
                    self.users += 1
                    self.last_error = ""
                    return True, info
                if self.proc is not None and self.proc.poll() is not None:
                    info["reason"] = f"the reasoning model exited while loading (code {self.proc.poll()})"
                    self.proc = None
                    break
                if self.clock() - t0 > self.start_sec:
                    info["reason"] = f"the reasoning model was not serving after {self.start_sec:.0f} s"
                    self._kill()
                    break
                self.sleep(self.poll_sec)
            self.last_error = info["reason"]
            return False, info

    def release(self):
        with self.op:
            self.users = max(0, self.users - 1)
            self.last_used = self.clock()

    # -- stopping
    def _kill(self):
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            p.wait(timeout=20)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def stop(self):
        with self.op:
            self._kill()

    def reap(self):
        """Stop the server we started if nobody holds it and it has been idle long enough. -> True if stopped."""
        if not self.op.acquire(blocking=False):
            return False                      # a start or a case is in progress
        try:
            if self.owned() and self.users == 0 and self.clock() - self.last_used >= self.idle_sec:
                self._kill()
                return True
            return False
        finally:
            self.op.release()

    def _reap_loop(self):
        while True:
            time.sleep(5)
            try:
                self.reap()
            except Exception:
                pass


# ---------------------------------------------------------------- the cascade

class ReasonerUnavailable(Exception):
    """The reasoning tier could not answer; the message says why."""


class Cascade:
    """pre-screen -> 2B screen -> (on escalation) 8B reasoner. `run` returns a vlm.Explanation.

    `screen`, `reason` (backends with .ask / .model / .name) and `server` (an OnDemandServer) can be injected; by
    default they are built from the settings, so tests need no network."""

    def __init__(self, data_dir, settings=None, screen=None, reason=None, server=None, model_dir=None):
        self.data_dir = Path(data_dir)
        self.settings = {**vlm.DEFAULTS, **(settings or {})}
        self.model_dir = model_dir
        self._screen, self._reason_b, self._server = screen, reason, server
        self._injected = bool(screen or reason or server)
        # Progress reporting: `on_stage(info)` is called on the explaining thread at each step of one case
        # (see _stage). Durations of finished steps are kept so the estimate shown to the user is the average of
        # what this machine actually did, not a guess.
        self.on_stage = None
        self._cb = None
        self._hist = {}

    # Typical seconds per step on the CPU-only dev machine, used until a step has been timed once.
    DEFAULT_ETA = {"screen": 15, "visual_check": 8, "load_reasoner": 90, "reason": 150}

    def eta(self, step):
        h = self._hist.get(step) or []
        return round(sum(h) / len(h)) if h else self.DEFAULT_ETA[step]

    def _timed(self, step, seconds):
        self._hist.setdefault(step, []).append(float(seconds))
        del self._hist[step][:-5]

    def _stage(self, cb, violation, stage, message, step=None, **extra):
        """Tell the dashboard where this case is. Never raises: progress must not break an explanation."""
        cb = cb or self.on_stage
        if cb is None:
            return
        info = {"violation_id": violation.get("id"), "stage": stage, "message": message, "at": time.time(), **extra}
        if step:
            info["eta_s"] = self.eta(step)
            info["eta_basis"] = "average of recent runs" if self._hist.get(step) else "typical, not yet measured here"
        try:
            cb(info)
        except Exception:
            pass

    # -- configuration -----------------------------------------------------------------------------------------
    def server(self, s):
        if self._server is None:
            self._server = OnDemandServer(s["vlm_reason_url"], ProcessLauncher("reason"),
                                          idle_sec=s["vlm_reason_idle_sec"], start_sec=s["vlm_reason_start_sec"],
                                          need_bytes=lambda: tier_need_bytes("reason", self.model_dir))
        self._server.idle_sec, self._server.start_sec = s["vlm_reason_idle_sec"], s["vlm_reason_start_sec"]
        return self._server

    def screen_backend(self, s):
        return self._screen or vlm.OpenAIChat(s["vlm_screen_url"], s["vlm_screen_model"], name="local",
                                              timeout=s["vlm_screen_timeout_sec"], max_tokens=48)

    def cloud_backend(self, s):
        return vlm.resolve_backend({**s, "vlm_backend": "cloud"})

    def reasoner_kind(self, s):
        """"local" (8B, on demand), "cloud", or "" when no reasoner can exist. A request can pin the choice through
        vlm_backend (local / cloud); otherwise vlm_reason_backend decides, auto = local 8B if installed, else cloud."""
        want = s.get("vlm_backend") if s.get("vlm_backend") in ("local", "cloud") else s["vlm_reason_backend"]
        if self._reason_b is not None:
            return "injected"
        local = tier_installed("reason", self.model_dir) or (self._server is not None)
        if want == "local":
            return "local" if local else ""
        if want == "cloud":
            return "cloud" if self.cloud_backend(s) else ""
        if local:
            return "local"
        return "cloud" if self.cloud_backend(s) else ""

    def active(self, settings=None):
        """Use the cascade for this request? mode single = never, cascade = always, auto = when a screen server is up
        and a reasoner can exist (otherwise the single-model path keeps working unchanged)."""
        s = {**self.settings, **(settings or {})}
        mode = s.get("vlm_mode", "auto")
        if mode == "single" or s.get("vlm_backend") == "off" or not s.get("vlm", True):
            return False
        if mode == "cascade":
            return True
        if self._injected:
            return True
        return bool(self.reasoner_kind(s)) and self.screen_backend(s).available()

    def usable(self, settings=None):
        """Could a case be answered at all (the 503 check)?"""
        s = {**self.settings, **(settings or {})}
        return self.active(s) and (self.screen_backend(s).available() or bool(self.reasoner_kind(s)))

    def status(self, settings=None):
        s = {**self.settings, **(settings or {})}
        scr = self.screen_backend(s)
        server = self._server
        return {
            "mode": s["vlm_mode"], "active": self.active(s),
            "screen": {"model": s["vlm_screen_model"], "url": s["vlm_screen_url"], "up": scr.available(),
                       "installed": tier_installed("screen", self.model_dir)},
            "reason": {"kind": self.reasoner_kind(s), "model": s["vlm_reason_model"], "url": s["vlm_reason_url"],
                       "installed": tier_installed("reason", self.model_dir),
                       "server": server.status() if server else {"state": "off", "idle_stop_in": None, "error": ""},
                       "need_gb": round(tier_need_bytes("reason", self.model_dir) / GB, 1),
                       "free_gb": round(available_bytes() / GB, 1),
                       "idle_sec": s["vlm_reason_idle_sec"]},
            "escalation": s["vlm_escalation"], "stop_score": s["vlm_stop_score"],
            "no_confidence": s["vlm_no_confidence"],
        }

    # -- one case ---------------------------------------------------------------------------------------------
    def run(self, violation, settings=None, on_stage=None):
        s = {**self.settings, **(settings or {})}
        self._cb = on_stage
        t_all = time.perf_counter()
        stages = []
        pre = screening.screen(violation)
        stages.append({"stage": "prescreen", "class": pre["class"], "score": pre["score"],
                       "send": pre["send_to_vlm"], "reason": pre["reason"]})
        self._stage(on_stage, violation, "prescreen_done",
                    "Pre-screen done: " + ("sent on to the screening model" if pre["send_to_vlm"] else "obviously empty, stopping"),
                    send=pre["send_to_vlm"])
        images = vlm.gather_images(violation, self.data_dir, s["vlm_max_images"], s["vlm_max_side"])
        if not images:
            return vlm.Explanation(ok=False, error="No evidence photographs were saved for this case.", stages=stages)

        if not pre["send_to_vlm"]:
            return self._stopped(stages, "prescreen", False, pre["reason"], len(images), t_all,
                                 "Not examined by any model: the kinematic pre-screen found the candidate obviously "
                                 "empty (" + pre["reason"] + ").", "", "")

        self._stage(on_stage, violation, "screening", "Screening on the small model", step="screen")
        screen_res, st = self._run_screen(violation, images, pre, s)
        stages.append(st)
        if not st["error"]:
            self._timed("screen", st["seconds"])
        escalate, why = decide_escalation(pre, screen_res, s)
        if not escalate:
            return self._stopped(stages, "screen", False, why, len(images), t_all,
                                 f"Screened out by the {st['model']} screening model, which saw no sign of the "
                                 f"event; the reasoning model was not asked. This is a screening decision, not a "
                                 f"reasoned verdict.", st["model"], st["backend"])

        self._stage(on_stage, violation, "escalated", "Escalated to the large reasoning model: " + why, reason=why)
        check = None
        if (violation.get("code") or "").upper() in vlm.VISUAL_CHECKS:
            self._stage(on_stage, violation, "visual_check", "Asking the small model one closed question about the "
                        "close-up", step="visual_check")
            check = vlm.visual_check(self.screen_backend(s), violation, images)
            stages.append({"stage": "visual_check", "answer": check["answer"] if check else "unavailable",
                           "seconds": check["seconds"] if check else 0.0})
            if check:
                self._timed("visual_check", check["seconds"])
        return self._reason(violation, images, stages, st, why, s, t_all, check)

    def _run_screen(self, violation, images, pre, s):
        backend = self.screen_backend(s)
        used = images[:max(1, int(s["vlm_screen_images"]))]
        st = {"stage": "screen", "model": backend.model, "backend": backend.name, "verdict": "unavailable",
              "image": 0, "confidence": 0.0, "seconds": 0.0, "error": ""}
        t0 = time.perf_counter()
        try:
            text = backend.ask(SCREEN_SYSTEM, build_screen_prompt(violation, used, pre["class"]), used,
                               max_tokens=48, schema=SCREEN_SCHEMA)
            st["seconds"] = round(time.perf_counter() - t0, 1)
            res = parse_screen(text)
            res["image"] = res["image"] if res["image"] <= len(used) else 0
            st.update(verdict=res["verdict"], image=res["image"], confidence=res["confidence"])
            if res["image"]:
                st["image_label"] = used[res["image"] - 1][0]
            return res, st
        except urllib.error.HTTPError as e:
            st["error"] = f"HTTP {e.code}"
        except Exception as e:
            st["error"] = f"unreachable: {e}"
        st["seconds"] = round(time.perf_counter() - t0, 1)
        return None, st

    def _reason(self, violation, images, stages, screen_stage, why, s, t_all, check=None):
        rs = {"stage": "reason", "model": "", "backend": "", "verdict": "", "seconds": 0.0, "load_seconds": 0.0,
              "started_server": False, "error": ""}
        stages.append(rs)
        kind = self.reasoner_kind(s)
        server = None
        backend = None
        if kind == "injected":
            backend = self._reason_b
        elif kind == "cloud":
            backend = self.cloud_backend(s)
        elif kind == "local":
            server = self.server(s)
            self._stage(self._cb, violation, "loading_reasoner",
                        "Starting the large model (loads into memory first)", step="load_reasoner")
            ok, info = server.acquire()
            if ok and info["started"]:
                self._timed("load_reasoner", info["load_seconds"])
            rs["started_server"], rs["load_seconds"] = info["started"], info["load_seconds"]
            if not ok:
                rs["error"] = info["reason"]
                server = None
            else:
                backend = vlm.OpenAIChat(s["vlm_reason_url"], s["vlm_reason_model"], name="local",
                                         timeout=max(s["vlm_timeout_sec"], s["vlm_reason_timeout_sec"]))
        else:
            rs["error"] = ("no reasoning model: the 8B is not installed (tools/setup_vlm.sh) and no cloud key is "
                           "set (ANTHROPIC_API_KEY)")
        if backend is None:
            return self._unavailable(stages, screen_stage, why, rs, len(images), t_all)
        rs["model"], rs["backend"] = backend.model, backend.name
        self._stage(self._cb, violation, "reasoning", f"Reasoning on {backend.model}", step="reason",
                    model=backend.model)
        t0 = time.perf_counter()
        try:
            reply = backend.ask(vlm.SYSTEM, vlm.build_prompt(violation, images, check), images)
        except urllib.error.HTTPError as e:
            rs["error"] = f"{backend.name} backend returned HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
        except Exception as e:
            rs["error"] = f"{backend.name} backend unreachable: {e}"
        finally:
            if server is not None:
                server.release()
        rs["seconds"] = round(time.perf_counter() - t0, 1)
        if rs["error"]:
            return self._unavailable(stages, screen_stage, why, rs, len(images), t_all)
        self._timed("reason", rs["seconds"])
        ex = vlm.to_explanation(vlm.parse_reply(reply), backend.model, backend.name, rs["seconds"], len(images),
                                vlm.guard_subject(violation), (violation.get("code") or ""))
        if not ex.ok:                                   # unusable reply: the screen result stays, nothing invented
            rs["error"] = ex.error
            return self._unavailable(stages, screen_stage, why, rs, len(images), t_all,
                                     lead="The reasoning model answered, but not in a usable form.")
        rs["verdict"] = ex.assessment
        ex.decided_by, ex.escalated, ex.escalation_reason = "reasoner", True, why
        ex.stages = stages
        ex.seconds = round(time.perf_counter() - t_all, 1)
        return ex

    def ask_reasoner(self, system, user_text, images, settings=None, schema=None, max_tokens=None):
        """One free-form question to the reasoning tier (the 8B, started on demand, or the cloud reasoner), with
        the caller's own prompt. Returns (reply text, info). Raises ReasonerUnavailable when it cannot run (not
        installed, not enough RAM, did not come up, timed out, unreachable) - the caller must then say so and
        never invent an answer. Used by surveillance/disputes.py; the Explain path above does not use it."""
        s = {**self.settings, **(settings or {})}
        kind = self.reasoner_kind(s)
        server = None
        if kind == "injected":
            backend = self._reason_b
        elif kind == "cloud":
            backend = self.cloud_backend(s)
        elif kind == "local":
            server = self.server(s)
            ok, info = server.acquire()
            if not ok:
                raise ReasonerUnavailable(info["reason"])
            backend = vlm.OpenAIChat(s["vlm_reason_url"], s["vlm_reason_model"], name="local",
                                     timeout=max(s["vlm_timeout_sec"], s["vlm_reason_timeout_sec"]))
        else:
            raise ReasonerUnavailable("no reasoning model: the 8B is not installed (tools/setup_vlm.sh) and no "
                                      "cloud key is set (ANTHROPIC_API_KEY)")
        t0 = time.perf_counter()
        try:
            reply = backend.ask(system, user_text, images, max_tokens=max_tokens, schema=schema)
        except urllib.error.HTTPError as e:
            raise ReasonerUnavailable(f"{backend.name} backend returned HTTP {e.code}")
        except Exception as e:
            raise ReasonerUnavailable(f"{backend.name} backend failed or timed out: {e}")
        finally:
            if server is not None:
                server.release()
        return reply, {"model": backend.model, "backend": backend.name,
                       "seconds": round(time.perf_counter() - t0, 1)}

    def _unavailable(self, stages, screen_stage, why, rs, n_images, t_all, lead=""):
        scr = screen_stage
        said = {"yes": "found the event shown", "cannot_tell": "could not tell", "no": "saw no event",
                "unavailable": "did not answer"}.get(scr["verdict"], scr["verdict"])
        summary = ((lead + " ") if lead else "") + (
            f"Reasoner unavailable ({rs['error']}). The {scr['model']} screening model {said}"
            f"{' (confidence %.0f%%)' % (100 * scr['confidence']) if scr['verdict'] != 'unavailable' else ''}; "
            f"no reasoned verdict exists for this case. Ask again when the reasoning model can run.")
        return vlm.Explanation(ok=True, assessment="insufficient", confidence=0.0, summary=summary,
                               model=scr["model"], backend=scr["backend"], seconds=round(time.perf_counter() - t_all, 1),
                               images=n_images, decided_by="screen", escalated=True, escalation_reason=why,
                               reasoner_unavailable=True, stages=stages)

    def _stopped(self, stages, by, escalated, why, n_images, t_all, summary, model, backend):
        return vlm.Explanation(ok=True, assessment="insufficient", confidence=0.0, summary=summary, model=model,
                               backend=backend, seconds=round(time.perf_counter() - t_all, 1), images=n_images,
                               decided_by=by, escalated=escalated, escalation_reason=why, stages=stages)
