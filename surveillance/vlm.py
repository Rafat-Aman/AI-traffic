"""Vision-language reasoning about a violation the rule engine has already found.

Why this exists
    The rule engine answers *whether* a rule was broken. It cannot say what the scene was doing, why the
    vehicle behaved that way, or whether the pictures actually back the finding up. An officer reviewing a
    case needs all three, and a notice that only says "V1 at 17:42" is contestable by anyone.

What it is asked to do
    The model never detects anything. Detection, tracking, speed and the rule decision are already done and
    are handed to it as facts; its job is to read the evidence photos, narrate the event and judge the
    finding. (Positioning the model over the perception stack rather than in place of it: the conventional
    tracker is far better at boxes than a 3B VLM is.)

    The answer is phase-aware - what the scene was doing before, the event itself, what followed, and the
    probable cause - because "why did this happen" is a question about the first and the last of those.

Declining is a first-class answer
    `assessment` may be "insufficient" and that is a success, not a failure. A blurred plate-sized vehicle at
    the far end of a zone, or a stop that could equally be a red light, must come back as insufficient rather
    than as an invented cause: the output can end up in front of a magistrate. Each rule below therefore
    carries the specific innocent explanation the model is told to look for, which is where most of the value
    is - the gap analysis of this project names red-light stops as the main false positive of every reviewed
    illegal-stopping detector.

Backends
    cascade - surveillance/vlm_cascade.py: a 2B model screens every case and an 8B reasons on the escalated
             ones; it calls into this module for the prompt, the schema and every guard.
    local  - llama.cpp's server with a quantised Qwen2.5-VL (tools/setup_vlm.sh, tools/vlm_server.sh).
             Runs on the CPU, needs no key and no network.
    cloud  - Anthropic or any OpenAI-compatible endpoint, for better reasoning when a key is configured.
    Both speak the same two shapes here; nothing else in the system knows which one answered.

Nothing in this module is on the per-frame path. Work is queued and answered on a background thread, the
same arrangement number-plate reading already uses.
"""

import base64
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------- what each rule claims

# code -> (what the engine decided, the innocent explanation that would overturn it)
RULE_CLAIM = {
    "V1": ("A bus stood still in a running traffic lane, away from any marked bus stop, long enough to count "
           "as loading passengers illegally.",
           "The whole road is stopped (red light, jam or a blocked junction ahead), so the bus had nowhere to "
           "go; or it is in fact pulled over at a stop, lay-by or kerb and out of the running lane."),
    "V2": ("The vehicle's measured ground speed was over the limit for this road.",
           "The box outlined is not the vehicle that was measured, or it is so far away or so blurred that no "
           "judgement about it is possible from these pictures."),
    "V3": ("The vehicle stood at a kerb marked no-parking, long enough to count as parking rather than a "
           "momentary halt.",
           "It is held in a queue of stopped traffic rather than parked, or it is picking up and setting down "
           "within seconds, or the kerb it sits at is a legal bay."),
    "V4": ("The vehicle travelled against the permitted direction of a one-way zone.",
           "It is passing a stopped obstruction and returning to its own side, turning, reversing into a "
           "space, or being pushed; or its direction cannot be told from these pictures."),
    "V5": ("The vehicle crossed a solid painted lane line.",
           "It was avoiding an obstruction, a pedestrian or a pothole, or the line under it is broken or worn "
           "away rather than solid."),
    "V6": ("Someone on this motorbike - rider or pillion - has no helmet on.",
           "Every head is in fact covered (a helmet dark against a dark background, a hood or cap mistaken "
           "for bare hair), or the heads are too small or too blurred to tell."),
    "V7": ("The vehicle stood still mid-road in a running lane while other traffic moved around it, blocking "
           "the lane.",
           "The traffic around it is also stopped, so this is a jam or a signal rather than one vehicle "
           "blocking a running lane; or it has broken down, which is not an offence of choice."),
    "V8": ("A class of vehicle banned from this road was using it.",
           "The vehicle is of a different class than the label says, or it is crossing rather than using the "
           "road, or it is an emergency or works vehicle."),
    "V9": ("A vehicle that is not a bus drove along a lane reserved for buses, far enough and long enough to "
           "be using the lane rather than crossing it.",
           "It is only crossing the lane to turn or to reach the kerb; or it is in fact a bus, an ambulance or "
           "another authorised vehicle; or the lane is not in force at that time of day, or its markings cannot "
           "be seen in the photographs, so nothing shows that this is a bus lane."),
    "V10": ("The vehicle swerved abruptly while moving at speed: it shifted sideways by about two metres or more "
            "within about two seconds, which the system flags as a possible unsafe or erratic manoeuvre. This is a "
            "candidate for you to judge, not an established offence.",
            "It swerved round something in its way - a pothole, a pedestrian or a stopped vehicle - or merged into "
            "a gap in the next lane, or the road bends or opens into a junction there, so the path is ordinary "
            "driving rather than a dangerous manoeuvre."),
    "V11": ("Two vehicles came dangerously close while closing on each other and one braked hard, or they touched "
            "and both then stood still. This is a candidate for you to judge, not an established accident.",
            "The two vehicles are only passing close to each other in dense, slow traffic, with no sign of an "
            "impact, a skid or a vehicle stopped askew; or the two outlined boxes have swapped or merged and "
            "really mark the same vehicle."),
}

ASSESSMENTS = ("supported", "insufficient", "contradicted")

# What the model might reasonably call each of the detector's classes, used to check its verdict against its
# own description of the photograph (see to_explanation). Deliberately generous: a word that is missing here
# costs a usable verdict, so the lists err toward matching.
SUBJECT_WORDS = {
    "bus": ("bus", "coach", "minibus"),
    "car": ("car", "sedan", "hatchback", "taxi", "saloon", "suv"),
    "mpv": ("mpv", "van", "minivan", "microbus", "people carrier", "suv"),
    "pickup": ("pickup", "pick-up", "ute", "truck"),
    "truck": ("truck", "lorry", "hgv"),
    "cng": ("cng", "auto-rickshaw", "autorickshaw", "auto rickshaw", "auto", "three-wheeler", "threewheeler",
            "tuk", "rickshaw"),
    "easybike": ("easybike", "easy bike", "e-bike", "electric rickshaw", "battery", "tomtom", "rickshaw"),
    "leguna": ("leguna", "laguna", "human hauler", "pickup", "van"),
    "motorbike": ("motorbike", "motorcycle", "motor bike", "bike", "scooter", "moped", "rider", "two-wheeler",
                  "twowheeler", "biker"),
    "bhotbhoti": ("bhotbhoti", "tractor", "engine"),
    "powertiller": ("tiller", "tractor", "power tiller"),
    "rickshaw": ("rickshaw", "cycle rickshaw", "pedal", "trishaw"),
    "bicycle": ("bicycle", "bike", "cyclist", "cycle"),
    "van": ("van", "cart", "trolley", "rickshaw van"),
    "wheelbarrow": ("wheelbarrow", "barrow", "cart", "trolley"),
}


def subject_missing(subject, visible):
    """True when the model's own description of the photograph never mentions the vehicle the case is about.

    Only a confident "no": an unknown class or an empty description returns False, so this can never
    manufacture a problem it has not actually seen."""
    words = SUBJECT_WORDS.get((subject or "").strip().lower())
    if not words or not visible:
        return False
    text = visible.lower()
    return not any(w in text for w in words)

SCHEMA_KEYS = ("pre_event", "event", "aftermath", "cause", "assessment", "confidence", "summary")

SYSTEM = """You are the check on an automated traffic-enforcement system in Dhaka, Bangladesh.

A camera system detected the vehicles, tracked them, measured their speed and applied one traffic rule. It
then produced a finding. That system is wrong often enough that checking it is the entire point of your job:
it mistakes red-light stops for illegal parking, boxes the wrong vehicle, and misreads one class of vehicle
as another. You are the only stage that looks at the photographs and can catch that.

Work in this order, and do not skip the first step.

STEP 1 - Look at the photographs and write down what is actually in them, before you read the finding
again. Name the vehicles you can make out and the road around them. If a vehicle is too small, too blurred
or too dark to identify, say so.

STEP 2 - Compare what you saw with what the system decided. The finding names a particular vehicle doing a
particular thing. Ask: is that vehicle visible in these photographs at all? Is it doing that?

STEP 3 - Give your verdict, and describe the event only to the extent the photographs show it.

The one mistake you must never make is agreeing with a finding the photographs do not show. If the vehicle
the case is about is not clearly visible, or the picture shows something else entirely, the verdict is
"contradicted" (the photographs show something else) or "insufficient" (they cannot settle it). Never
"supported". Do not narrate an event you cannot see merely because the finding describes one - that is
exactly how a wrong ticket is issued.

Also:
- Describe only what is visible. Never state a number plate, time, speed or street name you were not given.
- Dhaka traffic is dense, mixed and not lane-disciplined. Rickshaws, CNG auto-rickshaws, easy-bikes and
  leguna are ordinary road users here, not anomalies.
- "insufficient" is a correct, successful answer, and is always better than a confident guess.
- Answer with one JSON object and nothing else."""

USER_TEMPLATE = """Case {vid}.

THE PHOTOGRAPHS ATTACHED
{photos}

THE FINDING TO CHECK: {code} - {title}
{claim}

The system would be WRONG about this if:
{innocent}

MEASUREMENTS (from the detector and the rule engine, treat these as given)
{facts}{journey}

Now answer. Fill "visible" first, from the photographs alone, before judging anything.

{{
  "visible":   "what you can actually make out in the photographs: the vehicles, the road, the surroundings. Say plainly if something is too small, blurred or dark to identify. Do not mention the finding here.",
  "present":   true or false - is the {subject} that this case is about clearly identifiable in the photographs?,
  "pre_event": "what the scene was doing just before, 1-2 sentences, or \\"not visible\\"",
  "event":     "the behaviour the case is about, only as far as the photographs show it, or \\"not visible\\"",
  "aftermath": "what followed - traffic going round it, a queue forming, the vehicle moving off - or \\"not visible\\"",
  "cause":     "the most probable reason for the behaviour, 1-2 sentences, or \\"cannot be determined\\"",
  "assessment": "supported | contradicted | insufficient",
  "confidence": <a number from 0.0 to 1.0: how sure you are of your verdict>,
  "summary":   "one sentence an officer could paste into a notice",
  "other_cameras": "what the other cameras add, if anything, or \\"nothing\\" - and \\"not applicable\\" when none were given"
}}

Choosing "assessment":
- "supported"     - you can see the {subject} and it is doing what the finding says.
- "contradicted"  - the photographs show something else: a different kind of vehicle, a different situation,
                    or the innocent explanation above.
- "insufficient"  - you cannot tell: the vehicle is not identifiable, or both explanations fit.

If "present" is false then "assessment" must NOT be "supported". "confidence" is 0.0-1.0 in your verdict."""


@dataclass
class Explanation:
    """One reviewed case. `ok` is False when the model could not be reached at all."""
    ok: bool = True
    assessment: str = "insufficient"
    confidence: float = 0.0
    visible: str = ""                # what the model says it can actually see, written before it judged
    other_cameras: str = ""          # what the vehicle's passes on other cameras added
    present: bool = True             # could it identify the vehicle the case is about?
    downgraded: str = ""             # why the verdict was overruled here, if it was
    pre_event: str = ""
    event: str = ""
    aftermath: str = ""
    cause: str = ""
    summary: str = ""
    model: str = ""
    backend: str = ""
    seconds: float = 0.0
    error: str = ""
    images: int = 0
    # Two-tier cascade (surveillance/vlm_cascade.py). Empty / default on the single-model path.
    decided_by: str = ""             # "" (single model) | "reasoner" | "screen" | "prescreen"
    escalated: Optional[bool] = None
    escalation_reason: str = ""
    reasoner_unavailable: bool = False
    stages: list = field(default_factory=list)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------- evidence images

def encode_image(path, max_side, quality=85):
    """JPEG bytes of an evidence photo, scaled down so the vision encoder is not asked to read a 4K frame.

    Image size is the dominant cost on a CPU backend: every extra pixel becomes vision tokens that must be
    encoded before a single word is generated."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


CONTACT_CODES = ("V10", "V11")


def gather_images(violation, data_dir, max_images, max_side):
    """Evidence photos for a case, most informative first: the close-up, the whole scene with the vehicle
    boxed and its path drawn, then any extra moment the rule saved (e.g. when the vehicle stopped).

    For the two candidates the model has to decide (V10 swerve, V11 near-collision / collision) the frame that
    shows both vehicles boxed at the moment of contact is the one photograph that can settle it, so when the
    rule saved one it goes first and the default of two photographs keeps it: contact frame, then the close-up.

    Returns [(label, jpeg bytes)]."""
    data_dir = Path(data_dir)
    extras = [(e.get("label", "happened"), e["path"]) for e in (violation.get("evidence") or []) if e.get("path")]
    wanted = []
    contact = [x for x in extras if x[0] == "contact"] \
        if (violation.get("code") or "").upper() in CONTACT_CODES else []
    for _, path in contact[:1]:
        wanted.append(("the moment the two vehicles touched, both outlined", path))
    if violation.get("crop"):
        wanted.append(("close-up of the vehicle", violation["crop"]))
    if violation.get("snapshot"):
        wanted.append(("the whole scene, vehicle outlined in red, its recent path in yellow",
                       violation["snapshot"]))
    for label, path in extras:
        if (label, path) not in contact[:1]:
            wanted.append((f"the moment it {label}", path))
    if violation.get("plate_img"):
        wanted.append(("the number plate as read", violation["plate_img"]))
    out = []
    for label, rel in wanted:
        if len(out) >= max_images:
            break
        p = data_dir / rel
        if not p.exists():
            continue
        jpg = encode_image(p, max_side)
        if jpg:
            out.append((label, jpg))
    return out


def describe_facts(violation):
    """The rule engine's own numbers, written out for the prompt. Only what was actually measured."""
    v, lines = violation, []
    add = lines.append
    add(f"- vehicle: {v.get('cls') or 'unknown'} (classifier agreed on this class for "
        f"{round(100 * (v.get('class_conf') or 0))}% of the frames it was tracked)")
    if v.get("kph") is not None:
        add(f"- measured ground speed at the moment of the event: {v['kph']} km/h")
    if v.get("zone"):
        add(f"- road zone: {v['zone']}")
    if v.get("areas"):
        add(f"- marked areas the vehicle was inside: {v['areas'].replace('|', ', ')}")
    if v.get("wall_time"):
        add(f"- time of day: {v['wall_time']}")
    d = v.get("details") or {}
    if isinstance(d, dict):
        for k, val in d.items():
            add(f"- {str(k).replace('_', ' ')}: {val}")
    if v.get("plate_text_en"):
        add(f"- number plate read by the system: {v['plate_text_en']} (do not repeat this unless asked)")
    return "\n".join(lines) or "- no further measurements recorded"


def describe_journey(violation):
    """The same vehicle seen by other cameras, written out for the prompt.

    This is what one camera cannot tell you. A bus halted for forty seconds reads very differently when the
    camera before it also had it crawling - that is a jam it was caught in - than when it came through the
    previous junction at speed and then stopped dead. The model is given the passes and told what they do
    and do not establish, because the link itself is an inference and may be wrong."""
    passes = violation.get("journey") or []
    if len(passes) < 2:
        return ""
    here = violation.get("camera_id")
    lines = []
    prev_t = None
    for p in passes:
        when = p.get("wall_time") or (f"{p['t_cross']:.0f}" if p.get("t_cross") else "?")
        gap = f", {p['t_cross'] - prev_t:.0f} s later" if prev_t and p.get("t_cross") else ""
        mark = "  <- this case" if p.get("camera_id") == here else ""
        speed = f"{(p.get('speed') or 0) * 3.6:.0f} km/h"
        lines.append(f"- camera \"{p.get('camera_name', p.get('camera_id'))}\""
                     f"{' (' + p['camera_location'] + ')' if p.get('camera_location') else ''}: "
                     f"{speed}{gap}{mark}")
        prev_t = p.get("t_cross")
    return ("\n\nTHE SAME VEHICLE ON OTHER CAMERAS\n"
            "The system believes these are the same vehicle, matched on where and when it crossed, its\n"
            "speed, size, class and colour. That match is an inference and can be wrong, so treat it as\n"
            "supporting context, never as proof, and say so if it is what your explanation rests on.\n"
            + "\n".join(lines))


def build_prompt(violation, images):
    code = (violation.get("code") or "").upper()
    claim, innocent = RULE_CLAIM.get(code, ("The system flagged this vehicle.", "Nothing visible supports it."))
    photos = "\n".join(f"{i + 1}. {label}" for i, (label, _) in enumerate(images)) or "none available"
    subject = (violation.get("cls") or "vehicle").lower()
    return USER_TEMPLATE.format(vid=violation.get("id", "?"), code=code or "?",
                                title=violation.get("title") or "", claim=claim, innocent=innocent,
                                facts=describe_facts(violation), journey=describe_journey(violation),
                                photos=photos, subject=subject)


# ---------------------------------------------------------------- parsing

def parse_reply(text):
    """The model's JSON, however it wrapped it. Returns a dict or None."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        obj = json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)          # prose around the object
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    return obj if isinstance(obj, dict) else None


def to_explanation(obj, model, backend, seconds, images, subject=""):
    """Validate the model's object. Anything unrecognised degrades to a decline, never to a false
    'supported': an unparsable answer must not read as agreement."""
    ex = Explanation(model=model, backend=backend, seconds=round(seconds, 1), images=images)
    if not obj:
        ex.ok, ex.error = False, "the model did not return usable JSON"
        return ex
    for k in ("pre_event", "event", "aftermath", "cause", "summary", "visible", "other_cameras"):
        val = obj.get(k)
        setattr(ex, k, " ".join(str(val).split())[:600] if val not in (None, "") else "")
    a = str(obj.get("assessment", "")).strip().lower()
    ex.assessment = next((v for v in ASSESSMENTS if v in a), "insufficient")
    ex.present = obj.get("present", True) not in (False, "false", "False", "no", 0)
    try:
        # Models answer either as a fraction or as a percentage, so anything above 1 is read as a percentage.
        # That errs toward a lower confidence on an odd value, which is the safe direction for a case that
        # may support a prosecution: overstating certainty is the harmful mistake, understating it is not.
        c = float(obj.get("confidence", 0))
        ex.confidence = round(min(1.0, max(0.0, c / 100 if c > 1 else c)), 2)
    except (TypeError, ValueError):
        ex.confidence = 0.0
    # The guard that does not trust the model. A small model asked "does this photograph support the
    # finding?" tends to agree, because agreeing is what the prompt looks like it wants: on the probe in
    # tools/vlm_probe.py, Qwen2.5-VL-3B confirmed a bus violation from a photograph with no bus in it.
    # Asking first whether the vehicle is even identifiable is a far easier question, and answered more
    # honestly, so a "supported" that its own observation step does not back up is overruled here rather
    # than shown to an officer. It can only ever weaken a verdict, never strengthen one.
    if ex.assessment == "supported":
        why = ""
        if not ex.present:
            why = ("could not identify the " + (subject or "vehicle") + " the case is about in the "
                   "photographs")
        elif subject_missing(subject, ex.visible):
            # It answered "yes it is there" while describing a photograph in which it never mentions the
            # vehicle. Its description is the more trustworthy of the two: writing it came before judging.
            why = ("described the photographs without mentioning any " + subject + ", so its agreement is "
                   "not backed by its own observation")
        if why:
            ex.assessment = "insufficient"
            ex.downgraded = f"Verdict overruled by the system: the model {why}."
            ex.confidence = min(ex.confidence, 0.3)
    if not ex.summary:
        ex.summary = ex.event or ex.cause or "No description returned."
    return ex


# ---------------------------------------------------------------- backends

def _post(url, payload, headers, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class OpenAIChat:
    """Any endpoint speaking OpenAI's /v1/chat/completions with image_url content: llama.cpp's server
    locally, and OpenAI-compatible cloud providers."""

    _health = {}        # base url -> (checked at, was up): see available()

    def __init__(self, base_url, model, api_key="", name="local", timeout=300, max_tokens=420):
        self.base = base_url.rstrip("/")
        self.model, self.key, self.name = model, api_key, name
        self.timeout, self.max_tokens = timeout, max_tokens

    def available(self, probe_sec=1.5, cache_sec=3.0):
        """Is a server listening and answering? The answer is cached for a few seconds because the dashboard
        asks on every page load and several times per status poll.

        The probe is kept short on purpose: llama.cpp accepts connections while it is still mapping three
        gigabytes of weights, so an unlucky probe would otherwise hold up the page for its whole timeout."""
        hit = self._health.get(self.base)
        if hit and time.monotonic() - hit[0] < cache_sec:
            return hit[1]
        try:
            with urllib.request.urlopen(f"{self.base}/v1/models", timeout=probe_sec) as r:
                up = r.status == 200
        except Exception:
            up = False
        self._health[self.base] = (time.monotonic(), up)
        return up

    def ask(self, system, user_text, images, max_tokens=None, schema=None):
        """`schema`: a JSON schema the reply must follow (llama.cpp turns it into a grammar, so a 2B model
        cannot ramble); without it the reply is "some JSON object"."""
        content = [{"type": "text", "text": user_text}]
        for _, jpg in images:
            b64 = base64.b64encode(jpg).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        fmt = ({"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
               if schema else {"type": "json_object"})
        payload = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "temperature": 0.2,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                   "response_format": fmt}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post(f"{self.base}/v1/chat/completions", payload, headers, self.timeout)
        return data["choices"][0]["message"]["content"]


class AnthropicChat:
    """Anthropic's Messages API."""

    API = "https://api.anthropic.com/v1/messages"

    def __init__(self, model, api_key, name="cloud", timeout=180, max_tokens=420):
        self.model, self.key, self.name = model, api_key, name
        self.timeout, self.max_tokens = timeout, max_tokens

    def available(self):
        return bool(self.key)

    def ask(self, system, user_text, images, max_tokens=None, schema=None):
        content = []
        for _, jpg in images:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.b64encode(jpg).decode()}})
        content.append({"type": "text", "text": user_text})
        payload = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "temperature": 0.2,
                   "system": system,
                   "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        data = _post(self.API, payload, headers, self.timeout)
        return "".join(b.get("text", "") for b in data.get("content", []))


DEFAULTS = {
    "vlm": True,                       # answer "Explain" requests at all
    "vlm_backend": "auto",             # auto | local | cloud | off
    "vlm_local_url": "http://127.0.0.1:8081",
    "vlm_local_model": "qwen2.5-vl-3b",
    "vlm_cloud_model": "claude-sonnet-5-5",
    "vlm_max_images": 2,               # every image costs real seconds on a CPU backend
    "vlm_max_side": 768,               # longest side sent to the model, pixels
    "vlm_timeout_sec": 300,
    # ---- two-tier cascade (surveillance/vlm_cascade.py); the keys above are the single-model path
    "vlm_mode": "auto",                # auto: cascade when a reasoner can exist, else single | cascade | single
    "vlm_screen_url": "http://127.0.0.1:8081",
    "vlm_screen_model": "qwen3-vl-2b",
    "vlm_screen_images": 2,
    "vlm_screen_timeout_sec": 120,
    "vlm_reason_backend": "auto",      # auto (local 8B if installed, else cloud) | local | cloud
    "vlm_reason_url": "http://127.0.0.1:8082",
    "vlm_reason_model": "qwen3-vl-8b",
    "vlm_reason_idle_sec": 300,        # stop the on-demand 8B after this long without a case
    "vlm_reason_start_sec": 240,       # give up if it is not serving by then
    "vlm_reason_timeout_sec": 900,     # one 8B answer: ~4 min on a busy 6-core CPU with two photographs
    "vlm_escalation": "asymmetric",    # asymmetric | balanced | always (see vlm_cascade.py)
    "vlm_stop_score": 0.6,             # asymmetric: only a pre-screen score below this may be stopped by a "no"
    "vlm_no_confidence": 0.7,          # a "no" counts as confident from here
}


def resolve_backend(settings=None):
    """The backend to use, or None. `auto` prefers a local server that is actually up, then a cloud key.

    Keys come from the environment only - ANTHROPIC_API_KEY, or OPENAI_API_KEY with OPENAI_BASE_URL - so a
    key is never written into the database or shown in the dashboard."""
    s = {**DEFAULTS, **(settings or {})}
    want = (s.get("vlm_backend") or "auto").lower()
    if want == "off" or not s.get("vlm", True):
        return None
    local = OpenAIChat(s["vlm_local_url"], s["vlm_local_model"], name="local",
                       timeout=s["vlm_timeout_sec"])
    anth_key = os.environ.get("ANTHROPIC_API_KEY", "")
    oa_key, oa_base = os.environ.get("OPENAI_API_KEY", ""), os.environ.get("OPENAI_BASE_URL", "")
    def cloud():
        if anth_key:
            return AnthropicChat(s["vlm_cloud_model"], anth_key, timeout=s["vlm_timeout_sec"])
        if oa_key:
            return OpenAIChat(oa_base or "https://api.openai.com", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                              oa_key, name="cloud", timeout=s["vlm_timeout_sec"])
        return None
    if want == "local":
        return local
    if want == "cloud":
        return cloud()
    return local if local.available() else cloud()


def backend_status(settings=None):
    """What the dashboard shows: which backends could answer right now."""
    s = {**DEFAULTS, **(settings or {})}
    local = OpenAIChat(s["vlm_local_url"], s["vlm_local_model"])
    chosen = resolve_backend(settings)
    return {
        "enabled": bool(chosen),
        "backend": chosen.name if chosen else "",
        "model": chosen.model if chosen else "",
        "local_up": local.available(),
        "cloud_key": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "local_url": s["vlm_local_url"],
    }


# ---------------------------------------------------------------- the queue

class Explainer:
    """Explains cases one at a time on a background thread.

    One at a time on purpose: a local CPU backend saturates every core, and two of them at once would make
    both slow and would fight the detector if a camera is analysing at the same time.

    `on_result(vid, Explanation)` is called from the worker thread.
    """

    def __init__(self, data_dir, settings, on_result, max_queue=50, cascade=None):
        self.data_dir, self.on_result = Path(data_dir), on_result
        self.cascade = cascade         # surveillance.vlm_cascade.Cascade, or None for the single-model path
        self.settings = {**DEFAULTS, **(settings or {})}
        self.jobs = queue.Queue(max_queue)
        self.pending = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name="vlm")
        self.thread.start()

    def busy(self):
        with self.lock:
            return sorted(self.pending)

    def request(self, violation, settings=None):
        """Queue a case. `settings` overrides this one job only, so a case can be sent to the cloud backend
        while the rest stay local. Returns False if it is already queued or the queue is full."""
        vid = violation["id"]
        with self.lock:
            if vid in self.pending:
                return False
            self.pending.add(vid)
        try:
            self.jobs.put_nowait((violation, settings))
            return True
        except queue.Full:
            with self.lock:
                self.pending.discard(vid)
            return False

    def explain_now(self, violation, settings=None):
        """Run synchronously (command line, tests). Returns an Explanation."""
        settings = {**self.settings, **(settings or {})}
        if self.cascade is not None and self.cascade.active(settings):
            return self.cascade.run(violation, settings)
        backend = resolve_backend(settings)
        if backend is None:
            return Explanation(ok=False, error="No reasoning model configured. Start tools/vlm_server.sh or "
                                               "set ANTHROPIC_API_KEY.")
        images = gather_images(violation, self.data_dir, settings["vlm_max_images"],
                               settings["vlm_max_side"])
        if not images:
            return Explanation(ok=False, error="No evidence photographs were saved for this case.")
        t0 = time.perf_counter()
        try:
            reply = backend.ask(SYSTEM, build_prompt(violation, images), images)
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            return Explanation(ok=False, backend=backend.name, model=backend.model,
                               error=f"{backend.name} backend returned HTTP {e.code}: {detail}")
        except Exception as e:
            return Explanation(ok=False, backend=backend.name, model=backend.model,
                               error=f"{backend.name} backend unreachable: {e}")
        return to_explanation(parse_reply(reply), backend.model, backend.name,
                              time.perf_counter() - t0, len(images), violation.get("cls") or "")

    def _run(self):
        while True:
            v, settings = self.jobs.get()
            try:
                self.on_result(v["id"], self.explain_now(v, settings))
            except Exception as e:                      # never let the thread die
                self.on_result(v["id"], Explanation(ok=False, error=str(e)))
            finally:
                with self.lock:
                    self.pending.discard(v["id"])
                self.jobs.task_done()
