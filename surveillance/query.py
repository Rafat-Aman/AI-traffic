"""Natural-language query over the segment index (WP3, requirements R-3 and R-4; report section 3.1.5).

Retrieve-then-reason, in the VideoRAG style of the 499A slides:

  Stage A  grounding   question -> structured query (identity, verbalised motion, place, time window)
                       -> ranked 10-second segments from the persistent index. No video is read, no detector runs.
  Stage B  explanation the question is routed by rules:
                         DESCRIPTIVE ("how many buses stopped at camera 2 between 5 and 6 pm?")
                             answered exactly from the index rows, which are returned as the evidence;
                         CAUSAL ("why did traffic back up at 5:40?")
                             the few best segments' frames are pulled from the stored video ON DEMAND and handed
                             to an injected `reason_fn(question, segments)`; the answer must cite segments.
                       "Insufficient evidence" is a first-class answer: empty or weak retrieval, raw video gone,
                       a reasoner that does not cite, or no reasoner at all never produce an invented answer.

The parser is rule-based and is also the validator: an optional `llm_fn` may re-read the question, but whatever it
returns is checked field by field (vocabulary, ranges, and that a camera / time / plate really appears in the
question) before it is used. Times are Dhaka local (Asia/Dhaka), the same clock the cameras' start times use.

Injected callables (this module imports no model code)
    llm_fn(question: str, context: dict) -> str | dict
        returns a JSON object (or text containing one) with any of: classes, colours, codes, behaviours,
        camera_ids, t0, t1 (ISO local date-time), plate, direction, agg, intent.
    reason_fn(question: str, segments: list[dict]) -> dict
        segments: [{"id", "camera", "start", "end", "facts": [str], "frames": [{"label", "t", "jpeg": bytes}], ...}]
        returns {"answer": str, "sufficient": bool, "confidence": 0..1, "citations": [segment id], "limits": str}
    make_reason_fn(backend) / make_llm_fn(backend) build those from anything with the vlm.py backend method
        ask(system, user_text, images, max_tokens=None, schema=None) -> str   (OpenAIChat, AnthropicChat)
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from . import segment_index as si

# ---------------------------------------------------------------- limits

MAX_QUESTION_CHARS = 500
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_ROWS = 5000                  # vehicle rows scored for one question
POINT_PAD_S = 120.0              # "at 5:40" means 5:40 plus or minus this
SCENE_WINDOWS = 800              # windows scored for a scene question
CAUSAL_SEGMENTS = 4              # segments the reasoner sees
CAUSAL_FRAMES_PER_SEGMENT = 3
WEAK_SCORE = 0.30                # best segment below this = not enough to reason over
MAX_SPAN_S = 31 * 86400
MAX_HOURS_AGO = 400 * 86400

STOP_WORDS = set("""a an the of in on at to for from by with and or is are was were be been being did do does done
had has have it its this that these those there their them they we you i me my our your what which who whom how
when where why while than then so as if not no any all some many much more most show find list give tell get see
display me us please can could would should will just about over under into out up down off near around after
before during between camera cameras cam cctv vehicle vehicles traffic road area side time times day today
yesterday last past ago minutes minute hours hour pm am going go goes went leave leaves left leaving
came come comes looked look looking seen saw there here happen happened back up""".split())


# ---------------------------------------------------------------- Stage A: the parser

def _empty_query(question: str) -> dict:
    return {"question": question, "classes": [], "colours": [], "codes": [], "behaviours": [], "camera_ids": [],
            "unknown_cameras": [], "t0": None, "t1": None, "point": None, "time_text": "", "plate": None,
            "direction": None, "global_id": None, "min_kph": None, "max_kph": None, "agg": "list",
            "intent": "descriptive", "intent_reasons": [], "keywords": [], "notes": [], "source": "rules"}


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
PERIODS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 29), "noon": (11, 14)}
T_RE = r"(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?"
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
             "fifteen": 15, "twenty": 20, "thirty": 30, "half": 0.5}


def _blank(work: str, m: re.Match) -> str:
    return work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]


def _hour24(h: int, mer: Optional[str]) -> int:
    if mer:
        mer = mer[0]
        return (h % 12) + (12 if mer == "p" else 0)
    return h


def _clock(h, mi, mer):
    """(hour24, minute, had_meridiem) or None if it is not a time of day."""
    h = int(h)
    mi = int(mi) if mi else 0
    if mer and not 1 <= h <= 12:
        return None
    if not 0 <= h <= 24 or mi > 59:
        return None
    return (_hour24(h, mer) if mer else h, mi, bool(mer))


def _guess_meridiem(h24: int) -> tuple:
    """No am/pm given and the hour is 1..12: traffic questions about 1-6 mean afternoon, 7-11 morning."""
    if h24 == 12:
        return 12, "12 taken as noon"
    if 1 <= h24 <= 6:
        return h24 + 12, f"{h24} taken as {h24} pm"
    return h24, f"{h24} taken as {h24} am"


def parse_time(text: str, now: datetime) -> tuple:
    """-> (t0, t1, point, time_text, notes, work) ; t0/t1 naive Dhaka datetimes or None; `work` has the matched
    words blanked so the caller does not read them as keywords."""
    notes, work = [], text.lower()
    day = None                                 # a date (datetime at midnight)
    span = None                                # (start_hour, end_hour) of a named period or explicit range
    point_clock = None
    explicit_clock = None
    shown = []
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # relative spans: "last 10 minutes", "past 2 hours", "in the last hour"
    m = re.search(r"\b(?:in\s+the\s+)?(?:last|past|previous)\s+(a|an|one|two|three|four|five|six|ten|fifteen|twenty|thirty|half|\d+(?:\.\d+)?)?\s*(?:an\s+|a\s+)?(minute|min|hour|hr|day|week)s?\b", work)
    if m and not re.search(r"last\s+night", work):
        raw = m.group(1)
        n = 1 if raw is None else (float(raw) if re.match(r"\d", raw) else NUM_WORDS[raw])
        if m.group(0).strip().endswith("hour") and raw == "half":
            n = 0.5
        unit = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}[m.group(2)]
        secs = n * unit
        if 0 < secs <= MAX_SPAN_S:
            return (now - timedelta(seconds=secs), now, None, m.group(0).strip(), notes, _blank(work, m))
    m = re.search(r"\b(\d+|an?|half an?)\s*(minute|min|hour|hr)s?\s+ago\b", work)
    if m:
        raw = m.group(1)
        n = 0.5 if raw.startswith("half") else (1 if raw in ("a", "an") else float(raw))
        pt = now - timedelta(seconds=n * (60 if m.group(2).startswith("min") else 3600))
        return (pt - timedelta(seconds=POINT_PAD_S), pt + timedelta(seconds=POINT_PAD_S), pt, m.group(0), notes,
                _blank(work, m))

    # a date
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", work)
    if m:
        try:
            day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            shown.append(m.group(0))
            work = _blank(work, m)
        except ValueError:
            notes.append(f"'{m.group(0)}' is not a date")
    if day is None:
        mon = "|".join(sorted(MONTHS, key=len, reverse=True))
        for pat, order in ((rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({mon})\b(?:,?\s+(\d{{4}}))?", "dm"),
                           (rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", "md")):
            m = re.search(pat, work)
            if m:
                d, mo = (m.group(1), m.group(2)) if order == "dm" else (m.group(2), m.group(1))
                year = int(m.group(3)) if m.group(3) else now.year
                try:
                    day = datetime(year, MONTHS[mo], int(d))
                    if not m.group(3) and day > today:
                        day = day.replace(year=year - 1)
                        notes.append("no year given: taken as the most recent past date")
                    shown.append(m.group(0))
                    work = _blank(work, m)
                    break
                except ValueError:
                    notes.append(f"'{m.group(0)}' is not a date")
    if day is None:
        m = re.search(r"\bday before yesterday\b", work)
        if m:
            day = today - timedelta(days=2)
            shown.append("day before yesterday")
            work = _blank(work, m)
    if day is None:
        m = re.search(r"\b(last night|yesterday|today|tonight)\b", work)
        if m:
            w = m.group(1)
            shown.append(w)
            work = _blank(work, m)
            if w == "yesterday":
                day = today - timedelta(days=1)
            elif w == "today":
                day = today
            elif w == "tonight":
                day, span = today, (18, 29)
            else:
                day, span = today - timedelta(days=1), (18, 29)
    if day is None:
        m = re.search(r"\b(?:on\s+)?(?:(last|this|previous)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", work)
        if m:
            back = (today.weekday() - WEEKDAYS[m.group(2)]) % 7
            if back == 0 and m.group(1) in ("last", "previous"):
                back = 7
            day = today - timedelta(days=back)
            shown.append(m.group(0).strip())
            work = _blank(work, m)
            if back == 0:
                notes.append("weekday named is today")
    # a period of the day
    m = re.search(r"\b(?:(this|yesterday)\s+)?(morning|afternoon|evening|night)\b(?:\s+(?:rush|peak))?", work)
    if m and span is None:
        span = PERIODS[m.group(2)]
        if m.group(1) == "yesterday" and day is None:
            day = today - timedelta(days=1)
        shown.append(m.group(0).strip())
        work = _blank(work, m)
        if m.group(1) is None and day is None:
            notes.append(f"'{m.group(2)}' with no day: taken as today (or yesterday if that has not happened yet)")
    m = re.search(r"\b(morning|evening)\s+(?:rush|peak)\s*(?:hour)?s?\b|\b(?:rush|peak)\s+hours?\b", work)
    if m and span is None:
        span = (8, 10) if (m.group(1) == "morning") else (17, 20)
        if m.group(1) is None:
            notes.append("'rush hour' taken as 17:00-20:00 (say 'morning rush' for 08:00-10:00)")
        shown.append(m.group(0).strip())
        work = _blank(work, m)

    # an explicit range of clock times
    rng = re.search(rf"\b(?:between|from)\s+{T_RE}\s*(?:and|to|-|–|till|until)\s*{T_RE}", work)
    kind = "between"
    if not rng:
        rng = re.search(rf"\b{T_RE}\s*(?:-|–|to|till|until)\s*{T_RE}", work)
        kind = "bare"
    if rng:
        a = _clock(rng.group(1), rng.group(2), rng.group(3))
        b = _clock(rng.group(4), rng.group(5), rng.group(6))
        ok = a and b and (kind == "between" or any(g for g in (rng.group(2), rng.group(3), rng.group(5), rng.group(6))))
        if ok:
            ah, am_, amer = a
            bh, bm_, bmer = b
            if bmer and not amer and ah <= 12:
                bpm = rng.group(6).lower().startswith("p")
                cand = ah % 12 + (12 if bpm else 0)
                if cand > bh:                      # "11 and 1 pm": the first is the morning one
                    cand = ah % 12 + (0 if bpm else 12)
                ah = cand
            elif amer and not bmer and bh <= 12:
                apm = rng.group(3).lower().startswith("p")
                cand = bh % 12 + (12 if apm else 0)
                if cand < ah:
                    cand += 12
                bh = cand
            elif not amer and not bmer and ah <= 12 and bh <= 12:
                ah, n1 = _guess_meridiem(ah)
                bh, n2 = _guess_meridiem(bh) if bh <= 12 else (bh, "")
                if bh < ah:
                    bh += 12
                notes.append(f"no am/pm: {n1}" + (f", {n2}" if n2 else "") + " (write am/pm to be exact)")
            explicit_clock = ((ah, am_), (bh, bm_))
            shown.append(rng.group(0).strip())
            work = _blank(work, rng)
    if explicit_clock is None:
        pm = re.search(rf"\b(?:at|around|about|near|by)\s+{T_RE}", work)
        if pm and (pm.group(2) or pm.group(3)):
            c = _clock(pm.group(1), pm.group(2), pm.group(3))
            if c:
                h, mi, hadm = c
                if not hadm and h <= 12:
                    h, n1 = _guess_meridiem(h)
                    notes.append(f"no am/pm: {n1}")
                point_clock = (h, mi)
                shown.append(pm.group(0).strip())
                work = _blank(work, pm)
        if point_clock is None:
            pm = re.search(rf"\b(\d{{1,2}})[:.](\d{{2}})\s*(a\.?m\.?|p\.?m\.?)?|\b(\d{{1,2}})\s*(a\.?m\.?|p\.?m\.?)", work)
            if pm:
                if pm.group(1):
                    c = _clock(pm.group(1), pm.group(2), pm.group(3))
                else:
                    c = _clock(pm.group(4), None, pm.group(5))
                if c:
                    h, mi, hadm = c
                    if not hadm and h <= 12:
                        h, n1 = _guess_meridiem(h)
                        notes.append(f"no am/pm: {n1}")
                    point_clock = (h, mi)
                    shown.append(pm.group(0).strip())
                    work = _blank(work, pm)

    if day is None and span is None and explicit_clock is None and point_clock is None:
        return None, None, None, "", notes, work

    def at(d, h, mi):
        return d + timedelta(hours=h, minutes=mi)

    inferred_day = day is None
    base = day if day is not None else today
    point = None
    if explicit_clock:
        (ah, am_), (bh, bm_) = explicit_clock
        t0, t1 = at(base, ah, am_), at(base, bh, bm_)
        if t1 <= t0:
            t1 += timedelta(days=1)
    elif point_clock:
        point = at(base, *point_clock)
        t0, t1 = point - timedelta(seconds=POINT_PAD_S), point + timedelta(seconds=POINT_PAD_S)
    elif span:
        t0, t1 = at(base, span[0], 0), at(base, span[1], 0)
    else:
        t0, t1 = base, base + timedelta(days=1)
    if inferred_day and t0 > now:                  # "at 5:40" said now-ish: the last time that has happened
        shift = timedelta(days=1)
        t0, t1 = t0 - shift, t1 - shift
        point = point - shift if point else None
        notes.append("no day given and that time has not happened yet today: taken as yesterday")
    elif inferred_day and (explicit_clock or point_clock or span):
        notes.append("no day given: taken as today")
    return t0, t1, point, ", ".join(shown), notes, work


# ---- identity, behaviour, place

_CLASS_PHRASES = None


def _class_phrases() -> list:
    global _CLASS_PHRASES
    if _CLASS_PHRASES is None:
        out = []
        for cls, words in si.CLASS_WORDS.items():
            for w in words + [cls.lower()]:
                out.append((w, [cls]))
        for w, cls in si.AMBIGUOUS_CLASS_WORDS.items():
            out.append((w, list(cls)))
        _CLASS_PHRASES = sorted(set((w, tuple(c)) for w, c in out), key=lambda x: -len(x[0]))
    return _CLASS_PHRASES


BEHAVIOUR_RULES = [
    # (regex, codes, behaviours, note)
    (r"\bwrong[\s-]*(?:way|side|direction)\b|\bcontra[\s-]?flow\b|\b(?:against|opposite)\s+(?:the\s+)?(?:traffic|flow|direction)\b", ["V4"], [], ""),
    (r"\bspeeding\b|\bover[\s-]?speed(?:ing)?\b|\bexceed\w*\s+(?:the\s+)?(?:speed\s+)?limit\b", ["V2"], [], ""),
    (r"\bno[\s-]?helmet\b|\bwithout\s+(?:a\s+)?helmets?\b|\bhelmet(?:less)?\b", ["V6"], [], ""),
    (r"\bbus[\s-]?lane\b", ["V9"], [], ""),
    (r"\blane[\s-]?(?:line|change|changing|crossing)\b|\bcross\w*\s+(?:the\s+)?(?:solid\s+)?(?:lane\s+)?line\b|\bsolid\s+line\b", ["V5"], [], ""),
    (r"\bpark(?:ed|ing)\b|\bno[\s-]?parking\b|\billegally\s+parked\b", ["V3"], ["stopped"], ""),
    (r"\bblock(?:ed|ing)\s+(?:a\s+|the\s+)?(?:running\s+)?(?:lane|road|traffic)\b|\broad[\s-]?block\w*\b", ["V7"], ["stopped"], ""),
    (r"\bbanned\b|\brestricted\b|\bnot\s+allowed\b|\bprohibited\b", ["V8"], [], ""),
    (r"\berratic\b|\bswerv\w*\b|\bzig[\s-]?zag\w*\b|\bunsafe\s+(?:manoeuvre|maneuver|driving|move)\w*\b|\breckless\b", ["V10"], [], ""),
    (r"\bnear[\s-]?(?:collision|miss)\w*\b|\bcollisions?\b|\bcrash\w*\b|\baccidents?\b|\bcollided\b", ["V11"], [], ""),
    (r"\bbus\s+(?:stop\s+)?blocking\b|\bstopp\w+\s+outside\s+(?:the\s+)?bus[\s-]?stop\b", ["V1"], ["stopped"], ""),
    (r"\b(?:traffic\s+)?(?:back(?:ed)?[\s-]?up|jam(?:med|s)?|congest\w*|gridlock\w*|standstill|queue[sd]?|bottleneck\w*)\b", [], ["congestion"], ""),
    (r"\bstopp(?:ed|ing)\b|\bstationary\b|\bstanding\b|\bhalt(?:ed|ing)?\b|\bidling\b|\bwaiting\b|\bstuck\b|\bnot\s+moving\b", [], ["stopped"], ""),
    (r"\bslow(?:ly|-moving)?\b|\bcrawl\w*\b", [], ["slow"], ""),
    (r"\bviolations?\b|\boffen[cs]es?\b|\bbreaking\s+(?:the\s+)?rules?\b|\bflagged\b", ["ANY"], [], ""),
]
DIRECTION_RULES = [
    (r"\b(?:toward|towards|approaching|coming|incoming|heading\s+to(?:ward)?s?)\b(?:\s+the\s+camera)?", "incoming"),
    (r"\b(?:away\s+from|leaving|outgoing|going\s+away)\b(?:\s+the\s+camera)?", "outgoing"),
]
AGG_RULES = [
    (r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b|\bhow\s+much\s+traffic\b|\btotal\b", "count"),
    (r"\baverage\s+speed\b|\bmean\s+speed\b|\bhow\s+fast\b|\bspeed\s+of\b|\baverage\s+kph\b", "speed"),
    (r"\bwhen\b|\bwhat\s+time\b|\bat\s+what\s+time\b", "when"),
]
CAUSAL_RULES = [
    (r"\bwhy\b", "'why'"), (r"\bwhat\s+(?:caused|causes|led\s+to|made|happened|is\s+happening|was\s+happening)\b", "asks what happened or caused"),
    (r"\b(?:cause|causes|reason|reasons)\s+(?:of|for|behind)\b|\bwhat'?s\s+behind\b", "asks for a cause"),
    (r"\bexplain\b|\bhow\s+come\b|\bbecause\b|\bdue\s+to\b|\bresponsible\s+for\b|\bwho\s+(?:caused|started)\b", "asks for an explanation"),
    (r"\bwhat\s+went\s+wrong\b|\bhow\s+did\b.*\b(?:start|begin|happen)\b", "asks how it came about"),
]
DESCRIPTIVE_HINTS = [(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b", "a count"), (r"\bshow\b|\blist\b|\bfind\b|\bwhich\b|\bwhere\b|\bwhen\b|\bwhat\s+time\b", "asks to find or list"),
                     (r"\baverage\b|\bmean\b|\bhow\s+fast\b", "a statistic")]
SPEED_RE = re.compile(r"\b(?:faster\s+than|above|over|exceeding|more\s+than|at\s+least|>\s*=?)\s*(\d{2,3})\s*(?:km/?h|kph|kmh)\b|\b(\d{2,3})\s*(?:km/?h|kph|kmh)\s*(?:or\s+more|\+|and\s+(?:above|over))", re.I)
SLOW_RE = re.compile(r"\b(?:slower\s+than|below|under|less\s+than|at\s+most|<\s*=?)\s*(\d{1,3})\s*(?:km/?h|kph|kmh)\b", re.I)


def camera_mentions(text: str, cameras: list) -> tuple:
    """(ids found, unknown numbers, text with the mentions blanked). 'camera 2' is an id; otherwise a camera's
    name or location as a whole phrase. Generic names ('Camera') are never matched as a name."""
    work, ids, unknown = text.lower(), [], []
    for m in list(re.finditer(r"\b(?:camera|cam|cctv)s?\s*(?:#|no\.?|number)?\s*(\d{1,4})((?:\s*(?:,|and|&|or)\s*\d{1,4})*)\b", work)):
        nums = [int(m.group(1))] + [int(x) for x in re.findall(r"\d+", m.group(2) or "")]
        for n in nums:
            if any(c["id"] == n for c in cameras):
                ids.append(n)
            else:
                unknown.append(n)
        work = _blank(work, m)
    generic = {"camera", "cam", "cctv", "video", "cameras"}
    for c in cameras:
        for label in (c.get("name") or "", c.get("location") or ""):
            lab = label.strip().lower()
            if len(lab) < 3 or lab in generic:
                continue
            m = re.search(r"(?<![a-z0-9])" + re.escape(lab) + r"(?![a-z0-9])", work)
            if m:
                ids.append(c["id"])
                work = _blank(work, m)
    return sorted(set(ids)), sorted(set(unknown)), work


def parse_plate(text: str) -> tuple:
    """(plate fragment or None, text with it blanked)"""
    work = text
    m = re.search(r"\b(?:number\s*plate|licen[sc]e\s*plate|registration|plate|reg\.?)\b\s*(?:number|no\.?|#|:)?\s*"
                  r"((?:(?:is|ending|starting|with|in|containing|like|of|=)\s+)*)((?:[a-z0-9\-]{2,}\s*){1,4})", work, re.I)
    if m:
        toks = []
        for t in m.group(2).split():
            if t.lower() in STOP_WORDS | {"camera", "cam"} or re.fullmatch(r"(?:am|pm)", t, re.I):
                break
            toks.append(t)
        frag = " ".join(toks).strip("- ")
        if frag and (re.search(r"\d", frag) or len(frag) >= 3):
            return frag.upper(), work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    m = re.search(r"\b\d{2}[\s-]\d{4}\b", work)
    if m:
        return m.group(0).upper().replace(" ", "-"), _blank(work, m)
    return None, work


def classify_intent(text: str) -> tuple:
    """-> ("causal" | "descriptive", [reasons]). Rules: any causal cue wins, because a question that asks WHY is not
    answered by a count even if it contains one; otherwise descriptive."""
    low = text.lower()
    why = [d for pat, d in CAUSAL_RULES if re.search(pat, low)]
    if why:
        return "causal", why
    desc = [d for pat, d in DESCRIPTIVE_HINTS if re.search(pat, low)]
    return "descriptive", desc or ["no causal cue (why / what caused / explain): answered from the index"]


def parse_rules(question: str, cameras: Optional[list] = None, now: Optional[datetime] = None) -> dict:
    """The rule-based Stage A parse. Never raises; what it did not understand is in `notes` and `keywords`."""
    now = now or si.now_local()
    q = _empty_query((question or "").strip()[:MAX_QUESTION_CHARS])
    text = q["question"]
    if not text:
        q["notes"].append("empty question")
        return q
    cams = cameras or []
    low = text.lower()
    # time first: it contains digits that must not be read as camera numbers
    t0, t1, point, time_text, tnotes, work = parse_time(low, now)
    q["notes"] += tnotes
    if t0 is not None:
        q["t0"], q["t1"] = si.to_epoch(t0), si.to_epoch(t1)
        q["point"] = si.to_epoch(point) if point else None
        q["time_text"] = time_text
    ids, unknown, work = camera_mentions(work, cams)
    q["camera_ids"], q["unknown_cameras"] = ids, unknown
    if unknown:
        q["notes"].append("no such camera: " + ", ".join(str(n) for n in unknown))
    plate, work = parse_plate(work)
    q["plate"] = plate
    m = re.search(r"\b(?:vehicle|gid|global\s*id|id)\s*(?:#|no\.?|number)?\s*(\d{1,7})\b", work)
    if m:
        q["global_id"] = int(m.group(1))
        work = _blank(work, m)
    # behaviours
    for pat, codes, behaviours, _ in BEHAVIOUR_RULES:
        m = re.search(pat, work)
        if m:
            for c in codes:
                if c not in q["codes"]:
                    q["codes"].append(c)
            for b in behaviours:
                if b not in q["behaviours"]:
                    q["behaviours"].append(b)
            work = _blank(work, m)
    if "V1" in q["codes"] and not re.search(r"\bbus", work) and "Bus" not in q["classes"]:
        q["classes"].append("Bus")                     # 'bus blocking' is about buses
    # classes (longest phrase first), then colours
    for phrase, classes in _class_phrases():
        m = re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", work)
        if m:
            for c in classes:
                if c not in q["classes"]:
                    q["classes"].append(c)
            if len(classes) > 1:
                q["notes"].append(f"'{phrase}' can mean {' or '.join(classes)}: both are searched")
            work = _blank(work, m)
    for colour, words in si.COLOUR_WORDS.items():
        for w in words:
            m = re.search(r"(?<![a-z])" + w + r"(?![a-z])", work)
            if m:
                if colour not in q["colours"]:
                    q["colours"].append(colour)
                work = _blank(work, m)
    if q["colours"]:
        q["notes"].append("colour is estimated from the stored body colour and is approximate")
    # speed thresholds
    m = SPEED_RE.search(work)
    if m:
        q["min_kph"] = float(m.group(1) or m.group(2))
        work = _blank(work, m)
    m = SLOW_RE.search(work)
    if m:
        q["max_kph"] = float(m.group(1))
        work = _blank(work, m)
    if "ANY" in q["codes"]:
        q["codes"] = [c for c in q["codes"] if c != "ANY"] or ["ANY"]     # a specific code makes "any" redundant
    for pat, d in DIRECTION_RULES:
        m = re.search(pat, work)
        if m:
            q["direction"] = d
            work = _blank(work, m)
            break
    for pat, a in AGG_RULES:
        if re.search(pat, low):
            q["agg"] = a
            break
    q["intent"], q["intent_reasons"] = classify_intent(text)
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", work)
    q["keywords"] = [w for w in dict.fromkeys(words) if w not in STOP_WORDS]
    if q["intent"] == "causal" and "congestion" in q["behaviours"] and not q["classes"]:
        q["keywords"] = [k for k in q["keywords"] if k not in ("back", "backup")]
    return q


# ---- validation (used for LLM output and for edits made by the operator)

QUERY_FIELDS = ("classes", "colours", "codes", "behaviours", "camera_ids", "t0", "t1", "point", "plate", "direction",
                "global_id", "min_kph", "max_kph", "agg", "intent", "keywords")
BEHAVIOURS = ("stopped", "slow", "congestion")
AGGS = ("list", "count", "speed", "when")


def _epoch_from(value, now: datetime) -> Optional[float]:
    """Epoch seconds from an ISO local date-time or a number; None when it is not plausible."""
    t = None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        t = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            t = si.to_epoch(datetime.fromisoformat(value.strip().replace("Z", "").replace("T", " ")[:19]))
        except ValueError:
            try:
                t = float(value)
            except ValueError:
                return None
    if t is None or not math.isfinite(t):
        return None
    ref = si.to_epoch(now)
    if not (ref - MAX_HOURS_AGO <= t <= ref + 86400):
        return None
    return t


def validate_fields(obj, cameras: list, now: datetime, *, text: str = "", trusted: bool = False) -> tuple:
    """Check a structured query from outside (an LLM, or the operator's edit of the parsed query).
    -> (clean dict containing only fields that passed, [rejection messages]).
    trusted=True is the operator's own edit: vocabulary and ranges are still enforced, but a camera, time or
    plate need not be spelled in the question text. An LLM is not trusted."""
    clean, bad = {}, []
    if not isinstance(obj, dict):
        return clean, ["not a JSON object"]
    cam_ids = {c["id"] for c in cameras}
    low = text.lower()

    def listof(key, vocab, norm=lambda s: s):
        v = obj.get(key)
        if v in (None, "", []):
            return
        if not isinstance(v, list):
            v = [v]
        ok = [norm(x) for x in v if isinstance(x, str) and norm(x) in vocab]
        lost = [x for x in v if not (isinstance(x, str) and norm(x) in vocab)]
        if ok:
            clean[key] = list(dict.fromkeys(ok))
        if lost:
            bad.append(f"{key}: ignored {lost!r} (not in the vocabulary)")

    cls_lookup = {c.lower(): c for c in si.CLASS_WORDS}
    v = obj.get("classes")
    if v not in (None, "", []):
        v = v if isinstance(v, list) else [v]
        ok = [cls_lookup[x.lower()] for x in v if isinstance(x, str) and x.lower() in cls_lookup]
        if ok:
            clean["classes"] = list(dict.fromkeys(ok))
        if len(ok) != len(v):
            bad.append("classes: ignored values that are not detector classes")
    listof("colours", set(si.COLOUR_WORDS), lambda s: s.lower())
    listof("codes", set(si.CODE_WORDS) | {"ANY"}, lambda s: s.upper())
    listof("behaviours", set(BEHAVIOURS), lambda s: s.lower())
    v = obj.get("camera_ids")
    if v not in (None, "", []):
        v = v if isinstance(v, list) else [v]
        ints = [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
        known = [x for x in ints if x in cam_ids]
        if len(known) != len(v):
            bad.append("camera_ids: ignored ids that are not cameras")
        if not trusted:
            found, _, _ = camera_mentions(text, cameras)
            confirmed = [x for x in known if x in found]
            if len(confirmed) != len(known):
                bad.append("camera_ids: ignored cameras the question does not name")
            known = confirmed
        if known:
            clean["camera_ids"] = sorted(set(known))
    t0, t1 = _epoch_from(obj.get("t0"), now), _epoch_from(obj.get("t1"), now)
    if (obj.get("t0") not in (None, "") or obj.get("t1") not in (None, "")):
        cue = trusted or bool(re.search(r"\d|yesterday|today|tonight|morning|afternoon|evening|night|ago|last|past|"
                                        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|rush", low))
        if t0 is None or t1 is None or t1 < t0 or t1 - t0 > MAX_SPAN_S:
            bad.append("t0/t1: ignored (not a plausible, ordered, at most 31-day window)")
        elif not cue:
            bad.append("t0/t1: ignored (the question contains no time expression)")
        else:
            clean["t0"], clean["t1"] = t0, t1
            pt = _epoch_from(obj.get("point"), now)
            if pt is not None and t0 <= pt <= t1:
                clean["point"] = pt
    p = obj.get("plate")
    if p not in (None, ""):
        if not isinstance(p, str) or not re.fullmatch(r"[A-Za-z0-9\- ]{2,24}", p.strip()):
            bad.append("plate: ignored (not a plate fragment)")
        elif not trusted and re.sub(r"[\s-]", "", p).lower() not in re.sub(r"[\s-]", "", low):
            bad.append("plate: ignored (does not appear in the question)")
        else:
            clean["plate"] = p.strip().upper()
    if obj.get("direction") in ("incoming", "outgoing"):
        clean["direction"] = obj["direction"]
    elif obj.get("direction") not in (None, "", "unknown"):
        bad.append("direction: ignored")
    g = obj.get("global_id")
    if g not in (None, ""):
        if isinstance(g, int) and not isinstance(g, bool) and 0 < g < 10 ** 8 and (trusted or re.search(rf"\b{g}\b", low)):
            clean["global_id"] = g
        else:
            bad.append("global_id: ignored")
    for key, hi in (("min_kph", 250), ("max_kph", 250)):
        x = obj.get(key)
        if x not in (None, ""):
            if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= hi:
                clean[key] = float(x)
            else:
                bad.append(f"{key}: ignored")
    if obj.get("agg") in AGGS:
        clean["agg"] = obj["agg"]
    if obj.get("intent") in ("descriptive", "causal"):
        clean["intent"] = obj["intent"]
    kw = obj.get("keywords")
    if isinstance(kw, list):
        clean["keywords"] = [k.lower() for k in kw if isinstance(k, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{1,30}", k)][:10]
    return clean, bad


def extract_json(text) -> Optional[dict]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    for cand in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except ValueError:
            continue
    return None


def parse(question: str, cameras: Optional[list] = None, now: Optional[datetime] = None,
          llm_fn: Optional[Callable] = None) -> dict:
    """Rule parse, optionally refined by `llm_fn`. The rule parse always runs first and is the validator:
    an LLM value only fills a field the rules left empty, and only after validate_fields accepted it. If the
    rules and the LLM disagree about the time or the cameras the rules win (listed in `llm_rejected`)."""
    now = now or si.now_local()
    q = parse_rules(question, cameras, now)
    q["llm_used"], q["llm_rejected"] = False, []
    if llm_fn is None or not q["question"]:
        return q
    ctx = {"now": now.isoformat(timespec="minutes"), "timezone": si.DHAKA,
           "cameras": [{"id": c["id"], "name": c["name"]} for c in (cameras or [])],
           "classes": list(si.CLASS_WORDS), "colours": list(si.COLOUR_WORDS), "codes": si.CODE_WORDS,
           "behaviours": list(BEHAVIOURS)}
    try:
        raw = llm_fn(q["question"], ctx)
    except Exception as e:                  # noqa: BLE001
        q["llm_rejected"].append(f"LLM parse failed ({type(e).__name__}); rule parse used")
        return q
    obj = extract_json(raw)
    if obj is None:
        q["llm_rejected"].append("LLM reply was not a JSON object; rule parse used")
        return q
    clean, bad = validate_fields(obj, cameras or [], now, text=q["question"])
    q["llm_rejected"] += bad
    used = []
    for key in ("classes", "colours", "codes", "behaviours", "camera_ids", "keywords"):
        if key in clean and not q[key]:
            q[key] = clean[key]
            used.append(key)
        elif key in clean and clean[key] != q[key] and key in ("camera_ids",):
            q["llm_rejected"].append("camera_ids: the rules read the question differently; rules kept")
    if "t0" in clean:
        if q["t0"] is None:
            q["t0"], q["t1"], q["point"] = clean["t0"], clean["t1"], clean.get("point")
            used.append("time")
        elif abs(q["t0"] - clean["t0"]) > 1 or abs(q["t1"] - clean["t1"]) > 1:
            q["llm_rejected"].append("time: the rules read the question differently; rules kept")
    for key in ("plate", "direction", "global_id", "min_kph", "max_kph"):
        if key in clean and q[key] is None:
            q[key] = clean[key]
            used.append(key)
    if "agg" in clean and q["agg"] == "list":
        q["agg"] = clean["agg"]
    # the LLM may not turn a question the rules call causal into a descriptive one (or the reverse): the router is rule-based
    if "intent" in clean and clean["intent"] != q["intent"]:
        q["llm_rejected"].append("intent: the router is rule-based; kept " + q["intent"])
    q["llm_used"], q["source"] = bool(used), ("rules+llm" if used else "rules")
    return q


def normalise_edited(obj: dict, cameras: list, now: Optional[datetime] = None) -> dict:
    """The operator's edit of the parsed query (the UI sends it back). Same checks, trusted source; fields not
    sent are empty. Returns a full query dict."""
    now = now or si.now_local()
    q = _empty_query(str((obj or {}).get("question") or "")[:MAX_QUESTION_CHARS])
    clean, bad = validate_fields(obj or {}, cameras, now, text=q["question"], trusted=True)
    q.update(clean)
    q["notes"] = list(bad) + ["parsed query edited by the operator"]
    q["source"] = "edited"
    q["intent_reasons"] = ["set by the operator"] if "intent" in clean else classify_intent(q["question"])[1]
    if "intent" not in clean:
        q["intent"] = classify_intent(q["question"])[0]
    q["time_text"] = ""
    return q


# ---------------------------------------------------------------- Stage A: retrieval

IDENTITY_KEYS = ("classes", "codes", "plate", "global_id", "direction", "min_kph", "max_kph")


def has_identity(q: dict) -> bool:
    return bool(any(q.get(k) for k in IDENTITY_KEYS) or q.get("colours") or "stopped" in q.get("behaviours", [])
                or "slow" in q.get("behaviours", []))


def _vehicle_filter(q: dict, strict_colour: bool = False) -> tuple:
    where, args = ["1=1"], []
    if q.get("camera_ids"):
        where.append(f"sv.camera_id IN ({','.join('?' * len(q['camera_ids']))})")
        args += q["camera_ids"]
    if q.get("t0") is not None:
        where.append("w.wall=1 AND sv.t>=? AND sv.t<=?")
        args += [q["t0"], q["t1"]]
    if q.get("classes"):
        where.append(f"sv.cls IN ({','.join('?' * len(q['classes']))})")
        args += q["classes"]
    codes = [c for c in q.get("codes", [])]
    if codes:
        if "ANY" in codes:
            where.append("sv.codes != ''")
        else:
            where.append("(" + " OR ".join("sv.codes LIKE ?" for _ in codes) + ")")
            args += [f"% {c} %" for c in codes]
    if q.get("direction"):
        where.append("sv.direction=?")
        args.append(q["direction"])
    if q.get("global_id"):
        where.append("sv.global_id=?")
        args.append(q["global_id"])
    if q.get("plate"):
        where.append("REPLACE(REPLACE(UPPER(sv.plate),' ',''),'-','') LIKE ?")
        args.append("%" + re.sub(r"[\s-]", "", q["plate"].upper()) + "%")
    if "stopped" in q.get("behaviours", []):
        where.append("sv.stopped=1")
    if "slow" in q.get("behaviours", []):
        where.append("sv.kph IS NOT NULL AND sv.kph<10")
    if q.get("min_kph") is not None:
        where.append("sv.kph IS NOT NULL AND sv.kph>=?")
        args.append(q["min_kph"])
    if q.get("max_kph") is not None:
        where.append("sv.kph IS NOT NULL AND sv.kph<=?")
        args.append(q["max_kph"])
    if strict_colour and q.get("colours"):
        where.append(f"sv.colour IN ({','.join('?' * len(q['colours']))})")
        args += q["colours"]
    return " AND ".join(where), args


def _fts_scores(store, keywords: list) -> dict:
    """{seg_vehicle id: 0..1} for rows whose text matches any keyword (FTS5 bm25, else LIKE)."""
    toks = [re.sub(r"[^a-z0-9]", "", k.lower()) for k in keywords]
    toks = [t for t in toks if len(t) >= 3][:8]
    if not toks:
        return {}
    out = {}
    if si.has_fts(store):
        try:
            rows = store.q("SELECT rowid, bm25(seg_fts) AS r FROM seg_fts WHERE seg_fts MATCH ? ORDER BY r LIMIT 5000",
                           (" OR ".join(f'"{t}"' for t in toks),))
            if rows:
                worst = max(abs(r["r"]) for r in rows) or 1.0
                return {r["rowid"]: 0.4 + 0.6 * abs(r["r"]) / worst for r in rows}
            return {}
        except Exception:                                  # a build whose FTS5 table is unusable
            pass
    for t in toks:
        for r in store.q("SELECT id FROM seg_vehicle WHERE LOWER(text) LIKE ? OR LOWER(terms) LIKE ? LIMIT 5000",
                         (f"%{t}%", f"%{t}%")):
            out[r["id"]] = min(1.0, out.get(r["id"], 0.0) + 1.0 / len(toks))
    return out


def _seg_view(store, w: dict, cam_names: dict) -> dict:
    return {"id": w["id"], "camera_id": w["camera_id"], "camera": cam_names.get(w["camera_id"], f"camera {w['camera_id']}"),
            "t_start": w["w_start"], "t_end": w["w_end"], "wall": bool(w["wall"]),
            "start": si.day_clock(w["w_start"]), "end": si.clock(w["w_end"]),
            "video_t0": w["video_t0"], "video_t1": w["video_t1"], "run_id": w["run_id"],
            "n_pass": w["n_pass"], "counts": si._loads(w["counts"], {}), "mean_kph": w["mean_kph"],
            "n_stopped": w["n_stopped"], "events": si._loads(w["events"], {}), "state": si._loads(w["state"], {}),
            "impact_max": w["impact_max"], "summary": w["summary"]}


def _scene_score(w: dict, q: dict) -> tuple:
    """How much an ordinary window looks like what the question is about. -> (score, reasons)."""
    reasons, s = [], 0.0
    ev, stopped, n = w["n_events"], w["n_stopped"], w["n_pass"]
    mean = w["mean_kph"]
    if q.get("point") is not None:
        mid = (w["w_start"] + w["w_end"]) / 2
        d = abs(mid - q["point"])
        prox = math.exp(-d / 150.0)
        s += 0.30 * prox
        if d <= 150:
            reasons.append(f"within {int(d)} s of {si.clock(q['point'])}")
    elif q.get("t0") is not None:
        s += 0.05
    if ev:
        s += min(0.30, 0.15 + 0.05 * ev)
        reasons.append("events " + ", ".join(f"{c} x{k}" for c, k in sorted(si._loads(w['events'], {}).items())))
    if stopped:
        s += min(0.25, 0.12 + 0.04 * stopped)
        reasons.append(f"{stopped} vehicle{'s' if stopped != 1 else ''} stopped or crawling")
    if mean is not None and n >= 2 and mean < 12:
        s += 0.20 * (1 - mean / 12)
        reasons.append(f"slow traffic ({mean:.0f} km/h mean over {n} vehicles)")
    if w["impact_max"]:
        s += min(0.20, 0.20 * float(w["impact_max"]))
        reasons.append(f"event impact score {float(w['impact_max']):.2f}")
    if "congestion" in q.get("behaviours", []) and n >= 3:
        s += 0.05
    return s, reasons


def retrieve(store, q: dict, limit: int = DEFAULT_LIMIT) -> dict:
    """Stage A retrieval: structured filter + text match + ranking. -> {"segments": [...], "mode", "total_rows",
    "coverage": [...], "notes": [...]}. Every segment carries its score and the reasons it matched."""
    si.ensure_schema(store)
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    notes = []
    cam_names = {c["id"]: c["name"] for c in store.cameras()}
    cov = si.coverage(store, q.get("camera_ids") or [], q.get("t0"), q.get("t1")) if q.get("t0") is not None else []
    base = {"segments": [], "mode": "none", "total_rows": 0, "coverage": cov, "notes": notes}
    if q.get("unknown_cameras") and not q.get("camera_ids"):
        notes.append("the camera named in the question does not exist")
        return base
    kw_scores = _fts_scores(store, q.get("keywords") or []) if q.get("keywords") else {}
    identity = has_identity(q)
    if not identity and kw_scores and q.get("intent") != "causal":
        identity = True                                    # the text match is the filter
    if identity or (kw_scores and not q.get("behaviours")):
        where, args = _vehicle_filter(q)
        rows = store.q("SELECT sv.*, w.wall AS w_wall FROM seg_vehicle sv JOIN seg_windows w ON w.id=sv.window_id"
                       f" WHERE {where} ORDER BY sv.t LIMIT ?", args + [MAX_ROWS])
        if not identity:
            rows = [r for r in rows if r["id"] in kw_scores]
        base["mode"] = "vehicle"
        base["total_rows"] = len(rows)
        if len(rows) >= MAX_ROWS:
            notes.append(f"more than {MAX_ROWS} matching vehicle passes: the first {MAX_ROWS} by time were ranked")
        by_win = {}
        for r in rows:
            sc, why = 1.0, []
            if q.get("classes"):
                why.append(f"class {r['cls']}")
            if q.get("codes"):
                why.append("violation " + r["codes"].strip().replace(" ", "+"))
            if q.get("plate"):
                why.append(f"plate contains {q['plate']}")
            if q.get("global_id"):
                why.append(f"vehicle #{q['global_id']}")
            if "stopped" in q.get("behaviours", []):
                why.append("stopped or crawling" + (f" ({r['dwell_s']:.0f} s)" if r["dwell_s"] else ""))
            if "slow" in q.get("behaviours", []):
                why.append(f"slow ({r['kph']:.0f} km/h)")
            if q.get("min_kph") is not None or q.get("max_kph") is not None:
                why.append(f"{r['kph']:.0f} km/h")
            if q.get("direction"):
                why.append(f"heading {q['direction']}")
            if q.get("colours"):
                if r["colour"] in q["colours"]:
                    sc += 0.6
                    why.append(f"colour {r['colour']} (estimated)")
                elif r["colour"]:
                    sc -= 0.4
                    why.append(f"colour looks {r['colour']}, not {'/'.join(q['colours'])}")
                else:
                    sc -= 0.1
                    why.append("colour unknown")
            if r["id"] in kw_scores:
                sc += 0.4 * kw_scores[r["id"]]
                why.append("text match: " + ", ".join(q["keywords"][:4]))
            if q.get("point") is not None:
                sc += 0.2 * math.exp(-abs(r["t"] - q["point"]) / 150.0)
            if r["impact"]:
                sc += min(0.2, 0.2 * r["impact"])
            by_win.setdefault(r["window_id"], []).append((sc, r, why))
        scored = []
        for wid, items in by_win.items():
            items.sort(key=lambda x: -x[0])
            top = items[0][0] + 0.05 * math.log1p(len(items) - 1)
            scored.append((top, wid, items))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for top, wid, items in scored[:limit]:
            w = store.q("SELECT * FROM seg_windows WHERE id=?", (wid,))[0]
            seg = _seg_view(store, w, cam_names)
            seg["score"] = round(top, 3)
            seg["reasons"] = list(dict.fromkeys(items[0][2]))
            seg["matches"] = [{"id": r["id"], "text": r["text"], "pkey": r["pkey"], "global_id": r["global_id"],
                               "violation_id": r["violation_id"], "cls": r["cls"], "colour": r["colour"],
                               "kph": r["kph"], "t": r["t"], "score": round(sc, 3), "codes": r["codes"].split(),
                               "plate": r["plate"], "stopped": bool(r["stopped"]), "kind": r["kind"],
                               "clip": r["clip"], "snapshot": r["snapshot"]} for sc, r, _ in items[:8]]
            seg["n_matches"] = len(items)
            base["segments"].append(seg)
        return base
    # scene mode: nothing about a particular vehicle, so rank windows by how much is going on there
    base["mode"] = "scene"
    where, args = ["1=1"], []
    if q.get("camera_ids"):
        where.append(f"camera_id IN ({','.join('?' * len(q['camera_ids']))})")
        args += q["camera_ids"]
    if q.get("t0") is not None:
        where.append("wall=1 AND w_start>=? AND w_start<=?")
        args += [q["t0"] - 60, q["t1"]]
    order = "ABS(w_start-?)" if q.get("point") is not None else "(n_events*3+n_stopped*2+impact_max) DESC"
    if q.get("point") is not None:
        args2 = args + [q["point"], SCENE_WINDOWS]
    else:
        args2 = args + [SCENE_WINDOWS]
    wins = store.q(f"SELECT * FROM seg_windows WHERE {' AND '.join(where)} ORDER BY {order} LIMIT ?", args2)
    base["total_rows"] = len(wins)
    scored = []
    for w in wins:
        s, why = _scene_score(w, q)
        if s > 0:
            scored.append((s, w, why))
    scored.sort(key=lambda x: (-x[0], x[1]["w_start"]))
    for s, w, why in scored[:limit]:
        seg = _seg_view(store, w, cam_names)
        seg["score"] = round(s, 3)
        seg["reasons"] = why or ["in the requested time and place"]
        vs = store.q("SELECT * FROM seg_vehicle WHERE window_id=? ORDER BY stopped DESC, t LIMIT 8", (w["id"],))
        seg["matches"] = [{"id": r["id"], "text": r["text"], "pkey": r["pkey"], "global_id": r["global_id"],
                           "violation_id": r["violation_id"], "cls": r["cls"], "colour": r["colour"], "kph": r["kph"],
                           "t": r["t"], "score": 0.0, "codes": r["codes"].split(), "plate": r["plate"],
                           "stopped": bool(r["stopped"]), "kind": r["kind"], "clip": r["clip"],
                           "snapshot": r["snapshot"]} for r in vs]
        seg["n_matches"] = len(vs)
        base["segments"].append(seg)
    return base


# ---------------------------------------------------------------- frames, on demand

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def _under(root: Path, rel: str) -> Optional[Path]:
    """rel resolved inside root, or None (no traversal, no symlink out)."""
    if not rel:
        return None
    try:
        p = (root / rel).resolve()
        p.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None


def _jpeg(img, max_side: int = 640, quality: int = 80) -> Optional[bytes]:
    import cv2
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _video_path(cam: dict, data_dir: Optional[Path]) -> Optional[Path]:
    if not cam or cam.get("kind") != "file":
        return None
    src = Path(str(cam.get("source") or ""))
    for cand in (src, (Path(data_dir) / "videos" / src.name) if data_dir else None):
        if cand and cand.is_file():
            return cand
    return None


def pull_frames(store, window_id: int, n: int = CAUSAL_FRAMES_PER_SEGMENT, data_dir=None, max_side: int = 640) -> dict:
    """Frames for one segment, read from the stored video now (the index holds none).
    -> {"frames": [{"label", "t", "jpeg"}], "source": "video" | "clip" | "snapshot" | "gone", "note": str}.
    Order of preference: the camera's source video at the window's offsets; the evidence clip or snapshot of an
    event in the window; otherwise "gone" (the caller must say the raw video is no longer available)."""
    import cv2
    w = si.window(store, window_id)
    if w is None:
        return {"frames": [], "source": "gone", "note": "no such segment"}
    n = max(1, min(int(n), 8))
    cam = store.camera(w["camera_id"])
    vp = _video_path(cam, data_dir)
    if vp is not None and w["video_t0"] is not None:
        cap = cv2.VideoCapture(str(vp))
        try:
            if cap.isOpened():
                frames = []
                for i in range(n):
                    t = w["video_t0"] + (w["video_t1"] - w["video_t0"]) * (i + 0.5) / n
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok, img = cap.read()
                    if ok:
                        jp = _jpeg(img, max_side)
                        if jp:
                            frames.append({"label": f"segment {window_id} at video {t:.1f} s", "t": round(t, 2), "jpeg": jp})
                if frames:
                    return {"frames": frames, "source": "video", "note": ""}
        finally:
            cap.release()
    root = Path(data_dir) if data_dir else None
    if root is not None:
        for v in si.vehicles_in(store, window_id):
            clip = _under(root, v["clip"] or "")
            if clip is not None:
                cap = cv2.VideoCapture(str(clip))
                try:
                    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    frames = []
                    if cap.isOpened() and total > 0:
                        for i in range(n):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
                            ok, img = cap.read()
                            jp = _jpeg(img, max_side) if ok else None
                            if jp:
                                frames.append({"label": f"segment {window_id}, evidence clip frame {i + 1}", "t": None, "jpeg": jp})
                    if frames:
                        return {"frames": frames, "source": "clip",
                                "note": "the raw video is not available; frames come from the evidence clip"}
                finally:
                    cap.release()
            snap = _under(root, v["snapshot"] or "")
            if snap is not None and snap.suffix.lower() in IMAGE_EXT:
                img = cv2.imread(str(snap))
                jp = _jpeg(img, max_side) if img is not None else None
                if jp:
                    return {"frames": [{"label": f"segment {window_id}, evidence snapshot", "t": None, "jpeg": jp}],
                            "source": "snapshot", "note": "the raw video is not available; only the evidence snapshot"}
    return {"frames": [], "source": "gone",
            "note": "the stored video for this segment is no longer available" if vp is None
                    else "the video could not be read at this time"}


# ---------------------------------------------------------------- Stage B: descriptive answers

def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def describe_filter(q: dict, cam_names: dict) -> str:
    bits = []
    if q.get("colours"):
        bits.append("/".join(q["colours"]))
    bits.append("/".join(si.class_label(c) for c in q["classes"]) if q.get("classes") else "vehicle")
    if "stopped" in q.get("behaviours", []):
        bits.append("stopped or crawling")
    if q.get("codes"):
        bits.append("flagged " + ("with any violation" if "ANY" in q["codes"] else "/".join(q["codes"])))
    if q.get("camera_ids"):
        bits.append("at " + ", ".join(cam_names.get(c, f"camera {c}") for c in q["camera_ids"]))
    if q.get("t0") is not None:
        bits.append(f"between {si.day_clock(q['t0'])} and {si.day_clock(q['t1'])}")
    return " ".join(bits)


def answer_descriptive(store, q: dict) -> dict:
    """Computed from the index rows only. -> {"mode": "index" | "insufficient", "text", "value", "breakdown",
    "rows": [...supporting seg_vehicle rows...], "caveats": [...], "filter": str}"""
    cam_names = {c["id"]: c["name"] for c in store.cameras()}
    where, args = _vehicle_filter(q, strict_colour=True)
    rows = store.q("SELECT sv.* FROM seg_vehicle sv JOIN seg_windows w ON w.id=sv.window_id WHERE " + where +
                   " ORDER BY sv.t", args)
    desc = describe_filter(q, cam_names)
    caveats = ["counts are vehicles the tracker saw cross a camera's reference line; tracking misses and duplicates "
               "are possible"]
    if q.get("colours"):
        caveats.append("colour is estimated from the stored body colour; vehicles with no usable colour are not counted")
    if "stopped" in q.get("behaviours", []):
        caveats.append("'stopped' means a mean track speed under 1 m/s or a stationary-vehicle rule (V1, V3, V7); "
                       "the tracker stores no exact stop times")
    if q.get("t0") is not None:
        cov = si.coverage(store, q.get("camera_ids") or [], q["t0"], q["t1"])
        if not cov:
            return {"mode": "insufficient", "intent": "descriptive", "filter": desc, "rows": [], "value": None,
                    "text": "Insufficient evidence: no analysed video covers " + desc.split(" between ")[-1]
                            + (" on the requested camera" if q.get("camera_ids") else "")
                            + ". A count of zero would be a guess, so none is given.",
                    "caveats": caveats}
        lo, hi = min(c["t0"] for c in cov), max(c["t1"] for c in cov)
        if lo > q["t0"] + 1 or hi < q["t1"] - 1:
            caveats.append(f"analysed video only covers {si.day_clock(max(lo, q['t0']))} to {si.day_clock(min(hi, q['t1']))} "
                           "of the requested window")
    elif not store.q("SELECT 1 FROM seg_windows LIMIT 1"):
        return {"mode": "insufficient", "intent": "descriptive", "filter": desc, "rows": [], "value": None,
                "text": "Insufficient evidence: the index is empty (no video has been analysed yet).", "caveats": caveats}
    seen, passes = set(), []
    for r in rows:
        if r["pkey"] not in seen:
            seen.add(r["pkey"])
            passes.append(r)
    n = len(passes)
    by_cls, by_cam = {}, {}
    for r in passes:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
        by_cam[cam_names.get(r["camera_id"], str(r["camera_id"]))] = by_cam.get(cam_names.get(r["camera_id"], str(r["camera_id"])), 0) + 1
    out = {"mode": "index", "intent": "descriptive", "filter": desc, "value": n, "breakdown": {"by_class": by_cls, "by_camera": by_cam},
           "rows": [{k: r[k] for k in ("id", "window_id", "camera_id", "text", "cls", "colour", "kph", "t", "global_id",
                                       "violation_id", "codes", "stopped", "pkey")} for r in passes[:50]],
           "n_rows": n, "caveats": caveats}
    agg = q.get("agg")
    kphs = [r["kph"] for r in passes if r["kph"] is not None]
    if n == 0:
        out["text"] = f"0 {desc} in the analysed video."
    elif agg == "speed":
        if kphs:
            out["stats"] = {"mean_kph": round(statistics.fmean(kphs), 1), "min_kph": round(min(kphs), 1),
                            "max_kph": round(max(kphs), 1), "n": len(kphs)}
            out["text"] = (f"{desc}: mean {out['stats']['mean_kph']} km/h (min {out['stats']['min_kph']}, max "
                           f"{out['stats']['max_kph']}) over {_plural(len(kphs), 'vehicle')} with a measured speed.")
        else:
            out["text"] = f"{_plural(n, 'vehicle')} matched ({desc}) but none has a measured speed."
    elif agg == "when":
        out["text"] = (f"{_plural(n, 'vehicle')} matched ({desc}); first at {si.day_clock(passes[0]['t'])}, "
                       f"last at {si.day_clock(passes[-1]['t'])}.")
    else:
        extra = ""
        if len(by_cls) > 1:
            extra = " (" + ", ".join(f"{k} {si.class_label(c)}" for c, k in sorted(by_cls.items(), key=lambda kv: -kv[1])) + ")"
        out["text"] = f"{_plural(n, 'vehicle')} matched: {desc}{extra}."
    return out


# ---------------------------------------------------------------- Stage B: causal answers

def build_reasoner_segments(store, segs: list, data_dir=None, frames_each: int = CAUSAL_FRAMES_PER_SEGMENT) -> tuple:
    """The payload for reason_fn (with frames) and a light record for the response (no bytes).
    -> (payload list, evidence list)"""
    payload, evidence = [], []
    for s in segs:
        pulled = pull_frames(store, s["id"], frames_each, data_dir)
        facts = [s["summary"]] if s.get("summary") else []
        if s.get("state"):
            facts.append("traffic state around the worst event here: " + ", ".join(f"{k} {v}" for k, v in s["state"].items()))
        facts += [m["text"] for m in s.get("matches", [])[:6]]
        payload.append({"id": s["id"], "camera": s["camera"], "start": s["start"], "end": s["end"], "facts": facts,
                        "frames": pulled["frames"], "frame_source": pulled["source"], "reasons": s.get("reasons", [])})
        evidence.append({"segment_id": s["id"], "camera": s["camera"], "start": s["start"], "frames": len(pulled["frames"]),
                         "frame_source": pulled["source"], "note": pulled["note"]})
    return payload, evidence


def answer_causal(store, q: dict, retrieval: dict, reason_fn: Optional[Callable], data_dir=None,
                  max_segments: int = CAUSAL_SEGMENTS) -> dict:
    """Route a causal question to the reasoner over at most `max_segments` retrieved segments."""
    segs = [s for s in retrieval["segments"] if s["score"] >= WEAK_SCORE][:max(1, min(max_segments, 8))]
    limits = ["the reasoner sees a few frames of each segment, not the whole video",
              "causes it names are what is visible in those frames; a cause outside the view cannot be seen"]
    if not segs:
        why = ("nothing in the index matches the question" if not retrieval["segments"]
               else "the best matching segments are too weak to reason over")
        if q.get("t0") is not None and not retrieval.get("coverage"):
            why = "no analysed video covers the requested time and camera"
        return {"mode": "insufficient", "intent": "causal", "text": f"Insufficient evidence: {why}. No reasoner was asked.",
                "citations": [], "evidence": [], "limits": limits}
    payload, evidence = build_reasoner_segments(store, segs, data_dir)
    if not any(p["frames"] for p in payload):
        notes = "; ".join(sorted({e["note"] for e in evidence if e["note"]})) or "no frames could be read"
        return {"mode": "insufficient", "intent": "causal",
                "text": f"Insufficient evidence: the raw video for the matching segments is not available ({notes}). "
                        "The index facts are listed below, but they cannot show a cause.",
                "citations": [], "evidence": evidence, "facts": {p["id"]: p["facts"] for p in payload}, "limits": limits}
    if reason_fn is None:
        return {"mode": "insufficient", "intent": "causal",
                "text": "The visual reasoner is not available (no VLM backend is running), so no cause is offered. "
                        "The retrieved segments and their index facts are shown.",
                "citations": [], "evidence": evidence, "unavailable": True, "limits": limits}
    t0 = time.time()
    try:
        raw = reason_fn(q["question"], payload)
    except Exception as e:                       # noqa: BLE001
        return {"mode": "insufficient", "intent": "causal", "unavailable": True,
                "text": f"The visual reasoner failed ({type(e).__name__}); no cause is offered.",
                "citations": [], "evidence": evidence, "limits": limits}
    return finish_causal(raw, payload, evidence, limits, time.time() - t0)


def finish_causal(raw, payload: list, evidence: list, limits: list, seconds: float = 0.0) -> dict:
    """Validate the reasoner's reply. An answer that cites no retrieved segment is not an answer."""
    ids = {p["id"] for p in payload}
    if not isinstance(raw, dict):
        return {"mode": "insufficient", "intent": "causal", "text": "The reasoner returned nothing usable; no cause is offered.",
                "citations": [], "evidence": evidence, "limits": limits}
    text = str(raw.get("answer") or "").strip()
    cites = [c for c in (raw.get("citations") or []) if isinstance(c, int) and not isinstance(c, bool) and c in ids]
    dropped = [c for c in (raw.get("citations") or []) if c not in ids]
    conf = raw.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) else None
    lim = list(limits) + ([str(raw["limits"])] if raw.get("limits") else [])
    if dropped:
        lim.append(f"the reasoner cited segment ids that were not retrieved ({dropped}); those were ignored")
    sufficient = raw.get("sufficient", True) is not False
    if not text or not sufficient:
        return {"mode": "insufficient", "intent": "causal", "reasoner_said": text, "confidence": conf,
                "text": "Insufficient evidence: the retrieved frames do not show the cause." + (f" Reasoner: {text}" if text else ""),
                "citations": cites, "evidence": evidence, "limits": lim, "seconds": round(seconds, 1)}
    if not cites:
        return {"mode": "insufficient", "intent": "causal", "reasoner_said": text, "confidence": conf,
                "text": "Insufficient evidence: the reasoner gave an answer but cited none of the retrieved segments, "
                        "so it is not shown as a finding. Reasoner said: " + text,
                "citations": [], "evidence": evidence, "limits": lim, "seconds": round(seconds, 1)}
    return {"mode": "vlm", "intent": "causal", "text": text, "confidence": conf, "citations": cites,
            "evidence": evidence, "limits": lim, "seconds": round(seconds, 1)}


# ---------------------------------------------------------------- the whole question

def answer_question(store, question: str, *, now: Optional[datetime] = None, llm_fn: Optional[Callable] = None,
                    reason_fn: Optional[Callable] = None, data_dir=None, parsed: Optional[dict] = None,
                    limit: int = DEFAULT_LIMIT, max_segments: int = CAUSAL_SEGMENTS, intent: Optional[str] = None) -> dict:
    """Parse (or take the operator's edited parse), retrieve, route, answer. `intent` forces the route."""
    t0 = time.time()
    si.ensure_schema(store)
    now = now or si.now_local()
    cams = [{"id": c["id"], "name": c["name"], "location": c.get("location", "")} for c in store.cameras()]
    q = normalise_edited(dict(parsed, question=parsed.get("question") or question), cams, now) if parsed else parse(question, cams, now, llm_fn)
    if intent in ("descriptive", "causal") and q["intent"] != intent:
        q["intent"], q["intent_reasons"] = intent, ["set by the operator"]
    if not q["question"]:
        return {"question": "", "parsed": q, "route": {"intent": None, "why": ["empty question"]}, "segments": [],
                "answer": {"mode": "insufficient", "text": "Insufficient evidence: no question was asked."}, "seconds": 0}
    retrieval = retrieve(store, q, limit)
    if q["intent"] == "causal":
        answer = answer_causal(store, q, retrieval, reason_fn, data_dir, max_segments)
        cited = set(answer.get("citations") or [])
        used = {e["segment_id"] for e in answer.get("evidence", [])}
        for s in retrieval["segments"]:
            s["given_to_reasoner"] = s["id"] in used
            s["cited"] = s["id"] in cited
    else:
        answer = answer_descriptive(store, q)
    for k in ("t0", "t1", "point"):
        q[k + "_local"] = si.from_epoch(q[k]).isoformat(timespec="minutes") if q.get(k) is not None else ""
    return {"question": q["question"], "parsed": q,
            "route": {"intent": q["intent"], "why": q["intent_reasons"],
                      "stage_b": "VLM reasoner over retrieved segments" if q["intent"] == "causal" else "computed from the index"},
            "retrieval": {"mode": retrieval["mode"], "total_rows": retrieval["total_rows"], "notes": retrieval["notes"],
                          "coverage": retrieval["coverage"]},
            "segments": retrieval["segments"], "answer": answer, "seconds": round(time.time() - t0, 3)}


# ---------------------------------------------------------------- adapters to the VLM backends in vlm.py

REASON_SYSTEM = (
    "You analyse short video segments from fixed traffic cameras in Dhaka to answer an operator's question. "
    "Use ONLY the frames and the listed facts. Facts are measured by the system; frames are what you can see. "
    "If the frames do not show what the question asks about, set sufficient=false and say what is missing. "
    "Never guess a cause. Cite every segment your answer rests on by its id. Reply with one JSON object.")
REASON_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["answer", "sufficient", "confidence", "citations", "limits"],
                 "properties": {"answer": {"type": "string"}, "sufficient": {"type": "boolean"},
                                "confidence": {"type": "number"}, "citations": {"type": "array", "items": {"type": "integer"}},
                                "limits": {"type": "string"}}}
PARSE_SYSTEM = (
    "You convert a traffic question into a JSON object. Fields (omit what the question does not state): "
    "classes (from the list), colours, codes (rule codes V1..V11), behaviours (stopped, slow, congestion), "
    "camera_ids (only ids from the list, only if the question names that camera), t0 and t1 (ISO local date-times "
    "in Asia/Dhaka, only if the question states a time), plate (only text that appears in the question), "
    "direction (incoming/outgoing), agg (list/count/speed/when). Do not invent anything. Reply with JSON only.")


def make_reason_fn(backend, max_tokens: int = 500, max_frames: int = 10) -> Callable:
    """reason_fn from a vlm.py backend (anything with ask(system, user_text, images, max_tokens, schema))."""
    def reason_fn(question: str, segments: list) -> dict:
        images, lines = [], [f"Question: {question}", "", "Segments:"]
        for s in segments:
            lines.append(f"[segment {s['id']}] {s['camera']}, {s['start']} to {s['end']}")
            lines += [f"  - {f}" for f in s.get("facts", [])[:8]]
            for fr in s["frames"]:
                if len(images) < max_frames:
                    images.append((fr["label"], fr["jpeg"]))
                    lines.append(f"  image {len(images)}: {fr['label']}")
        reply = backend.ask(REASON_SYSTEM, "\n".join(lines), images, max_tokens=max_tokens, schema=REASON_SCHEMA)
        obj = extract_json(reply)
        if obj is None:
            return {"answer": "", "sufficient": False, "confidence": 0, "citations": [], "limits": "unreadable reply"}
        return obj
    return reason_fn


def make_llm_fn(backend, max_tokens: int = 300) -> Callable:
    def llm_fn(question: str, context: dict) -> str:
        return backend.ask(PARSE_SYSTEM + "\nContext: " + json.dumps(context), question, [], max_tokens=max_tokens)
    return llm_fn


# ---------------------------------------------------------------- X7: grounding metrics

def grounding_metrics(ranked: list, gold: list, ks=(1, 3, 5, 10)) -> dict:
    """MRR and Recall@k of retrieval.
    ranked: per query, a list of segments in rank order, each the set/list of vehicle-pass keys it matched.
    gold:   per query, the set of vehicle-pass keys that are truly relevant.
    A segment is a hit when it holds a gold pass. Reciprocal rank = 1/rank of the first hit (0 when none).
    Recall@k = share of the gold passes found in the top k segments (queries with no gold are skipped)."""
    rr, rec, hit, n = [], {k: [] for k in ks}, {k: [] for k in ks}, 0
    for segs, g in zip(ranked, gold):
        g = set(g)
        if not g:
            continue
        n += 1
        first = next((i for i, s in enumerate(segs, 1) if g & set(s)), None)
        rr.append(1.0 / first if first else 0.0)
        for k in ks:
            found = set().union(*[set(s) for s in segs[:k]]) & g if segs[:k] else set()
            rec[k].append(len(found) / len(g))
            hit[k].append(1.0 if first and first <= k else 0.0)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0                    # noqa: E731
    return {"n_queries": n, "mrr": round(mean(rr), 4), "recall_at": {k: round(mean(rec[k]), 4) for k in ks},
            "hit_at": {k: round(mean(hit[k]), 4) for k in ks}}


def evaluate_queries(store, cases: list, now: Optional[datetime] = None, ks=(1, 3, 5, 10), limit: int = 20) -> dict:
    """Run labelled questions through Stage A and score them.
    cases: [{"id", "question", "gold": [pass keys "camera:run:track"]}]. Uses the rule parser only (no LLM)."""
    cams = [{"id": c["id"], "name": c["name"], "location": c.get("location", "")} for c in store.cameras()]
    ranked, gold, per = [], [], []
    for c in cases:
        q = parse_rules(c["question"], cams, now)
        r = retrieve(store, q, limit)
        segs = [[m["pkey"] for m in s["matches"]] for s in r["segments"]]
        ranked.append(segs)
        gold.append(c["gold"])
        m = grounding_metrics([segs], [c["gold"]], ks)
        per.append({"id": c.get("id"), "mrr": m["mrr"], "n_gold": len(set(c["gold"])), "n_segments": len(segs)})
    out = grounding_metrics(ranked, gold, ks)
    out["per_query"] = per
    return out
