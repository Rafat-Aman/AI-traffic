# Natural-language query panel (WP3: R-3, R-4, U8, U9, X7)

Report section 3.1.5 and the abstract promise a query panel. This is it, built the way the 499A slides handle long
video: **retrieve, then reason** (VideoRAG), with analytics **computed once and stored** (Spatialyze).

```
question ─► Stage A  parse (rules; optional LLM, validated) ─► structured query ─► ranked 10 s segments
                                                                    (persistent index, no video read)
         ─► Stage B  router (rules)
                      descriptive ─► answer computed from the index rows (the rows are the evidence)
                      causal      ─► frames of the top few segments pulled from stored video ON DEMAND
                                     ─► reason_fn(question, segments) ─► answer that must cite segments
                      anything weak / missing / uncited ─► "insufficient evidence"
```

## Files
| File | Role |
|---|---|
| `surveillance/segment_index.py` | the index: schema, build (per run, per camera, incremental), status, worker hook |
| `surveillance/query.py` | parser, validator, retrieval, descriptive and causal answers, frame pulling, VLM adapters, X7 metrics |
| `surveillance/api_query.py` | router: `/query`, `/api/query`, segments, frames, index rebuild and status |
| `web/query.html`, `query.js`, `query.css` | the page |
| `tests/test_query.py`, `tests/data/queries.jsonl` | 81 tests; synthetic labelled queries for X7 |

## The index
Built only from rows the pipeline already wrote: `tracklets`, `violations`, `event_impact`, `runs`, `cameras`
(and `camera_geo` for compass words). It runs no detector and reads no pixels.

* **Window = 10 s**, aligned to multiples of 10 s on the camera wall clock. It equals the evidence clip (5 s + 5 s), is
  about one vehicle pass at Dhaka speeds, gives a VLM 3 frames per segment that cover it, and costs 360 rows per
  camera-hour. Stored in `seg_meta`; changing it needs a rebuild.
* `seg_windows`: per (camera, run, window) counts per class, crossings toward / away from the camera, mean and min
  speed, stopped vehicles, violation codes, traffic-state numbers around the worst event (queue, speed, ...), impact
  score, source-video offsets (`video_t0`, `video_t1`).
* `seg_vehicle`: per vehicle per window a **motion sentence**, e.g. *"red Bus crossed Mirpur Road heading
  southbound at 29 km/h at 17:00:03; flagged wrong-side driving (V4); vehicle #11, also seen at Karwan Bazar"*,
  plus global id, colour, class, direction, codes, plate, clip and snapshot paths.
  A violation whose tracklet falls in the same window is merged into that vehicle's row; otherwise it is its own
  `event` row (e.g. a V1 case with no tracklet).
* `seg_fts`: FTS5 over the sentences (checked: the SQLite here, 3.37, has it). Without FTS5 the search falls back to
  `LIKE`.
* `seg_runs`: what time spans were analysed, so "no video covers that hour" is not reported as "0 buses".
* `build_for_run` is a full idempotent rebuild of the run. `build_for_camera(cid, since=None)` is incremental by row id
  plus the last 20 minutes of data (so late global-id linking and impact scores arrive); with `since` it recomputes from
  that time. Rebuilding twice leaves the index identical (tested).

What the data can and cannot say, because tracking decides the quality:
* A tracklet is one row per vehicle per camera (the line-crossing moment, mean track speed, direction). There are no
  enter / exit times, so "entries / exits" are crossings toward / away from the camera and "stopped" means mean speed
  under 1 m/s or a stationary rule (V1, V3, V7, with its measured dwell). "Stopped 14 s in the bus-stop zone" appears
  only when a rule measured the dwell.
* Colour is estimated from the stored CIELAB body colour. For two-wheelers it is the rider's. It is a soft ranking
  signal in retrieval and a hard filter only in a descriptive count that names a colour (the answer says so).
* Compass words ("northbound") appear only when the camera has a heading in `camera_geo`; otherwise "toward / away
  from the camera".
* Windows exist only where something was seen; absence of a row means no tracked vehicle, not necessarily no vehicle.

## Stage A: the parser
Rule-based, Dhaka time (`Asia/Dhaka`, the same clock as camera start times). Covered: *yesterday, today, last night,
this morning/afternoon/evening, rush hour, weekday names, "on 25 October", ISO dates, "last 10 minutes", "past 2
hours", "an hour ago", "between 5 and 6 pm", "from 17:15 to 18:45", "9:30 am to 10", "at 5:40"* (a point, plus or
minus 2 minutes); the 16 detector classes with local names (CNG, auto-rickshaw, easy-bike, leguna / human hauler,
bhotbhoti, power tiller, rickshaw, van, ...) and ambiguous words ("bike" searches Motorbike and Bicycle and says so);
9 colours; behaviours mapped to rule codes (wrong way V4, speeding V2, no helmet V6, parked V3, blocking V7, bus lane
V9, lane line V5, banned V8, erratic V10, near-collision V11, bus outside stop V1, "any violation") and to motion terms
(stopped, slow, congestion, speed thresholds, toward / away); camera by number, name or location; plate fragments;
vehicle ids. Ambiguity is never silent: "5 and 6" with no am/pm is guessed (1-6 pm, 7-11 am) and listed in `notes`;
an unknown camera is reported, not ignored.

An optional `llm_fn` may re-read the question. Its JSON is **validated field by field**: classes, colours, codes and
behaviours must be in the vocabulary; a camera id must exist **and** be named in the question; a time window must be
plausible (ordered, at most 31 days, within the last 400 days) **and** the question must contain a time cue; a plate
must appear in the question. It can only fill fields the rules left empty; when it disagrees about camera or time the
rules win; it cannot change the router. Everything it dropped is shown (`llm_rejected`).

The operator sees the parse as chips, can edit every field and re-run (the edit is validated the same way, trusting
camera and time because the operator chose them).

## Stage B: routing and answers
The router is rules, with the reasons returned: any causal cue (*why, what caused, what happened, explain, because,
due to, how come*) makes the question causal, because a count does not answer "why"; otherwise descriptive. The
operator can force either.

* **Descriptive**: SQL over `seg_vehicle` with the structured filter; distinct vehicle passes are counted; the
  supporting rows, a breakdown and caveats are returned. Zero is answered as zero only when an analysed run covers the
  window; otherwise the answer is *insufficient evidence*. Partial coverage is stated.
* **Causal**: retrieval in scene mode ranks windows by proximity to the named time, events, stopped vehicles, slow
  traffic and impact score. Segments scoring under 0.30 are not reasoned over. The best 4 get up to 3 frames each,
  read from the camera's source video at the window offsets (JPEG, 640 px). If the raw video is gone the evidence
  clip or snapshot of an event in that window is used; if there is none the answer says the raw video is not
  available. The reasoner's reply must cite retrieved segment ids; an answer that cites none, a reply saying the
  frames do not show it, a crash, or no reasoner at all, all give *insufficient evidence* (with the retrieved segments
  still shown).

## Adapters to the VLM module
`query.make_reason_fn(backend)` and `query.make_llm_fn(backend)` accept anything with the `vlm.py` method
`ask(system, user_text, images, max_tokens=None, schema=None) -> str` (`OpenAIChat`, `AnthropicChat`; `images` is a
list of `(label, jpeg bytes)`). The reasoner prompt tells the model to use only the frames and the listed facts, to
answer `sufficient=false` rather than guess, and to cite segment ids; the reply schema is
`{answer, sufficient, confidence, citations[int], limits}`. The router takes `backend_fn=lambda: resolve_backend(...)`
and asks for the backend per request. Note for privacy: with a cloud backend the frames (which can show plates) leave
the machine. Prefer the local server; `use_llm` for parsing sends only the question text, and is off by default.

## Wiring (the edits left for the FS-A owner)
```python
# server.py, next to the other routers
from . import api_query
app.include_router(api_query.make_router(store, current_user, require, CONTROLLER, ADMIN, data_dir=DATA,
                                         backend_fn=lambda: vlm_mod.resolve_backend(vlm_settings())))
# index.html, sidebar next to Map
<a href="/query" data-route="query"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>Ask</a>
# worker.py, after end_run(...) of a finished run
segment_index.on_run_finished(self.store, run_id)     # never raises; import: from . import segment_index
```
Nothing indexes existing data until `POST /api/query/index/rebuild` (admin) or `segment_index.build_all(store)` runs once.

## X7: grounding metrics
`query.grounding_metrics(ranked, gold, ks)` returns MRR, Recall@k and Hit@k; `query.evaluate_queries(store, cases)`
runs labelled questions through Stage A (rule parser only). `tests/data/queries.jsonl` holds 13 hand-written
questions whose gold sets are predicates over the generator's **ground truth** (not over what the index inferred), in
`tests/test_query.py::build_world` (3 cameras, 2 h, ~330 passes, planted answers). On it: MRR 1.0, Recall@1 0.64,
Recall@3 0.95, Recall@10 0.99. **This is a synthetic world with clean colours and exact tracks: it shows the pipeline
and the metric code work and says nothing about real questions.** A real number needs real footage with labelled
questions (a query log does not exist yet; every question asked through the page is written to the audit log as
`query.ask` so one can be collected).

## Limits
Synthetic evaluation only; index quality is the tracker's quality; colour is approximate; no stop times unless a rule
measured them; the reasoner sees a few frames, not the video; free-form language outside the vocabulary is matched
only as keywords; Bangla questions are not parsed.
