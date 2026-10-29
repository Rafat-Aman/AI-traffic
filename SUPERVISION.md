# Supervision — Dhaka Traffic Surveillance (CSE499B)

My working tracker. It records what the plan promises, what actually exists, what is being built, by whom,
and what still needs a decision from the team. Updated after every work package. Nothing here is a claim of
done unless the Evidence column points at code and a test.

Legend: ✅ done · 🟡 partial · ❌ missing · 📐 design only (by plan) · ⛔ declined by team · 🔧 in progress

---

## North star — the team pipeline (2026-10-29)

```
DETECT → TRACK → UNDERSTAND → ENFORCE → ASSESS IMPACT → OPTIMIZE

Vehicles
  → Multi-camera city-wide trajectories
  → Violations & accidents
  → ANPR + VLM evidence / citation
  → Traffic impact assessment
  → City-wide routing & signal optimisation
```

Where the plan comes from:

| Source | What it commits to |
|---|---|
| `sdp (1).pdf` — 499A report | Ch.3 is the contract: 3 layers, Table 3.1 taxonomy (8 types), impact score with NNLS calibration + Spearman validation, VLM cascade, NL query panel (Stage A grounding / Stage B explanation), evidence bundles, plate-indexed history. **Ch.4 is empty** and must hold results in 499B. Ch.8.2 limitations, Ch.8.3 future work. |
| `Paper Presentation - 499A (5).pdf` | Expected outputs (dataset, YOLOv11 detector, BoT-SORT+Re-ID tracker, trajectory store, VLM module); next steps 1–7; software table (JetPack, Ultralytics, BoT-SORT, open-weights VLM, PostgreSQL, **SUMO**, CVAT). |
| Team pipeline (above) | Adds **OPTIMIZE** as a delivered stage, beyond the SDP's "design only" for signal control. |

---

## Snapshot — 2026-10-29

- git: clean at `6e0fe26 second commit`; tests **219 passing** across 9 modules.
- Running: dashboard `:8000` (`OTP_DELIVERY=echo INSECURE_COOKIES=1`), llama-server `:8081` (Qwen2.5-VL-3B Q4_K_M).
- Test accounts: `admin/admin`, `police/police`, driver licence `driver` (verified plate DHAKA METRO-GA 12-3456, demo case TV-1).
- Machine: i5-11400H 6C/12T AVX-512+VNNI, 23 GB RAM, **no usable GPU** (GTX 1650 on nouveau, torch CPU build).
  Anything needing a GPU (YOLOv11 retrain, Q-LoRA) has to be prepared here and run on rented compute.
- Footage: `side1_1.mp4` uploaded (camera 1). `side2_1.mp4` exists in `../../models/` (team's footbridge project).
- Footbridge ground truth available: `../../models/pipeline_output/matches.csv` (accepted pairs) +
  `feedback_wrong_pairs.csv` (human-rejected) — weak labels for cross-camera evaluation.

---

## Status matrix

### 1. DETECT
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| D1 | YOLOv11 fine-tuned on BD classes (SDP 3.1.2, slides) | 🟡 | `bd_vehicles_best.pt` | **It is YOLOv8s** (80 ep, 640 px), 16 classes. Retrain on YOLOv11 or correct the report → DEC-2 |
| D2 | Local classes (CNG, rickshaw, leguna, easy-bike, human hauler) | ✅ | detector names | No separate "human hauler" class (folded into Leguna) — say so in the report |
| D3 | Edge deployment / optimisation (slides: JetPack) | ❌ | CoreML export (Mac only) | ONNX/OpenVINO export for x86 CPU edge; Jetson is hardware we don't have |

### 2. TRACK
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| T1 | BoT-SORT per-camera tracking (SDP 3.1.2) | 🟡 | `vehicle_tracker.py`, setting `tracker` | **Default is ByteTrack**; BoT-SORT available. Measure both on CPU, then switch or justify → DEC-3 |
| T2 | Speed via homography | ✅ | `surveillance/speed.py`, `tools/bench_accuracy.py`, `tests/test_accuracy.py` | |
| T3 | Road-context annotation (lanes, bus stops, no-parking, direction) | ✅ | `violations/zones.py`, `web/editor.js`, `surveillance/autozone.py` | `bus_lane` area type with operating hours added by WP6 |
| T4 | Camera-motion compensation | ✅ | `surveillance/camera_motion.py` | |
| T5 | Cross-camera linking + Re-ID (SDP 3.1.3) | ✅ | `surveillance/reid.py`, `link.py`, `embed.py`, `tests/test_reid.py` | Validated on synthetic traffic only; real-footage validation pending (X6) |
| T6 | Trajectory database, queryable (SDP 3.1.3) | 🟡 | `tracklets` table | Summary per pass only (no per-frame boxes); SQLite not PostGIS (P1) |
| T7 | Multi-video site upload | ✅ | `/api/sites/upload`, `SiteRun` | |

### 3. UNDERSTAND
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| U1 | Geometric rule engine | ✅ | `violations/*`: V1 bus stopping, V3 kerb parking, V4 wrong side, V5 solid line, V7 lane blocking, V8 banned vehicle | Extras beyond SDP: V2 speed, V6 helmet → C-3 |
| U2 | Bus-lane violation (SDP Table 3.1 #5) | ✅ | `violations/bus_lane.py`, `tests/test_bus_lane.py` (50) | V9: ≥25 m and ≥4 s inside a drawn lane during its hours; jams excused. UI labels in INT-A |
| U3 | Unsafe / erratic manoeuvre (SDP #7, VLM-decided) | 🔧 | WP2 | Candidate from kinematics → VLM decides. V10 |
| U4 | Near-collision / collision (SDP #8, pipeline "accidents") | 🔧 | WP2 | TTC + post-event stasis → VLM decides. V11 |
| U5 | VLM reasoning, phase-aware, must decline | ✅ | `surveillance/vlm.py`, `tools/vlm_probe.py` (3/3), `tests/test_vlm.py` | 3B verdict is noisy — documented |
| U6 | VLM cascade: light screen → large reasoner (SDP 3.1.4) | 🔧 | WP2 | CPU-feasible screen; one local model only |
| U7 | Cross-camera journey given to the VLM | ✅ | `describe_journey` in `vlm.py` | |
| U8 | NL query Stage A: verbalised motion, identity + time window (SDP 3.1.5) | 🔧 | WP3 | |
| U9 | NL query Stage B: descriptive vs causal routing | 🔧 | WP3 | |
| U10 | Instruction-tuning corpus (self-instruct) + RAG over Road Transport Act 2018 / DMP rules (SDP 3.2.2) | ❌ | — | Needs the real statute text; never fabricate provisions |
| U11 | Q-LoRA rank 16 α 32 on 7B (SDP 3.3 step 5) | ❌ | — | Needs rented GPU → DEC-5 |

### 4. ENFORCE
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| E1 | ANPR (Bangla plates) | ✅ | `surveillance/anpr.py` | 0 wrong numbers on 167 held-out plates |
| E2 | Evidence bundle: snapshot, crop, 10 s clip, extra frames | ✅ | `worker._record`, `clips.py` | 6.28 MB / case measured |
| E3 | Tamper-evident evidence (hash at creation) | ✅ | `surveillance/citation.py` (seal), `tests/test_citation.py` (165) | Sealed at issue: SHA-256 per file + bundle hash on the notice; `/verify` endpoint. Evidence, not prevention: no external timestamp |
| E4 | Citation generation for e-prosecution | ✅ | `POST /api/violations/{vid}/citation`, `/api/citations*`, `/notice/{vid}` | Number, issuer, withdraw (admin, reason). **Fine amounts unset** (`FINE_SCHEDULE_BDT`); `paid` has no flow |
| E5 | Offender history + priority (impact + count + recency) | 🟡 | plate search | Priority depends on A6 |
| E6 | Roles (driver / controller / admin) + audit trail | ✅ | `auth.py`, `webauth.py`, `tests/test_auth.py` | |
| E7 | Driver portal | ✅ | `web/portal.html`, `/api/portal/*` | |
| E8 | Signed, expiring media links for offenders | ✅ | `/m/{token}/{path}` (HMAC, 1 h), `/media/*` staff-only | Key: `MEDIA_SECRET` env or `data/.media_secret` (gitignored). `data/` is fuseblk here → chmod ignored → prefer the env var |
| E9 | Dispute / appeal flow | ⛔ | — | Declined 2026-10-25; portal says contact the authority |
| E10 | Delivery transcode (≤720p) | ✅ | `clips.make_delivery_clip` (crf 28, veryfast, faststart) | 5.69→1.14 MB (1242×1440), 2.60→1.79 MB (720p). Originals kept for disputes. Stale `.part.mp4` blocks citing → WP13 cleanup |

### 5. ASSESS IMPACT — the SDP's central contribution
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| A1 | Rolling traffic state per approach: queue length, mean speed, lane occupancy, density, waiting time | 🟡 | `surveillance/traffic_state.py`, `tests/test_impact.py` (84) | Built + tested; wiring into pipeline/worker/UI = INT-A. 6–8 ms/frame at 80 vehicles |
| A2 | Per-approach, time-of-day baseline | 🟡 | `traffic_state.Baseline` | Falls back to the camera's own recent median until a time-of-day bin has data |
| A3 | Impact score S = Σ wᵢΔᵢ | 🟡 | `surveillance/impact.py` | Synthetic: busy stop 1.58 vs free road 0.34 vs quiet 0.11 |
| A4 | Calibration: NNLS to excess vehicle-delay (veh·s), equal-weight fallback labelled uncalibrated | 🟡 | `impact.nnls`, `impact.calibrate` | Lawson–Hanson in numpy. On synthetic n=9 calibrated ρ 0.90 < equal 0.95 → equal weights kept |
| A5 | Validation: Spearman ρ vs human severity ordering | 🟡 | `impact.spearman`, `impact.validate` | Needs human severity ratings of real cases (severity endpoint in INT-A) |
| A6 | Severity / priority = rule + impact + history (history as context, never a trigger) | 🟡 | `impact.priority` | History contribution capped |

### 6. OPTIMIZE
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| O1 | Signal control from measured state (SDP 2.2.1 review: MPLight pressure, CoLight, fog–cloud) | 🔧 | WP4 | SDP says design-only; pipeline wants it → simulation-validated controller → DEC-4 |
| O2 | City routing on live travel times | 🔧 | WP4 | |
| O3 | Emergency green corridor (EMVLight idea) | 🔧 | WP4 | |
| O4 | Evaluation vs fixed-time | 🔧 | WP4 | |

### 7. EVALUATE — SDP Chapter 4 is empty
| ID | Metric (SDP 3.3) | Status | Evidence | Gap / note |
|---|---|---|---|---|
| X1 | Detection mAP@0.5 | 🔧 | WP5 | Needs a labelled Dhaka val set |
| X2 | Tracking MOTA, IDF1, ID switches | 🔧 | WP5 | GT via CVAT MOT 1.1 export |
| X3 | Rule coverage, precision / recall, FP rate | 🟡 | `training/score_accuracy.py`, `tools/bench_accuracy.py` | Fold into one harness |
| X4 | Impact Spearman ρ | 🔧 | WP1/WP5 | Needs human severity ranking of real cases |
| X5 | Explanation quality incl. insufficient-evidence probes | 🟡 | `tools/vlm_probe.py` (3 probes) | Too few probes; no model-based scoring |
| X6 | Cross-camera IDF1 / pair precision | 🟡 | `tools/reid_eval.py`, synthetic tests | Run against footbridge labels |
| X7 | Grounding MRR / Recall@k | 🔧 | WP3/WP5 | |
| X8 | Chapter 4 results tables | ❌ | — | Generated from X1–X7 outputs |

### 8. PLATFORM / HOSTING
| ID | Item | Status | Gap |
|---|---|---|---|
| P1 | PostgreSQL + PostGIS (+ pgvector) — SDP Table 3.2 | ❌ | SQLite today; add a storage seam first |
| P2 | Object storage for media | ❌ | |
| P3 | Rate limiter in a shared store | ❌ | in-process memory |
| P4 | Password-change UI | ❌ | API exists (`/api/me/password`) |
| P5 | HTTPS / proxy / hosting config | ❌ | `TRUST_PROXY_HEADER`, `INSECURE_COOKIES` exist |

---

## Report ↔ code conflicts (must be resolved before the final report)

| ID | Report says | Code does | Recommendation |
|---|---|---|---|
| C-1 | YOLOv11 detector | YOLOv8s | DEC-2 |
| C-2 | BoT-SORT | ByteTrack default | Measure; DEC-3 |
| C-3 | Table 3.1 numbering; speed "never supports a speeding citation"; helmet listed as omitted | V1–V11 numbering differs; V2 now **advisory** (`ADVISORY_CODES`, never on a notice); V6 helmet implemented | Re-table the report to V-codes; say V6 was added |
| C-4 | NL query panel is in the abstract | not built | WP3 |
| C-5 | Impact score is the central argument | not built | WP1 |
| C-6 | PostgreSQL + PostGIS | SQLite | P1, before hosting |
| C-7 | Two-camera testbed | one camera loaded | Load `side2_1.mp4` and run a site |

V-code map, SDP Table 3.1 → code: #1 illegal parking → V3 · #2 bus stoppage → V1 · #3 road obstruction → V7 ·
#4 lane (class prohibited) → V8 · #5 bus lane → **V9 (new)** · #6 wrong-way → V4 · #7 unsafe manoeuvre →
**V10 (new)** · #8 near-collision/collision → **V11 (new)**. Extras: V2 speed, V5 solid line, V6 helmet.

---

## Decisions needed from the team

| ID | Question | My recommendation |
|---|---|---|
| DEC-1 | V2 speed: citable or advisory? | **Advisory** (detected, shown, never on a notice) — matches SDP 3.1.2 and 5.1.3 |
| DEC-2 | YOLOv11 retrain vs correct the report | Retrain on rented GPU *if* the training set is available; otherwise correct the report |
| DEC-3 | BoT-SORT vs ByteTrack | Decide on measured MOTA/IDF1 + FPS (X2) |
| DEC-4 | OPTIMIZE scope | Simulation-validated max-pressure + routing + EV corridor; SUMO bridge later |
| DEC-5 | Q-LoRA budget | Prepare corpus + scripts here; one rented run on Vast.ai |
| DEC-6 | When to move to PostgreSQL | Before any hosting; after features freeze |

---

## Revision 2 — 2026-10-29 (supersedes the wave-1 table below)

Wave 1 produced nothing: all seven agents hit the session limit within minutes. Relaunching with at most four
agents at a time on a lighter model, and one owner per shared file.

### New requirements from the team (2026-10-29)
| ID | Requirement | Plan source |
|---|---|---|
| R-1 | Use the team's CityFlowV2 notebook + its trained ReID weights (`cityflowv2.ipynb`, `reid_resnet50_ibn_a_320_syn_exclS02.zip`) | slides: city-scale MTMC |
| R-2 | Map in the dashboard: admin places each camera and its heading; point-to-point average speed of the same vehicle between two cameras | team; Google Maps asked for |
| R-3 | Query panel | SDP 3.1.5 |
| R-4 | Long-video retrieval and efficient VLM handling as in the slides (VideoRAG retrieve-then-reason, Spatialyze persistence) | slides p.8, SDP Fig 2.13 |
| R-5 | Production data handling: hundreds of videos a day — raw video, evidence, tracklets, retention | team |
| R-6 | 3D city view with cameras, traffic lights and vehicle densities; an AI that optimises signal flow | team pipeline OPTIMIZE; slides: SUMO, MPLight, CoLight |
| R-7 | Two-tier VLM as in the slides: **2B model screens, 8B model reasons** (TAU-R1) | slides p.7; SDP Table 3.3 |
| R-8 | Google Maps live traffic as a supplement to flow planning (team will supply API keys) | team |

R-8 design (folds into WP4 + WP9 + WP10): **Routes API** `computeRouteMatrix` / `computeRoutes` with
`TRAFFIC_AWARE` routing and `TRAFFIC_ON_POLYLINE` gives traffic-aware vs free-flow duration and congestion
intervals per link between camera points → fills link travel times where there is no camera, server-side key only,
cached with a TTL and rate-limited (billed per element; traffic-aware is the pricier SKU). **Maps JavaScript API**
`TrafficLayer` for the visual overlay, browser key restricted by referrer. Google's terms forbid showing Google Maps
content on a non-Google map, so Google traffic is drawn only on Google maps; the free MapLibre 3D view shows only the
system's own camera-measured densities. Keys stored server-side (admin page), never in the repo.

Facts that shape them: Qwen3-VL-2B-Instruct (1.1 G + 0.8 G mmproj) and Qwen3-VL-8B-Instruct (4.7 G + 1.1 G mmproj)
exist as official GGUFs → R-7 runs locally (8B started on demand; ~7.7 GB RAM free with the 3B stopped). The
CityFlow ReID model is ResNet50-IBN-a @320 px: held-out mAP 0.427 / rank-1 0.560; MTMC on CityFlowV2 S02 IDF1 46.5
(2-fold). Trained on US cars → must be measured on Dhaka footage before it replaces MobileNetV3. Google Maps
needs an API key with billing → map is provider-pluggable: Google when `GOOGLE_MAPS_API_KEY` is set, OpenStreetMap
otherwise; 3D uses MapLibre GL + OpenFreeMap (free vector tiles with building heights).

### Work packages (revision 2)
| WP | Covers | Files owned (edit) / created | Wave | Status |
|---|---|---|---|---|
| WP7 | E3 E4 E8 E10 citations, sealed evidence, signed links, delivery clips, V2 advisory | `server.py`, `store.py`, `clips.py`, `notice.html`, `portal.html`, `app.js` / `citation.py`, `tests/test_citation.py` | A | ✅ 288 tests across 5 modules; 42/42 headless-Chrome checks |
| WP6 | U2 bus lane V9 | `violations/zones.py`, `engine.py`, `web/editor.js`, `vlm.py` (RULE_CLAIM only) / `violations/bus_lane.py`, test | A | ✅ 50 + 170 tests |
| WP8 | R-1 trained ReID model in the matcher | `surveillance/embed.py`, `tools/reid_eval.py`, `.gitignore` / `reid_backbone.py`, `research/cityflow/`, `models/reid/`, test | A | 🔧 |
| WP1 | A1–A6 traffic state + impact | new files only | A | ✅ 84 tests; wiring = INT-A |
| INT-A | wire WP1 into pipeline/worker/server/UI; V9–V11 labels, colours, candidate badge; `summary()` never prints "undefined"; tests never open the live DB | FS-A + `web/app.css`, `tests/test_auth.py` / `tests/test_impact_integration.py` | B | 🔧 |
| WP4 | O1–O4 network (geo-aware), routing, max-pressure, EV corridor, simulator **+ R-8 Google Routes traffic provider** | new files (`surveillance/optimize/`, partial from wave 1) | B | 🔧 |
| WP2 | U3 U4 V10/V11 kinematic candidates + asymmetric pre-screen (`surveillance/screening.py`) | new files + `engine.py` RULES, `vlm.py` RULE_CLAIM, `test_vlm.py`, docs tables | B | 🔧 |
| WP5 | X1–X8 evaluation harness + CityFlow numbers into Chapter 4 | new files only | B | ⏳ |
| WP9 | R-2 map, camera geolocation, inter-camera average speed | FS-A after WP7 | B | ⏳ |
| WP11 | R-7 + U6 two-tier VLM (2B screen → 8B reasoner), escalation of V10/V11 | `vlm.py`, `tools/*vlm*` + server explain flow | C | ⏳ |
| WP3 | R-3 R-4 U8 U9 X7 segment index at ingest, retrieve-then-reason query panel | new files, then FS-A | C | ⏳ |
| WP13 | R-5 ingest queue, retention, compaction, disk monitor, storage page | FS-A | C | ⏳ |
| WP10 | R-6 3D city, signals, densities, optimiser UI | FS-A + new web files | D | ⏳ |
| INT | wire WP1 / WP2 / WP4 into pipeline, worker, engine, UI | FS-A, FS-B | D | ⏳ |

FS-A = `surveillance/server.py`, `store.py`, `worker.py`, `pipeline.py`, `web/app.js`, `web/index.html` — one owner at a time.

FS-A is the bottleneck, so later packages (WP9, WP3, WP13, WP10, WP11) build as **feature modules**: their own
`APIRouter` file (`make_router(store, auth, …)`), their own `ensure_schema(store)`, their own `web/*.js` page. The only
FS-A edits are one `include_router` line and one nav entry, done by whoever holds FS-A at the time.

Open items from WP7 for the team: drivers currently see every confirmed non-advisory case, cited or not (one-line
change to show only cited ones); fines are unset until the authority supplies amounts; uvicorn's access log records
signed-link tokens (redact before hosting, P5).

---

## Work packages (wave 1 — superseded)

Wave 1 runs as parallel background agents. **Rule for every agent: create new files only, never edit an
existing file, no new pip dependencies (numpy, opencv, torch-cpu, ultralytics, lap, fastapi only — no scipy,
no sklearn), never touch `data/surveillance.db` or the servers on :8000 / :8081.** I do all integration into
shared files myself, sequentially, so nothing collides.

| WP | Covers | Owner | Files (new) | Status |
|---|---|---|---|---|
| WP1 | A1–A6 traffic state + impact score + calibration + priority | agent | `surveillance/traffic_state.py`, `surveillance/impact.py`, `tests/test_impact.py` | 🔧 |
| WP2 | U3, U4, U6 behavioural violations + accident candidates + cascade screen | agent | `violations/erratic.py`, `violations/near_collision.py`, `surveillance/cascade.py`, `tests/test_behaviour.py` | 🔧 |
| WP3 | U8, U9, X7 NL query: motion verbalisation, parsing, retrieval, routing | agent | `surveillance/motion_text.py`, `surveillance/query.py`, `tests/test_query.py` | 🔧 |
| WP4 | O1–O4 network, routing, max-pressure signals, EV corridor, simulator | agent | `surveillance/optimize/*`, `tools/signal_sim.py`, `tests/test_optimize.py` | 🔧 |
| WP5 | X1–X8 evaluation harness + Chapter 4 report generator | agent | `evaluation/*`, `tools/evaluate.py`, `tools/export_tracks.py`, `tests/test_evaluation.py` | 🔧 |
| WP6 | U2 bus lane V9 | agent | edits `violations/zones.py`, `violations/engine.py`, `surveillance/vlm.py` (RULE_CLAIM), `web/editor.js`; new `violations/bus_lane.py`, `tests/test_bus_lane.py` | 🔧 |
| WP7 | E3, E4, E8, E10 + V2 advisory (DEC-1, per SDP 3.1.2) | agent | edits `surveillance/server.py`, `store.py`, `clips.py`, `web/notice.html`, `web/portal.html`, `web/app.js`; new `surveillance/evidence_seal.py`, `tests/test_citation.py` | 🔧 |
| INT-1 | Wire WP1–WP5 into pipeline / worker / store / server / UI | me | shared files | ⏳ after WP6 + WP7 land |
| WP8 | T1 tracker measurement + decision | me | — | ⏳ |
| WP9 | P4 password UI, small polish | me | shared | ⏳ |
| WP10 | P1–P3 storage seam → PostgreSQL / PostGIS / pgvector | agent (wave 3) | | ⏳ DEC-6 |
| WP11 | U10–U11 corpus pipeline, RAG, Q-LoRA scripts | agent (wave 3) | `training/vlm/*` | ⏳ DEC-5 |
| WP12 | D1 detector (YOLOv11) or report correction | team | | ⏳ DEC-2 |
| WP13 | Report: Chapter 4 results, taxonomy table, conflicts C-1…C-7 | me | report | ⏳ last |

### File ownership while wave 1 runs
Two agents (WP6, WP7) edit existing files; their sets are disjoint. WP1–WP5 create new files only. I touch no
shared file until WP6 and WP7 report, then integrate everything sequentially. Order of integration:
WP6 → WP7 → WP2 (needs engine + vlm from WP6) → WP1 (store + worker + server after WP7) → WP3 → WP4 → WP5.

### Integration checklist (INT-1) — fill as agents report back
- [ ] WP1: `TrafficStateCollector` fed from `Pipeline.process` (same observation list as the rule engine); impact computed when a violation is recorded; `impact` JSON + `priority` on `violations`; review list sortable by priority; impact panel in the case view
- [ ] WP2: register V10 / V11 in `violations/engine.RULES`; RULE_CLAIM entries in `vlm.py`; candidates marked "needs VLM decision"; cascade before the Explain queue
- [ ] WP3: `motion` text built in `TrackletCollector.build`; `/api/query`; query panel in the dashboard
- [ ] WP4: `/api/optimize/*`; network spec per site; Optimise page fed by A1 state
- [ ] WP5: `tools/evaluate.py` runs end to end; results JSON → `REPORT_RESULTS.md`
- [ ] full suite green; servers restarted; smoke-tested as admin, police, driver

---

## Log
- 2026-10-29 — Audit done against SDP + slides + pipeline. Found C-1…C-7. Wave 1 (WP1–WP5) dispatched.
- 2026-10-29 — Wave 1 lost to the session limit (no files written; a partial `surveillance/optimize/` left by WP4).
  Team added R-1…R-7 and the CityFlow notebook + ReID weights. Revision 2 planned; wave A (WP7, WP6, WP8, WP1) dispatched.
- 2026-10-29 — Team added R-8 (Google live traffic). WP1 ✅ (impact, 84 tests), WP6 ✅ (V9), WP7 ✅ (citations, signed
  links). WP4 relaunched with R-8; WP2 and INT-A dispatched. Added `data/.media_secret` and `.env` to `.gitignore`
  (WP7 found the signing key was not ignored).
