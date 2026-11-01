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

- git: last commit `80563e6 agent working`; everything since is uncommitted. Tests **1285 passing** (`unittest discover -s tests`, 143 s).
- Running: dashboard `:8000` via `./tools/run_dashboard.sh` (MEDIA_SECRET from `~/.config/dhaka-traffic/media_secret`, 0600, off the fuseblk mount). VLM: Qwen3-VL-2B screen on :8081 (`tools/vlm_server.sh --tier screen`), 8B started on demand on :8082.
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
| D3 | Edge deployment / optimisation (slides: JetPack) | 🟡 | CoreML (Mac), `Detector(backend="onnx")` opt-in | ONNX agreement 94–96 % vs torch, speed gain not established under load; GPU needed for hundreds of videos/day (`docs/CAPACITY.md`) |

### 2. TRACK
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| T1 | BoT-SORT per-camera tracking (SDP 3.1.2) | 🟡 | `vehicle_tracker.py`, setting `tracker` | **Default is ByteTrack**; BoT-SORT available. Measure both on CPU, then switch or justify → DEC-3 |
| T2 | Speed via homography | ✅ | `surveillance/speed.py`, `tools/bench_accuracy.py`, `tests/test_accuracy.py` | |
| T3 | Road-context annotation (lanes, bus stops, no-parking, direction) | ✅ | `violations/zones.py`, `web/editor.js`, `surveillance/autozone.py` | `bus_lane` area type with operating hours added by WP6 |
| T4 | Camera-motion compensation | ✅ | `surveillance/camera_motion.py` | |
| T5 | Cross-camera linking + Re-ID (SDP 3.1.3) | ✅ | `surveillance/reid.py`, `link.py`, `embed.py`, `reid_backbone.py`, `tests/test_reid*.py` | Team's CityFlow ResNet50-IBN-a is the default appearance model (async). Cross-camera accuracy on Dhaka unmeasured (X6) |
| T6 | Trajectory database, queryable (SDP 3.1.3) | 🟡 | `tracklets` table | Summary per pass only (no per-frame boxes); SQLite not PostGIS (P1) |
| T7 | Multi-video site upload | ✅ | `/api/sites/upload`, `SiteRun` | |

### 3. UNDERSTAND
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| U1 | Geometric rule engine | ✅ | `violations/*`: V1 bus stopping, V3 kerb parking, V4 wrong side, V5 solid line, V7 lane blocking, V8 banned vehicle | Extras beyond SDP: V2 speed, V6 helmet → C-3 |
| U2 | Bus-lane violation (SDP Table 3.1 #5) | ✅ | `violations/bus_lane.py`, `tests/test_bus_lane.py` (50) | V9: ≥25 m and ≥4 s inside a drawn lane during its hours; jams excused. UI labels in INT-A |
| U3 | Unsafe / erratic manoeuvre (SDP #7, VLM-decided) | ✅ | `violations/erratic.py`, `tests/test_behaviour.py` | Candidate from kinematics → VLM decides. V10 |
| U4 | Near-collision / collision (SDP #8, pipeline "accidents") | ✅ | `violations/near_collision.py`, `tests/test_behaviour.py` | TTC + post-event stasis → VLM decides. V11 |
| U5 | VLM reasoning, phase-aware, must decline | ✅ | `surveillance/vlm.py`, `tools/vlm_probe.py` (3/3), `tests/test_vlm.py` | 3B verdict is noisy — documented |
| U6 | VLM cascade: light screen → large reasoner (SDP 3.1.4) | ✅ | `surveillance/screening.py`, `vlm_cascade.py`, `tests/test_cascade.py` | Kinematic pre-screen → Qwen3-VL-2B → Qwen3-VL-8B on demand. Thresholds are policy (no labelled set, X5) |
| U7 | Cross-camera journey given to the VLM | ✅ | `describe_journey` in `vlm.py` | |
| U8 | NL query Stage A: verbalised motion, identity + time window (SDP 3.1.5) | ✅ | `surveillance/query.py`, `segment_index.py` | |
| U9 | NL query Stage B: descriptive vs causal routing | ✅ | `surveillance/query.py`, `api_query.py` | |
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
| E7 | Driver portal | ✅ | `web/portal.html`, `/api/portal/*` | "My citations": issued citations only (withdrawn shown as void); IDOR-checked by sealed plate |
| E8 | Signed, expiring media links for offenders | ✅ | `/m/{token}/{path}` (HMAC, 1 h), `/media/*` staff-only | Key: `MEDIA_SECRET` env or `data/.media_secret` (gitignored). `data/` is fuseblk here → chmod ignored → prefer the env var |
| E9 | Dispute / appeal flow | ✅ | `surveillance/disputes.py`, `tests/test_disputes.py` (51) | Driver disputes within 30 d → 8B re-check (advisory, injection-hardened, never shown to driver) → controller/admin upholds or accepts (four-eyes: not the issuer/confirmer). No uploads, no second appeal |
| E10 | Delivery transcode (≤720p) | ✅ | `clips.make_delivery_clip` (crf 28, veryfast, faststart) | 5.69→1.14 MB (1242×1440), 2.60→1.79 MB (720p). Originals kept for disputes. Stale `.part.mp4` blocks citing → WP13 cleanup |

### 5. ASSESS IMPACT — the SDP's central contribution
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| A1 | Rolling traffic state per approach: queue length, mean speed, lane occupancy, density, waiting time | ✅ | `surveillance/traffic_state.py`, `tests/test_impact.py` (84) | Wired by INT-A: pipeline → collector → worker → impact row + SSE → case view. 6–8 ms/frame at 80 vehicles |
| A2 | Per-approach, time-of-day baseline | ✅ | `traffic_state.Baseline` | Falls back to the camera's own recent median until a time-of-day bin has data |
| A3 | Impact score S = Σ wᵢΔᵢ | ✅ | `surveillance/impact.py` | Synthetic: busy stop 1.58 vs free road 0.34 vs quiet 0.11 |
| A4 | Calibration: NNLS to excess vehicle-delay (veh·s), equal-weight fallback labelled uncalibrated | ✅ | `impact.nnls`, `impact.calibrate` | Lawson–Hanson in numpy. On synthetic n=9 calibrated ρ 0.90 < equal 0.95 → equal weights kept |
| A5 | Validation: Spearman ρ vs human severity ordering | ✅ | `impact.spearman`, `impact.validate` | Needs human severity ratings of real cases (severity endpoint in INT-A) |
| A6 | Severity / priority = rule + impact + history (history as context, never a trigger) | ✅ | `impact.priority` | History contribution capped |

### 6. OPTIMIZE
| ID | Plan item | Status | Evidence | Gap / note |
|---|---|---|---|---|
| O1 | Signal control from measured state (SDP 2.2.1 review: MPLight pressure, CoLight, fog–cloud) | ✅ | `surveillance/optimize/`, `tools/signal_sim.py`, `tests/test_optimize.py` | Max-pressure, min green 15 s; simulator only, no real signal. API + 3D UI = WP10 |
| O2 | City routing on live travel times | 🟡 | `surveillance/optimize/`, `tools/signal_sim.py`, `tests/test_optimize.py` | |
| O3 | Emergency green corridor (EMVLight idea) | 🟡 | `surveillance/optimize/`, `tools/signal_sim.py`, `tests/test_optimize.py` | |
| O4 | Evaluation vs fixed-time | 🟡 | `surveillance/optimize/`, `tools/signal_sim.py`, `tests/test_optimize.py` | |

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
| P5 | HTTPS / proxy / hosting config | 🟡 | `deploy/` (push.sh, setup.sh, README): Ubuntu VM + systemd + Caddy HTTPS, secrets in /etc/dhaka-traffic.env. Scripts syntax-checked and rsync dry-run only; not yet run on a real VM |

---

## Report ↔ code conflicts (must be resolved before the final report)

| ID | Report says | Code does | Recommendation |
|---|---|---|---|
| C-1 | YOLOv11 detector | YOLOv8s | DEC-2 |
| C-2 | BoT-SORT | ByteTrack default | Measure; DEC-3 |
| C-3 | Table 3.1 numbering; speed "never supports a speeding citation"; helmet listed as omitted | V1–V11 numbering differs; V2 now **advisory** (`ADVISORY_CODES`, never on a notice); V6 helmet implemented | Re-table the report to V-codes; say V6 was added |
| C-4 | NL query panel is in the abstract | built (WP3), English only, rule parser + optional LLM parse | Report: say Bangla questions are not parsed |
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
| DEC-2 | YOLOv11 retrain vs correct the report | Retrain on rented GPU *if* the training set is available; otherwise correct the report. **Team (2026-10-30): training-video provenance of `bd_vehicles_best.pt` unknown** → build any test set only from footage recorded after the model (weights dated 2026-10-24), e.g. the 2026-10-29 `videos/` drop, and confirm it was never given to whoever trained it |
| DEC-3 | BoT-SORT vs ByteTrack | Proxy on 60 s of Dhaka footage (WP5): 152 vs 147 tracks, mutual IDF1 0.976, ByteTrack 3.6 ms vs BoT-SORT 25.4 ms/frame; detector dominates (~1.8 s/frame under load). Accuracy undecidable without MOT ground truth → keep ByteTrack, report BoT-SORT as equivalent here, or label 2 min in CVAT |
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
| WP8 | R-1 trained ReID model in the matcher | `surveillance/embed.py`, `tools/reid_eval.py`, `.gitignore` / `reid_backbone.py`, `research/cityflow/`, `models/reid/`, test | A | ✅ 74 tests. On Dhaka footage (1 camera, 62 vehicles) same-class rank-1 98.4% vs MobileNet 75.8%, but ~1 s vs 8 ms per vehicle → opt-in `EMBED_MODEL=auto`. Default switch needs: `embed_model` column (store), tag in reid.py/link.py, embedding off the frame thread, test_reid:342 → INT-B |
| WP1 | A1–A6 traffic state + impact | new files only | A | ✅ 84 tests; wiring = INT-A |
| INT-A | wire WP1 into pipeline/worker/server/UI; V9–V11 labels, colours, candidate badge; `summary()` never prints "undefined"; tests never open the live DB | FS-A + `web/app.css`, `tests/test_auth.py` / `tests/test_impact_integration.py` | B | ✅ 410 tests, 17/17 browser checks. Calibrated weights adopted only with ≥10 graded cases and strictly higher ρ |
| INT-B | R-1 ReID model as default: MobileNet sync + ReID async (bounded queue, drop never block), per-tag cues, `embed_model` columns, measured FPS | `reid.py`, `link.py`, `embed.py`, `store.py`, `worker.py`, tests, docs | B | ✅ 297 tests. Default `auto`; 45 s Dhaka clip: 51/51 vehicles got ReID, 0 dropped (~2.2 s each under load). FPS comparison confounded by machine load |
| INT-C | segment-index hook after runs + periodic live indexing; V10/V11 clip pre-roll from `started_s`; ReID thread vs detector cores; ReID queue line in camera view | `worker.py`, `pipeline.py`, `clips.py`, `reid.py` (doc), `app.js` (one line), tests | D | ✅ 7 tests. Index after each run + every 5 min live; V10/V11 clips reach back to contact −2 s (≤12 s ring, ~30–45 MB/camera at 720p). No per-thread torch cap exists → documented |
| WP4 | O1–O4 network (geo-aware), routing, max-pressure, EV corridor, simulator **+ R-8 Google Routes traffic provider** | new files (`surveillance/optimize/`, partial from wave 1) | B | ✅ 138 tests. Synthetic 3×3 grid, 5 held-out seeds: max-pressure mean delay 43–52% below tuned fixed-time at every load; EV corridor −34…−44% EV time, others +2…4%. Google never called live. API router + UI → WP10 |
| WP2 | U3 U4 V10/V11 kinematic candidates + asymmetric pre-screen (`surveillance/screening.py`) | new files + `engine.py` RULES, `vlm.py` RULE_CLAIM, `test_vlm.py`, docs tables | B | ✅ full suite 763 OK. 60 veh × 300 frames 0.46 s. V10/V11 advisory by default (never cited). FP rate on Dhaka unmeasured. V11 clip pre-roll fixed by INT-C |
| WP5 | X1–X8 evaluation harness + CityFlow numbers into Chapter 4; DEC-3 proxy | new files only (`evaluation/`, `tools/evaluate.py`, `docs/EVALUATION.md`, `REPORT_RESULTS.md`) | B | ✅ 26 metric tests. Real Dhaka numbers only for X2 (proxy) and X6 (within-camera). Everything else "not measured — needs …" (labelling plan in `docs/EVALUATION.md`) |
| WP9 | R-2 map, camera geolocation, section speed, Google keys (admin, server-side) | feature module: `geo.py`, `section_speed.py`, `api_map.py`, `web/map.*`, test | B | ✅ 30 tests; router mounted + nav link. Not yet opened in a browser; straight-line × 1.4 until a road network is defined; clocks per camera via manual `clock_offset_s` |
| WP11 | R-7 + U6 two-tier VLM (2B screen → 8B reasoner on demand), pre-screen → screen → reason, contact frame first | `vlm.py`, `screening.py`, new `vlm_cascade.py`, server explain flow, app.js VLM block, `tools/*vlm*`, `docs/VLM.md` | C | ✅ 40 cascade tests; probes 8/8 (sycophancy 3/3 kept). Loaded machine: 2B ~85 s/case, 8B ~275 s/case, 8B RSS ~10 GB (needs ~10 GB free). 2B is noisy → nearly every case escalates under the asymmetric rule. Live: 2B on :8081, 8B on demand :8082 |
| WP3 | R-3 R-4 U8 U9 X7 segment index (from stored tracklets/violations), Stage A parse + retrieval, Stage B descriptive vs causal (VLM via injected reason_fn) | feature module: `segment_index.py`, `query.py`, `api_query.py`, `web/query.*`, `tests/test_query.py`, `docs/QUERY.md` | C | ✅ 81 tests; router mounted + "Ask" nav. 10 s windows from stored analytics only. Synthetic X7: MRR 1.0, R@1 0.64, R@3 0.95. Worker hook wired by INT-C; reasoner should use the cascade's 8B tier after WP11 |
| WP13 | R-5 ingest queue, retention (dry run → admin apply), compaction, disk monitor, storage page; D3 ONNX backend + capacity numbers | feature module: `storage_mgr.py`, `ingest.py`, `api_storage.py`, `web/storage.*`, tests, `docs/STORAGE.md`, `docs/CAPACITY.md`; edits `detector.py` | C | ✅ 46 tests; router + queue + 507 upload guard + nav wired. Raw video 7 d, dismissed 30 d, cited evidence never, confirmed evidence only after admin confirms legal period. Capacity: machine saturates ~3–4 fps at 960 px → ~3 video-h/day at 30 fps, ~9 at 10 fps; 300 × 10-min videos/day needs ~9–16 such CPUs or a GPU. ONNX not reliably faster → opt-in |
| WP10 | R-6 3D city (MapLibre + OpenFreeMap), network editor, signals, densities, simulation playback, recommendation, routing; Google values never on MapLibre | feature module: `api_optimize.py`, `web/city.*`, `tests/test_city.py`, `docs/OPTIMIZE.md`; may edit `optimize/*`, `web/map.*` | C | ✅ 22 tests; router mounted + nav link; headless-Chrome render checked. Editor drag/click and Google overlay not exercised |
| INT | wire WP1 / WP2 / WP4 into pipeline, worker, engine, UI | FS-A, FS-B | D | ⏳ |
| WP14 | E7 + E9: portal shows only issued citations; driver disputes → 8B re-check (advisory, injection-safe) → human decides | `disputes.py` (new), FS-A, `citation.py`, `portal.html`, `app.js`, tests | E | ✅ 360 + 391 tests; not yet opened in a browser |

FS-A = `surveillance/server.py`, `store.py`, `worker.py`, `pipeline.py`, `web/app.js`, `web/index.html` — one owner at a time.

FS-A is the bottleneck, so later packages (WP9, WP3, WP13, WP10, WP11) build as **feature modules**: their own
`APIRouter` file (`make_router(store, auth, …)`), their own `ensure_schema(store)`, their own `web/*.js` page. The only
FS-A edits are one `include_router` line and one nav entry, done by whoever holds FS-A at the time.

Open items from WP7 for the team: (resolved 2026-10-30: drivers see only issued citations); fines are unset until the authority supplies amounts; uvicorn's access log records
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
- 2026-10-30 — All agents hit the session limit, then the editor closed mid-run. Team committed `80563e6 agent working`.
  WP8 ✅ (report received before the close). INT-A, WP2, WP4 resumed from their saved transcripts. `videos/` (team's
  3.6 GB footage zips) added to `.gitignore`.
- 2026-10-30 — WP4 ✅, INT-A ✅ (impact live in the UI after restart). Dispatched WP9 (map), WP5 (evaluation), INT-B (ReID default).
- 2026-10-30 — WP2 ✅ (V10/V11 + pre-screen). Set V10/V11 advisory in `citation.DEFAULT_ADVISORY`. Dispatched WP11 (two-tier VLM).
- 2026-10-30 — WP9 ✅ (map). Mounted `api_map` router in server.py, Map link in index.html.
- 2026-10-30 — Dispatched WP10 (3D city + optimiser UI).
- 2026-10-30 — Machine rebooted (servers down). WP10 ✅ (3D city). Mounted `api_optimize` router, 3D city nav link.
- 2026-10-30 — Dispatched WP3 (query panel, retrieve-then-reason).
- 2026-10-30 — WP5 ✅ (evaluation harness, `REPORT_RESULTS.md`). V10/V11 cost test now measures thread CPU time (was flaky under load). Dispatched WP13 (storage, ingest, capacity).
- 2026-10-30 — WP3 ✅ (query panel). Mounted `api_query` router, "Ask" nav link.
- 2026-10-30 — INT-B ✅ (ReID default `auto`). Dashboard restarted with `tools/run_dashboard.sh`; smoke test as admin/police: all new pages 200, admin-only endpoints 403 for police. Dispatched INT-C.
- 2026-10-30 — INT-C ✅. Fixed a timing race in `test_mobilenet_now_reid_later` (gate the fake ReID job).
- 2026-10-30 — WP13 ✅ (storage, ingest queue, capacity). Wired into server.py + nav. Full suite: **1031 tests OK** (214 s, loaded machine).
- 2026-10-30 — WP11 ✅. Live switch-over done: 2B on :8081, dashboard restarted with all packages. Smoke test: admin/police/driver access correct (driver 403 on every staff API, 302 on staff pages). Single-model label set to qwen3-vl-2b. E2E browser check dispatched (read-only, live server).
- 2026-10-30 — Team decisions: portal shows only issued citations; drivers may dispute (8B or human checks) → WP14 dispatched; fines + legal retention kept as is; detector training-data provenance unknown (see DEC-2).
- 2026-10-30 — WP14 ✅ (cited-only portal + disputes). Restart pending the E2E browser check.
- 2026-10-30 — E2E browser check (headless Chrome, live :8000, all roles, 1440 px + 390 px): every page loads, no uncaught exceptions; driver isolation correct. Found B1 /city 404 polling noise, B2 phone overflow (storage, admin tables, stats), B3 admin-only camera controls visible to police + start/stop/auto-zone routes without role check, B4 no Explain stage progress (escalated case 278.6 s; 8B contradicted demo case 1), B5 unexplained disabled buttons, B6 raw 403 on /notice for drivers, B7 no favicon. FIX-1 dispatched.
- 2026-10-30 — FIX-1 ✅ (B1–B7): police lose all camera/zone/settings controls (server `require(ADMIN)` + UI `isAdmin()`), start/stop = controller; /city returns an empty state instead of 404 and camera columns now render (data-driven opacity was invalid); no overflow at 390 px; Explain shows 5 stages with measured ETAs over SSE; /notice redirects drivers; favicon. Made the simulation-timeout test deterministic. Full suite **1096 OK**. Dashboard restarted; all-role smoke test passed. Open judgement: `POST /api/link` stays staff-level.
- 2026-10-30 — Compared with Shihab's repo (cloned read-only to `../Shihab-AI-Traffic-System`, not run): see `docs/COMPARISON_SHIHAB.md`. Same vision base; theirs is stronger on enforcement (payments, email, registry, plate for every vehicle, scalable cross-camera IDs, measured V3/V6 fixes, VLM prompt fixes), ours covers impact, optimisation, query, storage, two-tier VLM, trained ReID, V9–V11, disputes, tests. Ports P1–P6 proposed, none started.
- 2026-10-30 — User: "do all those upgrades" (P1–P6). Dispatched WP15 (P1+P2 helmet + kerb rules), WP16 (P3 VLM prompt fixes), WP17 (P5 passed vehicles, plate for every vehicle, search, look-alikes, retention), WP18 (P6 fines table, registry, SSLCommerz sandbox + test gateway, invoice PDF, mail outbox). WP19 (P4 gate/link topology) waits for WP17 (both touch worker.py). Server wiring (server.py, nav, PUBLIC_API/DRIVER_API) stays with me.
- 2026-10-30 — WP15 ✅ (V6 toward-camera only, in-road only, 4 votes ≥3/4, P≤0.5 = helmet; KERB_AREAS; creep_m 10 m; duplicate suppression V1/V3/V7; tests/test_violations helmet sims updated for heading). Side effect to watch: `road_at` now returns a road for kerb-strip points, so every rule sees those vehicles. WP16 ✅ (label inpainted before every VLM call, found by template match since rows carry no bbox; V6 head pre-question as `visual_check` stage on escalation, ~24–28 s cold on 2B; V2 photo cannot contradict speed; speed talk stripped for non-speed rules, stopped/slow talk for V2/V4/V5/V9; V10/V11 exempt). Open: store bbox + frame_size on violation rows for exact label hiding (pipeline owner). WP18 ✅ wired by me: router, mailer start/stop, `cit.on_issued`, `disputes_mod.on_decided`, SSLCommerz return/IPN paths in PUBLIC_API, admin-only nav "Fines & payments". Fixed a route collision: WP18's owner registry reused `/api/registry` (existing account↔vehicle registry wins) → renamed `/api/owners`, tab "Owner contacts". Portal list/dispute now use `registry.driver_plate_keys` (verified vehicles + owner contacts with the driver's licence; both admin-set, driver accounts admin-created) so see/dispute/pay agree; registry table created on demand for swapped stores. 498 targeted tests OK.
- 2026-10-30 — WP17 ✅ wired: page `/passed` ("Passed vehicles"; `#/vehicles` stays the cross-camera journeys page), API `/api/passed/*`, daily purge loop (PASSED_RETENTION_DAYS 30, also in Storage dry-run), camera settings `passed` (on) and `anpr_all` (off: ~0.5 s CPU per vehicle, −9 % fps) in DEFAULT_SETTINGS + settings dialog. 50 px confirm rule in anpr.vote (no effect while MIN_PLATE_PX = 110). Look-alikes 2.8 ms @10k, 11 ms @100k (MobileNet), suggestions only. WP19 ✅ (gates, links with map-suggested windows, two-link fallback, out-of-order merge, `vehicle_merges`; routes inside api_map). First defaults gave synthetic precision 0.81 → sent back; tuned to 1-link 3.0/3.0, 2-link 1.5/3.0: P 0.974 / kept 0.908 (all cameras see all), P 0.956 / kept 0.664 (middle camera missed); no-topology P 1.00 / kept 0.07. Synthetic and tuned on the same seeds: re-tune on real footage. I tightened the test guard from 0.7 to 0.9. Cross-camera ids feed journeys/section speed only, never citations. Added `display_id` (DV-000123) + merged-id resolution to `/api/vehicles`, shown on the vehicles page. `tools/run_dashboard.sh` sets PUBLIC_BASE_URL=http://localhost:8000 by default. Full suite **1285 OK**; all web/*.js pass `node --check`; dashboard restarted; smoke test: admin/police/driver access to /payments, /api/fines|owners|payments|outbox (admin only), /passed + /api/passed* + /api/topology (staff), driver 403 on all staff APIs, SSLCommerz IPN reachable without a session (returns ok:false for unknown tran_id), anon /api/fines 401.
- 2026-10-30 — Hosting for web testing: team chose Azure for Students (no credit card; Oracle Free Tier needs one). Added `deploy/` (rsync push of the working tree, one-time `setup.sh`: CPU torch, swap, systemd, Caddy, generated MEDIA_SECRET/admin password, `VLM_BACKEND=off`, `OTP_DELIVERY=log`). Found: `find_helmet_model()` prefers `helmet_cls.mlpackage` on every OS, and CoreML predict raises on Linux ("only supported on macOS"), uncaught in `violations/helmet.py:94`. Push excludes `*.mlpackage`; code fix proposed, not applied.
