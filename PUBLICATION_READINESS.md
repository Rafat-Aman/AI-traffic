# Publication readiness audit: Dhaka traffic-violation system

Audit date: 1 November 2026. Branch `rafat` at `2a6b63e`, plus the uncommitted `SUPERVISION.md` and `deploy/`.
Scope: this repository, plus the files outside it that the repository points to (footage folders, the footbridge
labels and the 499A report sources). Read-only: the only file this audit created is this one.

Conventions

- **pd** means person-day, 8 focused hours.
- The feasibility judgements assume about 8 pd per week across the team (4 people at about 2 days a week each).
  If the team has more time than that, the plan gets easier.
- Every number below cites a file, a log or a command. If a number could not be found, the text says
  **not found** and the item is listed in section 8.
- *Re-run* means I ran the command during this audit and got the same output (the commands are in Appendix A).
  *Copied* means the number comes from a file and was not re-run.
- Venue facts are marked **verified** (checked on the official site), **partly verified** (a sponsor or secondary
  listing) or **unverified**. Sources are in Appendix B.

---

## 1. Verdict

**First submission target: P1 (evidence-grade enforcement) at IEEE ITSC 2027. Regular-paper deadline 1 March 2027,
submissions open 1 December 2026 (verified on ieee-itsc.org/2027). Realistic: yes, if two conditions hold.**

Today nothing in the repository is ready to submit:

- No enforcement metric has been measured against ground truth on real Dhaka footage. That covers:
  - detector mAP (not found)
  - per-rule precision and recall
  - plate reading on real video
  - cross-camera matching
  - VLM verdicts
- The only real-footage number with a reference is within-camera ReID rank-1, which is not an enforcement metric.
- Every rule-accuracy figure comes from the team's own simulator (`REPORT_RESULTS.md`, X3).

The engineering is well ahead of the evidence:

- 1,285 tests pass.
- There is an evaluation harness with the metric code (`evaluation/`, `tools/evaluate.py`) and a written labelling
  protocol (`docs/EVALUATION.md`).
- The citation path is precision-first: human confirmation, a readable plate and a sealed clip are all required
  (`surveillance/citation.py:294`).

The shortest path to ITSC is a small but real **gold citation set**, a citation-level metric script and oracle hooks.
That is about 40–50 pd, which fits the 17 weeks if two conditions hold:

1. Recording permission and detector provenance are settled in November.
2. At least 2 hours of new footage is recorded by early December, at cameras where plates are at least 120 px wide.

The Kakoli footbridge camera alone cannot produce citations, because every plate there is unreadable
(`README.md:160`).

The two November deadlines are not realistic: IEEE IV 2027 on 15 Nov (verified) and CVPR 2027 on 16 Nov (verified).
ICME 2027 for P2 (30 Dec, partly verified) is a stretch, with a go/no-go on 13 Dec.

The fastest *real* publication is a paper this audit proposes: **P6, precision-first Bangla ANPR**. It has the most
held-out evidence today (167 test plates, an EasyOCR baseline, reproducible training scripts). It could go to
IET Image Processing (rolling submissions) by early February.

---

## 2. Module inventory

| Component | Where (paths) | Model / method; where the weights come from | Training data (source, size, licence) | Eval script | Metrics found (source) | Latency / hardware | Known limitations (source) |
|---|---|---|---|---|---|---|---|
| **Vehicle and person detection** | `surveillance/detector.py`, `vehicle_tracker.py`, `bd_vehicles_best.pt`, `models/bd_vehicles_best_{640,960}.mlpackage` | YOLOv8s, 80 epochs, 640 px, 16 classes (`SUPERVISION.md` D1). The 499A report says YOLOv11 (conflict C-1). **Who trained it, and on what, is unknown** (`SUPERVISION.md` DEC-2). Classes: Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, Leguna, Motorbike, MPV, Pedestrian, Pickup, PowerTiller, Rickshaw, Van, Truck, Wheelbarrow (`docs/EVALUATION.md` §1). CNG, rickshaw, easybike and leguna are covered. There is **no human-hauler class**: those vehicles are folded into Leguna (`SUPERVISION.md` D2). | **Not found.** Licence of the weights: AGPL-3.0, inherited from Ultralytics. | `tools/evaluate.py x1`, `evaluation/detection.py` (needs a labelled set) | mAP **not found** (`results/x1_detection.json`: "not measured"). Torch vs ONNX box agreement: 94.1 % / 96.5 % matched at IoU 0.5; this is agreement, not accuracy (`docs/CAPACITY.md`). | CPU i5-11400H, 960 px: 1,759 ms per frame mean at load 19 (`results/x2_tracking.json`). 0.917 s per frame at load ~7 (`docs/CAPACITY.md`). The whole machine saturates at about 3–4 fps (`docs/CAPACITY.md`). Mac M5 CoreML: 90 FPS at 640 px (`README.md`, Performance), not reproducible on this machine. | Without provenance, no test set can be shown to be held out. `docs/EVALUATION.md` §1 says use only footage recorded after the weights (dated 2026-10-24). |
| **Per-camera tracking** | `vehicle_tracker.py`, `surveillance/camera_motion.py` | ByteTrack by default; BoT-SORT optional, without ReID. Optical-flow camera-motion compensation. | n/a | `tools/evaluate.py x2`, `evaluation/tracking.py` (MOTA, IDF1, IDSW implemented) | Proxy only, 60 s of Kakoli side 1: 152 vs 147 tracks; BoT-SORT against ByteTrack IDF1 0.976, which is agreement, not accuracy (`results/x2_tracking.json`) | Tracker cost 3.58 ms/frame (ByteTrack) vs 25.42 ms/frame (BoT-SORT) (`results/x2_tracking.json`) | No MOT ground truth. The ByteTrack vs BoT-SORT choice is undecided (DEC-3). |
| **Helmet violation (V6)** | `violations/helmet.py`, `models/helmet_cls.pt` / `.mlpackage`, `training/{extract_motorbikes,label_sheets,build_helmet_data,train_helmet,eval_helmet_dhaka,eval_helmet_heldout}.py`, `training/helmet_labels.txt` | YOLO11s-cls at 224 px. Outputs P(*someone* on the motorbike is bare-headed) for the whole bike crop (`models/README.md`). **It does not count riders or pillions separately** and has no rule for more than 2 riders. Rule (`violations/helmet.py`): the bike must be coming toward the camera, inside the road zone, moving at 5 km/h or more, with a box of at least 100 px, and not stacked behind another bike. The decision is made once, when the track ends: at least 4 bare-head sightings at P ≥ 0.8, and at least 3/4 of all confident sightings (helmet means P ≤ 0.5). | Indonesian crops (revitotan/helmet-threerider, MIT): 1,372 helmeted / 227 bare. Dhaka crops from the Kakoli clip `TC_00006.MOV`: 157 H / 11 N (`models/README.md`). **The Dhaka labels were made by the AI assistant, by eye** (`models/README.md`). The label file has 352 H / 24 N / 224 X out of 1,232 crops (`training/helmet_labels.txt`, `training/helmet_crops_meta.json`). | `training/eval_helmet_dhaka.py`, `eval_helmet_heldout.py` (the crops are not on this machine, so not re-run) | Per crop, at threshold 0.9: 3/11 bare-headed caught, 0/155 helmeted flagged (`models/README.md`, copied). As a rule on held-out video: 0 cases, 0 of 11 bare-headed caught (`models/README.md`; that was an *older* rule). Current rule on Kakoli Part-1 (18 min): 16 reports, of which 7 helmeted, 1 bare-headed, 8 unclear. Those tracks were also used to tune the rule (`violations/helmet.py:19`). | Classified at most every 0.25 s per bike. Latency not found. | Recall is very low. One camera. Riders going away from the camera are not judged. On Linux, CoreML is chosen and then raises an error (`SUPERVISION.md` log, 2026-10-30; the fix is not applied). `models/README.md` describes a superseded rule. |
| **Lane violations: V5 solid line, V9 bus lane, V4 wrong side, V8 banned class** | `violations/lane_change.py`, `bus_lane.py`, `wrong_side.py`, `restricted_vehicle.py`, `violations/zones.py`, `web/editor.js`, `surveillance/autozone.py` | Geometry comes from **manually drawn polygons and lines** in the zone editor. An automatic proposal is also available: carriageways are bands of lateral position on the road plane, and lanes are peaks in lateral vehicle density (`surveillance/autozone.py` docstring). Solid lines can only be drawn by hand. There is no learned segmentation. Rules: V5 needs the vehicle at least 0.6 m clear of the line on both sides for at least 0.5 s, and straddling does not count. V9 needs at least 25 m and 4 s inside a drawn bus lane during its hours (`SUPERVISION.md` U2). V4 needs at least 8 m of travel against the allowed direction (`README.md`). | n/a (rules, not trained) | `tests/test_violations.py`, `tests/test_bus_lane.py`, `tools/bench_accuracy.py` (V4 only) | V4 synthetic: 2/2 caught with default errors, 12/12 clean, 0 false alarms (re-run). V5, V8 and V9 on real footage: **not found**. | Rules run per frame; part of `Pipeline.process` | There is no labelled lane set. `videos/ForLaneViolation/` holds 2 unlabelled 8K clips, 104.5 s in total (ffprobe). The committed `violation_zones.json` has no lane lines. |
| **Overspeeding (V2)** | `violations/speed_violation.py`, `surveillance/speed.py`, `surveillance/autozone.py` | **Homography** from a hand-measured 4-corner road zone (width and length typed in), or **automatic self-calibration**: the vanishing point comes from straight track lines via RANSAC, and scale and lens come from typical vehicle sizes (`surveillance/autozone.py`). Speed is a least-squares fit over the last 1 s of road-plane positions, with outliers refitted (`surveillance/speed.py`). Limits: 60 km/h city, 80 km/h highway and expressway (`violations/zones.py:68`). A vehicle is flagged above limit + max(3 km/h, 5 %), sustained for 1 s, and its section average over at least 10 m and 1.5 s must also be over. Auto-calibrated zones use a margin of at least 1.5 × the uncertainty, which defaults to 15 % (`speed_violation.py`). **V2 is advisory and never cited** (`surveillance/citation.py:77`). | n/a | `tools/bench_accuracy.py`, `tests/test_accuracy.py` | Synthetic: MAE 2.08 km/h, bias 0.77, p95 7.11, MAPE 12.4 % (n = 4,907). Clean: MAE 0.50, p95 2.54 (re-run). **There is no ground-truth speed anywhere: no GPS, GNSS, radar or LiDAR** (a repository search for these terms finds nothing). | n/a | Two documents disagree on auto-calibration accuracy: "±8–9 %" (`README.md:71`) vs "±15 %" (`surveillance/autozone.py:15`). |
| **Number plates (ANPR)** | `surveillance/anpr.py`, `models/plate_detector.onnx`, `models/bd_plate_chars.pt`, `training/{build_ocr_ds,train_chars,eval_ocr,grid_anpr,calibrate_anpr}.py` | Plate detector: morsetechlab YOLOv11 ONNX, not fine-tuned, **AGPL-3.0**. Characters: YOLO11n at 320 px, 45 epochs (`models/README.md`). **Character set: 22 tokens**: digits ০–৯, 3 districts (ঢাকা, খুলনা, যশোর), মেট্রো, and 8 class letters (ক গ চ ট ল হ ব ন) (`surveillance/anpr.py`). Layout: **two rows assumed** (BRTA style: line 1 is city [metro]-letter, line 2 is series-number); the digits are taken from below the words (`parse_plate`). Single-row plates are not handled. Reading rule: at least 3 crops agree on all 6 digits, each digit at confidence ≥ 0.7, plate at least 110 px wide (`surveillance/anpr.py:36-37`). | NightLPBD (CC BY 4.0, cite COMPAS 2025): 963 / 216 / 189 train / val / test, night images, with daytime simulated (`models/README.md`) | `training/calibrate_anpr.py`, `eval_ocr.py` (NightLPBD is not on this machine, so not re-run) | Characters: val mAP50 0.968, mAP50-95 0.72. On 167 held-out test plates, with video degradation simulated: **0 % wrong numbers** at 120, 150 and 200 px; read 62 / 60 / 75 %; wrong city or letter 2.9 / 4.1 / 6.6 %. EasyOCR baseline: 0–11 % at 50–100 px, 25 % at 220 px (`models/README.md`, copied). Read rate on real Dhaka video: **not found** (`docs/VEHICLES.md:63`). | About 0.5 s CPU per vehicle when every vehicle's plate is read (`SUPERVISION.md` WP17) | Only 3 districts. Training plates are night-time. All Kakoli plates are 35–100 px and unreadable (`README.md:160`). |
| **Owner lookup and citation** | `surveillance/registry.py`, `citation.py`, `disputes.py`, `payments.py`, `invoice.py`, `mailer.py`, `web/notice.html`, `web/portal.*` | The registry is **admin-entered or CSV, with demo/synthetic rows**. There is no BRTA connection. Lookup is an exact plate-key match (`registry.py:90`). A citation requires: (a) the code is not advisory (V2, V10 and V11 are), (b) a **controller has confirmed the case**, (c) the plate was read, and (d) the clip is on disk (`citation.py:294`). The citation record holds: number, issuer, case facts, a SHA-256 for every evidence file plus a bundle hash printed on the notice, the fine at issue (no fines are set: `FINE_SCHEDULE_BDT`), and the plate. Disputes are allowed within 30 days: the 8B model gives an advisory re-check and a different person decides (four-eyes) (`SUPERVISION.md` E9). | n/a | `tests/test_citation.py`, `test_disputes.py`, `test_registry.py` (functional tests) | Citation-level precision **not found**. Evidence bundle: 6.28 MB per case (`SUPERVISION.md` E2) | n/a | The tamper evidence has no external timestamp (`citation.py` docstring). There is no real registry, so owner-match error cannot be measured except through plate error. |
| **VLM explanation** | `surveillance/vlm.py`, `vlm_cascade.py`, `screening.py`, `tools/{setup_vlm.sh,vlm_server.sh,vlm_check.py,vlm_probe.py}`, `docs/VLM.md` | Cascade: a kinematic pre-screen, then **Qwen3-VL-2B** as screen, then **Qwen3-VL-8B** as reasoner on demand. Both are Q4_K_M GGUF on llama.cpp, Apache-2.0 (checked on Hugging Face). There is a single-model fallback, **Qwen2.5-VL-3B, whose licence is `qwen-research`, not Apache-2.0 as `models/README.md` says** (Hugging Face model card). A cloud backend (Anthropic or OpenAI-compatible) is optional and not used. **Inputs**: up to 2 photographs at 768 px (close-up and scene, with the drawn label painted out; for V10/V11 the contact frame goes first) plus **structured evidence** as text: the rule claim, the specific innocent explanation, the detector and rule measurements (speed removed except for V2, V10 and V11), and the cross-camera journey (`vlm.py:146` SYSTEM, `:472` build_prompt). **Output**: JSON with `visible` (written first), `present`, `pre_event`, `event`, `aftermath`, `cause`, `assessment` (supported / contradicted / insufficient), `confidence`, `summary` and `other_cameras`. **Checks** (`vlm.py:586`): unparsable output becomes "insufficient"; `present = false` or the vehicle class missing from `visible` downgrades "supported"; speed talk is stripped; a V6 one-word head check sets the verdict. **The output never decides a citation**: a human confirms. | Not fine-tuned. Q-LoRA was planned but not done (`SUPERVISION.md` U11). | `tools/vlm_probe.py` (needs live servers, so not re-run) | 3B: 1/3 probes, then 3/3 after the prompt and guard change, with the change not ablated (`models/README.md`). Cascade: 8/8 rendered probes, sycophancy set 3/3 (`docs/VLM.md`). **None of the probes is a real Dhaka case.** Accuracy, hallucination rate and agreement: **not found**. | i5 CPU under load 14–26: 2B median 84.7 s per case (n = 5), 8B median 274.6 s (n = 4); 8B RSS about 10 GB (`docs/VLM.md`) | Verdicts are noisy: the same case came back "supported 90 %" and then "contradicted 0 %" (`models/README.md`). 8B confidence is uncalibrated. |
| **Multi-camera tracking and ReID** | `surveillance/reid.py`, `link.py`, `embed.py`, `reid_backbone.py`, `topology.py`, `section_speed.py`, `geo.py`, `api_map.py`, `tools/reid_eval.py`, `research/cityflow/` | Matching happens in **metres on the road plane** with log-likelihood-ratio cues: class (a hard rule), transit time, lateral position, speed and width, colour, and an embedding. Assignment is one-to-one and can refuse. Topology: gates and links with travel-time windows (`docs/TOPOLOGY.md`). Section speed is distance over time, with plausibility bounds (`section_speed.py`). ReID: **ResNet50-IBN-a** trained by the team's CityFlowV2 notebook; weights are gitignored, sha256 in `research/cityflow/README.md`. MobileNetV3-Small (ImageNet) is the fallback. | AI City 2021 ReID: 295 ids, 44,378 images. VehicleX: 1,362 ids, 192,150 images. CityFlow S02 held out (`research/cityflow/README.md`). These datasets are under AI City data terms; redistribution of derived weights is **unverified**. | `tools/reid_eval.py`; `evaluation/retrieval.py` (pair P/R, MTMC IDF1 implemented); `tests/test_reid.py`, `test_topology.py` | CityFlow S02 (US roads, **the notebook's own pipeline, not `reid.py`**): ReID mAP 0.427, rank-1 0.560; MTMC IDF1 46.50 (2-fold) (copied). Dhaka, *within one camera*: ReID same-class rank-1 98.4 % vs MobileNet 75.8 % (copied). Synthetic matcher precision 95.3–99.3 % (`README.md`). Topology, synthetic: P 0.974 / kept 0.908 (`SUPERVISION.md` WP19). Cross-camera on Dhaka: **not found**. | ReID about 1 s per vehicle (6 crops), MobileNet 8–12 ms (`research/cityflow/README.md`) | **Cameras tested on real Dhaka footage: 1.** Two 2-camera clip pairs exist but are unlabelled: Kakoli 2 × ~121 s and Kuril 2 × ~72–73 s (ffprobe). Weak footbridge labels exist outside the repo (section 2.1). |
| **Other rules: V1 bus stopping, V3 kerb parking, V7 lane blocking, V10 erratic, V11 near-collision** | `violations/{bus_blocking,illegal_parking,road_blocking,stationary,erratic,near_collision}.py`, `engine.py` | Dwell timers with a "head of queue" check, at least 3 vehicles passing, and exemptions for jams and signals (`README.md`, Accuracy). V10 and V11 are kinematic *candidates* that the VLM judges; they are advisory. | n/a | `tools/bench_accuracy.py`, `tests/test_behaviour.py` | V1/V7 synthetic: 2/2 and 4/4, 0 false alarms (re-run). V3 on Kakoli Part-2: 0 → 32 cases after the kerb fix, precision not given (`docs/COMPARISON_SHIHAB.md`). Real precision and recall: **not found**. | V10/V11: 60 vehicles × 300 frames in 0.46 s (`SUPERVISION.md` WP2) | False-positive rate of V10/V11 on Dhaka not measured (`SUPERVISION.md` WP2) |
| **Impact and optimisation** (not enforcement) | `surveillance/traffic_state.py`, `impact.py`, `optimize/*`, `tools/signal_sim.py` | Impact score S = Σ wᵢΔᵢ, with NNLS calibration and a Spearman check. Max-pressure signals on the team's own mesoscopic simulator (not SUMO). | Synthetic | `tools/evaluate.py x4 / optimize`, `tools/signal_sim.py` | Synthetic ρ: 0.90 calibrated vs 0.95 equal weights (`results/x4_impact.json`). Max-pressure mean delay 71.0 ± 1.3 s vs 135.4 ± 1.4 s for tuned fixed-time (re-run, `--quick`). | 6–8 ms per frame at 80 vehicles (`SUPERVISION.md` A1) | No human severity ratings exist. Nothing has driven a real signal. |
| **System** | `surveillance/server.py` (+ `api_*.py` routers), `store.py`, `worker.py`, `ingest.py`, `storage_mgr.py`, `sources.py`, `web/*`, `deploy/` | One FastAPI process, SQLite, a thread per camera, an ingest queue, a retention manager, roles (driver / controller / admin) with an audit log, and a web dashboard. Inputs are files or HTTP/RTSP phone streams. Edge: the Mac CoreML path. Cloud: an Azure test VM via `deploy/`, with the VLM off; the scripts have not yet been run on a VM (`SUPERVISION.md` P5). | n/a | `python -m unittest discover -s tests` | **1,285 tests, OK, 161.1 s** (re-run) | Capacity: about 3 video-hours per day per CPU at 30 fps analysis (`docs/CAPACITY.md`) | SQLite, not PostGIS (C-6). Rate limiting is in process. **Not deployed live.** |

### 2.1 Datasets, labels, logs, notebooks and drafts

| Item | Location | Format / size | Notes |
|---|---|---|---|
| Own footage, Kakoli side 1 (uploaded copy) | `data/videos/20261025_092449_side1_1.mp4` (gitignored) | 121.0 s, 1242×1440, 30 fps | The only clip any Dhaka measurement used (X2, X6). The file inside the zip is dated 2026-09-26, so it *may predate* the detector. |
| Own footage, 2026-10-29 drop | `videos/ForNRP/` (3 clips), `videos/ForLaneViolation/` (2), `videos/ForMulticameraTrack/{kakoli,kuril}/` (2 + 2), plus `videos/for shihab.zip` (4.5 GB). All gitignored. | ForNRP: 297 s at 4K, 60 fps. ForLaneViolation: 104.5 s at 8K. Kakoli: 2 × ~121 s. Kuril: 72.0 s + 73.5 s. **About 13 minutes of unique footage, all unlabelled.** | One afternoon at 4 sites. Night and rain: not found. |
| Kakoli Part-1 / Part-2, `TC_00006.MOV`, `DataSet_2min.mp4` | Not on this machine | — | Used for helmet training and testing and for the V3/V6 fixes (`models/README.md`, `violations/helmet.py`) |
| Footbridge cross-camera weak labels | `/home/rafat/Storage/STUDY/spring26/models/pipeline_output/` (outside the repo) | `matches.csv` 236 accepted pairs, `feedback_wrong_pairs.csv` 32 human-rejected pairs, `vehicles.csv` 1,236 rows (row counts, header excluded) | Produced by a matcher with partial human review, so precision-biased. Recall cannot be computed from them. |
| Helmet crops and labels | `training/helmet_crops_meta.json`, `training/helmet_labels.txt` | 1,232 crop records; labels H 352 / N 24 / X 224 | The labels were made by the AI assistant (`models/README.md`). No human spot-check is recorded. |
| NightLPBD, revitotan helmet, AI City 2021 ReID, VehicleX, CityFlowV2 | External, not on this machine | see section 2 | Licences: CC BY 4.0, MIT, AI City terms (unverified), VehicleX terms (unverified) |
| X7 query test data | `tests/data/queries.jsonl` | 16 lines, marked SYNTHETIC | Not real operator questions |
| Tracker outputs | `results/x2_tracks/{bytetrack,botsort}.txt` | MOT format | Usable as CVAT pre-annotation |
| **Human-labelled test set** | **None in the repo** | — | The 32 rejected footbridge pairs are the only human judgements found |
| Experiment logs | `results/*.json` (x1–x7, optimize, optimize_raw) | JSON with provenance | There are no TensorBoard or W&B exports, and `runs/detect` is empty |
| Notebooks | `research/cityflow/cityflowv2.ipynb` | 19.3 MB **with outputs**, including a 15.7 MB embedded demo video of CityFlow footage (`research/cityflow/README.md`) | On public branches; see section 7 |
| Drafts | `REPORT_RESULTS.md` (generated Chapter 4). 499A report LaTeX at `~/Downloads/rafat/final paper/NSU_CSE_499_Dhaka_Traffic_Management_local/` (73 pp PDF; the publication list is empty; Chapter 4 is empty per `SUPERVISION.md`). Slides at `~/Downloads/Paper Presentation - 499A*.pdf`. `~/Downloads/rafat/{literature_review.md,references.bib}` (18 bib entries). | — | The intro and related work can be reused. There is no paper draft. |

---

## 3. Evidence checklist A–H

### A. End-to-end citation accuracy: **missing**

| Item | Status | Evidence | Gap / work | Effort |
|---|---|---|---|---|
| Gold citation set (every true violation, plate and owner match in held-out video) | missing | `training/ground_truth_template.csv` is a header only. `evaluation/ground_truth/` does not exist. | Record at least 2 h at 4+ sites where plates are ≥ 120 px, after 2026-10-24, and never given to whoever trained the detector. Label `events.csv` per `docs/EVALUATION.md` §3 (4–6 h per 30 min of footage). Add a `plates.csv` for every true violator. Build a test registry from the gold plates plus near-miss distractor plates (one digit off), so that misreads show up as wrong-owner matches. | Recording 3 pd; events 2–3 pd; plates 1 pd; registry 0.5 pd |
| Citation precision, false citations per 1,000 vehicles, recall per type | partial | `evaluation/events.py` does per-rule event P/R/F1 and false alarms per hour (`tools/evaluate.py x3 --duration`). It has no citation-level view and no per-vehicle denominator. | New `evaluation/citations.py` plus a `tools/evaluate.py e2e` subcommand. Join cases, then confirm, plate, registry and citation status (store tables). Take the denominator from the `passed_vehicles` counts (`surveillance/passed.py`, setting `passed` is on by default). | 3–4 pd |
| Oracle ablation (swap each module for ground truth) | missing | Pattern exists: `tools/bench_accuracy.py` already feeds `Pipeline` a simulated detector. | Hooks: **detection and tracking** at `surveillance/pipeline.py:160` (`self.det.track(...)`), replaced by a `GTReplayDetector` reading MOT `gt.txt`. **Class**: the same replay with ground-truth classes. **Calibration**: compare measured with auto zones (`road.scale_source`). **Helmet**: `NoHelmet.detector` callable (`violations/helmet.py:61`) set to a ground-truth lookup. **Plate**: `CameraWorker._plate_result` (`surveillance/worker.py:380`). **Owner**: `registry.by_plate` (`surveillance/registry.py:90`). **Review**: run two reviewers' confirm/reject logs as alternative "human" modules. **VLM veto** (P2 only): `vlm_cascade` verdict. Tracking ground truth costs 10–20 h per 60 s (`docs/EVALUATION.md` §2), so run the tracking oracle on 2 × 60 s only. | Hooks 5–7 pd; MOT ground truth 2–3 pd |

### B. Public benchmarks

| Module | Benchmark | Reads the format today? | Converter / work needed | Effort | Notes |
|---|---|---|---|---|---|
| Helmet | AI City 2023/2024 Track 5 (mAP) | No | Our model classifies a whole motorbike as "anyone bare-headed". It does not detect riders, so **mAP cannot be reported** without a new rider-level detector. A converter from AIC boxes to one bike crop with an "any NoHelmet" label gives a classification accuracy. | 3–4 pd for classification; 8–10 pd + GPU for a detector | AIC data terms and download size unverified. A 2026 helmet track was not found (Appendix B). |
| Plates | Bangla LPDB-A, Zenodo 4718238: **CC BY 4.0, 874.1 MB** (verified); 1,928 vehicle images, 2,662 crops, 102 classes | No | Map the LPDB-A labels onto our 22 tokens. Report the 6-digit number accuracy everywhere, and district and letter accuracy only where they fall in our vocabulary. Out-of-vocabulary districts are themselves a finding. The download is under 1 GB. | 2–3 pd | Cite the DICTA 2021 paper (the record asks for it). UFPR, RodoSol and CCPD can only test the plate *detector*: Latin and Chinese characters do not test Bangla OCR. |
| Vehicle detection | Dhaka-AI (3,953 images, 21 classes) | Partly: `tools/evaluate.py x1` reads YOLO `data.yaml` | Convert VOC XML to YOLO with a 21 → 16 class map (for example human hauler → Leguna, minibus → Bus, scooter → Motorbike). Report the ambiguous classes separately. | 2 pd | Licence reported as GPL-2.0 by Dataset Ninja (unverified). **Contamination risk** if Dhaka-AI trained `bd_vehicles_best.pt` (unknown). |
| Speed | BrnoCompSpeed (LiDAR ground truth) plus own GNSS drives | No | Build zone configs from BrnoCompSpeed's calibration, run `Pipeline`, and match to ground-truth vehicles by time and lane. Report mean, median and p95 km/h error. On CPU this is roughly 14 h of compute per hour of 50 fps video, from 3–4 fps machine throughput (`docs/CAPACITY.md`), so use a subset or a rented GPU. Own drives: a phone or GNSS logger in a probe car passing the camera 10+ times. | 5–7 pd + compute; GNSS 2–3 pd | Access is by request; licence reported as non-commercial (aggregator, unverified). The download is likely over 1 GB, so **ask before downloading**. |
| Multi-camera | CityFlow / AI City MTMC (IDF1) | Partly: the notebook scores CityFlow, but with **its own pipeline**. `reid.py` has never run on CityFlow. | Adapter from CityFlow's per-camera calibration to zone configs in metres, then run `link.run_linking` (`surveillance/link.py:67`) and `evaluation.retrieval.mtmc_idf1`. | 6–8 pd + GPU for detection | Without this, quoting IDF1 46.5 next to our matcher is misleading |
| VLM | WTS / AI City safety captioning and VQA; counterfactual tests | No | Our prompt checks a finding; it does not caption. WTS shows only general ability (low value). Counterfactual pairs are the stronger test: extend `tools/vlm_probe.py` into a `tools/vlm_eval.py` runner (see D). | WTS 3–4 pd (optional); counterfactual 3–4 pd | WTS size and licence unverified |
| Lane violation | No public benchmark | No internal labelled set | Label V4, V5, V8 and V9 events on `videos/ForLaneViolation/` and the new footage, as part of A | in A | Dhaka has few painted solid lines (`violations/lane_change.py` docstring), so expect small counts |

### C. Baselines and ablations

| Already present | Source |
|---|---|
| ByteTrack vs BoT-SORT (proxy, no ground truth) | `results/x2_tracking.json` |
| Our Bangla OCR vs EasyOCR Bengali | `models/README.md` |
| Helmet: Indonesian-only classifier flagged 82 % of helmeted Dhaka riders. A head detector found no head on 43/73 riders. | `models/README.md` |
| Helmet rule settings: Kakoli Part-1 reports 85 → 16, helmeted false positives 48 → 7 (tuned on the same tracks) | `violations/helmet.py:14-19` |
| Appearance cues (colour histogram, CIELAB, MobileNet, ReID) and ReID settings (size, crop, flip) | `research/cityflow/README.md` |
| Speed estimator, least-squares vs end-points: MAE 2.6 → 2.2 km/h, p95 10.9 → 8.3 | `surveillance/speed.py` docstring |
| VLM 3B before and after the prompt and guard change: 1/3 → 3/3 (not ablated) | `models/README.md` |
| Detector backend torch vs ONNX (agreement) | `docs/CAPACITY.md` |
| Fixed-time vs max-pressure signals (synthetic) | `results/optimize.json` |

Missing for P1 and P2:

- **A second detector.** An off-the-shelf YOLO11 on COCO with a class map: 1 pd. A retrain needs the unknown training data.
- **A second Bangla ANPR.** One open baseline: 2 pd.
- **Commercial ANPR.** Paid, so ask first.
- **A second open VLM of similar size:** 2 pd plus compute.
- **The VLM with and without structured evidence**, by removing the facts and claim blocks in `vlm.build_prompt`: 1 pd plus compute.
- **The VLM with and without the guard**, by turning off the downgrade in `to_explanation`: 0.5 pd.
- **A cloud VLM as reasoner.** Paid, so ask first.

### D. VLM evaluation: **partial (smoke tests only)**

| Item | Status | Evidence | Work | Effort |
|---|---|---|---|---|
| Hallucination rate | missing | — | Two raters mark each claim in 60 explanations as supported or unsupported by the photographs | 2 pd |
| Agreement with detector outputs | missing | `present` field and subject guard exist (`vlm.py:586`) | Score `present` and `visible` against the ground-truth class from the gold set | 1 pd |
| Counterfactual consistency | missing | Rendered probes only (`docs/VLM.md`, 8/8) | Pair each supported real case with: (a) the same photographs but a claim about another vehicle or rule, (b) photographs of a non-violating vehicle of the same class from `passed_vehicles`, (c) a frame from before the event. Measure the verdict flip rate. | 3–4 pd + about 4–5 CPU-days for 2B + 8B on about 200 cases at the measured 85 s and 275 s per case |
| Human or officer ratings | missing | — | Two raters, 1–5 faithfulness on 30 or more explanations (`docs/EVALUATION.md` §5). Ideally one traffic officer. | 1–2 pd |
| Labelled case set | missing | 3 + 8 probes, none from Dhaka | At least 120 real cases labelled supported / contradicted / insufficient by 2 raters | 2–3 pd |

### E. Robustness: **missing**

There is no breakdown of any kind. All footage is from one afternoon (file names `20261029_15xx`). No night or rain
footage was found. Plate widths are known per read (`PlateRead.plate_px`). Work:

- Tag every gold-set event with day/night, rain, site, resolution, plate width, occlusion and class: 1 pd.
- Write a breakdown script in `evaluation/`: 1 pd.
- Record night and rain sessions as part of A, if permitted.

### F. Efficiency: **partial**

What exists:

- CPU per-stage timings, all taken under load (`results/x2_tracking.json`, `docs/CAPACITY.md`, `docs/VLM.md`).
- Mac M5 figures (`README.md`), which cannot be reproduced on this machine.

GPU latency and a cost per camera are **not found**. Work:

- Re-time on an idle CPU: 1 pd.
- One rented-GPU run for detector, ReID and VLM per-stage latency: 1 pd. This is paid, so ask first.
- A cost-per-camera model built from `docs/CAPACITY.md` and cloud prices: 0.5 pd.

### G. Ethics, privacy and legal: **partial**

| Item | Status | Evidence / gap |
|---|---|---|
| Permission to record and use footage | **not found** | No permission letter, ethics approval or consent note anywhere in the repo or the drafts folder → section 8 |
| Redaction of non-violators | **missing** | No face or plate blurring code (a repository search for redact, anonymise, blur and pixelate finds nothing). `passed_vehicles` stores every vehicle; plates of all vehicles are off by default (`anpr_all`). Passed-vehicle records are purged after 30 days (`PASSED_RETENTION_DAYS`, `SUPERVISION.md` WP17). |
| Retention policy | present | Raw video 7 days, dismissed cases 30 days, confirmed evidence 1,095 days only after the authority confirms, embeddings 30 days (`docs/STORAGE.md`) |
| Human-in-the-loop review | present | Citation requires controller confirmation (`citation.py:294`); four-eyes on disputes (`SUPERVISION.md` E9) |
| Appeal path | present (prototype) | `surveillance/disputes.py`: 30 days, an advisory 8B re-check, a human decides. No uploads, no second appeal. |
| Licences of models and datasets | partial, **errors found** | See section 7. `models/README.md` lists Qwen2.5-VL-3B as Apache-2.0, but it is `qwen-research`. AGPL-3.0 applies to the detector, plate detector and helmet weights. The repo has **no LICENSE file**. |

Effort: draft an ethics and data statement: 1 pd. Add a redaction pass for any released frames: 3–4 pd (needed for
P3 and for paper figures).

### H. Reproducibility: **partial**

| Item | Status | Evidence / gap |
|---|---|---|
| README and environment | partial | `README.md` is detailed. `requirements.txt` gives version *ranges* only and was tested on macOS / M5. There is no lock file, Dockerfile or environment file. |
| Seeds and configs | partial | Seeds are fixed in the training scripts and benchmarks (`training/build_ocr_ds.py`, `calibrate_anpr.py`, `bench_accuracy.py` seed 0, `signal_sim.py` seeds 10+). Settings live in `DEFAULTS` dicts. |
| Scripts that regenerate tables | present | `tools/evaluate.py report` → `REPORT_RESULTS.md` from `results/*.json`. The OCR and helmet numbers come from `training/*`. The bench and signal numbers re-ran identically during this audit. |
| Detector training | missing | There is no script or data for `bd_vehicles_best.pt` (DEC-2) |
| What can legally be released | unclear | Code: yes once a licence is chosen (AGPL-compatible, because the code uses Ultralytics). Weights: AGPL-derived; ReID weights depend on AI City terms. Footage: not without permission and redaction. |

Effort: lock file, LICENSE, model cards, strip notebook outputs, fix the README licence line: 2–3 pd.

---

## 4. Paper readiness

Scale: 0 = idea only, 1 = components but no evaluation, 2 = partial evaluation, 3 = submission-ready.

### P1: Evidence-grade city-scale enforcement (flagship). **Score 1**

**Research question (sharpened):** in a precision-first pipeline where every stage may abstain, how often is an issued
citation wrong, how many true violations become citations, and which module limits each?

The code already abstains at every stage:

- ANPR returns "unreadable" (`anpr.py:36-37`).
- The VLM returns "insufficient".
- ReID refuses to link.
- Some codes are advisory.

That makes risk–coverage curves a natural headline.

**Evidence present:**

- The citation gate (`citation.py:294`).
- Synthetic rule benchmark (re-run).
- ANPR on held-out NightLPBD plates (`models/README.md`).
- Helmet per-crop test (`models/README.md`).
- Evaluation harness and labelling protocol (`evaluation/`, `docs/EVALUATION.md`).
- Retention, review and appeal design (section 3G).

**Blocking gaps:**

- No gold set.
- No real per-rule precision or recall.
- No detector mAP.
- No plate read rate on real video.
- No citation-level metric.
- No oracle ablation.
- Detector provenance unknown.
- 13 minutes of footage from one afternoon.

| Experiment | Where to implement | Expected output | Effort |
|---|---|---|---|
| E1.1 Gold set (≥ 2 h, ≥ 4 sites, plates ≥ 120 px) | `evaluation/ground_truth/violations/events.csv`, new `.../plates.csv`, `README.txt` (protocol in `docs/EVALUATION.md` §3) | Counts per rule, site and condition | 6–7 pd |
| E1.2 Detector mAP, held-out Dhaka plus Dhaka-AI | `tools/evaluate.py x1`; new `tools/convert_dhaka_ai.py` | Table: AP@0.5 per class, 2 test sets | 3–3.5 pd |
| E1.3 Per-rule precision / recall / false alarms per hour | `tools/evaluate.py x3 --cases ... --duration ...` | Table: P, R, FA/h per V-code with Wilson CIs | 1 pd |
| E1.4 Plate reading on real video | new `plates.csv`; the `anpr.vote` outputs in the store | Table: read rate and wrong-number rate by plate width | 1.5 pd |
| E1.5 Citation-level metrics | new `evaluation/citations.py`, `tools/evaluate.py e2e` | **Headline table**: citation precision, false citations per 1,000 vehicles, recall per type | 3–4 pd |
| E1.6 Oracle ablation | `pipeline.py:160`, `helmet.py:61`, `worker.py:380`, `registry.py:90` | **Figure**: error budget (share of misses and false citations per module) | 7–10 pd |
| E1.7 Reviewer agreement (2 reviewers, blind) | export from the Review screen | Cohen's κ, and reviewer-caused errors | 1 pd |
| E1.8 Robustness breakdown | `evaluation/` breakdown script | Table: P / R by condition and plate width | 2 pd |
| E1.9 Efficiency | `docs/CAPACITY.md` method | Table: per-stage latency (CPU idle, plus GPU if approved), cost per camera | 2–3 pd |
| Writing, ethics and data statement | — | 6-page ITSC paper | 8–10 pd |
| **Total** | | | **≈ 40–50 pd** |

**Main risk:** too few readable plates and too few events per rule. `docs/EVALUATION.md` asks for at least 20 true
events per rule. Without them, report counts, not rates. Also, recording permission is undocumented.

### P2: Faithful VLM violation explanations. **Score 1**

**Evidence present:**

- The grounded prompt with structured evidence and an innocent-explanation slot (`vlm.py:146`, `:472`).
- Guards that can only weaken a verdict (`vlm.py:586`).
- The two-tier cascade with recorded stages (`vlm_cascade.py`).
- 8/8 rendered probes and a 3/3 sycophancy set (`docs/VLM.md`).
- Measured CPU cost.

**Blocking gaps:** no real cases, no labels, no counterfactuals, no ablations, no hallucination rate.

| Experiment | Where | Output | Effort |
|---|---|---|---|
| E2.1 Real case set, ≥ 120, labelled by 2 raters | cases from the E1.1 runs; `evaluation/ground_truth/vlm/cases.csv` | Confusion matrix; **false-support rate** | 2–3 pd |
| E2.2 Counterfactual pairs (claim swap, evidence swap, time shift) | new `tools/vlm_eval.py` (extends `tools/vlm_probe.py`) | Table: verdict flip rate per counterfactual type | 3–4 pd |
| E2.3 Ablations: ± structured evidence, ± guard, 2B / 8B / 3B, ± cross-camera journey | flags in `vlm.build_prompt`, `to_explanation`, cascade settings | Table: false support, coverage, latency | 3 pd + ~4–5 CPU-days |
| E2.4 Claim-level hallucination | 60 explanations, 2 raters | Hallucinated-claim rate | 2 pd |
| E2.5 Veto value | VLM "contradicted" vs reviewer reject | False positives removed at a given recall cost | 1 pd |
| Writing | — | ICME (6 pp, unverified) | 6–8 pd |
| **Total** | | | **≈ 20–25 pd** |

**Main risk:** E2.1 depends on P1's gold footage, so the two papers compete for the same weeks. CPU inference takes
minutes per case. A cloud reasoner baseline needs paid API approval.

### P3: Dhaka multi-violation benchmark. **Score 1 (barely)**

**Evidence present:** about 13 minutes of unlabelled 4K/8K footage from 4 sites; a labelling protocol and metric code
(`docs/EVALUATION.md`, `evaluation/`).

**Blocking gaps:**

- No recording permission.
- No redaction.
- No labels.
- No data licence.
- No datasheet.
- At least 2–5 h of footage across times and weather is needed.
- Baselines are needed per task.

| Experiment | Where | Output | Effort |
|---|---|---|---|
| Permission, ethics approval, release licence | — | Signed approvals; CC BY-NC or similar | 3–5 pd plus waiting time |
| Redaction of faces and non-violator plates | new `tools/redact.py` | Released frames with faces and plates blurred | 3–4 pd |
| Record and label 3–5 h (boxes, tracks, events, plates, helmet per rider) | CVAT; formats from `docs/EVALUATION.md` | Dataset plus splits | 40–60 pd |
| Baselines per task, datasheet, Croissant metadata, hosting | `tools/evaluate.py` | Benchmark tables | 10–15 pd |
| **Total** | | | **≈ 60–85 pd + legal** |

**Main risk:** legal and ethical clearance to release public-road footage of identifiable people.

### P4: Cross-camera violations. **Score 1**

**Evidence present:**

- The matcher with refusal and topology (`reid.py`, `topology.py`).
- Section speed with plausibility bounds (`section_speed.py`).
- Synthetic precision and recall (`README.md`, `SUPERVISION.md` WP19).
- CityFlow numbers from the notebook's pipeline (`research/cityflow/README.md`).
- Within-camera Dhaka ReID.
- Two unlabelled 2-camera pairs (Kakoli, Kuril).
- Weak footbridge pair labels.

**Blocking gaps:**

- No Dhaka cross-camera ground truth.
- No section-speed ground truth.
- Camera spacing and clock offsets unknown.
- Our matcher has never been run on CityFlow.

| Experiment | Where | Output | Effort |
|---|---|---|---|
| E4.1 Cross-camera ids for Kakoli and Kuril (≥ 40 shared vehicles each) | `evaluation/ground_truth/crosscam/ids.csv` (`docs/EVALUATION.md` §6) | Pair P/R; MTMC IDF1 where MOT ground truth exists | 2–3 pd |
| E4.2 Error propagation: injected wrong links → section-speed error | `section_speed.py` + new `tools/link_noise.py` | **Figure**: section-speed error vs wrong-link rate | 2 pd |
| E4.3 GNSS probe-car runs between two cameras ≥ 50 m apart | `section_speed.py` with `clock_offset_s` | Table: section-speed error vs GNSS | 3–5 pd |
| E4.4 Our matcher on CityFlow S02 | adapter + `link.run_linking` | IDF1 comparable with the notebook's 46.5 | 6–8 pd + GPU |
| Writing (workshop, 6 pp) | — | — | 5 pd |
| **Total** | | | **≈ 18–23 pd** |

**Main risk:** the clips are short (~72–121 s), so there are few shared vehicles. The Kakoli views may be under 50 m
apart, which `section_speed.py` rejects as `cameras_too_close`.

### P5: Supervised pilot study. **Score 0**

Nothing pilot-like exists. The system is not deployed, there is no agreement with an authority, and the Azure VM is not
yet set up (`SUPERVISION.md` P5).

**Needed:**

- A memorandum of understanding (MoU) with DMP/BRTA or a city office.
- 4–8 weeks of supervised operation on 2+ cameras.
- Review-workload logging (time per case, which the audit log already partly records).
- A failure-mode taxonomy.

Effort: ≈ 30–40 pd plus the partner's time. **Main risk:** no partner.

### P6 (proposed): Precision-first Bangla ANPR from traffic video. **Score 2**

**Research question:** under CCTV conditions (small, blurred, compressed plates), can multi-frame agreement voting buy
near-zero wrong plate numbers, and at what coverage cost per plate width?

**Evidence present:**

- The 167 held-out plates: 0 % wrong numbers at ≥ 120 px, 60–75 % read.
- The EasyOCR baseline.
- Character mAP50 0.968 (`models/README.md`).
- Reproducible scripts in `training/`.

**Gaps:**

- Real daytime Dhaka video: `videos/ForNRP/` was recorded for this and holds 297 s at 4K60, unlabelled.
- Bangla LPDB-A cross-dataset test.
- Vocabulary covers 3 of 64 districts.
- One more Bangla ANPR baseline.

Effort ≈ 12–18 pd. **Main risk:** daytime real plates may read worse than simulated degradations of night images.

### P7 (proposed, optional): Violation impact assessment. **Score 1**

This is the 499A report's central argument (`SUPERVISION.md` A1–A6). The code and synthetic validation exist
(`results/x4_impact.json`). Human severity ratings are cheap: about 3 h for 2 raters on 40 cases
(`docs/EVALUATION.md` §4). They need real cases from E1.1. Fold it into P1 as one table, or into the T-ITS extension.
Effort ≈ 3–4 pd on top of P1.

---

## 5. Venue × paper matrix

### 5.1 Primary and backup per paper

| Paper | Primary venue (deadline, status) | Feasible? | Backup | Journal after | What to emphasise |
|---|---|---|---|---|---|
| P1 | **IEEE ITSC 2027**: 1 Mar 2027 (**verified**) | **Yes**, conditional on footage and permissions by early Dec | IET Intelligent Transport Systems (rolling) | IEEE T-ITS, as a substantial extension: more sites, BrnoCompSpeed or GNSS speed, P7 impact | Measured citation error budget in mixed traffic; abstention at every stage; human-in-the-loop; Dhaka classes |
| P2 | **IEEE ICME 2027**: 30 Dec 2026 (**partly verified**) | **Stretch**: go/no-go 13 Dec | ACM MM 2027, about April (unverified); no parallel submission with ICME | Communications in Transportation Research | Grounding on structured evidence, false-support rate, counterfactual flips, guards that only weaken |
| P3 | WACV 2028 Round 1, about June 2027 (unverified) | Only if permission and redaction are done by Mar 2027 | IEEE Open Journal of ITS (rolling) | IEEE OJ-ITS | New task (mixed-traffic multi-violation); developing-country ITS |
| P4 | IEEE IV 2027 **workshops**: 1 Feb 2027 (**verified**); needs a matching workshop once workshops are announced (proposals due 1 Dec) | Yes for a workshop-size study on Kakoli and Kuril | ITSC 2027 (1 Mar) or IET ITS | Transportation Research Part C, after real section-speed ground truth | How wrong links become wrong section speeds; refusal before error |
| P5 | IAAI-28 Emerging Applications; deadline **unverified** (IAAI-26 full papers were due 18 Aug 2025; IAAI-27 around 27 Jul 2026), so expect **late July–August 2027**, not September | Only if a pilot starts by about April 2027 | IET Smart Cities | Sustainable Cities and Society | Review workload, failure modes, lessons learned |
| P6 | **IET Image Processing** (rolling) | **Yes**, submit about early Feb 2027 | IET Intelligent Transport Systems | — | Precision vs coverage by plate width; cross-dataset; Bangla two-row layout |
| P7 | Inside P1 (ITSC) | Yes | T-ITS extension | — | Human severity vs impact score (Spearman) |

### 5.2 Conferences (deadline order)

| Venue | Deadline (status) | Fits | Ready by the deadline? | Emphasis / format notes |
|---|---|---|---|---|
| IEEE IV 2027 (Perth, 15–18 Jun 2027) | Papers **15 Nov 2026**. The official page gives **no time zone**; the AWST in your notes is unverified. Workshop papers 1 Feb 2027 and industry papers 15 Mar 2027 (**verified**, ieee-iv.org/2027). Workshop and tutorial *proposals* are due 1 Dec 2026 on the official site; one aggregator says 1 Nov. | P4 (workshop) | Main track: **no**, 14 days away and nothing measured. Workshop: yes. | Vehicle and infrastructure focus. Page limit not stated on the page (unverified). |
| CVPR 2027 | Registration 10 Nov, paper 16 Nov 2026, AoE (**verified** via the CFP page in search; Appendix B) | — | **No** | Needs a state-of-the-art result on public data; nothing qualifies |
| IEEE ICME 2027 (Xiamen, 13–17 Jul 2027) | 30 Dec 2026 (**partly verified**: IEEE SPS listing; official site not checked) | P2 | Stretch | Multimodal grounding; format unverified (check 2027.ieeeicme.org) |
| **IEEE ITSC 2027** (Boston, 21–24 Sep 2027) | Submissions open 1 Dec 2026; special-session proposals 1 Feb 2027 (the site calls them "Special Sessions", not invited sessions); workshop and tutorial proposals 1 Mar; **papers 1 Mar 2027**; notification 1 May; final 31 May (**verified**) | **P1**, P4 (backup), P7 | **Yes** | Implementation and deployment welcome. Page limit not checked; confirm in the CFP. |
| AI City Challenge 2027 | **Unverified**. A news report places the 2026 (10th) edition at ECCV 2026; it had fisheye traffic-violation (Track 7) and safety captioning/VQA tracks. | P2-adjacent | Low priority | Fisheye intersection data does not match our domain |
| ICCV 2027 (Hong Kong, 2–8 Oct) | Projected 5–8 Mar 2027 (**unverified**; aggregators disagree) | P3 | No | — |
| ACM MM 2027 | About April (**unverified**) | P2 backup | Yes, if ICME is skipped | Must be multimodal; no parallel submission |
| NeurIPS 2027 Datasets & Benchmarks | About mid-May (**unverified**) | P3 | **No**, given the release and legal prerequisites | Needs a released dataset, datasheet, licence |
| WACV 2028 | Round 1 about June 2027 (**unverified**) | P3 | Possible | Applications-first; a new task counts |
| IAAI-28 | See P5 (**unverified**; likely earlier than September) | P5 | Conditional | 6-page Emerging track in IAAI-26 (search result) |
| KDD 2027 ADS | Cycle 2 about Feb 2027 (**unverified**) | — | **Not applicable**: it requires production deployment | — |

### 5.3 Journals

The impact factors are the team's figures and were not re-checked.

| Journal | Fits | Notes |
|---|---|---|
| IEEE Transactions on ITS (8.4, Q1) | P1 extension | Third-party guides agree with 10 pp regular and 6 pp short; the overlap rule (> 40 %) and extension policy are **unverified**. There are no prior team publications (the 499A publication list is empty), so self-overlap is limited to the ITSC paper. |
| Transportation Research Part C (7.9) | P4 | Needs real section-speed ground truth (GNSS or BrnoCompSpeed) |
| Communications in Transportation Research (14.5, OA) | P2 | APC about $1,250 per the team (unverified) |
| IEEE Internet of Things Journal (8.9) | — | Only after edge or cloud measurements at scale exist (not now) |
| Engineering Applications of AI (8.0) / Expert Systems with Applications (7.5) | P1 fallback | Applied AI on real data, so they need E1 |
| Sustainable Cities and Society (12.0) | P5 | Needs a pilot |
| Accident Analysis & Prevention (6.2) | — | Needs a quantitative safety analysis; V11 is not validated |
| IEEE Open Journal of ITS (4.6, OA) | P3 | Lists "ITS in developing countries" (team note) |
| **IET Intelligent Transport Systems** (2.7, Q3) | P1 case-study fallback, P4, P6 backup | **APC unverified.** Secondary sources give about USD 2,200 (Editage, JournalMetrics; the IET 2020 price list's 2021 base rate). DOAJ lists "up to USD 2,800". |
| **IET Smart Cities** (3.4, Q2, ESCI) | P5 / system overview | Unverified |
| **IET Image Processing** (2.4, Q3) | **P6 primary** | Unverified |
| **IET Computer Vision** (1.4, Q4, APC about USD 2,200 per the team) | single-module fallback (helmet or ANPR) | Unverified |

**IET author-guideline check (IET ITS page, read in a real browser on 1 Nov 2026; the plain fetcher got HTTP 403):**

- **Article types:** Original Research Paper, Review, Case Study, Industry Article, Commentary, Opinion and
  Perspective. Some of these are by invitation.
- **Charges:** every IET journal is gold open access. There is **no charge for submitting**. An APC applies only if
  the paper is accepted, and the guideline says waivers or discounts may come "through your institution, funder, or a
  country waiver".
- **Conference extensions are accepted.** A manuscript that was previously a conference paper "must include
  information on how it has been expanded".

Still **unverified**:

- length limits (not found on that page)
- the APC amount (secondary sources say about USD 2,200)
- whether Bangladesh qualifies for a country waiver
- the guidelines of IET Smart Cities, IET Image Processing and IET Computer Vision (only the IET ITS page was read)

---

## 6. Action plan

Owners are placeholders (OWNER-1 to OWNER-4). The suggested split is:

- OWNER-1: data and labelling
- OWNER-2: evaluation code
- OWNER-3: VLM (P2)
- OWNER-4: ANPR (P6) and writing

### 6.1 Week by week to ITSC (1 Mar 2027)

| Week | Task | Files to touch | Expected output | Owner |
|---|---|---|---|---|
| W1, 2–8 Nov | Settle recording permission and ethics status; confirm detector provenance (DEC-2); freeze the held-out footage list. Decide no-go on IV/CVPR. | `SUPERVISION.md` (decisions), new `evaluation/ground_truth/README.txt` | A signed or emailed permission record; a held-out list with dates | OWNER-1 |
| W1 | Fix licence facts: Qwen2.5-VL-3B is `qwen-research`; add a LICENSE; decide on stripping the notebook outputs (section 7) | `models/README.md`, `LICENSE`, `research/cityflow/cityflowv2.ipynb` | Corrected licence table | OWNER-4 |
| W2, 9–15 Nov | Plan sessions at plate-readable sites (≥ 120 px plates; 4+ sites; include night). Start X1 labelling: 400 frames from the 2026-10-29 clips with CVAT pre-annotation. | `evaluation/ground_truth/detection/` | `data.yaml` + labels | OWNER-1 |
| W3, 16–22 Nov | Recording sessions 1–2. Run X1. Write the Dhaka-AI converter. | `tools/evaluate.py x1`, new `tools/convert_dhaka_ai.py` | AP@0.5 per class on 2 test sets | OWNER-1, OWNER-2 |
| W4, 23–29 Nov | Label `events.csv` for the first 60 min. Label plates on ForNRP. Download Bangla LPDB-A (874 MB) and write a converter. | `evaluation/ground_truth/violations/events.csv`, `.../plates.csv`, new `training/eval_lpdb.py` | Event counts per rule; plate list | OWNER-1, OWNER-4 |
| W5, 30 Nov–6 Dec | Write `evaluation/citations.py` and `tools/evaluate.py e2e` (precision, FC/1k vehicles, recall per type). ITSC submissions open on 1 Dec. | new `evaluation/citations.py`, `tools/evaluate.py` | First E2E table on 60 min | OWNER-2 |
| W6, 7–13 Dec | Oracle hooks (`GTReplayDetector`; helmet, plate and registry oracles). Build the test registry with distractor plates. **P2 go/no-go on 13 Dec.** | `surveillance/pipeline.py:160` (injection only), `violations/helmet.py:61`, `surveillance/worker.py:380`, `surveillance/registry.py:90`, new `evaluation/oracle.py` | Oracle runs working on 1 clip | OWNER-2, OWNER-3 |
| W7, 14–20 Dec | Blind reviewer study (2 people). P2: label the case set (≥ 120). MOT ground truth for 2 × 60 s. | Review export; `evaluation/ground_truth/vlm/cases.csv`, `.../tracking/gt.txt` | κ; VLM labels; MOT ground truth | OWNER-1, OWNER-3 |
| W8, 21–27 Dec | P2: counterfactuals and ablations (CPU runs overnight). P1: label the second hour. | new `tools/vlm_eval.py`; `events.csv` | False-support and flip-rate tables | OWNER-3, OWNER-1 |
| W9, 28 Dec–3 Jan | **ICME submission on 30 Dec, if go.** P1: per-rule table and ANPR-on-real-video table. | `REPORT_RESULTS.md` via `tools/evaluate.py report` | P/R per V-code; plate read rate by width | OWNER-3, OWNER-2 |
| W10, 4–10 Jan | Full oracle ablation; robustness breakdowns | `evaluation/oracle.py`, breakdown script | Error-budget figure; condition table | OWNER-2 |
| W11, 11–17 Jan | Efficiency re-timing on an idle CPU (and on a GPU if approved); cost per camera. P4: `ids.csv` for Kakoli and Kuril. | `docs/CAPACITY.md` method; `evaluation/ground_truth/crosscam/ids.csv` | Latency table; pair P/R | OWNER-4, OWNER-1 |
| W12, 18–24 Jan | P1 draft v1. P4: link-noise → section-speed experiment. P6: LPDB-A and real-plate tables. | new `tools/link_noise.py` | P1 draft; P4 figure; P6 tables | OWNER-4, OWNER-2 |
| W13, 25–31 Jan | Supervisor review of P1. P4 workshop draft. P6 draft. | — | Reviewed draft | all |
| W14, 1–7 Feb | **IV workshop paper (1 Feb), if a workshop fits.** Optional ITSC special-session proposal (1 Feb). **Submit P6** to IET Image Processing. | — | Submissions | OWNER-1, OWNER-4 |
| W15, 8–14 Feb | Close review gaps; ethics and data statement; release package (code licence, model cards) | `README.md`, model cards | Ethics paragraph; release checklist | OWNER-4 |
| W16, 15–21 Feb | **Experiment freeze on 15 Feb.** Regenerate every table from `results/*.json`. Figures. | `tools/evaluate.py report` | Final tables | OWNER-2 |
| W17, 22–28 Feb | Polish and PDF check. **Submit P1 by 28 Feb** (confirm the deadline time zone). | — | ITSC submission | all |

### 6.2 March to September 2027 (coarse)

| When | What |
|---|---|
| Mar | If ICME was skipped or rejected, revise P2 for ACM MM (about April; no parallel submission). Start P3 legal work (permission, ethics, redaction tool). Approach DMP or BRTA about a pilot (P5). |
| Apr | ACM MM submission (P2, if applicable). Pilot MoU target. Decide on NeurIPS D&B: go only if release rights and anonymisation are done (expect no-go). |
| May | ITSC notification 1 May, final paper 31 May. Begin the T-ITS extension: more sites, BrnoCompSpeed or GNSS speed (ask before the >1 GB download), P7 human severity, longer footage. |
| Jun | WACV 2028 Round 1 (P3), if the data is ready. P4 journal version (TR-C or IET ITS) once GNSS section speed exists. |
| Jul–Aug | IAAI-28 Emerging (P5): **check the deadline in June**, because earlier cycles fell in late July to mid-August. Needs at least 6–8 weeks of pilot logs. |
| Sep | ITSC in Boston (21–24 Sep). Submit the T-ITS extension of P1. |

---

## 7. Risks and ethics

1. **Recording permission is undocumented.** Footage of public roads from at least 4 sites (Kakoli, Kuril, and the
   ForNRP and ForLaneViolation sites) shows identifiable faces and plates. No permission, ethics approval or consent
   record was found. Reviewers at ITSC, NeurIPS D&B and IAAI will ask for one.
2. **Personal data on disk, not in git.**
   - Raw footage (`videos/`, 8.5 GB) and the runtime database and evidence (`data/`) are gitignored, and none of it is
     tracked.
   - Tracked code and tests contain only placeholder plates, `example.test` emails and `01700…` phone numbers. I
     checked this with pattern searches, counts only.
   - Keep the footage off any public host, and redact anything that goes into a paper figure.
3. **Third-party footage on public branches.** `research/cityflow/cityflowv2.ipynb` was committed in `80563e6`. It is on
   `origin/rafat` and `myrepo/rafat`, and both GitHub repositories are publicly readable (checked with
   `git ls-remote`). It embeds a 15.7 MB demo video and a 3.5 MB image rendered from CityFlow footage
   (`research/cityflow/README.md`). The AI City data terms may forbid redistribution (unverified). Strip the outputs:
   the README already has the snippet. Removing them from history needs a rewrite, which is the team's decision.
4. **Licence errors and obligations.**
   - **Qwen2.5-VL-3B uses the `qwen-research` licence, not Apache-2.0** as `models/README.md` states (Hugging Face
     model card). Qwen3-VL-2B and 8B are Apache-2.0 (verified).
   - Ultralytics YOLO weights (detector, helmet, characters) and the plate detector are **AGPL-3.0**. Releasing the
     code or weights brings AGPL obligations. The repo has **no LICENSE**.
   - The ReID weights are derived from AI City 2021 ReID and VehicleX data, whose redistribution terms are unverified.
   - NightLPBD and Bangla LPDB-A are CC BY 4.0: cite them.
   - Dhaka-AI is GPL-2.0 per Dataset Ninja (unverified).
   - BrnoCompSpeed is by request and reportedly non-commercial (unverified).
5. **Detector provenance unknown (DEC-2).** No mAP can be called held-out. If Dhaka-AI or the Kakoli clips were used in
   training, the cross-dataset numbers are contaminated.
6. **Label validity.**
   - The Dhaka helmet labels were made by the AI assistant (`models/README.md`), and no human verification is recorded.
     Reviewers will reject AI-made test labels.
   - The current V6 rule's Kakoli result is 1 confirmed bare-headed rider out of 16 reports (`violations/helmet.py:19`),
     on the same tracks used to tune it.
   - `models/README.md` still describes an older V6 rule.
7. **No ground-truth speed.** V2 is advisory by design (`citation.py:77`), so P1 must not claim speed citations, and P4
   needs GNSS or BrnoCompSpeed.
8. **Footage volume.** About 13 minutes from one afternoon. Per-rule rates need at least 20 events
   (`docs/EVALUATION.md` §3).
9. **VLM verdict noise and CPU cost.** The same case flipped from supported to contradicted
   (`models/README.md`). An 8B run takes about 275 s per case under load (`docs/VLM.md`).
10. **Inconsistent claims to fix before writing.**
    - Auto-calibration accuracy: ±8–9 % (`README.md:71`) vs ±15 % (`autozone.py:15`).
    - YOLOv11 in the report vs YOLOv8s in the code (C-1).
    - The `README.md` section "What this does not do yet" says there is no dispute flow, but `disputes.py` exists.
11. **Credential hygiene.** `README.md:201-206` shows a 16-character admin password in a first-run example. If it was
    ever a real password, rotate it. `SUPERVISION.md` lists `admin/admin` and `police/police` test accounts: never use
    them on the Azure VM. A gitignored file named `API` sits at the repo root and was not opened.
12. **Linux helmet crash.** On Linux the CoreML model is chosen, and V6 then fails at `violations/helmet.py:94`
    (`SUPERVISION.md` log, 2026-10-30). This affects reproducibility on a Linux reviewer machine.

---

## 8. Open questions for the team

1. Who trained `bd_vehicles_best.pt`, on which images and videos (Dhaka-AI? the Kakoli clips?), and with what split?
   Its validation mAP was **not found**.
2. Do you have written permission from DMP, DTCA, the site owners or NSU ethics to record at Kakoli, Kuril and the
   ForNRP and ForLaneViolation sites, and to publish frames or release footage?
3. Can you get any real BRTA registry access, or will the paper use a synthetic test registry? (This audit assumes
   synthetic.)
4. Has a person checked the helmet labels in `training/helmet_labels.txt` (352 H, 24 N, 224 X)?
5. Where are `TC_00006.MOV`, `DataSet_2min.mp4`, Kakoli Part-1 and Part-2, and the NightLPBD copy? None are on this
   machine, so the helmet and ANPR numbers could not be re-run.
6. How far apart are the Kakoli side 1 and side 2 cameras, and the two Kuril cameras? Are their clocks synchronised?
   (`section_speed.py` rejects cameras less than 50 m apart.)
7. Who produced and reviewed the footbridge `matches.csv` (236 rows) and `feedback_wrong_pairs.csv` (32 rows), and
   with which matcher?
8. Is a GNSS-logged probe-car drive possible (a phone logger is enough)?
9. Is there a budget for a rented GPU (BrnoCompSpeed, CityFlow, VLM ablations) and for paid APIs (cloud VLM reasoner,
   commercial ANPR)? I ran none of these.
10. Do the AI City 2021 ReID and VehicleX terms allow sharing the ReID weights? Should the CityFlow-derived notebook
    outputs stay on public GitHub?
11. Which licence do you want for the code? It must be AGPL-compatible because of Ultralytics.
12. Who owns each paper, and does the supervisor agree on ITSC for P1 and on the 13 Dec ICME go/no-go for P2?
13. What is the APC budget for IET gold open access, and do the authors qualify for a Bangladesh or Research4Life
    waiver? I could not verify this; the Wiley pages blocked my fetcher.
14. For the IET journals, still check: length limits and the current APC amount. Article types and conference-paper
    extensions are now confirmed for IET ITS (section 5.3). The other three IET journals' pages were not read.
15. For ITSC 2027, IV 2027 and ICME 2027: confirm the page limits and the deadline time zones. The IV page states no
    time zone.
16. Was the first-run admin password in `README.md:204` ever live?
17. Metrics that were **not found**:
    - detector mAP
    - real per-rule precision and recall
    - real plate read rate
    - Dhaka cross-camera IDF1
    - VLM accuracy and hallucination rate on real cases
    - speed error against real ground truth
    - helmet accuracy at a second site
    - GPU latency
    - cost per camera

    Do any of these exist outside the repo (Kaggle, Colab, a teammate's machine)?

---

## Appendix A: commands run during this audit

All were run from the repo root on the i5-11400H (load average about 4 at the start). `git status` was the same before
and after: `M SUPERVISION.md`, `?? deploy/`.

| Command | Wall time | Output (key lines) | Matches the repo? |
|---|---|---|---|
| `./.venv/bin/python tools/bench_accuracy.py` | 12.9 s | speed n 4907, mae 2.08, bias 0.77, p95 7.11, mape 12.4; V2 6/6 caught, 0 FA (of 122 legal); V1/V7 2/2, 0 FA; V4 2/2, 0 FA | Yes: `results/x3_rules.json`, `REPORT_RESULTS.md` X3 |
| `./.venv/bin/python tools/bench_accuracy.py --clean` | 12.2 s | speed n 5024, mae 0.5, bias −0.18, p95 2.54, mape 4.5; V2 9/10, 0 FA; V1/V7 4/4; V4 12/12 | Yes |
| `./.venv/bin/python tools/signal_sim.py --quick` | 2.6 s | load 0.8: fixed (equal) 183.5 ± 4.4 s, fixed (tuned) 135.4 ± 1.4 s, max pressure 71.0 ± 1.3 s mean delay; EV hard corridor −33.3 % (max pressure) | Yes: `results/optimize.json` |
| `./.venv/bin/python -m unittest discover -s tests` | 2 min 46.9 s | `Ran 1285 tests in 161.109s` / `OK` | Yes: `SUPERVISION.md` says 1285. (`REPORT_RESULTS.md` X3 showed 1 failure in `test_behaviour` on 2026-10-29; it is fixed now.) |
| `ffprobe` on every clip in `videos/` and `data/videos/` | <1 s each | Durations and resolutions in section 2.1 | n/a |
| `git ls-remote` on the three GitHub remotes (anonymous) | — | All three are readable without credentials | n/a |
| `git grep` counts for phone, email and plate patterns | — | Placeholders only (counts, not values) | n/a |

**Not run, and why:**

- `tools/evaluate.py`: it rewrites `results/*.json` and `REPORT_RESULTS.md`, which this audit must not change.
- `tools/vlm_probe.py`: it needs the live VLM servers, and the 8B run takes minutes per case.
- `tools/reid_eval.py`: 962 s for 25 s of video (`REPORT_RESULTS.md`), which is over the 10-minute limit.
- `training/calibrate_anpr.py` and `training/eval_helmet_*.py`: their datasets are not on this machine.

## Appendix B: online sources checked (1 Nov 2026)

- ITSC 2027 important dates: https://ieee-itsc.org/2027/ (verified)
- IV 2027 dates: https://ieee-iv.org/2027/ (verified); aggregator https://www.beri.net/events/ieee-iv-2027
- CVPR 2027 CFP: https://cvpr.thecvf.com/Conferences/2027/CallForPapers (via search extract)
- ICME 2027: IEEE SPS listing https://signalprocessingsociety.org/events/attend-an-event?c=45&page=2 ; official https://2027.ieeeicme.org/ (not checked)
- ICCV 2027 projection: https://www.opencurious.com/ai-conference-deadlines/iccv-2027 (unverified)
- IAAI-27 call: https://aaai.org/conference/aaai/aaai-27/iaai-27-call/ ; IAAI-26 call: https://aaai.org/conference/aaai/aaai-26/iaai-26-call/
- AI City 2026: https://www.scu.edu/engineering/stories/research/traffic-anomaly-reasoning-tar-moving-beyond-detection-in-the-2026-ai-city-challenge.html ; https://biz.heraldcorp.com/article/10881563
- Qwen2.5-VL-3B licence: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct (README front matter: `license_name: qwen-research`)
- Qwen3-VL-2B and 8B licence: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct , https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct (`license: apache-2.0`)
- Bangla LPDB-A: https://zenodo.org/records/4718238 (CC BY 4.0, 874.1 MB)
- Dhaka-AI: https://datasetninja.com/dhaka-ai (GPL-2.0 claim, unverified)
- BrnoCompSpeed: https://hyper.ai/datasets/9228 , https://github.com/RasoulZardeyni/BrnoCompSpeed (unverified)
- IET ITS APC (secondary): https://doaj.org/toc/1751-9578 , https://www.editage.com/journal/iet-intelligent-transport-systems/ , https://www.theiet.org/media/6944/article-processing-charges_october-2020.pdf
- T-ITS page limits (secondary): https://manusights.com/blog/ieee-transactions-on-intelligent-transportation-systems-submission-guide
- Not reachable: https://ietresearch.onlinelibrary.wiley.com/hub/journal/17519578/homepage/author-guidelines (HTTP 403); Wiley waiver page (empty after redirect)

## Appendix C: the teammate repository (`../Shihab-AI-Traffic-System`, commit `2137abc`)

Checked on 1 Nov 2026. The code was read, not run: it needs Windows, an NVIDIA GPU, MongoDB and Redis.

**How similar the two projects are (confirmed):**

- The vehicle detector and the helmet classifier are byte-identical. `bd_vehicles_best.pt` SHA-256 starts
  `200295f8…` in both repos; `helmet_cls.pt` starts `57bb75cb…` in both.
- Of the 25 vision Python files the two repos share, **12 are byte-identical**: `autozone.py`, `camera_motion.py`,
  `speed.py`, and most of `violations/`. The rest differ by 2–117 lines.
- `docs/COMPARISON_SHIHAB.md` already describes the split. Our repo covers impact, optimisation, query, storage, the
  two-tier VLM, the trained ReID model, V9–V11 and disputes. Theirs covers the enforcement web system.

**What only the teammate repo has, and what it measured:**

| Part | What it is | Measured evidence (source: its `README.md`) | Publishable on its own? |
|---|---|---|---|
| Plate reader | BLPD plate detector (`LPD_Model.pt`, YOLOv12m, not redistributed) plus its own `bd_plate_ocr_v2_best.pt`. Vocabulary: all districts (105 classes in BLPD's naming). | Test clip, 104 vehicles over a test speed limit: a plate was found on 33, and **0 were confirmed**. Plates there are 25–45 px. | No: no held-out accuracy. A useful baseline for P6. |
| VLM | Qwen3.5-2B, 4-bit, on a GPU; 10–25 s per case | 7 of 8 verdicts right, one case per rule (`vision/service/vlm.py` says 6/8 in an earlier run) | No: n = 8. A second model for P2's ablation. |
| Cross-camera IDs | Gates, links and time windows (after Liu et al., AI City 2021), DINOv2-small + colour histogram, HNSW look-alike index, two-link fallback, out-of-order merge | KAK-SIM, which is the *same* clip re-coloured and shifted 90 s: 93.9 % of kept IDs correct, 78.6 % kept. Synthetic 2,000-camera network: 3–5 µs per sighting at 0.1–1 M stored; HNSW recall@10 100 %. | No: the "second camera" is not a real second view. Use it as P4's scalability section and as a baseline matcher. |
| Enforcement web system | NestJS + MongoDB + Next.js; tickets, review requests, SSLCommerz sandbox payments, invoices, multi-factor login, Android APK | Functional tests only (~480 TS tests, mostly platform) | Not as research. It supports P1 (the citation life cycle) and P5 (a pilot needs this UI). |
| V3 / V6 tuning on Kakoli Part-1 and Part-2 | Kerb areas, creep, and the toward-camera helmet rule; already ported to our repo | V3 0 → 32 on Part-2; V6 85 → 16 reports on Part-1 (7 helmeted, 1 bare-headed). Labelled by eye, on the tuning footage. | No: the tuning data is the test data |

**Verdict:** nothing in the teammate repository is publishable on its own. Its unique results are either very small
(8 VLM cases), synthetic (KAK-SIM, the 2,000-camera benchmark) or measured on the data used for tuning. It
strengthens the shared papers:

- **P1:** the full citation life cycle (ticket, review, payment).
- **P4:** the scalability numbers, plus a second matcher (DINOv2) to compare against our ResNet50-IBN ReID.
- **P5:** the portal and Android app are the pilot front-end.
- **P6:** BLPD as a plate-reading baseline.

Pick **one canonical codebase per paper**, and say in each paper which repository and commit produced each number.

Hygiene: `infra/s3.json` holds a development S3 key pair, labelled "development only", in a public repo. Rotate it
if it was ever used outside a laptop.

## Appendix D: related published papers (1 Nov 2026)

### D.1 Other venues (IEEE Xplore searched in a browser; web search for the rest)

Your network has no IEEE subscription (the Xplore session reports `institute: false`), so only open-access or
author-posted copies were downloaded. Files are in `../papers/04-…` to `../papers/09-…`.

| Venue | Paper | Why it matters | Copy |
|---|---|---|---|
| WACV 2025 | Rawat et al., *DashCop: automated e-ticket generation for two-wheeler traffic violations* | The closest published analogue of P1, at a P3 target venue. It runs from detection all the way to e-ticket and releases a dataset (RideSafe-400). | **downloaded** (CVF open access) |
| T-ITS 2022 | Charran and Dubey, *Two-wheeler vehicle traffic violations detection and automated ticketing for Indian road scenario* (10.1109/TITS.2022.3186679) | The T-ITS precedent for P1. Reports 77 of 93 real violations caught with 0 false positives. | paywalled. The authors' 2026 follow-up (arXiv 2601.07845, edge-AI enforcement node) is **downloaded** |
| ITSC 2021 | Abdelhalim et al., *Data Enforced: impact of automated speed enforcement in Washington DC* | How ITSC frames enforcement *impact*: a model for P7 and for P1's motivation | **downloaded** (arXiv 2106.09933) |
| IEEE OJ-ITS 2025 | *Vehicle re-identification and tracking: algorithmic approach, challenges and future directions* | Related-work base for P4 | **downloaded** |
| IEEE OJ-ITS 2020 | Rademeyer, Barnard and Booysen, *Optoelectronic and environmental factors affecting crowd-sourced vehicle-mounted LPR accuracy* | Plate-reading accuracy versus optics and environment, with a developing-country framing. For P6 and P1's robustness section. | **downloaded** |
| IEEE OJ-ITS 2022 | *Multi-intersection traffic optimisation: a benchmark dataset and a strong baseline* | What a dataset/benchmark paper looks like in OJ-ITS, the backup venue for P3 | **downloaded** |
| T-ITS 2018 | Sochor et al., *Comprehensive data set for automatic single camera visual speed measurement* (BrnoCompSpeed) | The speed ground-truth benchmark for P1 and P4 | not downloaded (arXiv 1702.06441, 5.8 MB) |
| CVPR Workshops 2024 | AI City Track 5 helmet papers (Vo et al., rank 1, Co-DETR; Chen et al.; Luong et al.) | Helmet-violation state of the art; our classifier cannot report this track's mAP (section 3B) | not downloaded (CVF open access) |
| ITSC 2024 | Sural et al., *ContextVLM* | P2 related work | not downloaded (arXiv 2409.00301) |
| ITSC 2024 / 2025 | *TrafficLens: multi-camera traffic video analysis using LLMs*; *SliDeR-VLM: interpretable collision prediction*; *video-based traffic anomaly detection with VLMs: a survey* | P2 and P4 related work | paywalled; arXiv copies not checked |
| ITSC 2021 | Kacmaz et al., *Fully automated, realistic license plate substitution* | Plate anonymisation and augmentation, relevant to redaction for P3 | author PDF at ini.rub.de |
| Communications in Transportation Research 2026 | Zhang et al., *Distilling vision-language models for explainable vehicle collision prediction* (10.26599/COMMTR.2026.9640047) | The closest CommTR paper to P2 | not downloaded (open access, 12.0 MB) |
| IEEE OJ-ITS 2025 | *Pedestrian vision language model for intentions prediction* | P2 | not downloaded (open access, 16.0 MB) |
| Deployment (news, not peer-reviewed) | IIIT Hyderabad's VIOLA: dashcam enforcement feeding the Telangana e-Challan system. The institute reports that police approved about 96 % of the flagged violations. | The nearest real deployment: a model for P5. The number is self-reported. | — |

The IEEE Open Journal of ITS has almost no camera-based violation-enforcement papers, which leaves room for P3 there. No
close open-access match turned up in TR-C, ESWA or EAAI.

### D.2 IET (search on ietresearch.onlinelibrary.wiley.com)

These came from 9 searches in a real browser. **Downloaded** (P1 bundle, chosen by the team) to
`../papers/01-…`, `../papers/02-…` and `../papers/03-…`:

1. Wrong-way driving detection in dense urban Bangladesh (Hough + YOLOv5), IET Image Processing 2026,
   doi 10.1049/ipr2.70322, open access. Shows that IET publishes Dhaka violation work. Compare with our V4.
2. Özkul and Çapuni, police-less multi-party traffic violation detection and reporting with privacy preservation,
   IET ITS 2018, doi 10.1049/iet-its.2017.0122. Relevant to the ethics and privacy section of P1.
3. Ali et al., real-time object detection for unmanned vehicles in Bangladesh: dataset, implementation and evaluation,
   The Journal of Engineering 2024, doi 10.1049/tje2.70033. A Bangladesh detection dataset: compare with our 16
   classes and with P3.

Found but not downloaded:

- **P6:** layout-independent YOLO ALPR (IET ITS 2021, 10.1049/itr2.12030); LPR survey (IET CV 2026,
  10.1049/cvi2.70071); LPR systematic review (IET ITS 2019, 10.1049/iet-its.2018.5151).
- **P2:** AID-VLM for traffic incident understanding (IET ITS 2026, 10.1049/itr2.70271); TrafficRobot agent framework
  (IET ITS 2026, 10.1049/itr2.70189); vision-language hazard reasoning (Electronics Letters 2025, 10.1049/ell2.70466).
- **P4:** vision-based vehicle speed estimation survey (IET ITS 2021, 10.1049/itr2.12079); infrastructure camera
  calibration with GNSS (IET ITS 2022, 10.1049/itr2.12262); person and vehicle ReID survey (IET CV 2024,
  10.1049/cvi2.12316); vehicle ReID with fine-grained features (IET ITS 2026, 10.1049/itr2.70173).
- **Helmet:** helmet wear analysis of motorcycle riders (IET ITS 2019, 10.1049/iet-its.2018.5241); real-time helmet
  detection in urban traffic (IET IP 2021, 10.1049/ipr2.12295).
- **Other:** ground-plane homography intrusion prediction in smart-city zones (IET Smart Cities 2026,
  10.1049/smc2.70024; close to our zone method); driver decisions in informal mixed traffic, Kinshasa (IET ITS 2026,
  10.1049/itr2.70301; developing-country framing).
