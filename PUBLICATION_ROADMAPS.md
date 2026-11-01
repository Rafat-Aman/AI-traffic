# Publication roadmaps: four ways to get the Dhaka system published

Date: 1 November 2026. This builds on `PUBLICATION_READINESS.md` (the audit). Paper IDs (P1–P7), effort figures and
venue checks come from there. **pd** means person-day, 8 focused hours.

**Where we stand.** The system is built and tested (1,285 tests pass), but no enforcement metric has ground truth on
real Dhaka footage yet. Readiness by paper:

| Paper | What it is | Score |
|---|---|---|
| P6 | Bangla ANPR | 2 (partial evaluation) |
| P1 | City-scale enforcement | 1 (components, no evaluation) |
| P2 | VLM explanations | 1 |
| P3 | Dhaka benchmark | 1 |
| P4 | Cross-camera | 1 |
| P7 | Violation impact | 1 |
| P5 | Pilot | 0 |

Every roadmap below starts with the same groundwork. They differ in how much the team takes on and how soon the first
paper goes out.

---

## Step 0: groundwork every roadmap needs (weeks 1–2, 2–15 Nov)

None of the papers can use real footage until these are done.

| # | Task | Why | Effort |
|---|---|---|---|
| 0.1 | Get written permission to record and use the footage: DMP/DTCA or the site owners, plus NSU ethics approval if the university requires it | Reviewers ask for it. Without it, real footage cannot appear in a paper. | 1–2 pd + waiting time |
| 0.2 | Find out who trained `bd_vehicles_best.pt`, and on which videos (DEC-2) | Otherwise no test set can be called held-out | 0.5 pd |
| 0.3 | Pick **one codebase per paper** (ours or Shihab's) and freeze a commit | The two repos share the same models but have different plate readers, VLMs and matchers | 0.5 pd |
| 0.4 | Fix the licence facts: Qwen2.5-VL-3B is research-only; add a LICENSE; strip the CityFlow notebook outputs | Avoids desk-reject and takedown problems | 0.5 pd |
| 0.5 | Assign an owner per paper, agree a weekly check-in with the supervisor, and confirm when each member graduates | Work after the 499B semester depends on who stays available | — |
| 0.6 | Use **Kaggle's free GPUs** for anything a GPU needs; the team already trained the ReID model there (`research/cityflow/README.md`) | No credit card is needed, and the laptop is CPU-only | — |

Capacity assumption: about 8 pd per week across 4 people. From now to 1 March that is about 136 pd, or realistically
100–120 pd after exams and holidays.

---

## Roadmap A: quick win (plate reading first)

**Idea:** publish the module that already has held-out evidence (P6), then take the flagship to a journal later.

| When | Work | Output |
|---|---|---|
| Nov | Step 0. Label about 300 real plates in `videos/ForNRP/` (297 s of 4K60 footage, recorded for this). Download Bangla LPDB-A (CC BY 4.0, 874 MB) and map its labels to our 22 tokens. | `plates.csv`, an LPDB-A converter |
| Dec | Experiments: NightLPBD test (already done), LPDB-A cross-dataset test, real Dhaka video. Baselines: EasyOCR (done) and BLPD (from Shihab's repo). Curves of wrong-number rate and read rate by plate width. | 3 result tables, 1 figure |
| Jan | Write. **Submit to IET Image Processing**, which takes submissions year-round. | Submission by about 31 Jan |
| Feb–Jun | Build P1's gold set at an easier pace, then submit P1 to **IET Intelligent Transport Systems** (year-round) | Second submission by about June |

- **Effort to March:** about 15–20 pd.
- **Submissions by September 2027:** 2, both in Q3 IET journals.
- **Pros:** lowest risk. The first paper does not depend on violation labels.
- **Cons:** lower-tier venues, and the first paper does not show the system's main contribution.
- **Positioning:** precision first. It reports zero wrong numbers at a measured coverage cost, on two-row Bangla plates
  under CCTV conditions. Compare it with the layout-independent YOLO ALPR paper (IET ITS 2021) and the LPR surveys
  in IET.
- **Main risk:** real daytime plates may read worse than the simulated degradations of night images.

---

## Roadmap B: flagship plus quick win (recommended)

**Idea:** the plate-reading paper (P6) and the flagship (P1) share the same plate labels. Do both, and fold the impact
score (P7) into P1.

| When | Work | Output |
|---|---|---|
| Nov | Step 0. **Record at least 2 h at 4 or more sites where plates are at least 120 px wide**, including one night session if allowed. Label 400 detection frames (CVAT pre-annotation, about 8–10 h). Start the plate labels. | Detection mAP on held-out Dhaka footage |
| Dec | Label every violation in the first hour (`events.csv`). Write `evaluation/citations.py` (citation precision, false citations per 1,000 vehicles, recall per type). Add the oracle hooks at `pipeline.py:160`, `helmet.py:61`, `worker.py:380` and `registry.py:90`. Run the LPDB-A test for P6. | First end-to-end numbers |
| Jan | Label the second hour. Run the oracle ablation and robustness breakdowns. Two blind reviewers check the cases. Collect human severity ratings for P7 (about 3 h). Draft P6, then P1. | Error-budget figure; reviewer agreement κ |
| Feb | **Submit P6** to IET Image Processing in early February. Freeze P1 experiments on 15 Feb. **Submit P1 to ITSC 2027 by 28 Feb** (deadline 1 Mar, checked on the official site). | 2 submissions |
| Mar–May | ITSC decision on 1 May. Meanwhile add sites, a BrnoCompSpeed subset on Kaggle and GNSS test drives. | Extension material |
| Jun–Sep | Turn P1 into a **T-ITS** journal paper with substantial new content. Present at ITSC (21–24 Sep, Boston) if accepted. | T-ITS submission by about September |

- **Effort to March:** about 55–65 pd. That fits the capacity with some slack.
- **Submissions by September 2027:** 3 (IET Image Processing, ITSC, T-ITS).
- **Pros:** the flagship goes to the venue the audit identified as the best fit. The quick win comes nearly free from
  shared labels.
- **Cons:** it depends on recording permission and footage arriving in November.
- **Positioning against the closest published work:**
  - DashCop (WACV 2025): dashcam, two-wheelers only, e-tickets.
  - Charran & Dubey (T-ITS 2022): two-wheelers, 77 of 93 caught, 0 false positives.
  - Ours: fixed CCTV, 11 rules in mixed traffic, abstention at every stage, human confirmation, sealed evidence, and
    an error budget that blames each module. Report real-world counts the way Charran & Dubey did.
- **Main risk:** too few readable plates or violation events. If a rule has fewer than 20 events, report counts, not
  rates.

---

## Roadmap C: full portfolio (aggressive)

**Idea:** Roadmap B plus the VLM paper (P2) and the cross-camera paper (P4) before March, then the benchmark and pilot
from spring.

| When | Extra work on top of B | Venue (deadline, check status) |
|---|---|---|
| Nov–Dec | P2: about 120 real cases rated by 2 people; counterfactual pairs; ablations with and without the structured evidence and the guard, comparing the 2B, 8B and 3B models. VLM runs on Kaggle GPUs. | **ICME 2027, 30 Dec** (sponsor's listing). **Go/no-go on 13 Dec**; otherwise ACM MM about April (unchecked) |
| Dec–Jan | P4: cross-camera identities for Kakoli and Kuril (at least 40 vehicles each); experiment injecting wrong links to see the section-speed error; Shihab's DINOv2 matcher as a baseline; his 2,000-camera benchmark for scale | **IEEE IV 2027 workshops, 1 Feb** (official site), if a matching workshop is announced; otherwise IET ITS |
| Mar–Jun | P3 legal work and recording (see Roadmap D). P5 pilot agreement with DMP. | WACV 2028 / IAAI-28 (unchecked) |

- **Effort to March:** about 100–115 pd. That is at or above realistic capacity.
- **Submissions by September 2027:** 4–6.
- **Pros:** the most output, and it uses everything built so far.
- **Cons:** it takes the same footage and people as P1, and every paper risks being thin. The ICME go/no-go is the
  safety valve.
- **Main risk:** spreading the team too thin, so that P1, the most valuable paper, misses ITSC.

---

## Roadmap D: benchmark first (long game)

**Idea:** build a released, anonymised Dhaka multi-violation dataset (P3) first, then evaluate P1, P2 and P4 on it.
This is the slowest route and the one most likely to be cited.

| When | Work | Output |
|---|---|---|
| Nov–Jan | Step 0, plus ethics approval for a *released* dataset. A recording plan of 3–5 h across 6 or more sites, at different times of day and in rain. A redaction tool (`tools/redact.py`) that blurs faces and the plates of vehicles that broke no rule. Plate-reading paper P6 as a side paper (IET IP, early February). | Approvals; redaction tool; P6 submitted |
| Feb–Apr | Labelling: boxes, tracks, violation events, plates, and helmet status per rider (40–60 pd). Sample checks by a second annotator. | Dataset version 1 |
| May–Jun | Baselines per task (our pipeline is the first). Datasheet, licence, hosting. **WACV 2028 Round 1, about June (unchecked)**, or IEEE Open Journal of ITS (year-round). NeurIPS Datasets & Benchmarks (about mid-May, unchecked) only if everything is ready by April. | Dataset paper |
| Jul 2027 → | P1 and P2 evaluated on the benchmark, aiming at T-ITS and Communications in Transportation Research in 2028 | — |

- **Effort to June:** about 90–120 pd plus legal time.
- **Submissions by September 2027:** 1–2.
- **Pros:** the highest long-term impact. The IEEE Open Journal of ITS has almost no camera-enforcement datasets, and
  a public benchmark gets cited.
- **Cons:** there is no flagship paper in 2027, and it depends on legal clearance and on the team staying together after
  graduation.
- **Positioning:** DashCop's RideSafe-400 covers dashcam two-wheelers only. The Bangladesh detection dataset in *The
  Journal of Engineering* (2024) covers detection only. Ours would add multiple rules, plates and tracks from fixed
  cameras in Dhaka mixed traffic.
- **Main risk:** no permission to release footage of identifiable people.

---

## Optional add-on: pilot (P5)

This can attach to Roadmap B or D from March. The path:

1. An agreement with DMP or a city office.
2. A supervised pilot on 2 or more cameras for 6–8 weeks.
3. Logs of review workload and failure modes.
4. A paper for IAAI-28 Emerging Applications. Its deadline is unchecked; earlier editions point to late July or August.
   IET Smart Cities is the backup.

Shihab's portal and Android app are the natural front end. The closest precedent is IIIT Hyderabad's VIOLA, which
feeds dashcam detections into the Telangana e-Challan system. The institute reports that police approved about 96 % of
flagged violations; that figure is self-reported and not peer-reviewed.

Effort: about 30–40 pd plus the partner's time. **Main risk:** no partner.

---

## Side-by-side

| | A: quick win | **B: flagship + quick win** | C: full portfolio | D: benchmark first |
|---|---|---|---|---|
| Papers before Sep 2027 | P6, P1 (journal) | **P6, P1 (+P7), T-ITS extension** | P6, P1, P2, P4 (+P3/P5 started) | P6, P3 |
| First submission | ~31 Jan 2027 | **early Feb 2027 (P6); 28 Feb (P1)** | 30 Dec 2026 (if ICME is a go) | early Feb 2027 (P6); ~Jun 2027 (P3) |
| Team effort before March | 15–20 pd | **55–65 pd** | 100–115 pd | 90–120 pd by June |
| Highest venue tier | Q3 journal | **ITSC → T-ITS (Q1)** | ITSC, ICME, IV workshop | WACV / OJ-ITS |
| Needs recording permission by | Dec | **Nov** | Nov | Jan, *with a release licence* |
| Biggest risk | Daytime plate accuracy | Footage volume | Spreading too thin | Legal clearance, time |
| Choose it if | Little time this semester, or you want a CV line fast | The team can give about 2 days a week each until March | The team can give 3 or more days a week each, and the supervisor wants breadth | Permission to release is likely, and the team stays together after graduation |

## Recommendation

**Roadmap B.** It is the only plan that puts the system's main contribution, a measured error budget for automated
citations, in front of a Q1 audience within a year, while still getting a paper out in February. Two optional upgrades:

- If November goes well (permission granted, at least 2 h recorded), add P2 or P4 from Roadmap C.
- If permission to *release* footage arrives, start Roadmap D's benchmark in spring as the follow-on.

The week-by-week plan for Roadmap B is section 6 of `PUBLICATION_READINESS.md`.

## Decisions the team needs to make this week

1. Which roadmap: A, B, C or D?
2. Who asks DMP and NSU for recording permission, and by when?
3. Who trained the vehicle detector, and on which videos?
4. Which repo is canonical for each paper?
5. How many days a week can each member give until March, and who stays on after graduating?
