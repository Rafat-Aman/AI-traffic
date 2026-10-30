# Comparison: Shihab's "Dhaka Traffic Enforcement" vs this repository

Compared on 2026-10-30. Theirs: https://github.com/Shihab-cloud/AI-Traffic-System-for-Dhaka-City (3 commits,
2026-10-29/30), cloned to `../Shihab-AI-Traffic-System`. Ours: branch `rafat`, working tree as of the same day.
Sources: their README, code and git history. Their code was read, not run: it needs Windows, an NVIDIA GPU, MongoDB,
Redis, SeaweedFS and Mailpit, and this machine is Linux and CPU-only.

## At a glance

| | Theirs | Ours |
|---|---|---|
| Stack | Next.js + MUI (:3000), NestJS (:5001), MongoDB replica set, Redis, SeaweedFS (S3), Mailpit, Python FastAPI vision service (:8601), Android app (APK only) | One FastAPI process, SQLite, plain HTML/JS; llama.cpp servers for the VLMs |
| Runs on | Windows (PowerShell scripts, `.exe` tools) with an NVIDIA GPU (the VLM refuses to load without CUDA), 8 GB RAM is tight | Linux, CPU only |
| Size | ~73k lines TS/TSX, of which ~12k are traffic-specific (backend traffic module 6.2k, frontend traffic components 5.6k, vision service 2.7k); the rest is a general web platform (auth, MFA, RBAC, notifications, FAQ, support tickets) | ~38k lines Python (tests included), ~5.5k JS/CSS/HTML |
| Tests | ~480 TS test cases, mostly the platform (auth, rate limits, RBAC, MFA); 15 on cross-camera matching, plus the HNSW index; none on the Python rules | 1096 Python tests: rules, impact, optimiser, query, storage, disputes, permissions |
| 499A pipeline stages | DETECT, TRACK, UNDERSTAND, ENFORCE | DETECT, TRACK, UNDERSTAND, ENFORCE, ASSESS IMPACT, OPTIMIZE |

Both repositories start from the same vision code. `surveillance/autozone.py`, `camera_motion.py`, `speed.py` and
most of `violations/` are byte-identical; the differences below are what each side added.

## What they do better

1. **Enforcement right through to payment.** Officers send a ticket. The owner gets an email, sees the case and
   pays online through SSLCommerz (card, bKash, Nagad, Rocket; sandbox by default). The backend counts a payment
   only after SSLCommerz's validation API confirms it, and an invoice PDF is emailed. There is also a vehicle
   registry (plate → owner email) and an admin fines table. We stop at the citation and the driver portal: no
   payments, no email and no owner registry.
2. **A plate for every passing vehicle.** Both sides read violators' plates with the same method. They also read the
   plate of every passing vehicle and list it under **Passed Vehicles**, with class, first and last seen, a photo, a
   two-word look from the VLM ("grey SUV") and a CSV export. A read counts as "confirmed" only if the plate is at
   least 50 px wide. This rule came from three false "confirmed" reads on 25–44 px plates.
3. **Rule fixes measured on the Kakoli footage.** These are in the shared code, so they can be ported:
   - **V6 helmet** (`violations/helmet.py`): it judges only motorbikes coming toward the camera, only inside a road
     zone, and needs 4 confident sightings with at least 3/4 of them bare-headed. On Part-1 the cases fell from 85 to
     16, and the helmeted riders wrongly flagged fell from 48 to 7. The cost is that bare-headed riders going away
     from the camera are no longer caught.
   - **V3 parking** (`violations/zones.py` `KERB_AREAS`): a no-parking strip, bus stop or bay drawn just outside a
     road zone now belongs to that road. Before this, vehicles standing there were never checked; V3 on Part-2 went
     from 0 to 32.
   - **Kerb queues** (`violations/stationary.py`): `creep_m` (10 m) keeps one stand alive while a CNG row inches
     forward. A vehicle already reported is not reported again under a new track ID at the same spot (V1, V3, V7).
4. **VLM prompt tricks.** The red label is painted out of the scene image before the model sees it. The model had
   been reading "#20 2 km/h" as a speed and arguing with it. V6 first asks one question ("helmet, hair, cap or
   unclear?") and that answer sets the verdict. For V2 a photo can never contradict a measured speed. Talk about
   speed is removed from answers for rules that are not about speed. Result: 7 of 8 verdicts right, one per rule.
   Our `surveillance/vlm.py` draws on the same evidence images and does not hide the label. Their model also answers
   faster (10–25 s against our ~85 s for the 2B), but that comes from the GPU, not the method.
5. **Cross-camera tracking built to scale.**
   - Admins draw each camera's entry and exit gates and the links between cameras, each with a travel-time window.
   - If no match comes through a direct link, it tries cameras two links back, in case a camera in between missed
     the vehicle.
   - Late or out-of-order records are re-matched, and IDs that turn out to be the same vehicle are merged.
   - Every vehicle keeps a persistent ID (`DV-000123`). Officers can search by ID, plate or look.
   - An HNSW index suggests look-alike vehicles, but never joins them automatically.
   - Admins can rebuild every ID after changing the settings.
   - Benchmark: 3–5 µs per new sighting, flat from 100k to 1M stored vehicles.
   - On a simulated second camera (the same clip with changed colour and a 90 s offset): 93.9 % of the IDs kept
     across cameras were the right vehicle, and 78.6 % of the vehicles kept their ID.

   Ours links sightings with likelihood scores and learned routes, and shows each vehicle's journey. We have no gate
   or link topology, no two-link fallback, no plate or look search and no look-alike index.
6. **Platform hardening.**
   - Login: MFA, step-up authentication and rate limiting.
   - Roles: permission-based, with ranks.
   - The first admin is created with a one-time secret, and the backend refuses to start while any placeholder secret
     is left in its settings.
   - Evidence is kept in S3 object storage.
   - Events from the vision service go through a retry outbox.
   - Swagger documentation for the API, and an Android app.
7. **A more polished interface.** Charts on the dashboard and a zone editor with dragging and undo.

## What we do better

1. **Coverage of the plan and the revision-2 requirements.** They have nothing for ASSESS IMPACT, OPTIMIZE or R-2
   to R-8:
   - Impact assessment: traffic state, baseline, impact score and NNLS calibration.
   - Optimisation: max-pressure signals, emergency-vehicle corridor, routing and a simulator.
   - The map with camera placement and average speed between cameras.
   - The 3D city and the Google traffic supplement.
   - The query panel over long videos (segment index, descriptive and causal answers).
   - Storage and ingest for hundreds of videos a day: queue, retention, disk monitor and capacity numbers.
2. **The two-tier VLM (R-7) runs without a GPU.** A cheap kinematic pre-screen comes first, then the 2B model, and
   the 8B reasoner is started only when needed. Theirs is a single 2B model that needs CUDA.
3. **A trained ReID model (R-1).** The team's CityFlow ResNet50-IBN-a gets 98.4 % same-class rank-1, against 75.8 %
   for MobileNet. They use DINOv2, a general image model with no vehicle training; their own README says a trained
   vehicle ReID model would help.
4. **More rules.** V9 (bus lane, with hours), V10 (erratic driving) and V11 (near-collision); V10 and V11 are
   advisory only. They have V1–V8.
5. **Evidence integrity and disputes.**
   - Evidence is sealed with SHA-256, and media links are HMAC-signed.
   - A dispute must be filed within 30 days and gets an advisory re-check by the 8B model, with prompt-injection
     defences.
   - Four-eyes rule: the officer who issued the citation cannot decide its dispute.

   In their review flow an officer accepts or rejects with no AI check, and nothing stops that officer from being the
   one who issued the ticket: `cases.service.ts` records who decided but never compares it with `issuedById`.
6. **An evaluation harness.** `evaluation/`, `tools/evaluate.py` and `REPORT_RESULTS.md` produce the numbers for
   report Chapter 4. Their numbers are careful but appear only in the README.
7. **Tests on the vision and analysis code.** Their rule changes are checked by eye on footage. Ours are covered by
   unit tests.
8. **Easy to run.** One command on any Linux laptop. Theirs needs six services, Windows and an NVIDIA GPU.

## Problems on both sides

- Cross-camera tracking has been measured on one real camera view only, on both sides.
- Fine amounts are placeholders on both sides.
- Their Android app is committed as a 110 MB APK through Git LFS, with no source code, so it cannot be reviewed or
  rebuilt from the repository.
- Their test clip uses test thresholds and test zones (V3 after 7 s, a lane line that is not painted on the road),
  so its case counts are not real-world rates. Their README says so.

## Worth porting (not started; for the team to accept or decline)

| # | Port | Effort | Why |
|---|---|---|---|
| P1 | V6 helmet: toward-camera only, inside the road zone, 4 sightings with 3/4 bare-headed | small, same file | Measured: helmeted riders wrongly flagged fell from 48 to 7 |
| P2 | `KERB_AREAS`, `creep_m` and duplicate suppression for V1/V3/V7 | small to medium; `zones.py` also carries our V9 bus-lane changes | Measured: V3 went from 0 to 32 real stands |
| P3 | Hide the evidence label from the VLM, the V6 head question, and drop speed talk for non-speed rules | small, `surveillance/vlm.py` | Removes a known way for the VLM to go wrong |
| P4 | Gate/link topology with a two-link fallback for cross-camera matching | medium | Narrows the candidates as the network grows |
| P5 | Passed-vehicle list with a plate per vehicle, plate search and CSV | medium; costs CPU | Their "find a vehicle" feature |
| P6 | Payments and email | large, outside the 499A plan | Better left to their system, or added at the end if needed |
