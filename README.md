# Dhaka Traffic Surveillance

Detects traffic violations on Dhaka roads from recorded CCTV/phone videos and from live phone cameras, with a
web dashboard for setting up camera zones, watching live, reviewing evidence and viewing statistics.

| Code | Violation | Needs |
|---|---|---|
| V1 | Bus stopped anywhere but a bus stop that is just after an intersection (loading on a straight road) | road zone (+ intersections, bus stops) |
| V2 | Over the speed limit: 60 km/h on city roads, 80 km/h on highways and expressways | road zone with real size |
| V3 | Waiting / parked at a no-parking kerb in peak hours or heavy traffic (ride-share bikes, CNG stands) | no-parking kerb strips |
| V7 | Blocking a running lane: any other vehicle standing mid-road while traffic goes round it | road zone |
| V4 | Wrong-side driving | one-way road zone |
| V5 | Crossing a solid lane line | solid lane lines drawn |
| V6 | Motorbike rider or pillion without helmet | included (`models/helmet_cls`) |
| V8 | Vehicle banned on this road (rickshaw on a main road, CNG on a flyover, ...) | banned classes per zone |

Every case also gets an attempt at **reading the number plate** (Bangla, BRTA layout). See "Number plates" below.

## Start

```bash
./start.sh            # first run creates .venv and installs packages
```

Open **http://localhost:8000**. The terminal also prints an address for opening the dashboard from your
phone on the same Wi-Fi.

## Set up a new camera (once per camera position)

1. **Cameras → Add camera**
   - Choose **Automatic** zones (the AI learns them from traffic; see below) or **Manual** (draw them, step 2).
   - *Recorded video*: give the file path (fastest for large files) or upload it (mp4, mov, mkv, avi, ts, webm,
     ogv, mpg, wmv, 3gp). The recording time is read
     from the file name (`VID_20260922_173600.mp4`, NVR exports) or metadata; check it against the clock shown
     in the video.
   - *Phone camera (Wi-Fi)*: Android **IP Webcam** → `http://PHONE-IP:8080/video`, **DroidCam** →
     `http://PHONE-IP:4747/video`, or any `rtsp://` app. Use *Test connection*.
   - *iPhone / Mac camera*: iPhone **Continuity Camera** appears as a camera device. Use *Find cameras*.
     macOS asks once for camera permission for Terminal.
2. **Zone editor** (opens automatically):
   - **Road zone**: click the 4 corners of a rectangle on each carriageway that you can measure, then type its
     real width and length. Measure on site, or use Google Maps' *Measure distance*.
   - **Top-down view check**: lane lines must come out straight and parallel. If they don't, move a corner.
   - Set the **allowed direction**. After a first run the editor suggests it from observed traffic.
   - Set the **road type** (city / highway / expressway, which picks the speed limits), a lower local limit if
     there is one (e.g. a school zone), and **banned classes**.
   - Add **no-parking kerb** strips (select a road zone → *+ Left edge* / *+ Right edge*; 3 m wide by default).
   - Mark **intersections** (junction mouths) and then **bus stops** just after them in the direction of traffic.
     A bus stop that is not within 120 m after an intersection is shown in red and does not count: buses
     standing there are reported. Tick *officially approved mid-block bus bay* to accept one anyway.
   - Mark **signal queue areas**, **legal parking** and **U-turn openings**. Draw **solid lane lines** if they
     are painted.
   - The cyan line in each zone shows where measurement becomes reliable (explained under "Accuracy" below).
     If less than 20 m of a zone is reliable, extend the zone toward the camera.
   - **Save**. Every save is a new version, so you can tell which zones produced which case.
3. **Analyse video** / **Start live**. Watch on the dashboard. Violations pop up as they happen.
4. **Review**: confirm (`C`) or reject (`R`) each case using the snapshot, close-up and 10 s video clip.
   **Notice** prints a violation notice for a confirmed case (Print → Save as PDF).

## Automatic zones

With **Automatic**, the camera watches traffic for 1.5-7 minutes (a short recording: all of it), then:

- finds the road direction, each carriageway and its direction of travel, lanes, kerbs (marked no-parking),
  junctions, U-turn openings, signal queues, and bus stops (only just after a junction; buses stopping elsewhere
  are reported);
- calibrates the camera with no measuring: lens from vertical lines (poles, buildings), height from vehicle sizes.
  Speeds are accurate to about +-8-9%, so automatic zones flag speeding only above a wider margin (e.g. 67 km/h in
  a 60 zone);
- saves the zones as a new version and starts analysing.

The camera page shows what was found. **Learn again automatically**, or open the zone editor to adjust anything
(editing a zone's real width/length turns it into a measured zone with the normal speed margin). In the editor,
**Auto-detect zones** loads a proposal as a draft without saving it. Legal facts cannot be seen in video: banned
vehicle classes, legal parking bays, approved bus bays and lower local limits must be set in the editor.

## When the camera moves

Poles sway, cameras get knocked or re-aimed, PTZ presets don't return exactly, and handheld footage drifts. The
system keeps the frame the zones were drawn on (or learned from) and follows the view against it: optical flow of
background points every frame, plus a direct feature match to that frame at least once a second (vehicles are
masked out). Vehicle positions are mapped back onto the zones, so speeds, stops and directions stay correct, and
the zones on the live view move with the road. Cases recorded meanwhile note the correction in their details.

- **Camera moved: zones re-aligned**: corrected automatically (tested on a handheld clip drifting 190 px).
- **View changed: enforcement paused**: the zones have left the picture, or the view is moving and cannot be
  matched. No violation is judged until it matches again; point the camera back or redraw / relearn the zones.
- **View not verified**: the picture could not be matched for a moment (night, rain, glare). The camera was
  steady before, so it is assumed not to have moved and enforcement continues.

## Recorded video vs live

| | Recorded video | Live camera |
|---|---|---|
| Frames | every frame at 30 fps, none skipped, so the result is complete | always the newest frame, so it stays real time |
| Detector | 960 px (more far/small vehicles) | 640 px (~60 FPS) |
| Clock | recording start + video time | the Mac's clock |
| If the stream drops | n/a | shows *Reconnecting* and reconnects automatically |

All rules work on timestamps, so dropped live frames do not change the results (there is a test for this).
If a live camera runs below 6 FPS, the speed rule pauses and the dashboard says so.

## Accuracy: how wrong tickets are avoided

- **Human review**: nothing becomes a notice without a reviewer's confirmation. Rejections are counted, so
  each rule's precision is visible under Statistics.
- **Measurement reliability gate**: far from the camera, one detector pixel covers more than half a metre of
  road, so box jitter looks like movement. Positions there are not used for speed, direction or stopping
  (this removed a false wrong-side case and an 85 km/h bus spike on the Banani Kakoli footage).
- **Speed (V2)**: limit 60 km/h (city) or 80 km/h (highway / expressway); flagged only above the limit +
  max(3 km/h, 5%) measurement margin (63 / 84 km/h), sustained for 1 s, *and* the average speed through the zone
  must also be over.
- **Stopping (V1 / V3 / V7)**: a red-light queue or a jam is never a violation. A stand counts only when
  - at least 3 cars/buses/CNGs actually **drove past** the stopped vehicle (in a jam nobody passes; motorbikes
    filtering through a queue do not count), and
  - for V1 / V7, it is at the **head of its queue**: time spent right behind another standing vehicle in the same
    lane does not count (the one in front is the blocker), and
  - its position is measured: a vehicle that drives into the unmeasurable far end does not keep a timer.

  Thresholds: bus 10 s (V1), kerb 20 s (V3), running lane 20 s (V7), editable per camera. Intersections and
  signal queue areas are exempt from all three; V3 is enforced in peak hours *and* whenever traffic is heavy
  (15+ vehicles in the zone on average). When the tracker gives a stopped vehicle a new ID (pedestrians walking
  in front of a parked bike), the new ID keeps the timer, even if the class flips between look-alikes
  (Motorbike/Bicycle, CNG/Easybike/Leguna, Car/MPV).
- **Evidence for stops**: a photo 2 s after the vehicle stopped, the scene when it is reported (with the path it
  came along), a close-up, a 10 s clip and the plate, plus a timeline (stopped → vehicles passed → reported).
- **Camera movement**: positions are corrected when the view shifts, and enforcement pauses when it cannot be
  (see "When the camera moves").
- **Wrong side (V4)**: at least 8 m of fairly straight travel against the allowed direction; U-turn openings
  and intersections (turning traffic) are exempt.
- **Solid lines (V5)**: the vehicle must be clearly on one side of the line, then clearly on the other.
  Straddling the line is not a crossing.
- **Banned vehicles (V8)**: the vehicle class must be at least 75% consistent across frames (CNG, Easybike and
  Leguna look alike).

No system is 100% accurate. Speed accuracy depends directly on the road-zone measurements you enter. The
reliable part of a zone, and the detector's view of distant vehicles, depend on camera height and resolution.

## Helmets (V6) and number plates

Both run on full-resolution crops, so recorded videos are now analysed at full resolution by default (camera
settings: *Analysis resolution*). Model sources, licences and the full evaluation are in `models/README.md`.

**Helmets (V6).** A classifier looks at each moving motorbike 4 times per second and estimates whether anyone on it
is bare-headed. It decides once per motorbike, when it leaves the view: 3 confident bare-head sightings, at least half of all its confident sightings. Sightings are skipped when the motorbike is parked,
too far away (box under 100 px), or right behind another bike, where the next bike's riders' heads would be in the
crop. Public helmet data did not transfer to a Dhaka overpass camera (the best public model flagged 82% of helmeted
riders), so the classifier was also trained on 1,232 motorbikes from this camera. On held-out Dhaka riders it
flagged 0 of 155 helmeted riders but caught only 3 of 11 bare-headed ones per frame; on the held-out demo video it raised no false alarm but also caught none of the 11 bare-headed riders. So V6 is precise but misses most cases. More labelled Dhaka data is the
way to raise that (see `training/README.md`).

**Number plates.** For every case, the vehicle's best full-resolution crops (taken as it comes closest) are read in
the background: plate detector → Bangla character detector → BRTA layout (`ঢাকা মেট্রো-গ ১২-৩৪৫৬`) → vote across
frames. A number is shown only if 3 frames agree on all 6 digits with high confidence on plates at least 110 px
wide. On 167 held-out plates this gave **0 wrong numbers** while reading 60-75%. City and class letter are a
suggestion to check on the image: the training plates only cover Dhaka, Khulna and Jashore. Otherwise the case says
**"Not readable in the video"** and shows the best plate image. On the Banani Kakoli footbridge camera, plates are only
35-100 px wide and blurred, so every plate is (correctly) reported unreadable. To read plates, use a camera that
sees 1-2 lanes so plates are at least 120 px wide.

## Accounts: who can see what

Everything the dashboard exposes used to be open to anyone who could reach the port - the review queue, the
camera list, and a CSV of every case with its number plate. There are now three kinds of account.

| Role | Sees | Does |
|---|---|---|
| **Driver** | Only violations against vehicles registered to them | Views the evidence, adds a plate to claim |
| **Traffic controller** | Review queue, cameras, statistics, cross-camera vehicles | Confirms or rejects cases, runs *Explain*, prints notices |
| **Administrator** | Everything | Cameras, zones, accounts, the vehicle registry, the activity log |

Each lands on its own surface: staff on the dashboard, drivers on `/portal`, everyone else at `/login`.
The driver portal is a **separate page**, not a filtered dashboard, so there is no state in which the wrong
thing can be shown to the wrong person.

### Signing in

Staff use a username and password. **Drivers use their driving licence number and a one-time code sent to
the mobile registered against it**, and that distinction is the point: a licence number is printed on a
card, handed to police and copied onto rental forms. It identifies a person but keeps no secret. On its own
it would let anyone who has seen a licence read that person's violation history, and licence numbers are
regular enough to walk through one by one.

So the licence number only says *who is being asked about*; the code is what authenticates. Both paths are
rate limited, and the reply to a code request is **word-for-word identical** whether or not the licence
exists - otherwise the endpoint answers "does this person hold a licence?", which is the question it is
meant to refuse. That is pinned in `tests/test_auth.py`, because the first version did leak it: it returned
a masked phone number for a licence that existed and nothing for one that did not.

There is no SMS gateway wired up. By default the code is written to the server log for an operator to read
out. `OTP_DELIVERY=echo` also returns it in the response so the login page can display it, which is useful
for a demonstration and must never be set on a hosted instance - it warns about itself every time.

### First run

A database with no accounts gets an administrator, whose password is printed to the console once:

```
  No accounts existed, so an administrator has been created:
      username  admin
      password  NAldYyi3uXc38t3d
  This is shown once. Sign in and change it.
```

Set `ADMIN_USER` / `ADMIN_PASSWORD` to choose them instead. Change the password from the dashboard; doing so
ends every session on that account.

### Registering a vehicle to a driver

A driver only sees cases for plates **verified** as theirs, because the system reads plates rather than
drivers. Two routes in, and the portal says as much to the driver: a case shows the vehicle that was
involved, not who was at the wheel.

* an administrator links a plate to a licence under **Accounts → Vehicles**, standing in for the registry
  import a real deployment would do;
* a driver claims a plate in their portal, which shows nothing until an administrator verifies it -
  otherwise claiming a plate would be enough to read its owner's record.

### Who decided what

Every confirm, reject, registration, verification and sign-in is written to an activity log with the name
of whoever did it, visible under **Accounts → Activity**. A case with nobody's name on it is not evidence
of anything. The name is stored alongside the account id and survives the account being closed, so
attribution cannot go missing later.

### What this does not do yet

There is **no dispute flow**: a driver who believes a plate was misread has to contact the authority out of
band, and the portal tells them so. Everything it needs is in place - the case, the evidence and the
audit trail - so it is a contained addition rather than a redesign.

Sessions and rate limits live in this process's memory and in SQLite, which suits one server. Behind
several workers, the rate limiter needs to move to Redis or the database, or each worker will allow the
full quota on its own.

## The same vehicle on another camera

Each camera's tracker numbers vehicles independently, so track 7 here has nothing to do with track 7 two
junctions away. **Vehicles** in the dashboard links them into one identity, so a journey across the city is
a query rather than a pile of separate per-camera logs.

### Using it

1. **Vehicles → Add camera views.** Pick every video of the same junction or stretch of road at once. Each
   becomes its own camera; they are analysed one after another (the detector already uses every core, so
   running them together would only halve each other's speed) and matched at the end.
   Recording times are read from the file names where possible - `VID_20260922_173615.mp4` and most NVR
   exports - which is what puts the cameras on a common clock. Matching is most accurate when the clips
   overlap in real time.
2. Watch the progress bar. Each camera learns its road layout automatically, then analyses.
3. When the last one finishes, matching runs by itself. Afterwards, **Match vehicles across cameras**
   re-runs it whenever you add more footage.
4. Click any vehicle for its journey: every camera it passed, in order, with the time between them.

Already have cameras set up? Just press **Match vehicles across cameras**. Nothing else is needed.

### How a vehicle is recognised

Every pass is summarised as a *tracklet*: when and where it crossed the zone's reference line, in metres
across the carriageway and seconds on the wall clock, with its speed, physical width, class, a colour
signature, and a 576-number appearance vector. Matching happens **in metres on the road plane**, which is
the whole trick - it is the same physics at every camera, so nothing is configured per site:

| Evidence | Where it comes from |
|---|---|
| Class must be **identical** | majority vote over the track; never Car↔Pickup or Motorbike↔Bicycle |
| Transit time = gap ÷ speed | the gap between two cameras is **learned from the traffic**, not measured |
| Lateral position maps linearly | learned per route, mirrored when cameras face opposite ways |
| Speed and physical width agree | the homography this project already uses for speed |
| Colour | HSV histogram + CIELAB of the body — or of the **rider's clothing** on a two-wheeler |
| Appearance vector | MobileNetV3 features over the clearest crops (`surveillance/embed.py`) |

Each cue becomes a log-likelihood ratio **fitted from the confident matches themselves**, so no threshold is
guessed and a cue that fails to separate is switched off automatically. The assignment is one-to-one, and
leaving a vehicle unmatched has a cost — most traffic at a junction never reaches the next camera, and
forcing pairs is how a matcher invents journeys. Routes are discovered automatically; nobody describes the
camera layout.

### Does the appearance vector earn its place?

It is **not** a vehicle re-identification model — MobileNetV3 was trained on ImageNet, not on telling one
car from another. Measured on real footage from this project with `tools/reid_eval.py`, which splits each
vehicle's crops into an early and a late half and asks whether the halves match each other better than they
match other vehicles:

| Cue | Same vehicle | Different | Separation | Rank-1 |
|---|---|---|---|---|
| Colour histogram | +0.927 | +0.584 | 2.37 | 61.3% |
| CIELAB colour | −0.696 | −3.238 | 2.12 | 38.7% |
| **Appearance vector** | +0.866 | +0.522 | **2.52** | **71.0%** |

So it is the strongest single appearance cue here, and on a fleet whose colour is identical it is the
difference between refusing and matching:

| Identical-colour fleet | Links | Precision | Recall |
|---|---:|---:|---:|
| Colour only | 0 | — | *refuses* |
| **+ appearance vector** | 90 | **98.2%** | **98.5%** |

Proper ReID weights would be better and would drop straight in — the interface would not change — but they
want a GPU and a labelled vehicle corpus, and this project has neither.

### When it refuses, and why that matters

Four safeguards exist because the matcher **failed these cases in testing**, not because they were
anticipated:

1. **A flow is never taken in both directions.** A dense stream of look-alikes produces a reverse peak too;
   accepting it linked every vehicle backwards.
2. **Routes are gated and ranked on how much traffic a fixed distance explains**, not on how sharp the peak
   is. Sharpness is biased against single-class traffic — which is most Dhaka roads: a correct 121 m route
   explaining 138 pairs against a fixed delay's 68 scored only 4.1σ and was discarded. Fixing this took
   recall on an all-CNG stream from 0% to 90%.
3. **A route must beat a fixed delay.** Two streams that merely overlap in time share a *delay*; vehicles
   that really travel share a *distance*, so a fast one arrives sooner.
4. **A class is refused outright** when no appearance cue separates it *and* its vehicles arrive closer
   together than the timing can resolve. Then vehicle *k* and *k+1* explain the same arrival equally well,
   and the matcher can be confidently, systematically wrong.

Measured on synthetic traffic with known pairings (`tests/test_reid.py`; synthetic traffic flatters a
matcher, so read these as a floor, not a result):

| Traffic | Linked | Precision | Recall |
|---|---:|---:|---:|
| Mixed classes, 1.2 s headway | 99 | **99.3%** | 98.7% |
| Half near-identical CNGs | 97 | **99.3%** | 96.7% |
| All CNGs, 1.2 s headway | 91 | **99.3%** | 90.0% |
| All CNGs, dense 0.5 s | 78 | **95.3%** | 74.3% |
| **Identical fleet, no vector** | **0** | — | *refuses* |

Recall falls before precision does, which is the intended trade: an unlinked vehicle is a gap, a wrongly
linked one is a false journey attached to a plate. Flow discovery also needs enough traffic — around 60
vehicles per camera in testing; below that it links nothing rather than guessing.

**Not validated on real footage yet.** There is no hand-labelled cross-camera ground truth for this project,
so the matching figures above are synthetic. The team's own two-camera footbridge study is the dataset to
check them against.

## Explaining a case (vision-language model)

The rules decide **whether** a rule was broken. They cannot say what the scene was doing, why the driver
behaved that way, or whether the photographs really back the finding up. **Explain**, on any case in Review,
asks a vision-language model those three questions and writes the answer onto the case.

It is deliberately **on demand**, not automatic: the model is the most expensive thing in the system, and an
officer only needs it on the cases they are actually deciding. Nothing runs until the button is pressed, so
it never competes with the detector.

The answer is split the way an officer reads a case - **before, the event, after, probable cause** - plus a
one-line summary for the notice, and a verdict on the evidence itself:

| Verdict | Meaning |
|---|---|
| Evidence supports the finding | the photographs show what the rule decided |
| Evidence contradicts the finding | they show the innocent explanation instead - a likely false positive |
| **Not enough evidence to judge** | they cannot settle it: too blurred, too far, or the innocent explanation fits equally well |

The third is the important one. For each rule the model is told both what the engine decided *and* the
specific innocent explanation that would overturn it - for a stopped bus, that the whole road is stopped at a
red light - and it is required to decline rather than invent a cause. A wrong explanation on an enforcement
notice is worse than no explanation, so an unparsable or unrecognised answer degrades to "not enough
evidence", never to agreement.

The model never detects anything. Detection, tracking, speed and the rule decision are done already and are
handed to it as given facts; it reads the evidence photographs and judges.

**It is also told where else that vehicle was seen.** Once cameras are linked, a case carries the vehicle's
whole journey into the prompt, and that changes what can be said about it: a bus halted for forty seconds
reads very differently when the camera before it also had it crawling — a jam it was caught in — than when
it came through the previous junction at speed and then stopped dead. The explanation gets an **Other
cameras** line when that evidence actually contributed. The link is itself an inference, and the model is
told to treat it as supporting context rather than proof. That is also why a 3B model is
enough: it is not being asked to find vehicles, only to explain them.

### Setting it up

```bash
./tools/setup_vlm.sh        # once: llama.cpp + Qwen2.5-VL-3B (~3.1 GB into models/vlm/)
./tools/vlm_server.sh       # leave running while you use Explain
```

Then press **Explain** on any case. On a laptop CPU an answer takes **about one to two minutes**; it
appears by itself when it is ready, so you can carry on reviewing other cases meanwhile. Check what it will
cost on *your* machine first:

```bash
.venv/bin/python tools/vlm_check.py            # cores, memory, disk, and whether it fits
.venv/bin/python tools/vlm_check.py --case 12  # explain a real case and measure the time and memory
```

For better reasoning - and for comparing the two in the report - set a key and the cloud backend is used
instead. No key is ever stored in the database or shown in the dashboard:

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY (+ OPENAI_BASE_URL for a compatible endpoint)
export VLM_BACKEND=cloud          # default "auto": the local server if it is up, otherwise the key
```

`auto` prefers whichever is actually available, so the dashboard keeps working with no model at all - the
button simply reports that none is configured.

### What it costs

Measured on the development machine (Intel i5-11400H, 6 cores, AVX-512/VNNI, no usable GPU) — measured,
not estimated:

| | |
|---|---|
| Resident memory | **4.7 GB** while the server runs, freed when `vlm_server.sh` is stopped |
| One photograph | 15-60 s |
| Two photographs (the default) | **~110 s** |
| Disk | 3.1 GB of weights in `models/vlm/` (git-ignored) |

Image size dominates the cost on a CPU backend, because every extra pixel becomes vision tokens that must be
encoded before a single word is produced. Two photographs at 768 px is the default because the close-up
identifies the vehicle and the wide shot shows whether the rest of the traffic was moving - which is the
question that separates an illegal stop from a red light. Trade accuracy for speed with `--images 1`, or the
other way with `--max-side`, in `vlm_check.py`.

### Does it actually check, or just agree?

A model that agrees with whatever the rule engine says is worse than no model: it turns a false positive into
a confident paragraph on a notice. `tools/vlm_probe.py` tests exactly that, by handing the model cases whose
photographs cannot support the finding:

```bash
.venv/bin/python tools/vlm_probe.py        # needs the server running
```

Qwen2.5-VL-3B **failed this on the first attempt** - shown a photograph with no bus in it and told a bus had
stopped illegally, it answered "supported, 90% confident" and invented a cause. Two changes fixed it, and
both are worth knowing about:

1. **The prompt asks what is visible before it mentions the finding.** Writing down what it can see, first,
   makes the model much harder to lead.
2. **The system overrules the model.** If it votes "supported" but says it could not identify the vehicle, or
   describes a photograph without ever mentioning that kind of vehicle, the verdict is forced to "not enough
   evidence" and the case says so. The guard can only ever weaken a verdict, never strengthen one.

It now passes 3/3. Re-run the probe after changing the prompt, the model or the backend - and treat a failure
as a reason not to trust that model's explanations.

### What the local 3B model is and is not good for

Its **verdict is noisy**. The same case, run twice, came back "supports the finding, 90%" and then
"contradicts it, 0%". The guards above stop it agreeing with the impossible, but they cannot make a 3B model
decide consistently. So:

- Use the **narrative** - what is in the photographs, before, during, after - which is steady and genuinely
  useful for writing up a case.
- Treat the **verdict as a prompt to look again**, not as a second opinion worth quoting.
- Use `VLM_BACKEND=cloud` for anything that goes in front of a person as a judgement, and for any figure
  that goes in the report.

None of this output is evidence. The photographs, the trajectory and the measurements are the evidence; the
model only describes them, and the dashboard says so under every explanation.

## Performance (MacBook Air M5, Banani Kakoli 4K/60 fps footage)

| | |
|---|---|
| Detector | auto-exported to CoreML (Neural Engine): 90 FPS at 640 px vs 66 FPS on the GPU, with the same boxes |
| Video decoding | ffmpeg with VideoToolbox hardware decoding, 4K → 1080p |
| Full pipeline | ~60 FPS at 640 px, ~30 FPS at 960 px (1080p analysis); ~16 FPS at full 4K with helmets + plates |
| Live phone stream | full 25 fps processed, 0.02 s delay |
| Both at once | a live camera at 25 fps alongside a video analysis at ~0.4× real time |

## Command line (no dashboard)

```bash
.venv/bin/python violation_tracker.py --source video.mp4 --zones violation_zones.json
.venv/bin/python violation_tracker.py --source http://192.168.0.105:8080/video --zones violation_zones.json
```

Export zones from the dashboard's editor (*Export JSON*) to use them here.

## Project layout

```
app.py                  dashboard server (FastAPI) - start here
violation_tracker.py    command-line runner + OpenCV zone calibration
violations/             the rules (V1-V8), zone geometry, evidence images
models/                 detector exports, helmet classifier, plate models (+ README with sources and accuracy)
models/vlm/             vision-language weights (git-ignored; ./tools/setup_vlm.sh fetches them)
training/               scripts and labels to rebuild / re-measure the helmet and plate models
surveillance/           video sources, detector, per-frame pipeline, clips, SQLite store, workers, web API
surveillance/vlm.py       the reasoner: prompts, backends, the decline rule, the background queue
surveillance/auth.py      password and code hashing, rate limits, what each role may do
surveillance/webauth.py   sessions, the route guards, first-run administrator
surveillance/reid.py      cross-camera Re-ID: tracklets, colour signatures, route discovery, matching
surveillance/embed.py     the appearance vector per vehicle
surveillance/link.py      batch and online linking, and multi-video site runs
tools/reid_eval.py        does appearance actually separate vehicles on this footage?
tools/                  setup_vlm.sh, vlm_server.sh, vlm_check.py (hardware + cost check)
web/                    dashboard, plus login.html (sign-in) and portal.html (a driver's own cases)
tests/                  unit tests:  .venv/bin/python -m unittest discover -s tests
data/                   database, uploaded videos, evidence (created at runtime)
vehicle_tracker.py, speed_tracker.py   original scripts (still work)
```
