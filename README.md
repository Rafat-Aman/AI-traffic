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
training/               scripts and labels to rebuild / re-measure the helmet and plate models
surveillance/           video sources, detector, per-frame pipeline, clips, SQLite store, workers, web API
web/                    dashboard (HTML/CSS/JS, no build step)
tests/                  unit tests:  .venv/bin/python -m unittest discover -s tests
data/                   database, uploaded videos, evidence (created at runtime)
vehicle_tracker.py, speed_tracker.py   original scripts (still work)
```
