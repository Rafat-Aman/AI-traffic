# Passed vehicles (WP17, P5)

A record for every vehicle that crosses a camera, not only the ones that broke a rule: its class, a colour name,
first and last seen, a small photo, and a number plate where one could be read. Plus search, CSV, "find a vehicle
anywhere" and look-alike suggestions. Ported in spirit from the Shihab system's *Passed Vehicles* page and fitted to
a CPU-only machine (see `docs/COMPARISON_SHIHAB.md`, P5).

Code: `surveillance/passed.py` (recorder, plate worker, colour, retention), `surveillance/lookalike.py` (look-alike
index), `surveillance/api_vehicles.py` (HTTP), `web/vehicles.{html,js,css}`, `surveillance/anpr.py`
(`read_crops`, the 50 px rule), hooks in `worker.py`, `pipeline.py`, `storage_mgr.py`.
Tests: `tests/test_passed.py`, `tests/test_lookalike.py`.

## What is recorded, and when

When a vehicle's track ends (gone for 1.5 s) and it was on a road zone for at least 5 frames, one row is written to
`passed_vehicles`: camera, run, the tracker's id, class (the pipeline's majority vote), first/last seen on the video
clock and on the wall clock (text and epoch), the width of its box in its best view, the colour name, the plate
status, and a thumbnail (`data/passed/cam<id>/run<id>_t<track>.jpg`, at most 160 px, JPEG 78).

The **global vehicle id is not copied**. It is read from `tracklets` (same camera, run, tracker id) whenever a row is
listed, because the cross-camera linker assigns and revises ids long after the vehicle left. Nothing here writes to
`tracklets`, and `link.py` is untouched.

Cameras can switch recording off with the setting `passed` (default on) and plate reading with `anpr_all`
(default off). Neither key is in `store.DEFAULT_SETTINGS`; the worker reads them with defaults, so no migration is needed.

## Colour

`passed.name_colour(crop)`: the middle half of the crop, at most 48 px a side, each pixel named from HSV (dark =
black; low saturation = white / silver / grey by brightness; otherwise by hue, with dark orange and dark yellow
called brown). If at least 30% of the pixels are coloured the commonest colour among them wins (a red car with dark
windows is red); otherwise the commonest of white / silver / grey / black. Under 40% agreement within the chosen
group the answer is `mixed`. Vocabulary: white, silver, grey, black, red, orange, yellow, green, blue, purple, brown,
mixed. Two-wheelers get the colour of what is in the middle of the box, usually the rider.
No VLM is called. The hook for an optional VLM look is `passed.set_vlm_look(store, id, "grey SUV")` and the
`look_vlm` column; search and the page already show it. Nothing calls it today.

## Plates of every vehicle

* After the violators. Their reader (`PlateCollector`) is untouched. The passing-traffic worker
  (`PassedPlateWorker`, one thread per camera, shares that camera's `PlateReader`) checks "does the violators' reader
  have work?" before **every crop** and waits (up to 10 s) while it does.
* Which vehicles: motorised classes only (bicycles, rickshaws, vans, wheelbarrows are `skipped: pedal`), best view at
  least 100 px wide (else `skipped: small`). Theirs: the 10 largest views; if none of the first 8 shows a plate,
  stop at 8 (`anpr.read_crops`).
* Never blocks: the queue holds 16 vehicles. `submit` returns False at once when full and the row is stored with
  `skipped: busy`; nothing is guessed. At the end of a run the queue gets 30 s to drain, the rest are marked
  `skipped: busy`.
* If the violators' reader reads the same vehicle, its result is reused (no second reading).
* Statuses: `read` (confirmed: city, class letter and the six digits), `partial` (number only), `unreadable` (a plate
  was seen but not accepted), `no plate seen`, `pending`, `skipped: busy|small|pedal`, `not read` (`anpr_all` off).
* **The 50 px rule.** "Confirmed" needs the agreeing reads to come from a plate at least `CONFIRM_MIN_PX = 50` px wide
  (theirs found three false confirmations on 25-44 px plates). It is in `anpr.vote()` and so applies to violators
  too. It changes nothing for them today because `MIN_PLATE_PX = 110` already keeps narrower plates out (all 13
  existing ANPR tests pass unchanged); it is the guard that stays if someone lowers `min_plate_px`. Passing vehicles
  use the same 110 px floor, so on 25-45 px Dhaka footage most will read `unreadable` or `no plate seen`, which is
  the point of not guessing.

### Measured cost (this 12-thread CPU dev machine; synthetic input)

No video was read. Synthetic frames and crops only (noise-textured, so JPEG work is pessimistic), plus the real
`plate_detector.onnx`, `bd_plate_chars.pt` and `bd_vehicles_best_640.onnx`. No real plate image was available, so the
character model was timed on a synthetic 300x100 image; **the accepted-read rate on real Dhaka plates is not
measured here.**

| What | Cost |
|---|---|
| Plate detector on one crop with no plate | 58 ms |
| Character model on one plate crop | 8.7 ms |
| One vehicle, no plate visible (8 crops) | ~460 ms of background-thread time |
| `observe()` per tracked vehicle per frame (1080p, growing boxes: worst case, every sample kept) | 0.38 ms; 5.6 ms/frame with 15 vehicles |
| Finishing one vehicle (decode, colour, thumbnail, insert) on the pipeline thread | 4.9 ms |
| `name_colour` | 0.19 ms |
| Detector alone (640 px ONNX, 1280x720) | 7.01 fps |
| Detector with the plate reader running flat out on another thread | 6.38 fps (-9%) |

The -9% is the ceiling: it needs the reader busy all the time, and the bounded queue means more traffic than the
reader can take becomes `skipped: busy` instead of more load. The recording itself (no plates) costs the
pipeline thread about 4% of a 143 ms frame in the worst case measured and less when samples are sparse (one per 0.2 s
per vehicle).

### Default for `anpr_all`: off

Each vehicle costs about half a second of reader time, so one reader keeps up with about two vehicles a second at
best, and busy traffic would mostly produce `skipped: busy`. It costs up to 9% of detector speed. And these are
people who broke no rule. So it is opt-in per camera (`anpr_all: true`); the list, colours, photos, search and CSV
work without it. `ANPR_ALL_DEFAULT` in `worker.py` is the single place to change that.

## API (staff only)

All under `/api/passed` (`/api/vehicles` already belongs to cross-camera journeys). Controller and admin; drivers get
403 (from `require` here and from the server's fail-shut `/api` guard: do not add these to `DRIVER_API`).

| Endpoint | |
|---|---|
| `GET /api/passed` | list: `camera_id`, `run_id` (default the camera's newest run; `all`), `q`, `plate`, `cls`, `colour`, `gid`, `since`, `until` (epoch or ISO), `status`, `limit` (<=500), `offset` |
| `GET /api/passed/export.csv` | same filter, streamed in 1000-row batches, UTF-8 with BOM, audited; cells starting `= + - @` are prefixed with `'` |
| `GET /api/passed/find` | across all cameras by `gid`, `plate`, `colour`, `cls`, `q`; grouped by global id; each with `last_seen` (camera, time), counting cross-camera sightings too |
| `GET /api/passed/{id}` , `/thumb` , `/similar?k=` | detail with journey; photo; look-alikes |
| `GET /api/passed/cameras` , `/runs?camera_id=` , `/lookalike/status` | pickers, index state |
| `GET /passed` | the page |

Plate search is partial and script-blind: Bangla digits become Latin ones, case and spaces/dashes are ignored, so
`১২-৩৪৫৬`, `123456` and `metro-ga 12` find `ঢাকা মেট্রো-গ ১২-৩৪৫৬`. The `q` box takes words: a colour, a class,
`#12` (tracker id), `g12` (global id), anything else is a plate fragment. Records without a wall clock are left out
of a `since`/`until` search.

## Look-alikes

`lookalike.LookalikeIndex` reads the appearance vectors already stored on `tracklets` (ReID 2048-d, else MobileNet
576-d; one space per model, never mixed) and returns the k most similar, same class by default, each with its
camera, time, thumbnail and whether it already shares the open vehicle's global id. **Suggestions only: the module
issues no write; a test compares `total_changes` before and after.** A vehicle with no tracklet (parked, too short)
has no vector and the answer says so.

Brute force, one float32 matrix per space, unit vectors so the dot product is the cosine. float32, not float16: numpy
has no half-precision matmul and float16 was 10-30x slower on this machine (576-d, 100k: 191 ms vs 6 ms for the
product). Built lazily, refreshed at most every 30 s (new rows appended, vanished rows dropped, late ReID vectors
noticed by id).

**Cap**: at most 100,000 vectors and 256 MB per space, so 100,000 for MobileNet (230 MB) and 31,250 for ReID (256
MB). Beyond it the oldest are left out and the answer says how many (`left_out_by_cap`).

Measured (`LookalikeIndex`, synthetic unit vectors in a temp database, class filter on, k=10, 30 queries):

| Space | Vectors indexed | Memory | Build from SQLite | Query median / p95 |
|---|---|---|---|---|
| MobileNet 576-d | 10,000 | 23 MB | 0.2 s | 2.8 / 3.6 ms |
| MobileNet 576-d | 100,000 | 230 MB | 2.1 s | 11.1 / 12.5 ms |
| ReID 2048-d | 10,000 | 82 MB | 0.2 s | 3.7 / 5.7 ms |
| ReID 2048-d | 100,000 rows, 31,250 indexed (cap) | 256 MB | 1.1 s | 8.1 / 10.0 ms |

No IVF was needed. `storage_mgr` strips vectors from tracklets older than `embed_days` (30), so old sightings drop
out of the index by themselves.

## Retention

These rows are records of drivers who broke no rule. They expire after **`PASSED_RETENTION_DAYS`** (environment
variable, default **30**; counted from when the record was made, so analysing an old video does not delete its
records at once). `passed.purge()` deletes the rows and their thumbnails, plus thumbnails no row refers to that are
over a day old. Two routes, same function:

1. `storage_mgr` plan: a dry run lists an item `purge_passed` ("N records of passing vehicles older than 30 days and M
   thumbnails"); an admin's apply deletes them and writes a `storage.purge_passed` audit row.
2. `PurgeLoop`: once at start and then daily, so the rule holds even if nobody opens the storage page
   (`passed.purge` audit row).

Deleting a camera deletes its records (foreign-key cascade); the purge picks up its leftover thumbnails.
`data/passed/` is not part of the storage inventory categories.

## Open points

* Plate-read quality on real footage is unmeasured (above).
* The per-camera switches `passed` and `anpr_all` have no checkbox in the camera settings dialog (`web/app.js` is not
  in this work package); PATCH the camera's settings to set them.
* A vehicle the tracker loses and finds again is recorded twice, as in the reference system.
