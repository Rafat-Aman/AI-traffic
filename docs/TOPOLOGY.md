# Camera-network topology (WP19)

Gates, links and travel-time windows for matching a vehicle across cameras. Code: `surveillance/topology.py`
(matching, tables), `surveillance/geo.py` (`link_window`), `surveillance/api_map.py` (`/api/topology/*`),
`web/map.*` (admin editor, police read-only), `surveillance/reid.py` (entry/exit recording),
`surveillance/link.py` (online and batch hooks). Ported in spirit from the teammate's system
(`docs/COMPARISON_SHIHAB.md`), after Liu et al., AI City 2021, "City-Scale Multi-Camera Vehicle Tracking Guided by
Crossroad Zones".

## What it does

* **Gate**: a named polygon on one camera's frame (fractions 0..1 of the frame, so snapshot size does not matter)
  where vehicles come in or go out; kind `in`, `out` or `both`. Every tracklet now records where it entered and left
  (bottom-centre of its first and last box, as a fraction of the frame), when (`t_enter`, `t_exit`), and which gate
  each point fell in (a point just outside a gate, within 0.05 of the frame, takes the nearest). Columns
  `entry_x, entry_y, exit_x, exit_y, t_enter, t_exit, entry_gate, exit_gate` are added by `topology.ensure_schema`
  (idempotent `ALTER TABLE`); no existing column changed. The gate is worked out in `Store.add_tracklet`, so
  `worker.py` needs no change. A camera without gates stores NULLs and is matched exactly as before.
* **Link**: camera A's out gate to camera B's in gate with a window `[min_s, max_s]` = time from the vehicle
  leaving A's view to entering B's.
* **Window from the map** (`geo.link_window`): earliest = (distance between the placed cameras - both view ranges)
  at 60 km/h; latest = distance x 1.4 (the road factor `section_speed.py` already assumes) at 5 km/h + 10 s, rounded
  up to 5 s, capped at 30 min. Why: the window is a filter. Too narrow silently drops the real vehicle; too wide only
  costs a few more candidates for the appearance score to rule out, so it is deliberately generous. Example: cameras
  500 m apart with 80 m view ranges give 20-515 s. The 5 and 60 km/h and 1.4 are assumptions, not Dhaka
  measurements; an admin edits the window once real travel times are known (stored as `suggested` or `manual`).
* **Matching**: when the new sighting's camera has links, its candidates are the tracklets that left a linked
  camera through the linked gate with the gap inside the window (an indexed range query on
  `(camera, exit gate, exit time)`). Scoring is `reid.pair_terms`, the existing speed / size / colour / embedding
  LLR cost, with the time term centred on the window middle (sigma = a quarter of the window, so it costs at most
  about 2 inside it) and lateral position switched off (meaningless across cameras). Class stays a hard rule.
  Acceptance (tuned for precision, see below): cost < 3 and the runner-up at least 3 worse (`TOPOLOGY_ACCEPT`, `TOPOLOGY_MARGIN`). Where *both* cameras have topology
  only this path may link them; if either has none, the old flow-based path applies.
* **Two-link fallback**: if nothing is accepted, routes through a camera in between that missed the vehicle are
  tried: window = both windows added, plus up to 60 s dwell inside the missed camera's view; stricter bar (cost < 1.5,
  runner-up 3 worse; `TOPOLOGY_SKIP_ACCEPT`, `TOPOLOGY_SKIP_MARGIN`). A candidate that the middle camera did see (it already has a sighting there) is not skipped
  over. The two-link routes are recomputed into `camera_routes` whenever a gate or link changes.
* **Out of order**: when a record arrives whose *next* camera was processed first, that next-camera sighting (if it
  is the first of its vehicle, linked or not) is asked again with the new record among the candidates. If it picks
  the new record the ids join; when both already had an id, the vehicle whose first sighting is earlier keeps its id
  (not the smaller number) and the other is recorded in `vehicle_merges (old_gid -> new_gid)` so the old id stays
  findable (`GET /api/topology/vehicle?q=DV-000009`). One-link only; the two-link case is not re-asked.
* **Batch = online**: `link.run_linking` runs the flow-based answer, drops its links between two topology cameras,
  and adds the topology links by *replaying the stored tracklets in exit-time order through the same `choose()`* the
  online path calls (over an in-memory index instead of the database). The colour LLRs are fitted from the data
  (`topology.fit_llr`), by `run_linking` and by `OnlineLinker.learn_from`.
* **Display id**: `topology.display_id(123)` is `DV-000123`; `parse_display_id` accepts `DV-000123`, `dv123`,
  `123`. Display only; the stored id stays an integer.

## HTTP and UI

`GET /api/topology` (staff), `PUT /api/topology/cameras/{id}/gates` (admin; removing a gate removes its links),
`GET /api/topology/suggest?from=&to=` (staff), `POST/PUT/DELETE /api/topology/links[/{id}]` (admin), all writes
audited; `GET /api/topology/vehicle?q=` (staff). Drivers get 403. On `/map`, pick a camera: the Network card draws
gates over a frame fetched from the existing `/api/cameras/{id}/frame` (the zone editor's endpoint; a live camera
must be stopped first), and the map draws each link as a dashed line labelled with its window. Police see gates and
links but no buttons.

## Synthetic measurements

**Synthetic data only.** There is no second real camera. `tests/test_topology.py` (`python -m tests.test_topology
measure`) generates: 3 cameras 300 m apart on one road, 150 through-vehicles per seed (3 seeds), arrivals every
~3 s, speeds log-normal around 7 m/s, travel time with 12% noise, link windows 10-150 s, a 64-d appearance vector
per vehicle seen through noise of norm 0.9 at each camera (two views of one vehicle have cosine about 0.55 against
about 0 for different vehicles). On top: 12% of vehicles vanish through the main gate, 12% turn off through a
side gate, 15% join a later camera through its main in-gate and 10% through a side gate, none with an earlier
sighting. *Precision* = of consecutive sightings given one id, the share that are one true vehicle. *Kept* = of
vehicles seen by 2+ cameras, the share whose sightings all carry one id.

| world | mode | links | precision | kept |
|---|---|---|---|---|
| all three cameras see everyone | no topology (existing flows) | 57 | 1.000 | 0.073 |
| all three cameras see everyone | topology, batch (= online) | 168 | 0.974 | 0.908 |
| middle camera missed every vehicle | no topology | 0 | n/a | 0.000 |
| middle camera missed every vehicle | topology (two-link), batch (= online) | 51 | 0.956 | 0.664 |

Tuning sweep (same seeds, topology on; accept/margin one link, accept/margin two links; P = precision, K = kept):

| 1-link | 2-link | all P / K | missed-middle P / K |
|---|---|---|---|
| 9 / 1 (first version) | 6 / 1.5 | 0.810 / 0.786 | 0.790 / 0.774 |
| 6 / 2 | 4 / 2.5 | 0.904 / 0.880 | 0.809 / 0.748 |
| 4 / 3 | 3 / 3 | 0.943 / 0.889 | 0.869 / 0.744 |
| 3 / 3 | 2 / 4 | 0.966 / 0.905 | 0.912 / 0.619 |
| **3 / 3** | **1.5 / 3 (chosen)** | **0.974 / 0.908** | **0.956 / 0.664** |
| 3 / 3 | 1 / 5 | 0.976 / 0.908 | 0.970 / 0.632 |
| 2 / 5 | 1.5 / 5 | 0.963 / 0.829 | 0.957 / 0.587 |

Both worlds reach precision >= 0.95; the chosen row has the highest kept share among rows that do. Stricter bars
cost kept share only past this point. The existing matcher is cautious on this look-alike traffic (all right, but
7% kept); the tuned topology path keeps 91% / 66% at 0.97 / 0.96. Precision here depends entirely on the invented
appearance noise; the sweep is on the same 3 seeds it is reported on, so some of the gain is fit to them.

**Candidates per sighting** (same world, constant arrival rate, a store that grows; probe = the last 60
sightings at the middle camera; 1 route, window 10-150 s):

| tracklets stored | candidates per sighting (topology) | tracklets within the old +-600 s window on other cameras | ms per sighting (candidate query) |
|---|---|---|---|
| 654 | 31.8 | 409 | 0.32 |
| 2,587 | 27.8 | 330 | 0.28 |
| 10,312 | 27.3 | 366 | 0.27 |

Candidates are set by the link window and the arrival rate, not by how much is stored (stored grew 16x; candidates
did not). The old online path is also time-bounded (+-10 min) but looks at 13 times more here (366 vs 27 at the largest store).

## Limits

* Verified on synthetic data and unit tests, not on real video; the editor page was syntax-checked but not driven in
  a browser here.
* A sighting that entered through no drawn in-gate (or a tracklet stored before gates existed) has no topology
  candidates; tracklets can be re-tagged after editing gates, but only if they carry entry/exit points (rows stored
  before this change do not).
* Camera clock offsets from the map are applied to the window test; uploaded videos with unaligned clocks break it
  exactly as they break section speeds.
* The accept bars were swept on synthetic data only and the 60 s dwell is a hand-chosen value; re-tune on real footage.
