# WP10: 3D city view and signal optimiser (R-6, pipeline stage OPTIMIZE)

Files: `surveillance/api_optimize.py` (router), `web/city.html|js|css` (page at `/city`), `tests/test_city.py`.
The optimiser itself is `surveillance/optimize/` (WP4); this package exposes it, it does not change its logic
(one addition: `Simulator.run(stop=...)` so a server can cancel an overrunning simulation).

## What the page shows (and where each number comes from)

| On the map | Source |
|---|---|
| Camera markers, FOV wedges | `camera_geo` (admin placements, WP9) |
| Columns at cameras (height = density) | our `TrafficStateCollector.snapshot()` per road zone, via `snapshots_fn` |
| Link colour (density / speed / delay ratio) | the same snapshots turned into link travel times by `network.from_traffic_state` |
| Grey link | no camera data: free flow is assumed |
| Dashed, faded link | camera data older than 120 s (`CAMERA_FRESH_S`): stale |
| Signal dots (green/red at each approach) | live: the phase with the most green in the **recommendation** (a suggestion, the real signal is not read); playback: the simulated phase |
| Simulation playback | `/api/optimize/simulate` frames: synthetic |
| Routes | `routing.k_alternatives`; incident delay from recorded violations' impact |

Freshness: `snapshots_from_workers` adds `wall` (the collector's `wall_at(t)`; for a recorded file with no clock, the
last time the sample time advanced), so a stalled camera is stale, not fresh.

## Network editor (admin)

Intersections (signal / uncontrolled / boundary) and roads (lanes, speed limit, optional length, two-way). Each road
is two directed links; a camera plus its road zone is attached to one direction. Validation lists every problem
(duplicate ids, unknown camera, same zone on two roads, positions, sizes). Stored in `optimize_networks`.
"Generate demo around my cameras" makes a **synthetic** layout (junctions at the cameras joined by straight roads,
invented 250 m stubs); it is labelled synthetic in the document, the API and the UI, and is not saved until the admin saves.
With nothing saved, reads fall back to the demo, labelled.

## Endpoints

See the docstring of `api_optimize.py`. Controller: read, simulate, route. Admin: edit, Google settings.

## Google rule (R-8)

Google Maps Platform terms forbid showing Google Maps content on a non-Google map.

* The MapLibre 3D view never sends `use_google`; with it false Google is not even called, so no Google-derived
  value can be in `/live`, `/route`, `/recommendation` or `/simulate` responses. `city.js` additionally draws a
  link whose source is "google" as "no camera data".
* Google is applied only when the admin enabled it (`PUT /api/optimize/google`) AND a server key exists, only through
  `GoogleTraffic` (in-memory cache, per-minute budget), and only for `/live?use_google=true` and `/route` with
  `use_google: true`. Those responses say `contains_google: true`. They are for text/tables and for the Google-provider
  view of `/map`, which has an "Optimiser network (with Google times)" overlay that exists only when the provider is Google.
* Recommendation and simulation never use Google.
* The server key is never returned or audited. Nothing Google returns is written to disk (the saved network has no state).

## Honest limits

Simulator results are synthetic (own mesoscopic queue model, not SUMO, not calibrated to Dhaka); default demand is
invented; max pressure is Varaiya's established rule, not a trained model; 1200 vph/lane saturation flow is an
assumption; a camera sees tens of metres of a link whose value is assumed to hold along it; nothing is applied to a
real signal. Not verified: real cameras, real signals, live Google calls, browsers other than headless Chrome.
