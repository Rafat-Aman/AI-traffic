# Storage, retention and the ingest queue (WP13, requirement R-5)

Files: `surveillance/storage_mgr.py` (inventory, retention plans, compaction, disk monitor), `surveillance/ingest.py`
(queue), `surveillance/api_storage.py` (router), `web/storage.{html,js,css}` (page `/storage`),
`tests/test_storage.py` (46 tests, temp folders and a temp Store only). Capacity numbers are in `CAPACITY.md`.

## What lives where

| Thing | Where | Referenced by | Typical size |
|---|---|---|---|
| Raw uploaded video | `data/videos/` | `cameras.source` (kind `file`) | 0.1-1 GB per 10 min (the dev clip: 148 MB for 2 min at 1242x1440) |
| Case stills, plate crop, extra frames | `data/evidence/cam*/run*/` | `violations.snapshot / crop / plate_img / evidence[]` | 10-300 KB each |
| Case clip (10 s, the original) | same | `violations.clip`; its hash is sealed by a citation | 1-6 MB |
| Driver's delivery copy (720p) | `<clip>_delivery.mp4` | `citations.delivery_clip` (not sealed) | 1-2 MB, regenerable |
| Clip being encoded | `<clip>.part.mp4` | nothing | - |
| Zone reference frames, auto-zone previews | `data/zone_refs/`, `data/autozone/` | zone configs | < 1 MB; inventory only |
| Tracklets | table `tracklets` | segment index, journeys, section speed, linking | row ~0.5 KB + vectors ~10 KB |
| Database | `data/surveillance.db` (+ `-wal`) | - | grows with tracklets |

A tracklet row is one vehicle's pass through one camera. The row (class, time, speed, direction, colour `lab`,
`global_id`) is what the query/segment index, the map's section speeds and journeys read. The two appearance
vectors (`embed` 2.3 KB, `embed_reid` 8 KB) are read only by the linker, which runs when the vehicle is first
seen and again in the batch link after a site upload.

## Retention policy

All values are editable by an administrator on `/storage` (or `PUT /api/storage/policy`) and audited.

| Setting | Default | Why |
|---|---|---|
| `raw_video_days` | 7 | Raw video is the bulk of the disk and everything of value is extracted: tracklets, plus for each case a 10 s clip and stills. A week covers re-analysis after a bug fix or a changed zone. Held back while the camera is running or queued, has a case pending review, or has a standing citation (disputes). |
| `dismissed_days` | 30 | Rejected cases are false alarms with no legal value; a month leaves room for a second look or an appeal. |
| `legal_retention_days` | 1095 (3 years) | Evidence of confirmed cases. **The authority's decision**, not ours: `legal_retention_confirmed` is false by default and while it is false nothing in this class is deleted. Minimum accepted: 365. |
| `delivery_clip_days` | 30 | The driver's copy is regenerable from the sealed original. Never removed while a standing citation points at it. |
| `embed_days` | 30 | Drop the two vector blobs of tracklets older than this; keep the rows. |
| `stale_part_minutes` | 60 | A `.part.mp4` this old is a crashed encode. `citation.blockers` says "still being encoded" while one exists, so it blocks citing for ever. |
| `orphan_days` | 0 (never) | Files in `evidence/` that no case refers to are reported, not deleted, unless you set a number. |
| `warn_pct` / `pause_pct` | 80 / 90 | Disk monitor. At 90 % the ingest queue starts nothing new. |

Hard rules (not settings):

* A file of a case with a **standing citation** (issued or paid, not withdrawn) is never deleted, whatever the
  settings: the seal lists its SHA-256 and `verify` would report it "missing". Test:
  `test_cited_evidence_is_never_deleted_whatever_the_policy` and
  `test_sealed_citation_still_verifies_after_a_full_apply`.
* A pending case keeps all its files; its camera's raw video is kept too.
* A file referenced by several rows is kept if any of them says keep.
* Only regular, non-symlink files under `data/videos` and `data/evidence` are ever removed. Autozone, zone refs and the
  database are inventory only. Plan items are validated again at apply time.
* The 24-hour plan lifetime and re-checking at apply mean a citation issued after the preview still protects its files.

What is not done: rows in `violations` whose files were purged keep pointing at them (the UI shows a broken image).
`storage_purged` records every file removed, with plan and rule, so a missing file can be explained. Compaction frees
space inside SQLite for reuse; the file shrinks only after a `VACUUM`, which is not run automatically (it locks the
database).

## Dry run, apply, audit

`POST /api/storage/plan` walks the data folder, applies the rules above and stores a plan (`storage_plans`): each item
has path, bytes, rule and a sentence of reasoning; the kept files are summarised per rule with an example. Nothing
is deleted. `POST /api/storage/apply {"plan_id": n}` deletes at most those items, asking the same question again for
each; items that are no longer deletable are skipped and reported. Audit rows: `storage.plan`, `storage.delete` (one per
file, with plan id, rule, bytes), `storage.compact`, `storage.apply`, `storage.policy`, `ingest.retry|cancel`.
A scheduled nightly job may call `StorageManager.plan()` and email the totals; applying stays a human decision.

## Ingest queue

`IngestQueue(store, start_fn, is_running_fn, max_parallel, ...)`, table `ingest_jobs`: queued, running, done, failed,
cancelled; priority; attempts (default 3) with exponential backoff (30 s, 60 s, ...); `running` becomes `queued` on boot;
one active job per camera; nothing starts while the disk monitor says pause; it does not start a camera that is
already running by hand. `NotReady` (camera still learning zones) defers without using an attempt.

### Integration (the only edits to server.py; worker.py needs none)

```python
# imports
from . import api_storage

# after `manager = WorkerManager(...)` and `current_user` / `require` exist (next to the api_map line):
storage = api_storage.build(store, DATA, current_user, require, CONTROLLER, ADMIN, manager, max_parallel=1)
app.include_router(storage.router)

# lifespan(): after media_key()
    storage.queue.start()
# ... and after `yield`, before manager.stop_all:
    storage.queue.stop()
```

`build` supplies the WorkerManager lambdas:

```python
start_fn      = lambda cid: api_storage._start(store, manager, cid)   # manager.start(cid), or start_autozone(...) for a new auto-zone camera; raises NotReady while learning
is_running_fn = lambda cid: manager.running(cid) or manager.learning(cid)
status_fn     = manager.status        # how a finished run ended (done / error / stopped)
stop_fn       = manager.stop
paused_fn     = storage.storage.ingest_paused
```

Upload endpoints. `/api/cameras/upload`: refuse when the disk is at the pause level, and enqueue instead of starting.

```python
# top of upload_camera() and upload_site():
    if storage.storage.ingest_paused():
        raise HTTPException(507, "The server disk is nearly full; uploads are paused until space is freed")
# upload_camera, replace
    if zone_mode == "auto":
        manager.start_autozone(cid, apply=True, start_after=True)
# with
    if zone_mode == "auto":
        storage.queue.enqueue(cid, label=name)
```

(`zone_mode == "manual"` starts nothing today, so it enqueues nothing; enqueue from the zone editor's save if wanted.)
`/api/sites/upload`: `SiteRun` already analyses a site's cameras one after another and runs the batch link at the end,
which the queue cannot do, so leave it; it counts as one worker. Set `max_parallel` accordingly if sites and single
uploads overlap (see CAPACITY.md: more than one parallel analysis does not raise throughput on this CPU).

Nav link in `web/index.html`, after the Map link:

```html
<a href="/storage" data-route="storage"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>Storage</a>
```

## Gitignore

The ONNX export (`models/bd_vehicles_best_<imgsz>.onnx`, 45 MB each) must not be committed. Add to `.gitignore`:

```
models/bd_vehicles_*.onnx
```
