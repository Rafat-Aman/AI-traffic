# Capacity: how much video can one machine process? (D3, R-5)

Machine: i5-11400H, 6 cores / 12 threads, CPU only, 23 GB RAM. Clip: `data/videos/20261025_092449_side1_1.mp4`
(1242x1440, 29.77 fps, 121 s). Cost measured = `Detector.track()` per frame: YOLO inference (letterbox, NMS) plus
ByteTrack, on consecutive full-resolution frames from frame 600, 5 warm-up frames discarded.

## Read this first: the measurements are noisy

The machine was shared with other jobs (a `llama-server` on about 4 cores and another Python job). The 1-minute load
average never dropped below 4 in the 30-minute poll (from 16.0 at 08:47 it fell to 6.9 at 09:18), so the timings were taken
anyway, and each later run started with an even higher load because it included my own previous runs (17 to 31 at start).
Load averages at the start of each run are in the table. Treat every number as a **floor for an idle machine**, with an
unknown error that is probably 1.3-2x. The single-process 12-thread numbers are the worst affected (a 12-thread
OpenMP team on contended cores loses far more than a 6-thread one), which is why one torch worker looks slower than
two. Repeat on an idle machine before quoting: the commands are at the end.

## Torch vs ONNX Runtime, imgsz 960 (what uploaded files use), one process, 12 threads

| Backend | frames | mean s/frame | median | fps | load avg at start |
|---|---|---|---|---|---|
| PyTorch (CPU) | 120 | 0.917 | 0.844 | 1.09 | 6.93 (poll expired) |
| ONNX Runtime 1.23 (CPU provider) | 120 | 0.633 | 0.629 | 1.58 | 17.7 (my own preceding run included) |

At imgsz 640 (live default), one process: torch 0.463 s (median 0.402), ONNX 0.379 s (median 0.390), load 22-25 at start.

Under identical thread counts but with several workers (below), ONNX was *slower* than torch (0.84 vs 0.63 s at 2 workers
x 6 threads). So the single-worker ONNX advantage above is not established: it is confounded by when each run happened. The
honest summary: **no reliable speed difference between the two backends was measured**; ONNX is not shown to be faster.

## Box agreement, torch vs ONNX (same 120 frames, conf 0.1, greedy IoU matching)

| IoU threshold | torch boxes | ONNX boxes | matched | matched / torch | matched / ONNX | same class | mean IoU of matches |
|---|---|---|---|---|---|---|---|
| 0.5 and 0.7 | 1125 | 1098 | 1059 | 94.1 % | 96.5 % | 99.3 % | 0.983 |
| 0.9 | 1125 | 1098 | 1025 | 91.1 % | 93.4 % | 99.5 % | 0.987 |

There are no labels, so this is agreement, not accuracy. Matched boxes are practically identical; about 6 % of torch's
boxes (probably the low-confidence ones near the 0.1 threshold) have no counterpart. The CoreML path was documented at 99.6 %
at IoU 0.7. 94 % is not that, so **the default backend is not switched**: `backend="auto"` still means PyTorch on CPU/CUDA;
`backend="onnx"` is opt-in (export once to `models/bd_vehicles_best_<imgsz>.onnx`, gitignored by the line in `STORAGE.md`).

## Parallel workers (imgsz 960; each worker pinned to 12/w logical CPUs, torch and ORT threads set to the same count)

All workers run at the same time on the same clip, 40 frames each; throughput is the sum of 1/mean over workers.

| Workers | threads each | torch s/frame (each) | torch total fps | ONNX s/frame (each) | ONNX total fps | load at start (torch / ONNX) |
|---|---|---|---|---|---|---|
| 1 | 12 | 0.92 | 1.09 | 0.63 | 1.58 | 6.9 / 17.7 |
| 2 | 6 | 0.63 | 3.16 | 0.83 | 2.42 | 20.0 / 18.9 |
| 3 | 4 | 0.83 | 3.63 | 0.96 | 3.14 | 19.8 / 22.0 |
| 4 | 3 | 1.03 | 3.88 | 1.22 | 3.26 | 24.1 / 30.9 |

Throughput saturates at about 3-4 fps for the whole machine whichever way the cores are split (the 1-worker row is the
contended outlier explained above). More workers give almost nothing past two; they only add memory. `max_parallel=2` is
the sensible setting for this class of machine.

## What that means in video time

The pipeline analyses `analysis_fps` frames per video second (default 30, capped by the file's 29.77, so every frame; a
setting of 10 analyses every third frame). Video-hours per day per machine = total fps / analysis_fps x 24. These use the
best measured total (torch, 4 workers, 3.88 fps) and are detector+tracker only; the full pipeline (plates, helmet, ReID
vectors, clip encoding, impact) costs more, so subtract perhaps a quarter (the end-to-end figure in `REPORT_RESULTS.md`,
0.57 fps at load 19, agrees in order of magnitude).

| analysis_fps | video-min per wall-min | video-hours / day (measured floor) | with 1.5-2x for an idle machine |
|---|---|---|---|
| 30 (default, every frame) | 0.13 | about 3 | 5-6 |
| 10 (every 3rd frame) | 0.39 | about 9 | 14-18 |
| 5 (every 6th frame) | 0.78 | about 19 | 28-37 |

Lower analysis rates cost accuracy for speed and for tracking at 5 fps; that trade-off is not measured here (the evaluation
in `docs/EVALUATION.md` used about 10 fps).

## Hundreds of videos a day

Assume 300 videos a day; the length is the unknown, so two cases:

| Case | video per day | machines like this (floor / idle estimate) at 30 fps | at 10 fps |
|---|---|---|---|
| 300 x 2 min | 10 h | 3.3 / 2 | 1.1 / 1 |
| 300 x 10 min | 50 h | 16 / 9 | 5.5 / 3 |
| 300 x 30 min | 150 h | 50 / 28 | 17 / 10 |

So: a few hundred **short** clips a day (or sampling at 10 fps) fit on one to three machines of this class; hundreds of
**10-minute** videos at full frame rate need roughly ten of them, or a GPU. Estimate for a GPU, not measured here (no GPU on
this machine): this 11 M-parameter YOLO at 960 px runs in the order of 50-150 frames/s on a mid-range CUDA card
(`Detector` already uses CUDA with half precision when present), i.e. one card replaces about 15-40 of these CPUs for the
detector, with tracking and the plate/ReID stages on CPU becoming the next limit. Practical paths, in order of cost:
(1) set `analysis_fps` to 10 for uploaded batches; (2) screen with a cheap motion/low-resolution pass and analyse only
segments with traffic; (3) one GPU server; (4) several CPU workers behind the ingest queue (`ingest.py` is already a
database-backed queue, so workers on other machines need a shared database first: the PostgreSQL step, P1).

Storage side: 50 h of 1080p video at about 0.3 GB per 10 min is about 90 GB a day raw; with the 7-day raw retention that is
about 630 GB steady state, before evidence (a few MB per case). This is why raw video retention is the first policy lever.

## Reproduce

```
.venv/bin/python - <<'PY'   # per run: pin CPUs with os.sched_setaffinity, torch.set_num_threads(k), ORT intra_op_num_threads=k
from surveillance.detector import Detector
d = Detector("bd_vehicles_best.pt", 960, "onnx")          # or "auto" for PyTorch; exports once to models/
PY
```
Check `uptime` first; time only when the 1-minute load average is below 4.
