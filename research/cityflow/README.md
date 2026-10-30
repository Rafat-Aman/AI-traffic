# CityFlowV2 notebook: multi-camera tracking and the vehicle ReID weights

`cityflowv2.ipynb` is the team's AI City Challenge notebook (CityFlowV2, track 1: multi-target multi-camera
vehicle tracking). It was run on Kaggle with two T4 GPUs. It does two jobs: it **trains a vehicle
re-identification network**, and it uses that network inside a complete tracking-and-association pipeline to
score itself against the challenge's ground truth.

This project uses one product of it, the trained weights. The notebook's tracker and association step are not
reused: cross-camera matching here is `surveillance/reid.py`, which works in metres on the road plane. The
notebook is kept as the record of where the weights came from, what they scored, and how to make new ones.

## What the notebook does

| Stage | Cells | What happens |
|---|---|---|
| Setup, configuration | 1-5 | installs `ultralytics` and `lap`; every setting lives in cell 4 |
| Data | 6-9 | pulls scenes S02 (4 cameras, ground truth: 145 vehicles, 20,956 boxes) and S06 (6 cameras, the test scene, no ground truth) out of the official CityFlowV2 zip with HTTP range requests; per camera: time offset, region of interest, ground-plane homography |
| ReID model | 10-11 | trains, or loads, the network described below; prints retrieval accuracy on held-out vehicles |
| Single-camera tracking | 12-13 | YOLO11x (COCO car / bus / truck, 1280 px, confidence 0.1, NMS IoU 0.6) with ByteTrack, one camera per GPU; results cached as JSON |
| Tracklets | 14-15 | drops tiny boxes and tracks outside the ROI, merges duplicate ids, takes the 10 clearest crops of each tracklet (up to 384 px) and averages their ReID vectors, stitches fragments of one vehicle, flags parked cars, finds entry and exit zones |
| Cross-camera association | 16-17 | camera-to-camera transit times learned from mutual-best ReID pairs; pair cost = (1 - cosine) + time misfit + ground-position misfit; k-reciprocal links merged cheapest first; optional per-camera mean removal |
| Scoring and tuning | 18-21 | repeats the official IDF1 evaluation; tunes the matching settings on S02; 2-fold estimate; error analysis |
| Review images, demo clip | 22-24 | contact strips of matched vehicles, a 20 s demo video |

## The ReID network

ResNet-50 with IBN-a normalisation, last stage at stride 1, generalised-mean (GeM) pooling and a BatchNorm
"neck"; label-smoothed cross-entropy plus batch-hard triplet loss; 320 px input. Each batch takes 8
identities from the real AI City 2021 ReID set and 8 from the synthetic VehicleX set (4 images each), Adam at
3.5e-4 with one warm-up epoch then cosine decay, 12 epochs of 693 iterations; afterwards every BatchNorm's
statistics are re-estimated on 100 batches of real images only. The 145 vehicles of S02 are left out of
training and used to measure cross-camera retrieval. The embedding at inference is the neck output,
L2-normalised: 2048 numbers.

Training data: 295 real identities (44,378 images) plus 1,362 synthetic ones (192,150 images).

## What it scored

All numbers are printed by the notebook; none were re-run here.

| | Result |
|---|---|
| ReID retrieval on the 145 held-out S02 vehicles (450 queries, 8,339 gallery images, cross-camera) | mAP **0.427**, rank-1 **0.560**; the line the notebook prints beside it for v1 reads 0.490 / 0.656 |
| MTMC on S02, default settings | IDF1 **40.54** (IDP 34.91, IDR 48.34) |
| MTMC on S02, settings tuned on S02's own ground truth | IDF1 **47.84**: optimistic, it was tuned on the scene it is scored on |
| MTMC on S02, 2-fold (tuned on one half of the video, scored on the other) | IDF1 **46.50**: the fairer estimate |
| v1 of the notebook on S02 | IDF1 36.36 |
| Challenge winners on the test scene S06 | 2021: 80.95, 2022: 84.86 |

How to read them honestly:

* The retrieval accuracy is **lower** than v1's, although v2 added synthetic data and a larger input. Whatever
  moved default-settings IDF1 from 36.4 to 40.5 was something else in the pipeline (10 crops per tracklet
  instead of 6, a higher detector NMS IoU, filling gaps in a track). The notebook does not say whether v1 and
  v2 were scored on the same held-out vehicles, so treat the comparison as indicative.
* The gap to the challenge winners is large, and the notebook's own error analysis says most of it is not the
  association step: raw tracks cover only 60.6% of the ground-truth boxes, and *perfect* matching of these
  tracklets would reach IDF1 51.0 (IDP 80.5, IDR 37.3). Of the 40 ground-truth vehicles seen by two or more
  cameras, 30 end up under one global id and 10 are split; of 157 global ids, 77 are one vehicle, 20 are
  wrong merges and 60 match no ground-truth vehicle.
* Everything was measured on American roads, mostly cars. Nothing here says how it does on a CNG.

## How this project uses the weights

| Piece | Where |
|---|---|
| Weights | `models/reid/reid_resnet50_ibn_a_320_syn_exclS02.pth`: 94,388,659 bytes, sha256 `314394d04e3913deba9b43b910a0df7148513be407467d14482fc5530447ddc1`. Git-ignored (`models/reid/*.pth`); copy the file there on a new machine. It arrived as `.zip` but is a PyTorch checkpoint (torch's zip serialisation: `data.pkl`, `data/0` ... `data/349`), so renaming it is all that was needed. |
| Network, loader, preprocessing | `surveillance/reid_backbone.py` |
| Choosing between this and MobileNet, vector labels, camera-bias helper | `surveillance/embed.py` |
| Measuring it on a clip | `tools/reid_eval.py --embedder reid` |
| Tests | `tests/test_embed.py` |

* **Strict loading.** The notebook's own loader used `strict=False`. `reid_backbone.load` uses `strict=True`:
  all 350 tensors must match by name and shape, and none may be NaN or infinite. The checkpoint carries
  every weight, including the batch-norm statistics (`neck.num_batches_tracked` is 100, the 100 re-estimation
  batches), so the ImageNet IBN-a weights are not needed and nothing is downloaded. The saved classifier head
  is not in the file (the notebook drops it) and not needed.
* **The IBN-a channel split was checked, not assumed.** Both halves of an instance/batch-norm layer have the
  same width, so a swapped split would also load strictly. The batch-norm running means were compared with
  the activations that reach each half on real Dhaka crops: across the six layers checked they correlate at
  0.91 to 0.98 with the second half and between -0.24 and 0.24 with the first. So instance norm takes the
  first half of the channels and batch norm the second, as in IBN-Net.
* **Input.** BGR to RGB, squashed (aspect ratio not kept) to 320 x 320 with PIL's bilinear filter, scaled to
  [0, 1], ImageNet mean and std: the notebook's `TEST_TF`, and `tests/test_embed.py` compares the two numerically.
* **One vector per vehicle.** The mean of the per-crop vectors, normalised again, as in the notebook's cell 15.
* **Differences from the notebook, on purpose.** No flip averaging (the notebook adds each crop's mirror
  image): it doubles the CPU time and measured no better here. Six crops of at most 128 px per vehicle,
  because that is what `reid.TrackletCollector` keeps, against the notebook's ten of up to 384 px. Per-camera
  mean removal exists (`embed.CameraBias`) but nothing calls it yet.
* **Provenance.** The checkpoint was trained with S02 held out, so its S02 scores are not contaminated by
  training on the scored vehicles.

### Measured on this project's footage

`tools/reid_eval.py` on 50 s of `data/videos/20261025_092449_side1_1.mp4` (62 vehicles tracked cleanly: 25
cars, 18 motorbikes, 8 CNGs, 8 MPVs, 3 buses; no rickshaw reached the six clean crops the test needs).
Each vehicle's early crops are compared with its late crops and with every other vehicle. Higher is better.

| Appearance cue | Everyone: separation / rank-1 | Same class only: separation / rank-1 |
|---|---|---|
| Colour histogram | 2.37 / 61.3% | 1.79 / 64.5% |
| CIELAB colour | 2.12 / 38.7% | 1.79 / 43.5% |
| MobileNetV3-Small (ImageNet) | 2.52 / 71.0% | 1.76 / 75.8% |
| **ReID network** | **4.38 / 96.8%** | **3.35 / 98.4%** |

Same class only is the question the matcher faces, because it never links a Car to a CNG. By class (same-class
rivals, ReID against MobileNet): Car 3.63 / 100% against 1.41 / 64%; Motorbike 4.60 / 94% against 3.33 / 94%;
MPV 2.84 / 100% against 0.95 / 50%; CNG 2.30 / 100% against 1.98 / 88%. There is no class where the ReID
network loses. With only eight CNGs the CNG row says "no worse", not "better".

What this does **not** show: both halves of a vehicle come from one camera, so it cannot say how the model copes
with a change of viewpoint, light and background, which is the job it was trained for. Cross-camera accuracy on
Dhaka needs two cameras with labelled pairs (SUPERVISION X6).

### Settings and cost

The same 62 vehicles under other settings of the ReID network, each produced by `tools/reid_eval.py` with the
flag shown. Cost is the median time to embed one vehicle of ten crops on the development CPU (6 cores, AVX-512,
PyTorch eager, no GPU), all cores busy and the machine shared with other work, so read it to about 20%.

| ReID setting | Same class: separation / rank-1 | Ten crops |
|---|---|---|
| 320 px, full-size crops (the default) | 3.35 / 98.4% | 2.0 s |
| 320 px, crops of at most 128 px, what `reid.TrackletCollector` keeps (`--max-crop-px 128`) | 3.39 / 96.8% | 1.9 s |
| 256 px, crops of at most 128 px (`--reid-size 256 --max-crop-px 128`) | 3.35 / 96.8% | 1.2 s |
| 224 px, crops of at most 128 px (`--reid-size 224 --max-crop-px 128`) | 3.27 / 96.8% | 0.9 s |
| 320 px with the notebook's mirror averaging (`--reid-flip`) | 3.32 / 98.4% | 3.6 s |
| MobileNetV3-Small, either crop size | 1.76 / 75.8% | 0.012 s |

* The collector's 128 px cap costs the ReID network nothing measurable (one vehicle of 62 in rank-1), and the
  crops are small to begin with: the median long side of the ones the test uses is 83 px.
* Mirror averaging nearly doubles the time and moved the same-class separation by 0.03 and no vehicle, so it
  is off. The notebook uses it.
* 256 and 224 px are 40% and 55% cheaper and score the same or a little lower overall, but the CNG rank-1
  falls from 8 of 8 to 6 of 8 at both (two vehicles). That is noise-sized, and 320 px is the size the weights
  were fitted to, so 320 stays the default; `embed.REID_SIZE` changes it if the CPU budget demands.
* Six crops, which is what the collector keeps, take 1.0 s at 320 px (0.7 s at 256) against 8 ms for MobileNet.
  A forward pass is 25 GFLOPs per crop and this CPU delivers about 135 GFLOPS.

## Retraining

1. Upload `cityflowv2.ipynb` to Kaggle. Accelerator: GPU T4 x2; Internet: on; Run All. Set `QUICK = True` in
   cell 4 first for a short run that proves the notebook works end to end (its scores mean nothing).
2. The notebook downloads the CityFlowV2 scenes, the AI City 2021 ReID set (2.15 GB) and the VehicleX
   simulation set (2.26 GB) from the Google Drive links in cell 4. If Drive refuses ("download quota", cell 7),
   attach the files as a Kaggle dataset or set `REID_DATA` / `SYN_DATA`.
3. The settings that matter are in cell 4: `REID_BACKBONE`, `REID_SIZE` (320), `REID_EPOCHS` (12),
   `REID_P` / `REID_K` (16 x 4), `USE_SYNTHETIC`, `SYN_SHARE`, `BN_ADAPT_BATCHES` (100), `SCENES`.
4. The weights are written as `reid_<backbone>_<size>[_syn]_excl<scenes>.pth` in `/kaggle/working`. The
   notebook's header estimates the whole run at about 1.5 h; the log of the run that made these weights shows
   about 1,010 s per ReID epoch, so 12 epochs took about 3.4 h on its own.
5. Copy the `.pth` into `models/reid/`. `surveillance/embed.py` uses `reid_resnet50_ibn_a_320_syn_exclS02.pth`
   when present, otherwise the last `*.pth` in that folder; the input size is read from the checkpoint.
   Before relying on a retrained model, run `./.venv/bin/python tools/reid_eval.py <clip> --seconds 50
   --embedder reid` and compare with the table above.
6. Vectors from different weights must not be compared. The loader labels each vector with
   `Embedder.tag` (it contains the file name) so the store can keep that label beside the vector.

Watch for one trap in the notebook. If the IBN-Net code cannot be fetched, `load_backbone` prints "IBN-Net
unavailable ... using torchvision resnet50" and trains a plain ResNet-50, but the checkpoint is still labelled
`resnet50_ibn_a`. The notebook's loader would accept it, because it loads with `strict=False`: the IBN
normalisation layers, whose keys the plain network does not have, are simply left at their starting values.
`reid_backbone.load` refuses such a file, because its keys do not match.

## Housekeeping

* The notebook is 19 MB, nearly all of it two outputs of cell 23 (a 15.7 MB embedded demo video and a 3.5 MB
  image). Its code and text are about 117 KB. Strip the outputs before committing it:

  ```python
  import json
  nb = json.load(open("cityflowv2.ipynb"))
  for c in nb["cells"]:
      if c["cell_type"] == "code":
          c["outputs"], c["execution_count"] = [], None
  json.dump(nb, open("cityflowv2.ipynb", "w"), indent=1)
  ```

  The outputs hold the only copy of the training log and the scores quoted above, which is why they are
  summarised here first.
* `research/` is not imported by anything; deleting it loses the record but breaks nothing.
