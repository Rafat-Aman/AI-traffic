# Models

| File | Task | Used by |
|---|---|---|
| `bd_vehicles_best_*.mlpackage` | Vehicle detection (CoreML export of `../bd_vehicles_best.pt`, created on first run) | everything |
| `helmet_cls.pt` / `.mlpackage` | Is anyone on this motorbike bare-headed? (image classifier) | V6 |
| `plate_detector.onnx` | Number-plate detection | plate reading |
| `bd_plate_chars.pt` | Bangla plate character detection (digits, city, "metro", class letter) | plate reading |

All numbers below were measured on data **not used for training or threshold tuning**, unless stated otherwise.

---

## Helmet classifier (`helmet_cls`)

**Model:** YOLO11s-cls (Ultralytics, ImageNet-pretrained), 224 px, fine-tuned for 14 epochs on the Apple M5 GPU.
It sees one crop per motorbike (riders included) and outputs P(someone on it is bare-headed).

**Training data:**

| Source | Images | Licence |
|---|---|---|
| [revitotan/helmet-threerider-vlm-instruct-dataset](https://huggingface.co/datasets/revitotan/helmet-threerider-vlm-instruct-dataset): crops from elevated traffic-enforcement cameras (Indonesia), with counts of riders with / without a helmet | 1,372 helmeted, 227 bare-headed | MIT |
| This project's footage (`TC_00006.MOV`, Kakoli, Banani, Dhaka): one crop per tracked motorbike, labelled H (everyone wears a helmet) / N (someone is bare-headed) / X (cannot tell, not used) | 157 H, 11 N (weighted ×3 and ×8) | own data |

The Dhaka labels were made **by eye from the crops by the AI assistant that built this system**, conservatively
(anything uncertain is X). They should be spot-checked by the team; see `training/helmet_labels.txt`.

**What did not work (kept for the report):**

| Approach | Result on Dhaka footage |
|---|---|
| Head detector trained on [DeBoniz/helmet-detection-dataset](https://huggingface.co/datasets/DeBoniz/helmet-detection-dataset) (18k images, mostly eye-level close-ups) | found no head on 43 of 73 riders, and drew whole-crop "helmet" boxes |
| Classifier trained on the Indonesian crops only | 100% precision on its *own* test set, but **flagged 82% of helmeted Dhaka riders** (camera angle, distance and blur differ) |

**Result (Dhaka test: 166 labelled motorbikes from 122-242 s of `TC_00006.MOV` - the stretch that is
`DataSet_2min.mp4` - never used for training; the training crops come from before 109 s and after 255 s):**

| P(bare) threshold | bare-headed caught | helmeted wrongly flagged |
|---|---|---|
| 0.5 | 4 / 11 | 4 / 155 |
| 0.75 | 4 / 11 | 1 / 155 |
| **0.9 (used)** | **3 / 11** | **0 / 155** |

**End-to-end (the V6 rule on video, decided once per motorbike when it leaves the view: ≥ 3 sightings at
P ≥ 0.8 and at least half of all confident sightings):**

| Clip | Motorbikes judged | Cases | Correct |
|---|---|---|---|
| `DataSet_2min.mp4` (held-out test stretch, 11 labelled bare-headed riders) | 237 | **0** | no false alarms, **no bare-headed rider caught** |
| `TC_00006.MOV` 30-110 s (overlaps the training stretch: not independent) | 111 | 3 | 2 clearly bare-headed, 1 plausible |

Two earlier versions of the rule were rejected on this footage: deciding at the first 3 bare sightings flagged a
parked rider and a column of bikes; a lower threshold flagged a cycle-van. The rule now ignores parked bikes,
bikes right behind another bike, and vehicles the detector is not sure are motorbikes.

Per single crop the classifier is precise but misses most bare heads, which is why the rule votes over many
sightings of each moving motorbike (4 per second). With only 11
bare-headed test riders, the recall figure is rough (±15%). Same camera, different riders: accuracy at another
location must be re-measured. **More labelled Dhaka motorbikes (especially bare-headed and pillion riders) are
the single most effective improvement.**

---

## Number plates

### Plate detector (`plate_detector.onnx`)
[morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
`license-plate-finetune-v1s.onnx`, licence **AGPL-3.0** (same as Ultralytics). ONNX format: cannot execute code
when loaded. Finds Bangladeshi plates in the Banani Kakoli footage without fine-tuning.

### Bangla character detector (`bd_plate_chars.pt`)
**Model:** YOLO11n, 320 px, 45 epochs. Classes: digits ০-৯, ঢাকা / খুলনা / যশোর, মেট্রো, class letters (ক গ চ ট ল হ ব ন).

**Data:** [NightLPBD](https://huggingface.co/datasets/thasan3003/NightLPBD_Nighttime_License_Plates_of_Bangladesh)
(CC BY 4.0) - 963 / 216 / 189 train / val / test images with every character labelled. Each training plate is
used once sharp and three times degraded to what a traffic camera delivers (60-400 px wide, motion blur, JPEG
compression, day/night brightness). Please cite:

> Md. Tahmid Hasan and Sk. Md. Masudul Ahsan, "NightLPBD: A YOLO-Based Single-Stage Framework for Nighttime
> License Plate Recognition", IEEE COMPAS 2025, doi:10.1109/COMPAS67506.2025.11381609.

Validation: mAP50 0.968, mAP50-95 0.72.

**Reading rule** (`surveillance/anpr.py`): each violating vehicle's best crops are read; a number is reported only
if **3 crops give exactly the same 6 digits, every digit at confidence ≥ 0.7, on plates at least 110 px wide**.
City and class letter are added only when the same crops agree on them confidently. Thresholds were tuned on the
validation plates; the 110 px minimum was raised after the test run below showed 2-4% wrong numbers at 70-100 px.

**Result on the 167 held-out test plates** (each seen in 4 frames with random blur / compression, as in a video):

| Plate width | Read | **Wrong number** | Wrong city / letter |
|---|---|---|---|
| ≤ 100 px | 0% (not attempted) | - | - |
| 120 px | 62% | **0%** | 2.9% |
| 150 px | 60% | **0%** | 4.1% |
| 200 px | 75% | **0%** | 6.6% |

For comparison, EasyOCR's general Bengali model got the 6-digit number right on 0-11% of plates at 50-100 px and
25% at 220 px.

**Limits:**
* The training plates contain only Dhaka, Khulna and Jashore. Plates from other districts will get a wrong city:
  the dashboard shows city and class letter as "check on the image"; the 6-digit number is the reliable part.
* Training images are night-time; daytime plates were only simulated (brightness / contrast augmentation).
* The Banani Kakoli camera sees plates at 35-100 px, blurred - **too small to read**, and the system correctly
  reports every plate there as "unreadable". Readable plates need a camera closer to the lane or zoomed in:
  plates should be ≥ 120 px wide in the video (e.g. a 4K camera covering 1-2 lanes).

---

## Reproducing

The scripts that built the datasets, trained the models and produced every number above are in `../training/`.
