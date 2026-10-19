# Training and evaluation scripts

Everything needed to rebuild the models in `../models/` and re-measure the numbers in `../models/README.md`.
Run with the project environment: `../.venv/bin/python <script> ...` (arguments are listed at the top of each script).

| Step | Script |
|---|---|
| **Plates** | |
| Plate-crop dataset from NightLPBD (sharp + camera-like degraded copies) | `build_ocr_ds.py <NightLPBD v1.0> <out>` |
| Train the Bangla character detector | `train_chars.py <out>/plate_chars.yaml <runs>` |
| Single-frame accuracy by plate width (also EasyOCR baseline) | `eval_ocr.py <out> yolo|easyocr [model]` |
| Tune acceptance thresholds on validation plates | `grid_anpr.py <out> <model> val` |
| Multi-frame accuracy on the held-out test plates | `calibrate_anpr.py <out> <model>` |
| **Helmets** | |
| One crop per tracked motorbike from a video, tagged train/test by time | `extract_motorbikes.py <out>/crops` |
| Numbered sheets of 24 crops for labelling | `label_sheets.py <folder>` |
| Labels: `idx H|N|X` (H all helmeted, N someone bare-headed, X cannot tell) | `helmet_labels.txt` (+ `helmet_crops_meta.json`) |
| Training set (Indonesian crops + Dhaka crops) | `build_helmet_data.py <threerider parquet> <folder> <out>` |
| Train the classifier | `train_helmet.py <out> <runs>` |
| Accuracy on the labelled Dhaka test motorbikes | `eval_helmet_dhaka.py <folder> <model.pt>` |

| **Accuracy of the whole system (test manual, Level 5)** | |
| List every true violation while watching a clip (copy `ground_truth_template.csv`) | `ground_truth_template.csv` |
| Score the system's cases (dashboard Export CSV or the runner's `violations.csv`) per rule | `score_accuracy.py ground_truth.csv violations.csv` |

**Adding Dhaka data (recommended):** record more footage (ideally other locations and times), run
`extract_motorbikes.py` and `label_sheets.py`, add lines to `helmet_labels.txt`, rebuild and retrain. Keep a
time- or location-separated test set so the measured accuracy stays honest.
