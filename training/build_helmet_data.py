"""Helmet classifier training set: Indonesian enforcement-camera crops + labelled Dhaka motorbike crops.

    python build_helmet_data.py <threerider train.parquet> <dhaka folder with crops/, meta.json, labels.txt> <out>

threerider = revitotan/helmet-threerider-vlm-instruct-dataset (MIT). no_helmet = someone on the bike is bare-headed.
The Dhaka test motorbikes (split "test" in meta.json) are never used here."""
import sys, ast, json, random, shutil, hashlib
from pathlib import Path
import pyarrow.parquet as pq

parquet, D, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
shutil.rmtree(OUT, ignore_errors=True)
random.seed(7)


def put(split, cls, data_or_path, name):
    d = OUT / split / cls
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(data_or_path, bytes):
        (d / name).write_bytes(data_or_path)
    else:
        shutil.copy(data_or_path, d / name)


t = pq.read_table(parquet).to_pydict()
seen = set()
rows = []
for i, (o, imgs) in enumerate(zip(t["output"], t["image"])):
    lab = ast.literal_eval(o)
    b = (imgs[0] if isinstance(imgs, list) else imgs)["bytes"]
    h = hashlib.md5(b).hexdigest()
    if h in seen or lab["person-with-helmet"] + lab["person-without-helmet"] == 0:
        continue
    seen.add(h)
    rows.append(("no_helmet" if lab["person-without-helmet"] > 0 else "helmet", b, i))
for cls in ("helmet", "no_helmet"):
    rs = [r for r in rows if r[0] == cls]
    random.shuffle(rs)
    keep = rs[int(0.27 * len(rs)):]  # the same 73% used for training here (15% test + 12% val held out)
    for _, b, i in keep:
        put("train", cls, b, f"etle_{i}.jpg")

meta = {m["idx"]: m for m in json.load(open(D / "meta.json"))}
labels = [(int(a), b) for a, b in (l.split() for l in open(D / "labels.txt") if l.strip() and not l.startswith("#"))]
train = sorted([(meta[i], y) for i, y in labels if meta[i]["split"] == "train" and y in "HN"], key=lambda x: x[0]["t"])
cut = int(0.8 * len(train))
for m, y in train[:cut]:  # Dhaka weighted: helmeted x3, bare-headed x8
    cls = "helmet" if y == "H" else "no_helmet"
    for k in range(3 if y == "H" else 8):
        put("train", cls, D / "crops" / m["file"], f"dk{k}_{m['file']}")
for m, y in train[cut:]:  # last 20% (by time) of the Dhaka training labels: checkpoint selection
    put("val", "helmet" if y == "H" else "no_helmet", D / "crops" / m["file"], m["file"])
for s in ("train", "val"):
    print(s, {c.name: len(list(c.glob("*.jpg"))) for c in (OUT / s).iterdir()})
