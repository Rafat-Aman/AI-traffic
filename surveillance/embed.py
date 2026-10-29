"""An appearance vector per vehicle, for recognising it on another camera and for showing a reviewer why.

Two models behind one interface
    "reid"       the team's ResNet50-IBN-a, trained for vehicle re-identification on CityFlowV2 plus synthetic
                 VehicleX data (`surveillance/reid_backbone.py`, weights in `models/reid/`). 2048 numbers.
    "mobilenet"  MobileNetV3-Small with ImageNet weights, 2.5M parameters. 576 numbers. It is **not** a
                 vehicle re-identification model - it was never trained to tell one car from another - and
                 nothing here pretends it was. It is the fallback, and what ran before the team's model
                 arrived.
    Either way a vehicle's clearest crops are reduced to one L2-normalised float32 vector, and the matcher
    scores it by a log-likelihood ratio fitted from the traffic itself (`reid.LLR`): a cue that fails to
    separate matched pairs from rejected ones is switched off automatically, so a model that carries no
    signal on some class costs nothing but CPU.

Measured on this project's footage
    `tools/reid_eval.py`, 50 s of `data/videos/20261025_092449_side1_1.mp4`: 62 vehicles (25 Car, 18
    Motorbike, 8 CNG, 8 MPV, 3 Bus), each one's early crops against its late crops and against every other
    vehicle. "Same class" lets only vehicles of the same class count as rivals, which is what the matcher
    faces, since it never links across classes. Separation above ~1 is useful; rank-1 is how often the other
    half is the nearest neighbour.

                          everyone                same class only
                          separation   rank-1     separation   rank-1
        colour histogram      2.37      61.3%         1.79       64.5%
        CIELAB colour         2.12      38.7%         1.79       43.5%
        mobilenet             2.52      71.0%         1.76       75.8%
        reid                  4.38      96.8%         3.35       98.4%

    The ReID model is ahead in every class (same-class separation / rank-1, reid against mobilenet): Car
    3.63 / 100% against 1.41 / 64%, Motorbike 4.60 / 94% against 3.33 / 94%, MPV 2.84 / 100% against 0.95 /
    50%, CNG 2.30 / 100% against 1.98 / 88%. Eight CNGs or MPVs are few: for CNGs read "no worse", not
    "better". No rickshaw was tracked cleanly enough to be tested. Shrunk to the 128 px the tracklet
    collector keeps, ReID scores 4.42 / 95.2% and 3.39 / 96.8% - the same within one vehicle - and at 256
    or 224 px instead of 320 it is also unchanged (3.35 and 3.27 same-class, rank-1 96.8%); mirror averaging
    (`REID_FLIP`) changed nothing either (3.32 same-class).

    What this cannot show: both halves of a vehicle come from one camera. How the model copes with another
    camera's viewpoint, light and background - what it was trained for, and where the team measured mAP 0.427
    and rank-1 0.560 on held-out American vehicles - is still unmeasured on Dhaka; that needs two cameras and
    labelled pairs. Treat the table as "it can tell these vehicles apart", not as the size of the gain across
    cameras.

Choosing the model
    `EMBED_MODEL=reid|mobilenet|auto` in the environment, or `Embedder.use(...)` in code. "auto" means the
    ReID model when its weights are in `models/reid/` and load, MobileNet otherwise. A model that cannot be
    loaded (weights missing or not matching the architecture key for key, torch missing) is reported once
    and the next one is used, so `encode` still answers. `Embedder.name`, `.dim` and `.tag` describe the
    model `encode` is using; read them after the first `encode`, because that is when a failed load falls
    back.

    The default (`DEFAULT_POLICY`) is still "mobilenet", although the ReID model is the better cue. That is
    about cost and about what gets stored, not accuracy. `reid.TrackletCollector` calls `encode` on the
    pipeline thread when a vehicle leaves the view, and the ReID model takes about a second for six crops
    (below) against 8 ms: on the 50 s test clip, 62 vehicles would add about a minute to the two the
    detector needs, in short stalls. It also writes 2048-number vectors and needs the store to say which
    model made each one. Whoever owns those should arrange both, then set `DEFAULT_POLICY = "auto"`;
    until then `EMBED_MODEL=auto` turns it on for a deployment without touching code.

Vectors from different models must never be compared
    A MobileNet vector and a ReID vector are unrelated number systems, and tracklets already in the store
    hold 576-number MobileNet vectors. `similarity` therefore returns NaN - "no evidence", which the matcher
    treats as a missing cue - when the shapes differ, and also when both vectors are labelled with a model
    `tag` and the tags differ. The shape test alone would not catch two models of the same width, such as
    the ReID network retrained and its weights replaced, which is why the `tag` carries the weights file's
    name and why the store should keep it next to each vector.

Cost
    Once per vehicle, when it has left the view, on its clearest crops (`reid.CLEAR_FRAMES`, six).
    MobileNet: 12 ms for ten crops. ReID, all six cores busy: 1.0 s for six crops and 1.9 s for ten at its
    trained 320 px, about 150 times as much; mirror averaging doubles that; 256 px takes 0.7 s for six. A
    forward pass is 25 GFLOPs per crop, and this CPU delivers about 135 GFLOPS. Concurrent calls are
    serialised, because two at once would only fight over the cores.
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS = ("reid", "mobilenet")
POLICIES = ("auto",) + MODELS
MOBILENET_TAG = "mobilenet:v3-small-imagenet1k"     # what every vector stored before the ReID model existed is
DEFAULT_POLICY = "mobilenet"        # what "nothing chosen" means; the reason is in the measurements above
EMBED_DIM = 576                     # MobileNet's width, kept for callers that imported it; use Embedder.dim
CROP_SIZE = 128                     # MobileNet's input; the crops kept for colour are already about this size
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "models" / "reid"
WEIGHTS_FILE = "reid_resnet50_ibn_a_320_syn_exclS02.pth"
REID_FLIP = False                   # the notebook adds each crop's mirror image; doubles the CPU time and
                                    # bought nothing measurable on Dhaka footage (see the table above)
REID_SIZE = None                    # None: the 320 px the network was trained at; 256 is ~40% faster


def weights_path():
    """The ReID checkpoint to use: the team's file when present, otherwise the last `*.pth` in
    `models/reid/` (a retrained model dropped in beside it). None when there is none."""
    given = WEIGHTS_DIR / WEIGHTS_FILE
    if given.is_file():
        return given
    found = sorted(WEIGHTS_DIR.glob("*.pth")) if WEIGHTS_DIR.is_dir() else []
    return found[-1] if found else None


# ---------------------------------------------------------------- the two models

class _MobileNet:
    name, dim, tag = "mobilenet", 576, MOBILENET_TAG

    def __init__(self):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

    def features(self, crops):
        """(N, 576): one L2-normalised row per crop."""
        import torch
        batch = []
        for c in crops:
            img = cv2.resize(c, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(np.transpose((img - MEAN) / STD, (2, 0, 1)))
        with torch.inference_mode():
            f = self.model.features(torch.from_numpy(np.stack(batch)))
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            return torch.nn.functional.normalize(f, dim=1).numpy()


class _ReID:
    name = "reid"

    def __init__(self):
        path = weights_path()
        if path is None:
            raise FileNotFoundError(f"no ReID weights in {WEIGHTS_DIR} (expected {WEIGHTS_FILE})")
        from . import reid_backbone
        self._backbone = reid_backbone
        self.net = reid_backbone.load(path)
        self.dim = reid_backbone.EMBED_DIM
        self.tag = f"reid:{path.stem}"
        self._one_at_a_time = threading.Lock()

    def features(self, crops):
        """(N, 2048): one L2-normalised row per crop."""
        with self._one_at_a_time:           # a forward pass already spreads over every core; two at once
            return self._backbone.embed(self.net, crops, flip=REID_FLIP, size=REID_SIZE)   # fight for them


_BUILD = {"reid": _ReID, "mobilenet": _MobileNet}


class _Meta(type):
    """Lets `Embedder.name`, `.dim` and `.tag` be read on the class, as `Embedder.encode` is called."""

    @property
    def name(cls):
        b = cls._backend()
        return b.name if b else None

    @property
    def dim(cls):
        b = cls._backend()
        return b.dim if b else 0

    @property
    def tag(cls):
        b = cls._backend()
        return b.tag if b else None


class Embedder(metaclass=_Meta):
    """Loads lazily and at most once per process: importing torch costs seconds and most runs never need it."""

    _lock = threading.RLock()
    _backends = {}          # model name -> loaded backend
    _failed = set()         # model names that could not be loaded; never retried
    _choice = None          # chosen with use(); None means the environment, then DEFAULT_POLICY
    _active = None

    @classmethod
    def use(cls, model=None):
        """Choose the model for this process: "reid", "mobilenet", "auto", or None to go back to the
        environment / default. Takes effect for the next `encode`."""
        if model is not None and model not in POLICIES:
            raise ValueError(f"unknown appearance model {model!r}; choose from {', '.join(POLICIES)}")
        with cls._lock:
            cls._choice, cls._active = model, None

    @classmethod
    def reset(cls):
        """Forget what was loaded and what failed (tests, or after the weights were replaced)."""
        with cls._lock:
            cls._backends, cls._failed, cls._choice, cls._active = {}, set(), None, None

    @classmethod
    def _order(cls):
        """Models to try, best first, for the current choice."""
        policy = cls._choice or os.environ.get("EMBED_MODEL", "").strip().lower() or DEFAULT_POLICY
        if policy not in POLICIES:
            policy = DEFAULT_POLICY
        if policy == "mobilenet" or (policy == "auto" and weights_path() is None):
            return ["mobilenet"]
        return ["reid", "mobilenet"]

    @classmethod
    def _backend(cls):
        if cls._active is not None:
            return cls._active
        with cls._lock:
            for name in cls._order():
                if name in cls._failed:
                    continue
                if name not in cls._backends:
                    try:
                        cls._backends[name] = _BUILD[name]()
                    except Exception as e:      # no weights / no network / torch missing / wrong checkpoint
                        later = "colour only" if name == "mobilenet" else "the next model"
                        print(f"appearance model {name!r} unavailable ({e}); matching will use {later}")
                        cls._failed.add(name)
                        continue
                cls._active = cls._backends[name]
                return cls._active
        return None

    @classmethod
    def available(cls):
        """False only once every model that could be used has failed to load."""
        return not set(cls._order()) <= cls._failed

    @classmethod
    def encode(cls, crops):
        """One L2-normalised vector for a vehicle, averaged over its crops. None when unavailable."""
        backend = cls._backend()
        if backend is None or not crops:
            return None
        try:
            good = [c for c in crops if c is not None and c.size and min(c.shape[:2]) >= 8]
            if not good:
                return None
            v = backend.features(good).mean(0)      # each crop was normalised first, so none can dominate
            return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
        except Exception as e:
            print(f"embedding failed: {e}")
            return None


def similarity(a, b, *, model_a=None, model_b=None):
    """Cosine similarity of two vehicle vectors; NaN when either is missing or they are not comparable.

    Not comparable means different shapes (a 576-number MobileNet vector against a 2048-number ReID one) or,
    when the caller passes the `Embedder.tag` each was made with, different tags."""
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    if model_a is not None and model_b is not None and model_a != model_b:
        return float("nan")
    return float(np.dot(a, b))


# ---------------------------------------------------------------- camera bias

class CameraBias:
    """Takes each camera's mean appearance vector out before vehicles are compared across cameras.

    The team's notebook (`research/cityflow/`, cell 17, `CAM_DEBIAS`) found that a camera's background and
    viewpoint add a component shared by every vehicle it sees, which makes vehicles look more alike within
    one camera than across cameras. Subtracting the camera's mean vector and normalising again removes it;
    their tuner switched it on for scene S02. That is the whole method: `v - mean(camera)`, then unit length.
    This class keeps the means as running sums so it works while cameras are still delivering vehicles.

    Below `min_count` vectors a camera's mean is mostly the vectors themselves - with two, subtracting it
    makes the pair exact opposites - so `centre` hands the vector back unchanged until there are enough;
    `min_count=2` reproduces the notebook, which had 115-175 tracklets per camera. Means are kept per
    (camera, vector width), so vectors from different models never mix; use one `CameraBias` per model
    when two models share a width.

    Using it from `reid.py`: call `update(camera_id, t.embed)` for every tracklet in the window, then
    compare `centre(camera_id, t.embed)` instead of `t.embed`. Whether it helps has to be measured with
    two or more real cameras; `tools/reid_eval.py` sees only one, where it cannot show a benefit.
    """

    def __init__(self, min_count=20):
        self.min_count = min_count
        self._sum, self._n = {}, {}
        self._lock = threading.Lock()

    @staticmethod
    def _usable(vec):
        return vec is not None and getattr(vec, "ndim", 0) == 1 and bool(np.all(np.isfinite(vec)))

    def update(self, camera_id, vec):
        """Add one vehicle's vector to its camera's mean. Returns False when the vector was ignored."""
        if not self._usable(vec):
            return False
        key = (camera_id, vec.shape[0])
        with self._lock:
            self._sum[key] = self._sum.get(key, 0.0) + vec.astype(np.float64)
            self._n[key] = self._n.get(key, 0) + 1
        return True

    def count(self, camera_id, dim):
        return self._n.get((camera_id, dim), 0)

    def mean(self, camera_id, dim):
        """The camera's mean vector, or None while it has fewer than `min_count` vectors of this width."""
        n = self._n.get((camera_id, dim), 0)
        if n < max(self.min_count, 1):
            return None
        return (self._sum[(camera_id, dim)] / n).astype(np.float32)

    def centre(self, camera_id, vec):
        """The vector with its camera's mean removed and unit length restored; unchanged when the camera
        has too few vectors yet, or when the result would be degenerate."""
        if not self._usable(vec):
            return vec
        m = self.mean(camera_id, vec.shape[0])
        if m is None:
            return vec
        v = vec - m
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return vec
        return (v / n).astype(np.float32)
