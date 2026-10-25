"""An appearance vector per vehicle, for recognising it on another camera and for showing a reviewer why.

What this is, and what it is not
    A small ImageNet-trained network (MobileNetV3-Small, 2.5M parameters) reduces a vehicle's clearest crops
    to one 576-number vector. It is **not** a vehicle re-identification model: it was never trained to tell
    one car from another car, and nothing here pretends it was. Proper ReID weights would be better and are
    a straightforward upgrade - the interface below would not change - but they want a GPU to train and a
    labelled vehicle corpus to train on, and this project has neither.

    It earns its place anyway, because of how it is used. The matcher scores every cue by a log-likelihood
    ratio fitted from the traffic itself (`reid.LLR`), and a cue that fails to separate matched pairs from
    rejected ones is switched off automatically. So the embedding can only help: where it carries signal it
    is used, and where it does not - a fleet of identical green CNGs is the obvious case - it contributes
    nothing rather than noise. `tools/reid_eval.py` measures which of those two it is on real footage.

Cost
    It runs once per vehicle, when the vehicle has left the view, on at most six small crops: about 90 ms on
    the development CPU. It is never on the per-frame path.
"""

import threading

import cv2
import numpy as np

EMBED_DIM = 576
CROP_SIZE = 128          # the crops kept for colour are already small; this is what the network wants
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Embedder:
    """Loads lazily and at most once per process: importing torch costs seconds and most runs never need it."""

    _lock = threading.Lock()
    _model = None
    _failed = False

    @classmethod
    def available(cls):
        return not cls._failed

    @classmethod
    def _load(cls):
        if cls._model is not None or cls._failed:
            return cls._model
        with cls._lock:
            if cls._model is not None or cls._failed:
                return cls._model
            try:
                import torch
                from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
                torch.set_grad_enabled(False)
                m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
                cls._model = m
            except Exception as e:      # no weights cached and no network, or torch missing
                print(f"appearance embedding unavailable ({e}); matching will use colour only")
                cls._failed = True
        return cls._model

    @classmethod
    def encode(cls, crops):
        """One L2-normalised vector for a vehicle, averaged over its crops. None when unavailable."""
        model = cls._load()
        if model is None or not crops:
            return None
        try:
            import torch
            batch = []
            for c in crops:
                if c is None or c.size == 0 or min(c.shape[:2]) < 8:
                    continue
                img = cv2.resize(c, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                batch.append(np.transpose((img - MEAN) / STD, (2, 0, 1)))
            if not batch:
                return None
            x = torch.from_numpy(np.stack(batch))
            f = model.features(x)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            f = torch.nn.functional.normalize(f, dim=1)         # per crop, so a bright crop cannot dominate
            v = f.mean(0)
            v = v / (v.norm() + 1e-9)
            return v.numpy().astype(np.float32)
        except Exception as e:
            print(f"embedding failed: {e}")
            return None


def similarity(a, b):
    """Cosine similarity of two vehicle vectors; NaN when either is missing."""
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    return float(np.dot(a, b))
