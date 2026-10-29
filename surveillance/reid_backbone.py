"""The team's trained vehicle re-identification network (ResNet50-IBN-a, 320 px), rebuilt for CPU inference.

Where it comes from
    `research/cityflow/cityflowv2.ipynb`, cell 11. The team trained it on the AI City Challenge 2021 track-2
    ReID data (295 real vehicle identities) plus 1,362 synthetic VehicleX identities, 12 epochs on two T4s,
    with the vehicles of CityFlowV2 scene S02 held out. This file is only the part needed to *use* it:
    the same layers with the same names, so the saved `state_dict` loads with `strict=True`, and none of
    the training code. It needs torch and nothing else new: no IBN-Net package and no download, because
    the checkpoint already holds every weight, including the batch-norm statistics re-estimated on real
    images after training (`load` refuses a checkpoint that does not fit, key for key).

The network, as trained
    ResNet-50 with instance-batch norm "a" (instance norm on the first half of the channels of the first
    norm in every block of stages 1-3, plain batch norm in stage 4), the last stage run at stride 1 so the
    final map is 20 x 20 at 320 px, generalised-mean pooling (a learned exponent, 2.96 in this checkpoint),
    then a batch-norm "neck". The embedding is the neck output L2-normalised - 2048 numbers. The classifier
    the network was trained with is not in the checkpoint and is not needed.

    Which half of the channels takes instance norm is not something the key names can confirm - both halves
    have the same width, so a swapped split would also load strictly. It was checked against the data: on
    real Dhaka crops the batch-norm running means of six layers correlate at 0.91-0.98 with the activations
    of the second half and between -0.24 and 0.24 with those of the first.

Input, exactly as in training
    A BGR crop is turned to RGB, squashed (aspect ratio not kept) to 320 x 320 with PIL's bilinear filter,
    scaled to [0, 1] and normalised with the ImageNet mean and standard deviation. The notebook also runs
    each crop mirrored and adds the two vectors; that is `flip=True` here, and costs twice as much on a CPU.

What to expect of it
    It was trained on American roads, mostly cars. The team measured on held-out CityFlowV2 vehicles
    (450 queries against 8,339 gallery images) mAP 0.427 and rank-1 0.560 - the notebook prints 0.490 / 0.656
    beside it for the previous version, trained without synthetic data at 256 px - and, inside their
    whole tracking pipeline on scene S02, IDF1 40.5 with default settings, 47.8 tuned on S02 itself and
    46.5 by 2-fold cross-validation, against 80.95 / 84.86 for the 2021 / 2022 challenge winners. How it does
    on Dhaka's vehicles is a different question: `tools/reid_eval.py --embedder reid` measures it on this
    project's footage and `surveillance/embed.py` records the result.

Cost
    A forward pass is about 25 GFLOPs per 320 px crop, so a CPU is slow at this. On the development machine
    (6 cores, AVX-512, PyTorch eager, channels-last, which is 8-28% faster than the default layout) one
    vehicle takes 1.0 s for six crops and 1.9 s for ten, all cores busy; with `flip` 3.5 s for ten; at 256 px
    1.2 s for ten. MobileNet needs about 45 ms for ten. Two of these running at once would only fight over
    the cores, which is why `surveillance.embed` lets one through at a time.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

BACKBONE = "resnet50_ibn_a"
SIZE = 320
EMBED_DIM = 2048
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


# ---------------------------------------------------------------- the network

class IBN(nn.Module):
    """Instance norm on the first half of the channels, batch norm on the second (IBN-Net "a")."""

    def __init__(self, planes):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, self.half, 1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), 1)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, ibn, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + (x if self.downsample is None else self.downsample(x)))


def _stage(inplanes, planes, blocks, stride, ibn):
    down = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))
    return nn.Sequential(Bottleneck(inplanes, planes, ibn, stride, down),
                         *[Bottleneck(planes * 4, planes, ibn) for _ in range(blocks - 1)])


class GeM(nn.Module):
    """Generalised-mean pooling over the whole map; the exponent was learned in training."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = nn.Parameter(torch.ones(1) * p), eps

    def forward(self, x):
        x = x.float().clamp(min=self.eps).pow(self.p)
        return F.avg_pool2d(x, x.shape[-2:]).pow(1.0 / self.p)


class ReIDNet(nn.Module):
    """ResNet50-IBN-a, last stride 1, GeM, BNNeck. Layer names follow the notebook's `ReIDNet`."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            _stage(64, 64, 3, 1, True), _stage(256, 128, 4, 2, True), _stage(512, 256, 6, 2, True),
            _stage(1024, 512, 3, 1, False))           # stride 1, not 2: the "last stride 1" of the notebook
        self.pool = GeM()
        self.neck = nn.BatchNorm1d(EMBED_DIM)

    def forward(self, x):
        return F.normalize(self.neck(self.pool(self.body(x)).flatten(1)), dim=1)


# ---------------------------------------------------------------- weights

def load(path):
    """The network with its trained weights, in eval mode. Raises if the checkpoint does not fit exactly.

    Strict on purpose: the notebook loaded with `strict=False`, which would let a renamed layer or a
    missing batch-norm statistic through and quietly give worse vectors. `surveillance.embed` decides
    where the weights live (`models/reid/`) and falls back to MobileNet when this raises."""
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("backbone") != BACKBONE:
        raise ValueError(f"{path.name}: backbone {ck.get('backbone')!r}, this module builds {BACKBONE!r}")
    model = ReIDNet()
    model.load_state_dict(ck["state_dict"], strict=True)
    bad = [k for k, v in ck["state_dict"].items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise ValueError(f"{path.name}: non-finite values in {', '.join(bad[:3])}")
    model.size = int(ck.get("size", SIZE))
    model.meta = {"file": path.name, "backbone": ck["backbone"], "size": model.size,
                  "scenes_held_out": ck.get("scenes", ""), "train_vehicles": len(ck.get("train_ids", ()))}
    return model.eval().to(memory_format=torch.channels_last)    # 8-28% faster on this CPU, same vectors


# ---------------------------------------------------------------- inference

def prepare(crops, size=SIZE):
    """BGR uint8 crops -> (N, 3, size, size) float32, the notebook's `TEST_TF` step for step: RGB, PIL
    bilinear squash to size x size, /255, ImageNet mean and std."""
    out = np.empty((len(crops), 3, size, size), np.float32)
    for i, c in enumerate(crops):
        img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((size, size), _BILINEAR)
        out[i] = ((np.asarray(img, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return out


@torch.inference_mode()
def embed(model, crops, flip=True, batch=16, size=None):
    """(N, 2048) float32: one L2-normalised vector per crop. With `flip`, the crop and its mirror image are
    both run and their vectors added before normalising, as the notebook does (twice the CPU time).
    `size` overrides the 320 px the network was trained at; a smaller one is faster and is not what the
    weights were fitted to."""
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(prepare(crops[i:i + batch], size or getattr(model, "size", SIZE)))
        x = x.contiguous(memory_format=torch.channels_last)
        f = model(x)
        if flip:
            f = F.normalize(f + model(torch.flip(x, dims=[3])), dim=1)
        out.append(f)
    return torch.cat(out).numpy() if out else np.zeros((0, EMBED_DIM), np.float32)
