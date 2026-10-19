"""Camera movement (surveillance/camera_motion.py) and how the pipeline keeps zones on the road when it happens.

    python -m unittest tests.test_camera_motion -v
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.camera_motion import CameraMotion  # noqa: E402
from surveillance.pipeline import Pipeline  # noqa: E402
from violations.zones import camera_config_from_dict  # noqa: E402

W, H = 1280, 720
ROAD = [[500, 250], [780, 250], [1100, 700], [180, 700]]


def scene(seed=0):
    """A textured street-like picture: buildings, windows, signs."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 110, np.uint8)
    for _ in range(400):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(8, 90)), int(rng.integers(8, 90))
        cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(cv2.GaussianBlur(img, (3, 3), 0) + noise, 0, 255).astype(np.uint8)


def similarity(dx, dy, zoom=1.0, rot_deg=0.0):
    """Reference -> new view."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_deg, zoom)
    M[:, 2] += (dx, dy)
    return M.astype(np.float32)


def view(ref, M, seed=1):
    out = cv2.warpAffine(ref, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    noise = np.random.default_rng(seed).normal(0, 3, out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def apply(M, pts):
    p = np.float32(pts).reshape(-1, 2)
    return p @ M[:, :2].T + M[:, 2]


class CameraMotionTests(unittest.TestCase):
    def setUp(self):
        self.ref = scene()

    def test_steady_camera_stays_identity(self):
        m = CameraMotion(self.ref, [ROAD])
        for i in range(10):
            m.update(view(self.ref, similarity(0, 0), seed=i), i * 0.1)
        self.assertEqual(m.state, "stable")
        self.assertTrue(m.identity)

    def test_shift_zoom_and_rotation_are_undone(self):
        M = similarity(35, -18, 1.06, 1.5)
        m = CameraMotion(self.ref, [ROAD])
        for i in range(4):
            m.update(view(self.ref, M, seed=i), i * 0.1)
        self.assertEqual(m.state, "compensated")
        pts = np.float32([[300, 600], [640, 400], [900, 650]])
        back = m.to_ref(apply(M, pts))
        np.testing.assert_allclose(back, pts, atol=2.0)
        np.testing.assert_allclose(m.to_frame(pts), apply(M, pts), atol=2.0)

    def test_traffic_moving_together_is_not_camera_movement(self):
        """A large block of vehicles moving as one must not be taken for the camera moving."""
        m = CameraMotion(self.ref, [ROAD])
        block = scene(seed=7)[0:260, 0:500]
        for i in range(6):
            img = view(self.ref, similarity(0, 0), seed=i)
            x = 300 + 12 * i
            img[380:640, x:x + 500] = block
            m.update(img, i * 0.1)
        self.assertTrue(m.identity, m.describe())

    def test_view_lost_after_moving_pauses_enforcement(self):
        m = CameraMotion(self.ref, [ROAD])
        m.update(view(self.ref, similarity(30, 0)), 0.0)
        blank = np.full_like(self.ref, 120)
        for i in range(1, 30):
            m.update(blank, i * 0.1)
        self.assertEqual(m.state, "lost")
        self.assertFalse(m.enforce)
        for i in range(3):   # pointed back: recovers
            m.update(view(self.ref, similarity(0, 0), seed=i), 3.0 + i * 0.1)
        self.assertEqual(m.state, "stable")

    def test_steady_camera_that_cannot_be_matched_keeps_enforcing(self):
        """Night / fog on a fixed camera: no evidence it moved, so it is assumed not to have."""
        m = CameraMotion(self.ref, [ROAD])
        for i in range(5):
            m.update(view(self.ref, similarity(0, 0), seed=i), i * 0.1)
        dark = (self.ref * 0.02).astype(np.uint8)
        for i in range(50):
            m.update(dark, 1 + i * 0.1)
        self.assertNotEqual(m.state, "lost")
        self.assertTrue(m.enforce)

    def test_zones_out_of_the_picture(self):
        m = CameraMotion(self.ref, [ROAD])
        for i in range(3):
            m.update(view(self.ref, similarity(-850, 0), seed=i), i * 0.1)
        self.assertLess(m.in_view, 0.5)
        self.assertEqual(m.state, "lost")


class FakeDetector:
    imgsz = 960

    def __init__(self):
        self.boxes = []

    def track(self, frame):
        return self.boxes


class PipelineFollowsCameraTests(unittest.TestCase):
    def run_parked_car(self, stabilize):
        """The camera tilts down 40 px and pans 30 px over one second (a pole swaying, a camera knocked), then holds;
        a car parked on the road moves with the picture. Returns (speeds, road positions in metres)."""
        ref = scene()
        cfg = {"frame_size": [W, H], "zones": [{"type": "road", "name": "Road", "points": ROAD, "width_m": 10,
                                                 "length_m": 60, "allowed_direction": "toward_camera"}]}
        det = FakeDetector()
        cam = camera_config_from_dict(cfg, W, H)
        p = Pipeline(det, cam, (W, H), reference=ref, auto_roi=False, stabilize=stabilize)
        car = np.float32([[560, 330], [640, 390]])   # x1y1, x2y2 in the reference view, far part of the road
        speeds, xy = [], []
        for i in range(40):
            t = i / 10
            k = min(1.0, t)
            M = similarity(-30 * k, 40 * k)
            b = apply(M, car).ravel()
            det.boxes = [((float(b[0]), float(b[1]), float(b[2]), float(b[3])), 1, "Car")]
            p.process(view(ref, M, seed=i), t)
            speeds.append(p.speeds.current(1))
            _, x, y = p.trails[1][-1]
            xy.append(cam.roads[0].to_metres((x, y)))
        return p, [s for s in speeds if s is not None], np.float32(xy)

    def test_parked_car_stays_parked_when_the_camera_moves(self):
        p, kph, xy = self.run_parked_car(stabilize=True)
        self.assertEqual(p.motion.state, "compensated")
        self.assertTrue(kph, "no speed measured")
        self.assertLess(max(kph), 3.0)                        # parked: about 0 km/h
        self.assertLess(float(np.ptp(xy[:, 1])), 0.5)         # and it stays on the same spot of the road

    def test_without_following_the_camera_the_parked_car_would_move(self):
        """The same scene with stabilisation off: proves the test above checks something real."""
        _, kph, xy = self.run_parked_car(stabilize=False)
        self.assertGreater(float(np.ptp(xy[:, 1])), 3.0)
        self.assertGreater(max(kph), 5.0)

if __name__ == "__main__":
    unittest.main()
