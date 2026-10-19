"""Tests for the time-based speed estimator, recording-time detection and the measurement-reliability gate."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.speed import SpeedEstimator  # noqa: E402
from surveillance.sources import detect_start_time, output_size  # noqa: E402
from violations import CameraConfig, RoadZone  # noqa: E402

POINTS = [[1340.0, 200.0], [1505.0, 196.0], [2508.0, 932.0], [1458.0, 908.0]]


class SpeedEstimatorTests(unittest.TestCase):
    def drive(self, fps, kph, seconds=4, jitter=0.0):
        import random
        random.seed(0)
        est, zone, v = SpeedEstimator(), object(), kph / 3.6
        out = None
        for i in range(int(seconds * fps)):
            t = i / fps
            out = est.update(1, t, zone, (2.0 + random.uniform(-jitter, jitter), 5 + v * t))
        return out

    def test_constant_speed(self):
        self.assertAlmostEqual(self.drive(30, 50), 50, delta=0.5)

    def test_same_speed_when_frames_are_dropped(self):
        self.assertAlmostEqual(self.drive(8, 50), self.drive(30, 50), delta=1.0)

    def test_jitter_is_smoothed(self):
        self.assertAlmostEqual(self.drive(30, 40, jitter=0.3), 40, delta=4)


class StartTimeTests(unittest.TestCase):
    def test_file_name_patterns(self):
        for name, want in [("VID_20260922_173600.mp4", datetime(2026, 9, 22, 17, 36, 0)),
                           ("PXL_20260922_173600123.mp4", datetime(2026, 9, 22, 17, 36, 0)),
                           ("ch01_20260922173600.mp4", datetime(2026, 9, 22, 17, 36, 0)),
                           ("2026-09-22 17.36.00.mov", datetime(2026, 9, 22, 17, 36, 0))]:
            self.assertEqual(detect_start_time(name)[0], want, name)

    def test_unknown(self):
        self.assertIsNone(detect_start_time("/nonexistent/traffic.mp4")[0])

    def test_output_size_is_even(self):
        self.assertEqual(output_size(3840, 2160, 1920), (1920, 1080))
        self.assertEqual(output_size(1281, 721, 0), (1280, 720))


class ReliabilityTests(unittest.TestCase):
    def test_far_end_is_less_precise(self):
        road = RoadZone("r", POINTS, 10.5, 90)
        self.assertGreater(road.metres_per_pixel(5), 5 * road.metres_per_pixel(85))
        start = road.reliable_from(0.25)
        self.assertTrue(0 < start < 90)
        self.assertLessEqual(road.metres_per_pixel(start + 1), 0.25)

    def test_validation_messages(self):
        with self.assertRaisesRegex(ValueError, "4 corners"):
            RoadZone("r", POINTS[:3], 10, 90)
        with self.assertRaisesRegex(ValueError, "width and length"):
            RoadZone("r", POINTS, 0, 90)
        with self.assertRaisesRegex(ValueError, "cross over"):
            RoadZone("r", [POINTS[0], POINTS[2], POINTS[1], POINTS[3]], 10, 90)

    def test_speed_limits(self):
        cam = CameraConfig([RoadZone("c", POINTS, 10, 90), RoadZone("e", POINTS, 10, 90, road_category="expressway"),
                            RoadZone("s", POINTS, 10, 90, speed_limit_kph=20)])
        city, exp, school = cam.roads
        self.assertEqual(cam.speed_limit(city, "Motorbike"), 60)
        self.assertEqual(cam.speed_limit(city, "Car"), 60)
        self.assertEqual(cam.speed_limit(exp, "Car"), 80)
        self.assertEqual(cam.speed_limit(exp, "Motorbike"), 80)
        self.assertEqual(cam.speed_limit(school, "Car"), 20)
        self.assertIsNone(cam.speed_limit(city, "Rickshaw"))


class ClipRecorderTests(unittest.TestCase):
    def test_burst_of_violations_gives_every_clip(self):
        """40 violations within a second (a flood) must all get a complete clip, none cut off at the end."""
        import tempfile
        import cv2
        import numpy as np
        from surveillance.clips import ClipRecorder
        with tempfile.TemporaryDirectory() as d:
            rec = ClipRecorder(pre_sec=1, post_sec=1)
            jpg = cv2.imencode(".jpg", np.full((72, 128, 3), 120, np.uint8))[1].tobytes()
            t = 0.0
            for i in range(30):  # 1 s of history
                t = i / 30
                rec.add(jpg, t)
            for k in range(40):
                rec.trigger(Path(d) / f"c{k}.mp4", t)
            for i in range(31, 45):  # the video ends 0.5 s later, before post_sec is over
                rec.add(jpg, i / 30)
            rec.close()
            clips = sorted(Path(d).glob("*.mp4"))
            self.assertEqual(len([c for c in clips if not c.name.endswith(".part.mp4")]), 40)
            self.assertFalse(any(c.name.endswith(".part.mp4") for c in clips))
            cap = cv2.VideoCapture(str(Path(d) / "c39.mp4"))
            self.assertGreater(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
            cap.release()


if __name__ == "__main__":
    unittest.main()


class AutoZoneStartTests(unittest.TestCase):
    def test_camera_starts_when_learning_finishes(self):
        """The learning job starts the camera from its own thread right after saving the zones: at that moment
        the job thread is still alive, but the camera must no longer count as learning."""
        import threading
        from surveillance import worker as W

        started = []
        mgr = W.WorkerManager(store=None, data_dir="/tmp", notify=None)

        class Job(threading.Thread):
            def __init__(self):
                super().__init__(daemon=True)
                self.status, self.error = {"state": "observing"}, None

            def run(self):
                self.status["state"] = "done"
                try:
                    mgr.start(7)
                except Exception as e:
                    self.error = e

        orig = W.CameraWorker
        W.CameraWorker = lambda *a, **k: type("Fake", (), {"start": lambda s: started.append(7),
                                                            "is_alive": lambda s: True})()
        mgr.store = type("S", (), {"camera": lambda s, cid: {"id": cid}})()
        try:
            j = Job()
            mgr.autozone[7] = j
            j.start()
            j.join(5)
        finally:
            W.CameraWorker = orig
        self.assertIsNone(j.error)
        self.assertEqual(started, [7])
