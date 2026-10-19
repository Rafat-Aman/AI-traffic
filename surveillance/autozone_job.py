"""Background job: watch a camera for a few minutes and propose its zones automatically (surveillance/autozone.py).

Two ways to use it:
* "Automatic" camera setup: the proposal is saved as the camera's zones (a new zone version, like a save in the
  editor) and the camera starts analysing / enforcing right away.
* "Auto-detect" in the zone editor: the proposal is only returned; the editor loads it as a draft to review.
"""

import json
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

from .autozone import TrafficObserver, propose
from .camera_motion import CameraMotion, save_reference
from .detector import Detector
from .sources import FileSource, LiveSource

TARGET_MOVING = 300       # moving motor vehicles: enough for a stable layout and calibration
MIN_SECONDS = 90.0        # observe at least this long (signals, buses at stops, side-road traffic)
MAX_SECONDS_FILE = 420.0  # recorded video: at most this much of it
MAX_SECONDS_LIVE = 900.0  # live camera: give up after 15 minutes of too little traffic
ANALYSIS_FPS = 15         # tracks stay continuous at 15 FPS; twice as fast as full analysis


class AutoZoneJob(threading.Thread):
    def __init__(self, store, cam, data_dir, apply=True, on_done=None, notify=None):
        super().__init__(daemon=True, name=f"autozone-{cam['id']}")
        self.store, self.cam, self.data_dir = store, cam, Path(data_dir)
        self.apply, self.on_done = apply, on_done
        self.notify = notify or (lambda msg: None)
        self.stop_event = threading.Event()
        self.status = {"state": "starting", "camera_id": cam["id"], "message": "Opening the video...",
                       "progress": 0.0, "seconds": 0.0, "moving_vehicles": 0, "apply": apply}
        self.result = None   # {"config", "report", "version"}

    def stop(self):
        self.stop_event.set()

    def _set(self, push=False, **kw):
        self.status.update(kw)
        if push:
            self.notify({"type": "autozone", "camera_id": self.cam["id"], "status": dict(self.status)})

    def run(self):
        cam, s = self.cam, self.cam["settings"]
        source = detector = None
        try:
            if cam["kind"] == "live":
                source = LiveSource(cam["source"], s["max_width"]).open()
                limit = MAX_SECONDS_LIVE
            else:
                source = FileSource(cam["source"], ANALYSIS_FPS, s["max_width"], False).open()
                limit = min(MAX_SECONDS_FILE, source.duration or MAX_SECONDS_FILE)
            self._set(push=True, message="Loading the detection model...")
            detector = Detector(s["model"], s["imgsz"], s["backend"], tracker=s["tracker"], fps=source.fps,
                                log=lambda m: self._set(message=m))
            obs = TrafficObserver((source.width, source.height))
            self._set(push=True, state="observing", message="Watching traffic to learn the road layout...")
            last_check, moving, t0 = 0.0, 0, None
            # Everything is learned in the view of the first frame (the reference): if the camera sways or is
            # moved meanwhile, positions are mapped back to it, and frames it cannot be matched to are skipped.
            reference = motion = None
            boxes, skipped = [], 0
            while not self.stop_event.is_set():
                got = source.read(timeout=1.0) if source.is_live else source.read()
                if got is None:
                    if source.is_live:
                        continue
                    break  # end of the video
                frame, t = got[0], got[1]
                t0 = t if t0 is None else t0
                if motion is None:
                    reference, ref_t = frame.copy(), t
                    motion = CameraMotion(reference)
                motion.update(frame, t, boxes)
                tracks = detector.track(frame)
                boxes = [b for b, _, _ in tracks]
                if not motion.enforce:
                    skipped += 1
                    continue
                if not motion.identity:
                    tracks = [(tuple(motion.to_ref([b[:2], b[2:]]).ravel()), tid, c) for b, tid, c in tracks]
                obs.add(t, tracks)
                obs.add_frame(frame if motion.identity else
                              cv2.warpAffine(frame, motion.A, (frame.shape[1], frame.shape[0])), t)
                elapsed = t - t0
                if elapsed - last_check >= 3.0:
                    last_check = elapsed
                    moving = obs.moving_tracks()
                    progress = min(1.0, 0.5 * min(1.0, moving / TARGET_MOVING) + 0.5 * min(1.0, elapsed / MIN_SECONDS))
                    self._set(push=True, seconds=round(elapsed, 0), moving_vehicles=moving, progress=round(progress, 3),
                              message=f"Watching traffic: {moving} moving vehicles in {elapsed:.0f} s")
                    if moving >= TARGET_MOVING and elapsed >= MIN_SECONDS:
                        break
                    if elapsed >= limit:
                        break
            if self.stop_event.is_set():
                self._set(push=True, state="cancelled", message="Cancelled")
                return
            self._set(push=True, state="analysing", progress=1.0,
                      message="Calibrating the camera and working out the zones...")
            cfg, report = propose(obs, log=lambda m: self._set(message=m))
            if skipped:
                report["notes"].append(f"The camera view changed too much to use {skipped} frame(s)")
            cfg["auto"]["reference_t"] = None if source.is_live else round(ref_t, 2)  # the editor shows this frame
            out = self.data_dir / "autozone"
            out.mkdir(parents=True, exist_ok=True)
            preview = None
            if obs.preview is not None:
                from violations.zones import camera_config_from_dict
                img = obs.preview.copy()
                h, w = img.shape[:2]
                camera_config_from_dict(cfg, w, h).draw(img, max(1.0, h / 900))
                preview = out / f"cam{cam['id']}.jpg"
                cv2.imwrite(str(preview), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cfg["auto"]["preview"] = str(preview.relative_to(self.data_dir))
            (out / f"cam{cam['id']}.json").write_text(json.dumps(cfg, indent=1, default=float))
            version = None
            if self.apply:
                cal = report["calibration"]
                note = (f"Automatic setup: {len(report['carriageways'])} carriageway(s), field of view "
                        f"{cal['hfov_deg']:.0f} deg, camera {cal['camera_height_m']:.1f} m up, speeds "
                        f"+-{cal['speed_uncertainty_pct']}%")
                _, version = self.store.save_zones(cam["id"], cfg, note)
                save_reference(self.data_dir, cam["id"], version, reference)  # the view these zones belong to
            self.result = {"config": cfg, "report": report, "version": version}
            self._set(push=True, state="done", version=version,
                      message=("Zones learned and saved as version %s" % version) if version else "Zones proposed")
            if self.apply and self.on_done:
                try:
                    self.on_done(cam["id"])
                except Exception as e:  # the zones are saved either way: learning itself succeeded
                    self._set(push=True, message=f"Zones saved as version {version}, but the camera did not start: {e}")
        except Exception as e:
            if not isinstance(e, ValueError):
                traceback.print_exc()
            self._set(push=True, state="error", message=str(e) or f"Automatic setup failed ({type(e).__name__})")
        finally:
            for obj in (detector, source):
                if obj:
                    try:
                        obj.close()
                    except Exception:
                        pass
