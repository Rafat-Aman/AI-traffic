"""Background worker per camera: source -> pipeline -> database + evidence + live view."""

import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from violations.evidence import event_record, save_extra_frames, save_images
from violations.zones import camera_config_from_dict

from .anpr import PlateCollector, PlateReader, find_models
from .autozone_job import AutoZoneJob
from .camera_motion import load_reference
from .clips import ClipRecorder
from .detector import MODELS_DIR, Detector, HelmetDetector, find_helmet_model
from .pipeline import Pipeline
from .reid import TrackletCollector
from .sources import FileSource, LiveSource

JPEG_QUALITY = 80


class CameraWorker(threading.Thread):
    def __init__(self, store, camera, data_dir, notify=None, start_sec=0.0, linker=None):
        super().__init__(daemon=True, name=f"camera-{camera['id']}")
        self.store, self.cam, self.data_dir = store, camera, Path(data_dir)
        self.linker = linker
        self.notify = notify or (lambda msg: None)
        self.start_sec = start_sec
        self.stop_event = threading.Event()
        self.jpeg, self.seq = None, 0      # newest annotated frame for the live view
        self.status = {"state": "starting", "camera_id": camera["id"], "fps": 0.0, "lag": 0.0, "progress": None,
                       "frames": 0, "tracks": 0, "violations": 0, "error": "", "message": "Starting...",
                       "inactive": {}, "speed_paused": False, "backend": "", "video_time": 0.0}
        self.pipeline = None
        self.plates = None
        self.tracklets = None
        self.run_id = None
        self.out_dir = None

    def stop(self):
        self.stop_event.set()

    def _set(self, **kw):
        self.status.update(kw)

    def run(self):
        cam, s = self.cam, self.cam["settings"]
        source = detector = clips = None
        run_id, t_begin, frames, t = None, time.time(), 0, 0.0
        try:
            zones = self.store.zones(cam["id"])
            if not zones:
                raise RuntimeError("No zones yet: open the zone editor and draw at least one road zone")
            self._set(message="Opening video source...")
            if cam["kind"] == "live":
                source = LiveSource(cam["source"], s["max_width"]).open()
            else:
                source = FileSource(cam["source"], s["analysis_fps"], s["max_width"], s["realtime"],
                                    self.start_sec).open()
            frame_size = (source.width, source.height)
            camera_cfg = camera_config_from_dict(zones["config"], *frame_size)
            self._set(message="Loading detection model...")
            detector = Detector(s["model"], s["imgsz"], s["backend"], tracker=s["tracker"], fps=source.fps,
                                log=lambda m: self._set(message=m))
            helmet_path = s.get("helmet_model") or (find_helmet_model() if s.get("helmet", True) else None)
            helmet = HelmetDetector(helmet_path) if helmet_path else None
            plate_model, char_model = find_models(MODELS_DIR)
            if s.get("anpr", True) and plate_model and char_model:
                self._set(message="Loading number-plate models...")
                self.plates = PlateCollector(PlateReader(plate_model, char_model), self._plate_result)
            start_time = None
            if cam["kind"] == "file" and cam.get("start_time"):
                start_time = datetime.fromisoformat(cam["start_time"])
            if s.get("reid", True):
                self.tracklets = TrackletCollector(cam["id"], self._tracklet_done)
            self.pipeline = Pipeline(detector, camera_cfg, frame_size, start_time, source.is_live, helmet,
                                     auto_roi=s["auto_roi"], plates=self.plates,
                                     reference=load_reference(self.data_dir, cam["id"], zones["version"]),
                                     stabilize=s.get("stabilize", True), tracklets=self.tracklets)
            clips = ClipRecorder(s["clip_pre_sec"], s["clip_post_sec"])
            run_id = self.store.start_run(cam["id"], zones["id"], "live" if source.is_live else
                                          ("realtime" if s["realtime"] else "full"))
            self.run_id = run_id
            out_dir = self.out_dir = self.data_dir / "evidence" / f"cam{cam['id']}" / f"run{run_id}"
            self._set(state="running", message="", backend=f"{detector.backend.upper()} {detector.imgsz}px",
                      features={"anpr": self.plates is not None, "helmet": helmet is not None},
                      inactive=self.pipeline.engine.inactive, run_id=run_id, roi=self.pipeline.roi,
                      frame_size=frame_size)
            last_learn = time.time()
            proc_fps, t_prev = 0.0, time.perf_counter()
            while not self.stop_event.is_set():
                got = source.read(timeout=1.0) if source.is_live else source.read()
                if got is None:
                    if not source.is_live:
                        break  # end of the video
                    # No frame for a second: show the outage instead of the last good numbers.
                    stalled = source.state != "online" or source.lag() > 2
                    self._set(state="reconnecting" if stalled else "running", lag=round(source.lag(), 1),
                              fps=0.0 if stalled else self.status["fps"], reconnects=source.reconnects,
                              message=source.error or ("No frames from the camera" if stalled else ""))
                    continue
                if source.is_live:
                    frame, t, wall = got
                else:
                    frame, t = got
                    wall = start_time + timedelta(seconds=t) if start_time else None
                res = self.pipeline.process(frame, t, wall)
                frames += 1
                now = time.perf_counter()  # how fast this Mac processes frames (wall clock)
                inst = 1.0 / max(1e-6, now - t_prev)
                proc_fps = inst if proc_fps == 0 else 0.95 * proc_fps + 0.05 * inst
                t_prev = now
                ok, buf = cv2.imencode(".jpg", res.display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                jpeg = buf.tobytes()
                self.jpeg, self.seq = jpeg, self.seq + 1
                clips.add(jpeg, t)
                for ev in res.events:
                    self._record(ev, frame, wall, out_dir, clips, run_id, zones["id"], t)
                self._set(fps=round(proc_fps, 1), stream_fps=round(res.fps, 1),
                          speed_x=round(proc_fps / source.fps, 2) if not source.is_live else None,
                          lag=round(source.lag(), 2), progress=source.progress(),
                          frames=frames, tracks=res.tracks, video_time=round(t, 1),
                          speed_paused=self.pipeline.speed_paused, view=self.pipeline.view_status(),
                          state="running", message="", reconnects=getattr(source, "reconnects", 0))
                if time.time() - last_learn > 5:
                    self._save_learning()
                    last_learn = time.time()
            done = not self.stop_event.is_set() and not source.is_live
            self._set(state="done" if done else "stopped", progress=1.0 if done else self.status["progress"],
                      message="Finished: full video analysed" if done else "Stopped")
            status = "done" if done else "stopped"
        except Exception as e:
            traceback.print_exc()
            self._set(state="error", error=str(e), message=str(e))
            status = "error"
        finally:
            if self.tracklets:
                self.tracklets.finish_all()
            if self.plates:
                self._set(message="Reading the last number plates...")
                self.plates.flush()
                self._set(message="Finished: full video analysed" if self.status["state"] == "done" else
                          self.status["message"] if self.status["state"] == "error" else "Stopped")
            if clips:
                clips.close()
            if self.pipeline:
                self._save_learning()
                self.pipeline.close()
            if run_id:
                elapsed = max(1e-6, time.time() - t_begin)
                self.store.end_run(run_id, status, frames, t, frames / elapsed, self.status.get("error", ""))
            for obj in (detector, source):
                if obj:
                    obj.close()
            self.notify({"type": "status", "camera_id": cam["id"], "status": self.status})

    def _record(self, ev, frame, wall, out_dir, clips, run_id, zone_version_id, t):
        if ev.frame is not None:  # decided after the vehicle left: its evidence is an earlier frame
            frame = cv2.imdecode(np.frombuffer(ev.frame, np.uint8), cv2.IMREAD_COLOR)
            if self.pipeline.start_time and not self.pipeline.is_live:
                wall = self.pipeline.start_time + timedelta(seconds=ev.obs.time_s)
        wall_text = wall.strftime("%Y-%m-%d %H:%M:%S") if wall else ""
        stem = f"{ev.code}_t{ev.obs.tid}_{ev.obs.time_s:.1f}s"
        when = wall_text or f"video {int(ev.obs.time_s // 60)}:{ev.obs.time_s % 60:05.2f}"
        snap, crop = save_images(ev, frame, out_dir, stem, when)
        clip = out_dir / f"{stem}.mp4"
        clips.trigger(clip, t)
        rel = lambda p: str(p.relative_to(self.data_dir)) if p else None
        start = self.pipeline.start_time
        wall_of = lambda ts: ((start + timedelta(seconds=ts)).strftime("%Y-%m-%d %H:%M:%S") if start and
                              not self.pipeline.is_live else
                              (wall - timedelta(seconds=ev.obs.time_s - ts)).strftime("%Y-%m-%d %H:%M:%S")
                              if wall else "")
        label_time = lambda ts: wall_of(ts) or f"video {int(ts // 60)}:{ts % 60:05.2f}"
        extra = [{**e, "path": rel(e["path"]), "wall_time": wall_of(e["time_s"])}
                 for e in save_extra_frames(ev, out_dir, stem, label_time)]
        rec = event_record(ev, wall_text)
        if ev.started_s is not None:
            rec["details"] = {**rec["details"], "started": label_time(ev.started_s)}
        vid = self.store.add_violation(self.cam["id"], run_id, zone_version_id, rec,
                                       rel(snap), rel(crop), rel(clip), "pending" if self.plates else "", extra)
        if self.plates:
            self.plates.request(ev.obs.tid, vid, t)
        self.status["violations"] += 1
        self.notify({"type": "violation", "camera_id": self.cam["id"], "violation": self.store.violation(vid)})

    def _tracklet_done(self, tl):
        """A vehicle has left this camera: store its pass, and link it straight away if another camera
        already saw it and the routes are known. The batch run revisits every link later."""
        try:
            tl.id = self.store.add_tracklet(self.cam["id"], self.run_id, tl)
            self.status["tracklets"] = self.status.get("tracklets", 0) + 1
        except Exception as e:          # storing a tracklet must never stop the analysis
            print(f"tracklet not stored: {e}")
            return
        if self.linker is None:
            return
        try:
            gid = self.linker.link(tl)
            if gid:
                self.status["linked"] = self.status.get("linked", 0) + 1
                self.notify({"type": "vehicle_linked", "camera_id": self.cam["id"], "global_id": gid,
                             "tracklet_id": tl.id, "cls": tl.cls})
        except Exception as e:
            print(f"online link failed: {e}")

    def _plate_result(self, vid, result):
        """Called from the plate-reading thread."""
        img = None
        if result.image is not None and result.image.size and self.out_dir is not None:
            path = self.out_dir / f"plate_{vid}.jpg"
            cv2.imwrite(str(path), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img = str(path.relative_to(self.data_dir))
        self.store.set_plate(vid, result.status, result.text_bn, result.text_en, result.confidence, img)
        self.notify({"type": "plate", "camera_id": self.cam["id"], "violation": self.store.violation(vid)})

    def _save_learning(self):
        self.store.update_camera(self.cam["id"], learning={
            "updated": datetime.now().isoformat(timespec="seconds"),
            "zones": self.pipeline.direction_suggestions()})


class WorkerManager:
    def __init__(self, store, data_dir, notify=None, linker=None):
        self.store, self.data_dir, self.notify = store, data_dir, notify
        self.linker = linker
        self.workers = {}
        self.autozone = {}   # cid -> AutoZoneJob (learning the zones automatically)

    # ------------------------------------------------------------ automatic zones
    def learning(self, cid):
        j = self.autozone.get(cid)
        # finished jobs still run their completion callback (which starts the camera): not learning any more
        return bool(j and j.is_alive() and j.status["state"] not in ("done", "error", "cancelled"))

    def start_autozone(self, cid, apply=True, start_after=True):
        """Learn the camera's zones from its traffic. apply: save them as the camera's zones; start_after: then
        start analysing (recorded video) / enforcing (live camera)."""
        if self.learning(cid):
            return self.autozone[cid]
        if self.running(cid):
            raise RuntimeError("Stop the camera first: it cannot analyse and learn its zones at the same time")
        cam = self.store.camera(cid)
        if cam is None:
            raise KeyError(cid)
        on_done = (lambda c: self.start(c)) if (apply and start_after) else None
        j = AutoZoneJob(self.store, cam, self.data_dir, apply=apply, on_done=on_done, notify=self.notify)
        self.autozone[cid] = j
        j.start()
        return j

    def stop_autozone(self, cid):
        j = self.autozone.get(cid)
        if j:
            j.stop()

    def autozone_status(self, cid):
        j = self.autozone.get(cid)
        if not j:
            return {"state": "idle", "camera_id": cid}
        st = dict(j.status)
        if j.result:
            st["result"] = j.result
        return st

    # ------------------------------------------------------------ analysis workers
    def start(self, cid, start_sec=0.0):
        if self.learning(cid):
            raise RuntimeError("The camera is still learning its zones")
        w = self.workers.get(cid)
        if w and w.is_alive():
            return w
        cam = self.store.camera(cid)
        if cam is None:
            raise KeyError(cid)
        w = CameraWorker(self.store, cam, self.data_dir, self.notify, start_sec, self.linker)
        self.workers[cid] = w
        w.start()
        return w

    def stop(self, cid, wait=False):
        w = self.workers.get(cid)
        if w:
            w.stop()
            if wait:
                w.join(timeout=15)

    def get(self, cid):
        return self.workers.get(cid)

    def running(self, cid):
        w = self.workers.get(cid)
        return bool(w and w.is_alive())

    def status(self, cid):
        w = self.workers.get(cid)
        learn = self.autozone_status(cid)
        learn.pop("result", None)
        if not w:
            return {"state": "idle", "camera_id": cid, "learning": learn}
        st = dict(w.status)
        if not w.is_alive() and st["state"] in ("running", "starting", "reconnecting"):
            st["state"] = "stopped"
        st["learning"] = learn
        return st

    def stop_all(self):
        for cid in list(self.autozone):
            self.stop_autozone(cid)
        for cid in list(self.workers):
            self.stop(cid, wait=True)
