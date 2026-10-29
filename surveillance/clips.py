"""Evidence video clips: the seconds before and after each violation, as browser-playable H.264 MP4.

The last `pre_sec` of display frames are kept in memory as JPEG bytes (~100 KB each at 1280 px). When a violation
happens the clip collects `post_sec` more seconds, then a background thread encodes it with ffmpeg (libx264).

A second, smaller copy is made for the person a citation is addressed to (make_delivery_clip). The original is
the evidence and is never changed; the copy is only so a phone on a mobile plan can watch it.
"""

import queue
import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


def write_mp4(path, jpegs, fps):
    frames = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    if imageio_ffmpeg is not None:
        gen = imageio_ffmpeg.write_frames(
            str(tmp), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24", quality=None, macro_block_size=2,
            output_params=["-preset", "veryfast", "-crf", "26", "-movflags", "+faststart"])
        gen.send(None)
        for f in frames:
            gen.send(np.ascontiguousarray(f))
        gen.close()
    else:
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
    tmp.rename(path)


DELIVERY_MAX_HEIGHT = 720
DELIVERY_CRF = 28


def make_delivery_clip(src, dst, max_height=DELIVERY_MAX_HEIGHT, crf=DELIVERY_CRF, preset="veryfast", timeout=180):
    """A smaller copy of an evidence clip for the person it is addressed to: at most `max_height` pixels
    tall (never enlarged), H.264, constant-quality `crf`, index first (+faststart) so it starts playing before
    it has all arrived.

    The original is not touched and is what a citation's hash seals; this copy is regenerated if it is lost.
    It is written as <name>.part.mp4 and renamed when complete - the convention write_mp4 uses - so a
    half-written file is never served. Measured on this project's footage: a 1242x1440 original of 5.7 MB
    becomes 1.1 MB, and a 1280x720 original loses about a third. The `veryfast` preset is the smallest at
    this quality as well as the quickest (about 2 s for a 10 s clip on the development CPU). Raise `crf`
    (30 gives under 1 MB for the first case) if a hard size target matters more than detail.
    Raises RuntimeError when ffmpeg is missing or fails; the caller decides whether that matters."""
    if imageio_ffmpeg is None:
        raise RuntimeError("ffmpeg is not available")
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("the delivery copy must not overwrite the original clip")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    # scale to an even height no taller than max_height (H.264 4:2:0 needs even sides); -2 keeps the width even
    # and the aspect ratio. The single quotes keep the comma inside min() from being read as a filter separator.
    scale = f"scale=-2:'trunc(min({int(max_height)},ih)/2)*2'"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-vf", scale, "-c:v", "libx264",
           "-preset", preset, "-crf", str(int(crf)), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
           str(tmp)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0:
            tail = (run.stderr.strip().splitlines() or ["no message"])[-1]
            raise RuntimeError(f"ffmpeg failed: {tail}")
        tmp.replace(dst)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg took more than {timeout} s") from None
    finally:
        tmp.unlink(missing_ok=True)


class ClipRecorder:
    """Encodes clips on a fixed number of background workers, in the order they finish collecting.

    A burst of violations (a jam, a rule misconfigured) queues its clips instead of starting an encoder each,
    and close() waits until every queued clip is written."""

    def __init__(self, pre_sec=5.0, post_sec=5.0, workers=2):
        self.pre_sec, self.post_sec = pre_sec, post_sec
        self.ring = deque()       # (t, jpeg)
        self.pending = []         # [path, end_t, [(t, jpeg)]]
        self.jobs = queue.Queue()
        self.failed = []          # paths that could not be written
        self.workers = [threading.Thread(target=self._run, daemon=True, name=f"clip-{i}") for i in range(workers)]
        for w in self.workers:
            w.start()

    def add(self, jpeg, t):
        self.ring.append((t, jpeg))
        while self.ring and t - self.ring[0][0] > self.pre_sec:
            self.ring.popleft()
        still = []
        for clip in self.pending:
            clip[2].append((t, jpeg))
            if t >= clip[1]:
                self._write(clip)
            else:
                still.append(clip)
        self.pending = still

    def trigger(self, path, t):
        """Start a clip around time t (the current frame); it is written to `path` once post_sec have passed."""
        self.pending.append([Path(path), t + self.post_sec, list(self.ring)])

    def _write(self, clip):
        path, _, frames = clip
        if len(frames) < 2:
            return
        span = frames[-1][0] - frames[0][0]
        fps = max(1.0, min(60.0, (len(frames) - 1) / span)) if span > 0 else 10.0
        self.jobs.put((path, [j for _, j in frames], fps))

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                write_mp4(*job)
            except Exception as e:  # one bad clip must not stop the others
                self.failed.append(str(job[0]))
                print(f"Clip not written ({job[0].name}): {e}")
            finally:
                self.jobs.task_done()

    def backlog(self):
        return self.jobs.qsize()

    def close(self):
        """Write clips that are still collecting (the video ended), and wait until every clip is encoded."""
        for clip in self.pending:
            self._write(clip)
        self.pending = []
        self.jobs.join()
        for _ in self.workers:
            self.jobs.put(None)
        for w in self.workers:
            w.join()
