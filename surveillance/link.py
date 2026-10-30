"""Turning stored tracklets into city-level vehicle identities.

Two paths, as a real deployment needs both:

batch   `LinkJob` reconsiders every tracklet in a time window at once. It discovers which cameras feed
        which, fits each route's parameters from the traffic itself, and solves a one-to-one assignment per
        route. This is the accurate answer and the one to quote, because a vehicle's speed, width and
        colour are only final once it has left the view.

online  `OnlineLinker` answers while the cameras are still running, using the routes the last batch run
        learned. It is deliberately timid - it links only when the best candidate clearly beats the next
        one - because a provisional link on a dashboard is worse than an honest blank, and the batch run
        will revisit it anyway.

Both write the same `global_id` onto the tracklet rows; `link_source` records which path set it.
"""

import threading
import time
import traceback
from collections import defaultdict

import numpy as np

from . import reid

ONLINE_WINDOW_SEC = 600.0     # how far back the online path looks on the other cameras


def to_tracklet(row):
    """A database row back into a reid.Tracklet."""
    t = reid.Tracklet(
        camera_id=row["camera_id"], track_id=row["track_id"], cls=row["cls"] or "",
        t_cross=row["t_cross"], lateral=row["lateral"], speed=row["speed"] or 0.0,
        direction=row["direction"] or "unknown",
        width=row["width"] if row["width"] is not None else float("nan"),
        extrap_m=row["extrap_m"] or 0.0, zone=row["zone"] or "", run_id=row["run_id"],
        id=row["id"], global_id=row["global_id"])
    t.hist = np.frombuffer(row["hist"], np.float32) if row["hist"] else None
    t.lab = np.frombuffer(row["lab"], np.float32) if row["lab"] else None
    t.embed = np.frombuffer(row["embed"], np.float32) if row.get("embed") else None
    t.embed_model = (row.get("embed_model") or None) if t.embed is not None else None
    t.embed_reid = np.frombuffer(row["embed_reid"], np.float32) if row.get("embed_reid") else None
    t.embed_reid_model = (row.get("embed_reid_model") or None) if t.embed_reid is not None else None
    return t


def load(store, since=None, until=None, camera_id=None):
    rows = store.tracklets(camera_id=camera_id, since=since, until=until)
    return [to_tracklet(r) for r in rows if r["t_cross"] is not None and (r["speed"] or 0) > 0]


def run_linking(store, since=None, until=None, log=lambda m: None):
    """Link everything in the window and write the global ids. Returns a report."""
    tracklets = load(store, since, until)
    if len(tracklets) < 2:
        return {"tracklets": len(tracklets), "linked": 0, "vehicles": 0, "flows": [],
                "message": "Not enough tracklets yet: analyse at least two cameras first."}
    cameras = {t.camera_id for t in tracklets}
    log(f"  {len(tracklets)} tracklets from {len(cameras)} camera(s)")
    if len(cameras) < 2:
        return {"tracklets": len(tracklets), "linked": 0, "vehicles": 0, "flows": [],
                "message": "Only one camera has tracklets; cross-camera linking needs at least two."}

    links, groups = reid.link_all(tracklets, log=log)
    # keep global ids stable across runs by reusing the smallest existing id in each group
    assignments, next_id = [], store.next_global_id()
    used = set()
    for members in groups.values():
        prior = [t.global_id for t in members if t.global_id and t.global_id not in used]
        gid = min(prior) if prior else next_id
        if gid >= next_id:
            next_id = gid + 1
        used.add(gid)
        for t in members:
            assignments.append((t.id, gid))
    store.set_global_ids(assignments, source="batch")

    flows, _ = reid.discover_flows(tracklets, log=lambda m: None)
    multi = sum(1 for g in groups.values() if len(g) > 1)
    return {
        "tracklets": len(tracklets), "cameras": sorted(cameras), "linked": len(links),
        "vehicles": len(groups), "multi_camera": multi,
        "flows": [{"name": f.name, "from_camera": f.src[0], "from_direction": f.src[1],
                   "to_camera": f.dst[0], "to_direction": f.dst[1], "gap_m": round(f.gap_m, 1),
                   "significance": round(f.significance, 1),
                   "refused": sorted(getattr(f, "refused", set()))} for f in flows],
        "message": (f"{multi} vehicle(s) recognised on more than one camera, from {len(tracklets)} passes"
                    if multi else "No vehicle could be matched across cameras with enough confidence."),
    }


class LinkJob(threading.Thread):
    """Runs `run_linking` in the background so the dashboard stays responsive."""

    def __init__(self, store, since=None, until=None, notify=None):
        super().__init__(daemon=True, name="link")
        self.store, self.since, self.until = store, since, until
        self.notify = notify or (lambda m: None)
        self.status = {"state": "running", "message": "Matching vehicles across cameras...", "log": []}
        self.result = None

    def _log(self, m):
        self.status["log"].append(m.strip())
        del self.status["log"][:-40]

    def run(self):
        t0 = time.perf_counter()
        try:
            self.result = run_linking(self.store, self.since, self.until, log=self._log)
            self.status.update(state="done", message=self.result["message"],
                               seconds=round(time.perf_counter() - t0, 1))
        except Exception as e:
            traceback.print_exc()
            self.status.update(state="error", message=str(e) or type(e).__name__)
        self.notify({"type": "link", "status": dict(self.status), "result": self.result})


class OnlineLinker:
    """Links a tracklet the moment its vehicle leaves view, using routes already learned.

    Holds nothing but the learned parameters and a short memory of what the other cameras saw, so it costs
    nothing on the per-frame path."""

    def __init__(self, store, window_sec=ONLINE_WINDOW_SEC):
        self.store, self.window = store, window_sec
        self.params = {}          # (src key, dst key) -> fitted parameters
        self.lock = threading.Lock()

    def learn_from(self, store=None, since=None):
        """Pick up the routes and parameters from the stored tracklets (after a batch run, or at startup)."""
        store = store or self.store
        tracklets = load(store, since=since)
        if len(tracklets) < 20:
            return 0
        flows, by = reid.discover_flows(tracklets, log=lambda m: None)
        learned = {}
        for f in flows:
            src, dst = by[f.src], by[f.dst]
            try:
                _, P = reid.match_flow(f, src, dst, log=lambda m: None)
            except Exception:
                continue
            if not getattr(f, "refused", set()):
                learned[(f.src, f.dst)] = P
        with self.lock:
            self.params = learned
        return len(learned)

    def link(self, tracklet):
        """Give this tracklet a global id if one candidate clearly stands out. Returns the id or None."""
        with self.lock:
            params = dict(self.params)
        if not params:
            return None
        rows = self.store.tracklets(since=tracklet.t_cross - self.window,
                                    until=tracklet.t_cross + self.window)
        recent = [to_tracklet(r) for r in rows
                  if r["camera_id"] != tracklet.camera_id and r["id"] != tracklet.id]
        if not recent:
            return None
        best = reid.match_online(tracklet, recent, params)
        if best is None:
            return None
        a, b, _ = best
        other = b if a.id == tracklet.id else a
        gid = other.global_id or self.store.next_global_id()
        pairs = [(tracklet.id, gid)]
        if not other.global_id:
            pairs.append((other.id, gid))
        self.store.set_global_ids(pairs, source="online")
        return gid


class SiteRun(threading.Thread):
    """Several videos of one place, taken through the whole pipeline and then linked together.

    Runs the cameras one after another. The detector already saturates every core, so two analyses at once
    would halve each other's frame rate without finishing any sooner; and linking needs every camera's
    tracklets anyway, so there is nothing to gain by overlapping them. When the last one is done the batch
    link runs, and the online linker picks up the routes it learned.
    """

    def __init__(self, manager, store, cameras, zone_mode="auto", notify=None, linker=None):
        super().__init__(daemon=True, name="site")
        self.manager, self.store, self.cameras = manager, store, list(cameras)
        self.zone_mode, self.linker = zone_mode, linker
        self.notify = notify or (lambda m: None)
        self.status = {"state": "queued", "message": "Waiting to start...", "done": [], "current": None,
                       "total": len(self.cameras)}
        self.result = None

    def _set(self, **kw):
        self.status.update(kw)
        self.notify({"type": "site", "status": dict(self.status), "cameras": self.cameras})

    def _wait_while(self, check, timeout=24 * 3600, poll=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not check():
                return True
            time.sleep(poll)
        return False

    def run(self):
        try:
            for n, cid in enumerate(self.cameras, 1):
                cam = self.store.camera(cid)
                label = cam["name"] if cam else f"camera {cid}"
                if self.zone_mode == "auto":
                    self._set(state="learning", current=cid,
                              message=f"[{n}/{len(self.cameras)}] learning the road layout of {label}...")
                    # start_autozone applies the zones and starts the analysis itself when it finishes
                    self.manager.start_autozone(cid, apply=True, start_after=True)
                    self._wait_while(lambda: self.manager.learning(cid))
                else:
                    self.manager.start(cid)
                self._set(state="analysing", current=cid,
                          message=f"[{n}/{len(self.cameras)}] analysing {label}...")
                time.sleep(2.0)                      # let the worker register before we watch for it
                self._wait_while(lambda: self.manager.running(cid))
                self.status["done"].append(cid)

            self._set(state="linking", current=None, message="Matching vehicles across the cameras...")
            self.result = run_linking(self.store, log=lambda m: None)
            if self.linker:
                self.linker.learn_from()
            self._set(state="done", message=self.result["message"])
        except Exception as e:
            traceback.print_exc()
            self._set(state="error", message=str(e) or type(e).__name__)
