"""V11: two vehicles that nearly collided, or did - a CANDIDATE for the vision-language model, never a citation alone.

The report's Table 3.1 (#8, near-collision and collision: convergent trajectories, abrupt deceleration, post-event
stasis) is decided by the VLM. This rule only nominates pairs, so every event carries decided_by="vlm" and
needs_review=True. Dhaka traffic is dense and slow: vehicles pass within centimetres of each other all day and creep
bumper to bumper, so closeness alone means nothing. Only a pair that is CLOSING FAST is a candidate.

Measured from fitted positions. Each vehicle's last 0.8 s of precise road-plane positions (obs.xy only; vehicles in the
imprecise far end are ignored) gives, by least squares, its position and velocity now, and the window before it
(0.8-1.6 s ago) gives its earlier speed. Deceleration = (earlier speed - speed now) / the time between the window
centres. With 0.2 m box jitter the velocity is good to about 0.25 m/s and the deceleration to about 0.5 m/s^2, so the
4 m/s^2 threshold is eight times the noise. A window whose fit residual exceeds max_fit_rms_m (0.6 m) is a tracker
glitch (ID swap, box jump) and the vehicle is skipped for that frame.

Footprints. Each class has a nominal width x length (FOOTPRINT, the sizes of surveillance/autozone.CLASS_SIZE plus
typical sizes for the local classes; +-15%). The bottom edge of a box is the part of the vehicle nearest the camera and
cameras here look along the road, so the body lies beyond the point by its full length (body_behind_frac = 1.0; 0.5 for
a camera looking across the road). Gaps are between these rectangles, which matters: a car behind a bus is 3 m closer
than the points alone suggest.

(a) near_collision - all must hold for min_hits = 2 consecutive frames, for a same-zone pair within range_m = 25 m:
  * time to collision <= 1.5 s, from the fitted relative position and velocity with footprints swept at constant
    velocity (exact for rectangles; 1.5 s is the usual "serious conflict" level at urban speeds, and means the
    deceleration needed to stop is about a third of the closing speed per second);
  * closing speed >= 2.5 m/s (9 km/h): relative speeds in creeping Dhaka traffic are mostly under 1.5 m/s and the
    noise on a relative velocity is 0.4 m/s, so slow squeezes and side-by-side passing never qualify. Two vehicles
    abreast at the same speed have closing speed 0 however close they are;
  * hard braking: one of the two decelerated at >= 4 m/s^2 (0.4 g; ordinary braking is 1.5-3, emergency 6-9) within
    the last 1.5 s. Gentle approaches to a queue never qualify however short their TTC.
  The event is held contact_wait_sec = 1.5 s: if the pair touches in that time it becomes the collision below, so one
  incident gives ONE event. No contact: reported as near_collision.
(b) collision: the footprints touch while closing (>= 2.5 m/s peak in the second before, so a bumper kiss in a jam is
  not one), then BOTH vehicles are stationary (< 0.8 m/s) within stop_within_sec = 4 s, stay so for stasis_sec = 3 s and
  still touch (gap <= 1 m, the position error). Not in parking bays, valid bus stops or zones where stopping is
  allowed. Reported at the end of the stasis; the photo of the moment of contact travels as an extra frame and
  started_s is the contact time, so a clip should reach back that far (about 8 s, more than the default 5 s).
A vehicle in an incident is quiet for cooldown_sec = 5 s (a pile-up is one incident); the event goes to the vehicle
that braked harder (collision: that approached faster) unless it already has a V11 case, then to the other.

Limits: collisions in which a vehicle drives away, a tracker that merges or swaps the two boxes at contact, and any
pair in the imprecise far end are missed. A hard stop that ends 0.5 m short of a queue can look like contact (the
footprints are +-15% and positions +-0.3 m), which is why the VLM decides. False-positive rates on real Dhaka footage
are unmeasured. Cost: one vectorised pass per frame over all vehicles (ring buffers in numpy), Python only for the few
pairs that are close and closing (peak closing is remembered from 1.5 m/s, 0.6 of the minimum)."""

import numpy as np

from .base import BaseViolation

# (width, length) in metres. Car..Leguna: surveillance/autozone.CLASS_SIZE; the rest typical catalogue sizes.
FOOTPRINT = {"Car": (1.75, 4.4), "MPV": (1.75, 4.6), "Pickup": (1.78, 5.0), "Bus": (2.45, 10.5),
             "Truck": (2.35, 6.5), "CNG": (1.35, 2.7), "Leguna": (1.5, 3.6), "Easybike": (1.3, 2.7),
             "Motorbike": (0.8, 2.0), "Bhotbhoti": (1.5, 3.5), "PowerTiller": (1.0, 2.8), "Rickshaw": (1.2, 2.5),
             "Bicycle": (0.6, 1.8), "Van": (1.1, 2.4), "Wheelbarrow": (0.6, 1.4)}
DEFAULT_FOOTPRINT = (1.8, 4.0)
INF = 1e9


def pair_photo(frame, boxes, max_width=1920):
    """JPEG of the scene with the two vehicles boxed (orange and cyan)."""
    import cv2
    h, w = frame.shape[:2]
    s = min(1.0, max_width / w)
    img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
    for box, col in zip(boxes, ((0, 140, 255), (255, 200, 0))):
        x1, y1, x2, y2 = (int(v * s) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, round(3 * img.shape[0] / 1080)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _ls(T, P, m, t_now):
    """Line fit of each row's masked (t, x, y): (n, span, position now, velocity, rms residual)."""
    n = m.sum(1)
    nz = np.maximum(n, 1)
    tb = np.where(m, T, 0.0).sum(1) / nz
    dt = np.where(m, T - tb[:, None], 0.0)
    mm = m[..., None]
    pb = np.where(mm, P, 0.0).sum(1) / nz[:, None]
    dp = np.where(mm, P - pb[:, None, :], 0.0)
    v = (dt[..., None] * dp).sum(1) / np.maximum((dt * dt).sum(1), 1e-9)[:, None]
    res = dp - v[:, None, :] * dt[..., None]
    rms = np.sqrt((res * res).sum((1, 2)) / nz)
    span = np.where(m, T, -INF).max(1) - np.where(m, T, INF).min(1)
    return n, span, pb + v * (t_now - tb)[:, None], v, rms, tb


class _Rings:
    """Per-vehicle ring buffers of (t, x, y), as numpy arrays so a whole frame is fitted at once."""

    def __init__(self, k=36, cap=64):
        self.k = k
        self.T = np.full((cap, k), -INF)
        self.P = np.zeros((cap, k, 2))
        self.head = np.zeros(cap, np.int64)
        self.last = np.full(cap, -INF)
        self.slot, self.free = {}, list(range(cap - 1, -1, -1))

    def _grow(self):
        cap = len(self.head)
        self.T = np.vstack([self.T, np.full((cap, self.k), -INF)])
        self.P = np.concatenate([self.P, np.zeros((cap, self.k, 2))])
        self.head = np.concatenate([self.head, np.zeros(cap, np.int64)])
        self.last = np.concatenate([self.last, np.full(cap, -INF)])
        self.free = list(range(2 * cap - 1, cap - 1, -1))

    def slots(self, tids):
        out = []
        for tid in tids:
            s = self.slot.get(tid)
            if s is None:
                if not self.free:
                    self._grow()
                s = self.slot[tid] = self.free.pop()
            out.append(s)
        return np.array(out, np.int64)

    def add(self, idx, t, xy, min_dt):
        keep = (t - self.last[idx]) >= min_dt
        idx, xy = idx[keep], xy[keep]
        h = self.head[idx]
        self.T[idx, h] = t
        self.P[idx, h] = xy
        self.head[idx] = (h + 1) % self.k
        self.last[idx] = t

    def drop(self, tid):
        s = self.slot.pop(tid, None)
        if s is not None:
            self.T[s] = -INF
            self.head[s] = 0
            self.last[s] = -INF
            self.free.append(s)


class NearCollision(BaseViolation):
    code = "V11"
    title = "Near-collision or collision"
    DEFAULTS = {
        "range_m": 25.0,             # only pairs closer than this are examined
        "fit_sec": 0.8,              # velocity window; the window before it gives the earlier speed
        "min_samples": 5,
        "max_fit_rms_m": 0.6,        # a worse line fit is a tracker glitch
        "min_purity": 0.5,
        "body_behind_frac": 1.0,     # share of the vehicle's length beyond its box's bottom edge (away from camera)
        "ttc_max_s": 1.5,
        "min_closing_mps": 2.5,
        "min_decel_mps2": 4.0,
        "decel_memory_sec": 1.5,
        "min_hits": 2,
        "contact_wait_sec": 1.5,
        "watch_gap_m": 6.0,          # pairs this close and closing remember their peak closing speed
        "closing_memory_sec": 1.0,
        "still_mps": 0.8,
        "stop_within_sec": 4.0,
        "stasis_sec": 3.0,
        "stasis_gap_m": 1.0,
        "lost_sec": 1.5,             # an incident whose vehicle is not seen this long is dropped
        "cooldown_sec": 5.0,
        "min_dt_sec": 0.045,        # about 22 Hz at most: 36 samples cover the 1.6 s of history
    }
    NO_COLLISION_AREAS = {"parking", "bus_stop"}

    def __init__(self, settings=None, camera=None):
        super().__init__(settings, camera)
        self.rings = _Rings()
        self.frame_key = None
        self.pending = {}      # tid -> ViolationEvent to hand back when the engine reaches that vehicle
        self.watch = {}        # (a, b) -> [last seen s, peak closing m/s]
        self.inc = {}          # (a, b) -> incident dict
        self.cool = {}         # tid -> quiet until
        self.reported = set()  # tids that already carry a V11 case
        self.peak_decel = {}   # tid -> [decel, when]

    def update(self, obs, ctx):
        key = (ctx.frame_idx, ctx.time_s)
        if key != self.frame_key:
            self.frame_key = key
            self.pending = {}
            self._batch(ctx)
        return self.pending.pop(obs.tid, None)

    # ------------------------------------------------------------------ one frame
    def _batch(self, ctx):
        cfg, t = self.cfg, ctx.time_s
        obs = [o for o in ctx.scene.observations if o.xy is not None and o.road is not None
               and o.purity >= cfg["min_purity"]]
        if not obs:
            return
        idx = self.rings.slots([o.tid for o in obs])
        self.rings.add(idx, t, np.array([o.xy for o in obs], np.float64), cfg["min_dt_sec"])
        T, P = self.rings.T[idx], self.rings.P[idx]
        age = t - T
        w = cfg["fit_sec"]
        nA, spanA, pos, vel, rms, tbA = _ls(T, P, (age >= 0) & (age <= w), t)
        nB, spanB, _, velB, _, tbB = _ls(T, P, (age > w) & (age <= 2 * w), t)
        ok = (nA >= cfg["min_samples"]) & (spanA >= 0.5 * w) & (rms <= cfg["max_fit_rms_m"])
        okB = ok & (nB >= cfg["min_samples"]) & (spanB >= 0.5 * w)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        decel = np.where(okB, (np.hypot(velB[:, 0], velB[:, 1]) - speed) / np.maximum(tbA - tbB, 1e-3), 0.0)
        for i in np.nonzero(okB & (decel >= cfg["min_decel_mps2"]))[0]:
            self.peak_decel[obs[i].tid] = [float(decel[i]), t]
        road = np.array([id(o.road) for o in obs])
        wl = np.array([FOOTPRINT.get(o.cls, DEFAULT_FOOTPRINT) for o in obs])
        half = wl / 2
        centre = pos + np.stack([np.zeros(len(obs)), (0.5 - cfg["body_behind_frac"]) * wl[:, 1]], 1)
        n = len(obs)
        iu, ju = np.triu_indices(n, 1)
        sel = (road[iu] == road[ju]) & ok[iu] & ok[ju]
        iu, ju = iu[sel], ju[sel]
        if len(iu):
            d = centre[ju] - centre[iu]
            near = np.hypot(d[:, 0], d[:, 1]) <= cfg["range_m"] + 0.5 * (half[iu, 1] + half[ju, 1])
            iu, ju, d = iu[near], ju[near], d[near]
        by_tid = {o.tid: o for o in obs}
        info = {}
        if len(iu):
            u = vel[ju] - vel[iu]
            h = half[iu] + half[ju]
            gapv = np.maximum(np.abs(d) - h, 0.0)
            gap = np.hypot(gapv[:, 0], gapv[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (-h - d) / u, (h - d) / u
            moving = np.abs(u) > 1e-6
            inside = np.abs(d) < h
            tin = np.where(moving, np.minimum(t1, t2), np.where(inside, -INF, INF))
            tout = np.where(moving, np.maximum(t1, t2), INF)
            tin_m, tout_m = tin.max(1), tout.min(1)
            ttc = np.where((tin_m < tout_m) & (tout_m > 0), np.maximum(tin_m, 0.0), INF)
            nvec = np.sign(d) * gapv
            nn = np.hypot(nvec[:, 0], nvec[:, 1])
            nvec = np.where((nn < 1e-6)[:, None], d, nvec)
            nn = np.maximum(np.hypot(nvec[:, 0], nvec[:, 1]), 1e-6)
            closing = -(nvec * u).sum(1) / nn
            dec = np.maximum(decel[iu], decel[ju])
            for k in np.nonzero((gap <= cfg["watch_gap_m"]) & (closing >= 0.6 * cfg["min_closing_mps"]))[0]:
                a, b = obs[iu[k]].tid, obs[ju[k]].tid
                w_ = self.watch.setdefault((a, b) if a < b else (b, a), [t, 0.0])
                w_[0], w_[1] = t, max(w_[1], float(closing[k]))
            cond = (ttc <= cfg["ttc_max_s"]) & (closing >= cfg["min_closing_mps"]) & (dec >= cfg["min_decel_mps2"])
            for k in np.nonzero(cond | (gap <= 0.0))[0]:
                ia, ib = obs[iu[k]], obs[ju[k]]
                key = (ia.tid, ib.tid) if ia.tid < ib.tid else (ib.tid, ia.tid)
                if t < self.cool.get(ia.tid, -INF) or t < self.cool.get(ib.tid, -INF):
                    continue
                inc = self.inc.setdefault(key, {"hits": 0, "near": None, "contact": None, "last": t, "still": None})
                inc["last"] = t
                if cond[k]:
                    inc["hits"] += 1
                    if inc["hits"] >= cfg["min_hits"] and (inc["near"] is None or ttc[k] < inc["near"]["ttc"]):
                        first = inc["near"]["t"] if inc["near"] else t
                        inc["near"] = {"t": first, "ttc": float(ttc[k]), "closing": float(closing[k]),
                                       "decel": float(dec[k]), "gap": float(gap[k]),
                                       "subject": ia.tid if decel[iu[k]] >= decel[ju[k]] else ib.tid}
                if (gap[k] <= 0.0 and inc["contact"] is None and not (ia.areas | ib.areas) & self.NO_COLLISION_AREAS
                        and not ia.road.stopping_allowed
                        and self.watch.get(key, (0, 0.0))[1] >= cfg["min_closing_mps"]):
                    inc["contact"] = {"t": t, "closing": self.watch[key][1],
                                      "subject": ia.tid if speed[iu[k]] >= speed[ju[k]] else ib.tid,
                                      "photo": self._photo(ctx, ia, ib)}
            for k in np.nonzero(gap <= cfg["stasis_gap_m"] + 1.0)[0]:
                a, b = obs[iu[k]].tid, obs[ju[k]].tid
                info[(a, b) if a < b else (b, a)] = (float(gap[k]), float(closing[k]))
        tid_i = {o.tid: i for i, o in enumerate(obs)}
        self._advance(ctx, t, by_tid, tid_i, speed, decel, info)
        for k in [k for k, v in self.watch.items() if t - v[0] > cfg["closing_memory_sec"]]:
            del self.watch[k]
        for tid in [k for k, v in self.peak_decel.items() if t - v[1] > cfg["decel_memory_sec"]]:
            del self.peak_decel[tid]

    def _photo(self, ctx, a, b):
        if ctx.frame is None or not getattr(ctx.frame, "size", 0):
            return None
        return pair_photo(ctx.frame, (a.bbox, b.bbox))

    # ------------------------------------------------------------------ incidents -> events
    def _advance(self, ctx, t, by_tid, tid_i, speed, decel, info):
        cfg = self.cfg
        for key, inc in list(self.inc.items()):
            a, b = key
            if t - inc["last"] > cfg["lost_sec"] + (cfg["stop_within_sec"] + cfg["stasis_sec"] if inc["contact"] else 0):
                del self.inc[key]
                continue
            con, near = inc["contact"], inc["near"]
            if con is not None:
                if a in tid_i and b in tid_i:
                    inc["last"] = t
                    still = speed[tid_i[a]] < cfg["still_mps"] and speed[tid_i[b]] < cfg["still_mps"]
                    inc["still"] = (inc["still"] or t) if still else None
                    gap = info.get(key, (INF, 0.0))[0]
                    if inc["still"] is not None and t - inc["still"] >= cfg["stasis_sec"] and gap <= cfg["stasis_gap_m"]:
                        self._emit(ctx, by_tid, key, "collision", con["subject"], inc, t - inc["still"], con["t"])
                        continue
                if t - con["t"] > cfg["stop_within_sec"] + cfg["stasis_sec"] + 1.0:
                    if near is not None:
                        self._emit(ctx, by_tid, key, "near_collision", near["subject"], inc, 0.0, near["t"])
                    else:
                        del self.inc[key]
            elif near is not None and t - near["t"] >= cfg["contact_wait_sec"]:
                self._emit(ctx, by_tid, key, "near_collision", near["subject"], inc, 0.0, near["t"])

    def _emit(self, ctx, by_tid, key, kind, subject, inc, still_s, started):
        cfg = self.cfg
        del self.inc[key]
        other = key[0] if subject == key[1] else key[1]
        if subject in self.reported and other not in self.reported:
            subject, other = other, subject
        so, oo = by_tid.get(subject), by_tid.get(other)
        if so is None or oo is None:
            return
        near, con = inc["near"], inc["contact"]
        pk = self.peak_decel.get(subject, [0.0])[0]
        details = {"kind": kind,
                   "ttc_s": round(near["ttc"], 2) if near else 0.0,
                   "closing_speed_mps": round((con or near)["closing"], 1),
                   "decel_mps2": round(max(pk, self.peak_decel.get(other, [0.0])[0], near["decel"] if near else 0.0), 1),
                   "other_tid": other, "other_cls": oo.cls,
                   "gap_m": round(near["gap"], 2) if near else 0.0,
                   "decided_by": "vlm", "needs_review": True}
        if kind == "collision":
            details["stationary_s"] = round(still_s, 1)
        ev = self.event(so, details, started_s=started)
        photo = (con or {}).get("photo")
        if photo is not None:
            ev.extra_frames.append(("contact", con["t"], photo))
        self.pending[subject] = ev
        self.reported.add(subject)
        until = ctx.time_s + cfg["cooldown_sec"]
        self.cool[subject] = self.cool[other] = until

    def forget(self, tid):
        self.rings.drop(tid)
        self.cool.pop(tid, None)
        self.peak_decel.pop(tid, None)
        self.reported.discard(tid)
        for key in [k for k in self.inc if tid in k]:
            del self.inc[key]
        for key in [k for k in self.watch if tid in k]:
            del self.watch[key]
