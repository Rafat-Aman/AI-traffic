"""Camera-network topology (WP19): gates, links, the candidate filter, the two-link fallback, out-of-order
records, batch == online, the HTTP surface, and the SYNTHETIC measurements quoted in docs/TOPOLOGY.md.

There is no real second camera. Every number from `measure_*` comes from the generator in this file, which
makes up vehicles, travel times and appearance vectors; it says how the method behaves on that world, not on
Dhaka. Run `python -m tests.test_topology` to print the tables.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRATCH = tempfile.mkdtemp(prefix="topology-tests-")
os.environ.setdefault("TRAFFIC_DATA_DIR", _SCRATCH)
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from surveillance import api_map, geo as geo_mod, link, reid, topology  # noqa: E402
from surveillance.store import Store  # noqa: E402
from tests.test_citation import asgi_call  # noqa: E402
from tests.test_map import FakeAuth, LAT, LON, lat_for  # noqa: E402

# gates drawn the same way on every synthetic camera (fractions of the frame)
MAIN_IN = [[0.0, 0.3], [0.15, 0.3], [0.15, 0.7], [0.0, 0.7]]
MAIN_OUT = [[0.85, 0.3], [1.0, 0.3], [1.0, 0.7], [0.85, 0.7]]
SIDE_IN = [[0.3, 0.0], [0.7, 0.0], [0.7, 0.12], [0.3, 0.12]]
SIDE_OUT = [[0.3, 0.88], [0.7, 0.88], [0.7, 1.0], [0.3, 1.0]]
PT = {"main_in": (0.05, 0.5), "main_out": (0.95, 0.5), "side_in": (0.5, 0.05), "side_out": (0.5, 0.95)}
WINDOW = (10.0, 150.0)


def make_store(tmp):
    s = Store(Path(tmp) / "t.db")
    geo_mod.ensure_schema(s)
    return s


def add_cameras(store, n):
    return [store.add_camera(f"Cam{i}", "file", f"/v/{i}.mp4", f"L{i}") for i in range(n)]


def draw_gates(store, cam, which):
    """which: {'name': kind}, drawn at the fixed polygons above. Returns {name: gate id}."""
    polys = {"main_in": MAIN_IN, "main_out": MAIN_OUT, "side_in": SIDE_IN, "side_out": SIDE_OUT}
    out = topology.save_gates(store, cam, [{"name": n, "kind": k, "points": polys[n]} for n, k in which.items()])
    return {g["name"]: g["id"] for g in out["gates"]}


def chain(store, cams, window=WINDOW):
    """Cameras in a row: each has main_in / main_out / side_in / side_out; main_out of one links to main_in of
    the next. Returns {cam: {gate name: id}}."""
    gates = {c: draw_gates(store, c, {"main_in": "in", "main_out": "out", "side_in": "in", "side_out": "out"})
             for c in cams}
    for a, b in zip(cams, cams[1:]):
        topology.save_link(store, {"from_camera": a, "from_gate": gates[a]["main_out"], "to_camera": b,
                                   "to_gate": gates[b]["main_in"], "min_s": window[0], "max_s": window[1]})
    return gates


def tl(cam, vid, t_in, t_out, enter="main_in", leave="main_out", cls="Car", speed=7.0, vec=None, tid=None):
    t = reid.Tracklet(camera_id=cam, track_id=vid if tid is None else tid, cls=cls, t_cross=(t_in + t_out) / 2,
                      lateral=0.0, speed=speed, direction="incoming", width=1.8)
    t.t_enter, t.t_exit = t_in, t_out
    t.entry_xy, t.exit_xy = PT[enter], PT[leave]
    if vec is not None:
        t.embed = vec
    return t


def unit(rng, dim=64):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------- the synthetic world

def simulate(seed=0, n=150, cams=3, headway=3.0, miss=(), vanish=0.12, turn_off=0.12, join_main=0.15,
             join_side=0.10, noise=0.9, dwell=3.0, spacing_m=300.0):
    """Tracklets of `n` through-vehicles driving camera 0 -> 1 -> ... plus the traffic that confuses a
    matcher: vehicles that turn off at a camera through a side gate, vanish through the main gate (parked,
    lost), join at a later camera through the main gate with no earlier sighting, or through a side gate.

    Appearance is a 64-d unit vector per vehicle seen through noise of norm `noise` at every camera (the
    cosine of two views of one vehicle is about 1 / (1 + noise^2)). Cameras in `miss` see nobody.
    Returns (tracklets, truth) with truth[(camera index, track id)] = true vehicle id."""
    rng = np.random.default_rng(seed)
    out, truth, vid = [], {}, 0
    t = 1000.0
    classes = ["Car", "Car", "CNG", "Motorbike", "Bus", "Motorbike"]

    def seen(ci, v, base, cls, speed, t_in, enter, leave, lane):
        if ci in miss:
            return
        vec = base + rng.normal(0, noise / np.sqrt(len(base)), len(base))
        k = reid.Tracklet(camera_id=ci, track_id=v, cls=cls, t_cross=t_in + dwell / 2,
                          lateral=lane + rng.normal(0, 0.3),
                          speed=speed * (1 + rng.normal(0, 0.05)), direction="incoming",
                          width={"Car": 1.8, "CNG": 1.4, "Motorbike": 0.8, "Bus": 2.5}[cls])
        k.t_enter, k.t_exit = t_in, t_in + dwell
        k.entry_xy, k.exit_xy = PT[enter], PT[leave]
        k.embed = (vec / np.linalg.norm(vec)).astype(np.float32)
        out.append(k)
        truth[(ci, v)] = v

    for _ in range(n):
        t += rng.exponential(headway)
        cls = classes[rng.integers(len(classes))]
        speed = float(np.clip(rng.lognormal(np.log(7.0), 0.35), 2.5, 16.0))
        base = unit(rng)
        lane = float(rng.uniform(1.0, 8.0))
        kind = rng.choice(["through", "vanish", "turn", "join_main", "join_side"],
                          p=[1 - vanish - turn_off - join_main - join_side, vanish, turn_off, join_main, join_side])
        start = 0 if kind in ("through", "vanish", "turn") else int(rng.integers(1, cams))
        te = t
        for ci in range(start, cams):
            enter = "side_in" if (ci == start and kind == "join_side") else "main_in"
            last = ci == cams - 1
            leave = "main_out"
            if kind == "turn" and ci == start:
                leave = "side_out"
            seen(ci, vid, base, cls, speed, te - dwell / 2, enter, leave, lane)
            if kind in ("vanish", "turn") and ci == start:
                break
            te += spacing_m / speed * (1 + rng.normal(0, 0.12))     # crossing line to crossing line
            if last:
                break
        vid += 1
    out.sort(key=lambda k: (k.t_exit, k.camera_id))
    return out, truth


def fill(store, cam_ids, tracklets):
    """Store tracklets (camera index -> real camera id) in order; returns {row id: (camera index, track id)}."""
    key = {}
    for k in tracklets:
        ci = k.camera_id
        k.camera_id = cam_ids[ci]
        k.id = store.add_tracklet(k.camera_id, None, k)
        key[k.id] = (ci, k.track_id)
    return key


def score_ids(store, key, truth):
    """precision: of the pairs of consecutive sightings given one id, how many are one vehicle.
    kept: of the vehicles seen by 2+ cameras, how many have all their sightings under one id."""
    rows = store.tracklets()
    by_gid = defaultdict(list)
    gid_of = {}
    for r in rows:
        gid_of[key[r["id"]]] = r["global_id"]
        if r["global_id"] is not None:
            by_gid[r["global_id"]].append(r)
    pairs = right = 0
    for members in by_gid.values():
        members.sort(key=lambda r: r["t_cross"])
        for a, b in zip(members, members[1:]):
            pairs += 1
            right += truth[key[a["id"]]] == truth[key[b["id"]]]
    seen = defaultdict(list)
    for k, v in truth.items():
        seen[v].append(k)
    multi = [v for v, ks in seen.items() if len(ks) > 1]
    kept = sum(1 for v in multi if all(gid_of[k] is not None and gid_of[k] == gid_of[seen[v][0]] for k in seen[v]))
    return {"links": pairs, "precision": right / pairs if pairs else float("nan"),
            "vehicles": len(multi), "kept": kept / len(multi) if multi else float("nan")}


def run_batch(tracklets, truth, with_topology, cams=3, miss=()):
    tmp = tempfile.mkdtemp(prefix="topo-run-")
    try:
        store = make_store(tmp)
        ids = add_cameras(store, cams)
        if with_topology:
            chain(store, ids)
        key = fill(store, ids, [_copy(k) for k in tracklets])
        link.run_linking(store)
        res = score_ids(store, key, truth)
        store.db.close()
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_online(tracklets, truth, cams=3):
    """The same world, one tracklet at a time through OnlineLinker, with the LLRs the batch run would fit."""
    tmp = tempfile.mkdtemp(prefix="topo-run-")
    try:
        store = make_store(tmp)
        ids = add_cameras(store, cams)
        chain(store, ids)
        lk = link.OnlineLinker(store)
        # the colour cue is learned from the data, as learn_from would after a batch run
        probe = make_store(tmp + "-fit")
        pids = add_cameras(probe, cams)
        chain(probe, pids)
        fill(probe, pids, [_copy(k) for k in tracklets])
        lk.topo_llr = topology.fit_llr(link.load(probe), topology.load(probe))
        probe.db.close()
        key = {}
        for k in [_copy(k) for k in tracklets]:
            ci = k.camera_id
            k.camera_id = ids[ci]
            k.id = store.add_tracklet(k.camera_id, None, k)
            key[k.id] = (ci, k.track_id)
            lk.link(k)
        res = score_ids(store, key, truth)
        store.db.close()
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp + "-fit", ignore_errors=True)


def _copy(k):
    import dataclasses
    return dataclasses.replace(k)


def measure_accuracy(seeds=(0, 1, 2), n=150):
    """The table in docs/TOPOLOGY.md: batch matching, without and with topology, with and without a middle camera."""
    rows = []
    for label, miss in (("all three cameras see everyone", ()), ("middle camera missed every vehicle", (1,))):
        for mode in ("no topology", "topology", "topology, online"):
            with_topology = mode != "no topology"
            acc = []
            for seed in seeds:
                trk, truth = simulate(seed, n=n, miss=miss)
                acc.append(run_online(trk, truth) if mode.endswith("online") else run_batch(trk, truth, with_topology))
            rows.append({"world": label, "mode": mode, "topology": with_topology,
                         "links": int(np.mean([a["links"] for a in acc])),
                         "precision": float(np.nanmean([a["precision"] for a in acc])),
                         "kept": float(np.nanmean([a["kept"] for a in acc]))})
    return rows


def measure_scaling(sizes=(300, 1200, 4800), sample=60, seed=0):
    """Candidates per sighting as the store grows, at a constant arrival rate."""
    rows = []
    for n in sizes:
        tmp = tempfile.mkdtemp(prefix="topo-scale-")
        try:
            store = make_store(tmp)
            ids = add_cameras(store, 3)
            chain(store, ids)
            trk, _ = simulate(seed, n=n)
            fill(store, ids, trk)
            topo = topology.load(store)
            probes = [link.to_tracklet(r) for r in store.tracklets(camera_id=ids[1])[-sample:]]
            idx = topology.StoreIndex(store, topo)
            t0 = time.perf_counter()
            for b in probes:
                for r in topo.into(b.camera_id, b.entry_gate, 1):
                    idx.candidates(b, r)
            ms = (time.perf_counter() - t0) / len(probes) * 1000
            total = len(store.tracklets())
            window_600 = np.mean([len([x for x in store.tracklets(since=b.t_cross - 600, until=b.t_cross + 600)
                                       if x["camera_id"] != b.camera_id]) for b in probes])
            rows.append({"stored": total, "candidates": idx.n_candidates / len(probes),
                         "existing_600s_window": float(window_600), "ms_per_sighting": ms})
            store.db.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rows


# ---------------------------------------------------------------- tests

class Base(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = make_store(self._dir.name)
        self.addCleanup(self.store.db.close)


class GateTests(Base):
    def test_gates_are_assigned_from_entry_and_exit_points(self):
        a, = add_cameras(self.store, 1)
        g = draw_gates(self.store, a, {"main_in": "in", "main_out": "out", "side_out": "out"})
        t = tl(a, 1, 100, 103)
        t.id = self.store.add_tracklet(a, None, t)
        self.assertEqual((t.entry_gate, t.exit_gate), (g["main_in"], g["main_out"]))
        row = self.store.tracklets()[0]
        self.assertEqual((row["entry_gate"], row["exit_gate"]), (g["main_in"], g["main_out"]))
        self.assertAlmostEqual(row["entry_x"], 0.05)
        self.assertAlmostEqual(row["exit_x"], 0.95)
        self.assertEqual((row["t_enter"], row["t_exit"]), (100, 103))

    def test_a_point_just_outside_a_gate_belongs_to_it_but_a_far_one_does_not(self):
        a, = add_cameras(self.store, 1)
        g = draw_gates(self.store, a, {"main_in": "in"})
        gates = topology.gates_of(self.store, a)
        self.assertEqual(topology.gate_at((0.18, 0.5), gates), g["main_in"])      # 0.03 outside the edge
        self.assertIsNone(topology.gate_at((0.5, 0.5), gates))                    # the middle of the road

    def test_an_entry_point_is_never_put_in_an_exit_only_gate(self):
        a, = add_cameras(self.store, 1)
        draw_gates(self.store, a, {"main_out": "out"})
        t = tl(a, 1, 100, 103, enter="main_out", leave="main_out")
        self.store.add_tracklet(a, None, t)
        self.assertIsNone(t.entry_gate)
        self.assertIsNotNone(t.exit_gate)

    def test_a_camera_without_gates_stores_exactly_what_it_did(self):
        a, = add_cameras(self.store, 1)
        t = tl(a, 1, 100, 103)
        t.entry_xy = t.exit_xy = t.t_enter = t.t_exit = None
        self.store.add_tracklet(a, None, t)
        row = self.store.tracklets()[0]
        for col in ("entry_x", "exit_x", "t_enter", "t_exit", "entry_gate", "exit_gate"):
            self.assertIsNone(row[col], col)

    def test_the_schema_can_be_applied_twice_and_to_an_older_database(self):
        path = Path(self._dir.name) / "old.db"
        s = Store(path)
        s.db.execute("ALTER TABLE tracklets RENAME TO tracklets_new")       # an old table without the columns
        s.db.execute("CREATE TABLE tracklets AS SELECT id, camera_id, run_id, track_id, cls, zone, t_cross, video_t,"
                     " lateral, speed, width, direction, extrap_m, hist, lab, embed, embed_model, embed_reid,"
                     " embed_reid_model, global_id, link_source, created_at, class_conf FROM tracklets_new")
        s.db.commit()
        topology.ensure_schema(s)
        s._topology_ready = False
        topology.ensure_schema(s)
        have = {r[1] for r in s.db.execute("PRAGMA table_info(tracklets)")}
        self.assertTrue({"entry_gate", "exit_gate", "t_enter", "t_exit", "entry_x"} <= have)
        s.db.close()

    def test_editing_gates_retags_stored_tracklets_and_drops_their_links(self):
        a, b = add_cameras(self.store, 2)
        g = chain(self.store, [a, b])
        t = tl(a, 1, 100, 103)
        self.store.add_tracklet(a, None, t)
        out = topology.save_gates(self.store, a, [])                  # remove every gate of camera a
        self.assertEqual(out["removed_links"], 1)
        self.assertEqual(topology.list_links(self.store), [])
        self.assertIsNone(self.store.tracklets()[0]["exit_gate"])


class CollectorTests(unittest.TestCase):
    def test_the_collector_records_first_and_last_contact_point_and_times(self):
        class Road:
            name = "z"

            def to_metres(self, p):
                return (p[0] * 0.05, p[1] * 0.05)

        got = []
        col = reid.TrackletCollector(1, got.append, embed=False, min_samples=5)
        frame = np.zeros((100, 200, 3), np.uint8)
        for i in range(12):
            t = i * 0.2
            x1 = 10 + 10 * i
            col.observe(7, frame, (x1, 20, x1 + 20, 30 + 5 * i), "Car", Road(), (1.0, 2.0 + 0.4 * i), 20.0, 1,
                        t, 5000.0 + t, 8.0)
        col.finish_all()
        self.assertEqual(len(got), 1)
        k = got[0]
        self.assertAlmostEqual(k.entry_xy[0], (10 + 10) / 200.0)
        self.assertAlmostEqual(k.entry_xy[1], 30 / 100.0)
        self.assertAlmostEqual(k.exit_xy[0], (120 + 10) / 200.0)
        self.assertAlmostEqual(k.exit_xy[1], 85 / 100.0)
        self.assertAlmostEqual(k.t_enter, 5000.0)
        self.assertAlmostEqual(k.t_exit, 5000.0 + 11 * 0.2)


class WindowTests(Base):
    def place(self, cid, metres, rng=80):
        geo_mod.save(self.store, cid, {"lat": lat_for(metres), "lon": LON, "range_m": rng})

    def test_the_window_comes_from_the_map_distance(self):
        a, b = add_cameras(self.store, 2)
        self.place(a, 0)
        self.place(b, 500)
        w = topology.suggest_window(self.store, a, b)
        # 500 m apart, 160 m already in view: 340 m at 60 km/h = 20.4 s -> 20;  500 x 1.4 m at 5 km/h + 10 s = 514.0 -> 515
        self.assertEqual((w["min_s"], w["max_s"]), (20.0, 515.0))
        self.assertAlmostEqual(w["distance_m"], 500, delta=1)

    def test_a_far_pair_is_capped_and_an_unplaced_one_has_no_suggestion(self):
        a, b, c = add_cameras(self.store, 3)
        self.place(a, 0)
        self.place(b, 5000)
        w = topology.suggest_window(self.store, a, b)
        self.assertEqual(w["max_s"], 1800.0)
        self.assertTrue(w["capped"])
        self.assertIsNone(topology.suggest_window(self.store, a, c))

    def test_a_link_without_a_window_takes_the_suggestion_and_edits_win(self):
        a, b = add_cameras(self.store, 2)
        self.place(a, 0)
        self.place(b, 500)
        ga = draw_gates(self.store, a, {"main_out": "out"})
        gb = draw_gates(self.store, b, {"main_in": "in"})
        row = topology.save_link(self.store, {"from_camera": a, "from_gate": ga["main_out"], "to_camera": b,
                                              "to_gate": gb["main_in"]})
        self.assertEqual((row["min_s"], row["max_s"], row["window_source"]), (20.0, 515.0, "suggested"))
        row = topology.save_link(self.store, {"from_camera": a, "from_gate": ga["main_out"], "to_camera": b,
                                              "to_gate": gb["main_in"], "min_s": 30, "max_s": 90}, link_id=row["id"])
        self.assertEqual((row["min_s"], row["max_s"], row["window_source"]), (30.0, 90.0, "manual"))

    def test_bad_links_are_refused(self):
        a, b = add_cameras(self.store, 2)
        ga = draw_gates(self.store, a, {"main_in": "in", "main_out": "out"})
        gb = draw_gates(self.store, b, {"main_in": "in", "main_out": "out"})
        ok = {"from_camera": a, "from_gate": ga["main_out"], "to_camera": b, "to_gate": gb["main_in"],
              "min_s": 5, "max_s": 50}
        for bad in ({"to_camera": a}, {"from_gate": ga["main_in"]}, {"to_gate": gb["main_out"]},
                    {"min_s": 50, "max_s": 5}, {"max_s": 99999}, {"to_gate": ga["main_in"]}):
            with self.assertRaises(topology.TopologyError, msg=str(bad)):
                topology.save_link(self.store, {**ok, **bad})


class CandidateTests(Base):
    def test_only_the_right_gate_inside_the_window_is_a_candidate(self):
        a, b = add_cameras(self.store, 2)
        chain(self.store, [a, b])
        want = tl(a, 1, 100, 103)                                    # leaves main_out at 103
        wrong_gate = tl(a, 2, 100, 103, leave="side_out")
        too_early = tl(a, 3, 20, 23)                                 # left 100 s before; but window max is 150: inside
        too_old = tl(a, 4, -80, -77)                                 # left 190 s earlier than b enters at 113: outside
        too_close = tl(a, 5, 108, 111)                               # b enters at 120: gap 9 s < 10: outside
        for k in (want, wrong_gate, too_early, too_old, too_close):
            k.id = self.store.add_tracklet(a, None, k)
        mine = tl(b, 9, 120, 123)
        mine.id = self.store.add_tracklet(b, None, mine)
        topo = topology.load(self.store)
        idx = topology.StoreIndex(self.store, topo)
        got = [x.track_id for r in topo.into(b, mine.entry_gate, 1) for x in idx.candidates(mine, r)]
        self.assertEqual(sorted(got), [1, 3])
        # a sighting that came in through the side gate has no upstream candidates at all
        side = tl(b, 10, 120, 123, enter="side_in")
        side.id = self.store.add_tracklet(b, None, side)
        self.assertEqual(topo.into(b, side.entry_gate, 1), [])


class MatchingTests(Base):
    def vec(self, seed):
        return unit(np.random.default_rng(seed))

    def setUp(self):
        super().setUp()
        self.cams = add_cameras(self.store, 4)
        self.gates = chain(self.store, self.cams)
        self.linker = link.OnlineLinker(self.store)

    def put(self, *ks):
        for k in ks:
            k.id = self.store.add_tracklet(k.camera_id, None, k)
        return ks

    def gid(self, k):
        return self.store.q("SELECT global_id FROM tracklets WHERE id=?", (k.id,))[0]["global_id"]

    def test_the_vehicle_that_left_the_linked_gate_is_linked_online(self):
        v = self.vec(1)
        a, = self.put(tl(self.cams[0], 1, 100, 103, vec=v))
        decoy, = self.put(tl(self.cams[0], 2, 100, 103, leave="side_out", vec=v))     # same look, wrong gate
        b, = self.put(tl(self.cams[1], 1, 140, 143, vec=v))
        gid = self.linker.link(b)
        self.assertIsNotNone(gid)
        self.assertEqual(self.gid(a), gid)
        self.assertIsNone(self.gid(decoy))

    def test_a_different_class_or_a_late_arrival_is_not_linked(self):
        a, = self.put(tl(self.cams[0], 1, 100, 103, cls="Bus"))
        b, = self.put(tl(self.cams[1], 1, 140, 143, cls="Car"))
        self.assertIsNone(self.linker.link(b))
        c, = self.put(tl(self.cams[1], 2, 400, 403, cls="Bus"))                    # 297 s later: outside the window
        self.assertIsNone(self.linker.link(c))

    def test_two_look_alikes_in_the_window_are_not_guessed_between(self):
        v = self.vec(2)
        self.put(tl(self.cams[0], 1, 100, 103, vec=v), tl(self.cams[0], 2, 101, 104, vec=v))
        b, = self.put(tl(self.cams[1], 1, 140, 143, vec=v))
        self.assertIsNone(self.linker.link(b))

    def test_two_link_fallback_finds_a_vehicle_through_a_camera_that_missed_it(self):
        v = self.vec(3)
        a, = self.put(tl(self.cams[0], 1, 100, 103, vec=v))
        # no sighting at cams[1]; the vehicle shows up at cams[2] 40 + 40 s later (window 2 x [10,150] + 60 dwell)
        c, = self.put(tl(self.cams[2], 1, 183, 186, vec=v))
        gid = self.linker.link(c)
        self.assertIsNotNone(gid)
        self.assertEqual(self.gid(a), gid)
        self.assertEqual(self.store.q("SELECT link_source FROM tracklets WHERE id=?", (c.id,))[0]["link_source"],
                         "online-skip")

    def test_two_link_fallback_stops_at_the_dwell_limit(self):
        v = self.vec(4)
        a, = self.put(tl(self.cams[0], 1, 100, 103, vec=v))
        mid, = self.put(tl(self.cams[2], 1, 103 + 190, 103 + 193, vec=v))          # middle of the 20-360 s window
        self.assertIsNotNone(self.linker.link(mid))
        top = 103 + 150 + 150 + topology.SKIP_DWELL_S                       # the latest a two-link trip can arrive
        topo = topology.load(self.store)
        idx = topology.StoreIndex(self.store, topo)
        a2, = self.put(tl(self.cams[0], 2, 1000, 1003, vec=v))
        edge = tl(self.cams[2], 2, 1003 + 150 + 150 + topology.SKIP_DWELL_S, 1500, vec=v)
        late = tl(self.cams[2], 3, 1003 + 150 + 150 + topology.SKIP_DWELL_S + 1, 1500, vec=v)
        for k, want in ((edge, [a2.track_id]), (late, [])):
            k.entry_gate = self.gates[self.cams[2]]["main_in"]
            got = [x.track_id for r in topo.into(k.camera_id, k.entry_gate, 2) for x in idx.candidates(k, r)]
            self.assertEqual(got, want)

    def test_a_camera_that_did_see_the_vehicle_is_not_skipped_over(self):
        v = self.vec(5)
        a, = self.put(tl(self.cams[0], 1, 100, 103, vec=v))
        b, = self.put(tl(self.cams[1], 1, 140, 143, vec=v))
        self.assertIsNotNone(self.linker.link(b))
        c, = self.put(tl(self.cams[2], 1, 190, 193, vec=v))
        self.assertEqual(self.linker.link(c), self.gid(a))       # one link back (b), not a two-link jump to a
        # a second vehicle arriving at cams[2] cannot also claim a: it already has a sighting at cams[1]
        d, = self.put(tl(self.cams[2], 2, 192, 195, vec=v))
        self.assertIsNone(self.linker.link(d))

    def test_out_of_order_records_merge_and_the_first_camera_keeps_its_id(self):
        v = self.vec(6)
        # cameras 1 and 2 were processed first: their records got an id of their own (1)
        b, = self.put(tl(self.cams[1], 1, 140, 143, vec=v))
        self.assertIsNone(self.linker.link(b))
        c, = self.put(tl(self.cams[2], 1, 190, 193, vec=v))
        self.assertEqual(self.linker.link(c), 1)
        # then camera 0's record arrives: camera 1's sighting is asked again, picks it, and they share id 1
        a, = self.put(tl(self.cams[0], 1, 100, 103, vec=v))
        gid = self.linker.link(a)                                  # asks camera 1's sighting again: picks a
        self.assertEqual(gid, 1)
        self.assertEqual({self.gid(a), self.gid(b), self.gid(c)}, {1})

    def test_two_existing_ids_merge_into_the_first_cameras_and_the_old_one_stays_findable(self):
        v = self.vec(7)
        # the later half of the journey already has id 1 (b, c) ...
        b, = self.put(tl(self.cams[1], 1, 140, 143, vec=v))
        self.linker.link(b)
        c, = self.put(tl(self.cams[2], 1, 190, 193, vec=v))
        self.assertEqual(self.linker.link(c), 1)
        # ... while a was given id 2 earlier by its own sighting at cams[0] (a stand-in earlier link: a2)
        a, a2 = self.put(tl(self.cams[0], 1, 100, 103, vec=v), tl(self.cams[0], 5, 90, 93, leave="side_out", vec=v))
        self.store.set_global_ids([(a.id, 2), (a2.id, 2)], source="online")
        a.global_id = 2
        self.assertEqual(self.linker.link(a), 2)                   # a's id (2) belongs to the earlier first sighting
        self.assertEqual({self.gid(x) for x in (a, a2, b, c)}, {2})
        self.assertEqual(topology.resolve(self.store, 1), 2)
        self.assertEqual(topology.merged_from(self.store, 2), [1])

    def test_without_topology_the_online_path_is_unchanged(self):
        s2 = make_store(self._dir.name + "-x")
        self.addCleanup(s2.db.close)
        lk = link.OnlineLinker(s2)
        self.assertIsNone(lk.topology())
        cams = add_cameras(s2, 2)
        k = tl(cams[0], 1, 100, 103)
        k.id = s2.add_tracklet(cams[0], None, k)
        self.assertIsNone(lk.link(k))                               # no learned routes, no topology: no answer


class BatchOnlineTests(unittest.TestCase):
    def test_batch_and_online_give_the_same_vehicles_on_in_order_data(self):
        trk, truth = simulate(seed=4, n=120)
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d1, True)
        self.addCleanup(shutil.rmtree, d2, True)
        batch, online = make_store(d1), make_store(d2)
        self.addCleanup(batch.db.close)
        self.addCleanup(online.db.close)
        ids1, ids2 = add_cameras(batch, 3), add_cameras(online, 3)
        chain(batch, ids1)
        chain(online, ids2)
        key1 = fill(batch, ids1, [_copy(k) for k in trk])
        link.run_linking(batch)

        lk = link.OnlineLinker(online)
        lk.topo_llr = topology.fit_llr(link.load(batch), topology.load(batch))
        key2 = {}
        for k in [_copy(k) for k in trk]:
            ci = k.camera_id
            k.camera_id = ids2[ci]
            k.id = online.add_tracklet(k.camera_id, None, k)
            key2[k.id] = (ci, k.track_id)
            lk.link(k)

        def partition(store, key):
            groups = defaultdict(set)
            for r in store.tracklets():
                if r["global_id"] is not None:
                    groups[r["global_id"]].add(key[r["id"]])
            return {frozenset(g) for g in groups.values() if len(g) > 1}

        p1, p2 = partition(batch, key1), partition(online, key2)
        self.assertGreater(len(p1), 20, "the world produced almost no cross-camera vehicles")
        self.assertEqual(p1, p2)


class MeasurementTests(unittest.TestCase):
    """Loose guards on the synthetic numbers (the numbers themselves are in docs/TOPOLOGY.md)."""

    def test_topology_beats_no_topology_and_survives_a_missed_middle_camera(self):
        rows = measure_accuracy(seeds=(0,), n=150)
        by = {(r["world"][:6], r["topology"]): r for r in rows if not r["mode"].endswith("online")}
        full_no, full_yes = by[("all th", False)], by[("all th", True)]
        miss_no, miss_yes = by[("middle", False)], by[("middle", True)]
        # topology keeps many more ids; the defaults were tuned for precision >= 0.95 (a wrong id is worse than a new one)
        self.assertGreater(full_yes["kept"], full_no["kept"] + 0.3)
        self.assertGreater(full_yes["precision"], 0.9)
        self.assertGreater(miss_yes["kept"], miss_no["kept"] + 0.3)
        self.assertGreater(miss_yes["precision"], 0.9)

    def test_candidates_per_sighting_stay_bounded_as_the_store_grows(self):
        rows = measure_scaling(sizes=(150, 600), sample=30)
        self.assertGreater(rows[1]["stored"], 3.5 * rows[0]["stored"])
        self.assertLess(rows[1]["candidates"], 1.5 * rows[0]["candidates"] + 3)
        self.assertLess(rows[1]["candidates"], rows[1]["existing_600s_window"])


class DisplayIdTests(unittest.TestCase):
    def test_display_ids(self):
        self.assertEqual(topology.display_id(123), "DV-000123")
        self.assertEqual(topology.display_id(12345678), "DV-12345678")
        for text in ("DV-000123", "dv123", " 123 ", "DV000123"):
            self.assertEqual(topology.parse_display_id(text), 123)
        for text in ("", "DV-", "ABC", "12x"):
            self.assertIsNone(topology.parse_display_id(text))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = make_store(self._dir.name)
        self.addCleanup(self.store.db.close)
        self.a, self.b = add_cameras(self.store, 2)
        geo_mod.save(self.store, self.a, {"lat": lat_for(0), "lon": LON})
        geo_mod.save(self.store, self.b, {"lat": lat_for(500), "lon": LON})
        fa = FakeAuth()
        app = FastAPI()
        web = Path(self._dir.name) / "web"
        web.mkdir()
        (web / "map.html").write_text("<html>map</html>")
        app.include_router(api_map.make_router(self.store, fa.current_user, fa.require, "controller", "admin",
                                               web_dir=web, env={}))
        self.app = app

    def call(self, method, path, role=None, body=None):
        headers = {"x-role": role} if role else {}
        data = b""
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        status, _, out = asgi_call(self.app, method, path, headers, data)
        try:
            return status, json.loads(out)
        except ValueError:
            return status, out.decode()

    def gates(self, cid, role="admin"):
        return self.call("PUT", f"/api/topology/cameras/{cid}/gates", role, {"gates": [
            {"name": "north out", "kind": "out", "points": MAIN_OUT},
            {"name": "south in", "kind": "in", "points": MAIN_IN}]})

    def test_roles(self):
        for role, code in (("admin", 200), ("controller", 403), ("driver", 403), (None, 401)):
            self.assertEqual(self.gates(self.a, role)[0], code, role)
        for role, code in (("admin", 200), ("controller", 200), ("driver", 403), (None, 401)):
            self.assertEqual(self.call("GET", "/api/topology", role)[0], code, role)
        s, _ = self.call("POST", "/api/topology/links", "controller", {})
        self.assertEqual(s, 403)
        s, _ = self.call("DELETE", "/api/topology/links/1", "driver")
        self.assertEqual(s, 403)

    def test_admin_draws_gates_links_with_a_suggested_window_and_it_is_audited(self):
        s, ga = self.gates(self.a)
        s, gb = self.gates(self.b)
        gate = lambda out, name: next(g["id"] for g in out["gates"] if g["name"] == name)
        s, sug = self.call("GET", f"/api/topology/suggest?from={self.a}&to={self.b}", "controller")
        self.assertEqual((s, sug["min_s"], sug["max_s"]), (200, 20.0, 515.0))
        s, lk = self.call("POST", "/api/topology/links", "admin", {
            "from_camera": self.a, "from_gate": gate(ga, "north out"), "to_camera": self.b,
            "to_gate": gate(gb, "south in")})
        self.assertEqual((s, lk["min_s"], lk["max_s"], lk["window_source"]), (200, 20.0, 515.0, "suggested"))
        s, lk2 = self.call("PUT", f"/api/topology/links/{lk['id']}", "admin", {
            "from_camera": self.a, "from_gate": lk["from_gate"], "to_camera": self.b, "to_gate": lk["to_gate"],
            "min_s": 25, "max_s": 80})
        self.assertEqual((s, lk2["min_s"], lk2["max_s"], lk2["window_source"]), (200, 25.0, 80.0, "manual"))
        s, view = self.call("GET", "/api/topology", "controller")
        self.assertEqual(len(view["links"]), 1)
        self.assertEqual(view["can_edit"], False)
        self.assertEqual(len(view["gates"][str(self.a)]), 2)
        actions = {r["action"] for r in self.store.audit_trail()}
        self.assertTrue({"topology.gates", "topology.link.add", "topology.link.edit"} <= actions)
        s, _ = self.call("DELETE", f"/api/topology/links/{lk['id']}", "admin")
        self.assertEqual(s, 200)
        self.assertEqual(self.call("GET", "/api/topology", "admin")[1]["links"], [])

    def test_bad_input_is_a_400_not_a_crash(self):
        s, _ = self.call("PUT", f"/api/topology/cameras/{self.a}/gates", "admin",
                         {"gates": [{"name": "x", "kind": "up", "points": MAIN_IN}]})
        self.assertEqual(s, 400)
        s, _ = self.call("PUT", f"/api/topology/cameras/{self.a}/gates", "admin",
                         {"gates": [{"name": "x", "kind": "in", "points": [[0, 0], [2, 0], [1, 1]]}]})
        self.assertEqual(s, 400)
        self.assertEqual(self.call("PUT", "/api/topology/cameras/999/gates", "admin", {"gates": []})[0], 404)
        self.assertEqual(self.call("POST", "/api/topology/links", "admin", {"from_camera": 1})[0], 400)

    def test_a_merged_display_id_resolves_to_the_vehicle_that_kept_it(self):
        for i in range(2):
            self.store.add_tracklet(self.a, None, tl(self.a, i, 100 + i, 103 + i))
        self.store.set_global_ids([(1, 5), (2, 9)], source="online")
        topology.merge_vehicles(self.store, 5, 9, "test")
        s, out = self.call("GET", "/api/topology/vehicle?q=DV-000009", "controller")
        self.assertEqual((s, out["display_id"], out["merged"], out["sightings"]), (200, "DV-000005", True, 2))
        self.assertEqual(out["merged_from"], ["DV-000009"])
        self.assertEqual(self.call("GET", "/api/topology/vehicle?q=xyz", "controller")[0], 400)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        print("accuracy (batch, mean of 3 seeds x 150 through-vehicles)")
        for r in measure_accuracy(seeds=(0, 1, 2), n=150):
            print(f"  {r['world']:<38} {r['mode']:<17} links={r['links']:>4} "
                  f"precision={r['precision']:.3f} kept={r['kept']:.3f}")
        print("candidates per sighting")
        for r in measure_scaling():
            print(f"  stored={r['stored']:>6} candidates={r['candidates']:.1f} "
                  f"existing +-600s window={r['existing_600s_window']:.0f} ms/sighting={r['ms_per_sighting']:.2f}")
    else:
        unittest.main()
