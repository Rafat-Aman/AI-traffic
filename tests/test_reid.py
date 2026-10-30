"""Cross-camera vehicle re-identification: the cost model, the safeguards, and end-to-end matching.

The end-to-end tests run synthetic traffic whose true pairings are known, because that is the only way to
measure precision here: the project has no hand-labelled cross-camera ground truth. Synthetic traffic
flatters a matcher, so these are a floor, not a result to quote - what they really pin down is the
behaviour that must never regress, namely that the matcher refuses rather than guesses when the vehicles
cannot be told apart.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import reid  # noqa: E402
from surveillance.reid import Tracklet, link_all  # noqa: E402

WIDTH = {"Car": 1.7, "CNG": 1.4, "Motorbike": 0.7, "Bus": 2.5, "Rickshaw": 1.2}


def colour(rng):
    h = np.abs(rng.normal(size=192))
    return (h / h.sum()).astype(np.float32), rng.normal(size=6).astype(np.float32)


def traffic(seed=0, n=60, headway=1.2, gap=120.0, l_a=10.0, l_b=-1.0, classes=None,
            identical=False, strays=20):
    """Two cameras, `gap` metres apart, with known pairings. Returns (tracklets, {src id: dst id})."""
    rng = np.random.default_rng(seed)
    classes = classes or ["Car", "CNG", "Motorbike", "Bus", "Rickshaw"]
    shared = colour(rng)
    out, truth, t = [], {}, 0.0
    for k in range(n):
        cls = classes[rng.integers(len(classes))]
        h, lab = shared if identical else colour(rng)
        t += abs(rng.normal(headway, headway * 0.4))
        sp = float(np.clip(rng.normal(8.0, 2.5), 2.0, 18.0))
        lat = float(rng.uniform(0.5, 9.5))
        a = Tracklet(1, k, cls, t, lat, sp, "outgoing", WIDTH[cls] * rng.normal(1, 0.05),
                     zone="A", hist=h, lab=lab)
        hb = h if identical else (lambda x: (x / x.sum()).astype(np.float32))(
            np.abs(h + rng.normal(0, h.std() * 0.15, 192)))
        b = Tracklet(2, 1000 + k, cls, t + (gap / sp) * rng.normal(1, 0.04) + rng.normal(0, 0.4),
                     l_a + l_b * lat + rng.normal(0, 0.5), sp * rng.normal(1, 0.10), "incoming",
                     WIDTH[cls] * rng.normal(1, 0.05), zone="B", hist=hb,
                     lab=lab if identical else lab + rng.normal(0, 0.2, 6).astype(np.float32))
        out += [a, b]
        truth[id(a)] = id(b)
    for k in range(strays):          # vehicles only one camera ever sees: most traffic at a junction
        cls = classes[rng.integers(len(classes))]
        # In the indistinguishable-fleet case these must look the same too, or they hand the matcher a
        # colour cue it would not have in reality and the scenario stops testing what it claims to.
        h, lab = shared if identical else colour(rng)
        cam, d = (1, "outgoing") if k % 2 else (2, "incoming")
        out.append(Tracklet(cam, 9000 + k, cls, rng.uniform(0, max(t, 1)), rng.uniform(0.5, 9.5),
                            float(np.clip(rng.normal(8, 2.5), 2, 18)), d, WIDTH[cls],
                            zone="A" if cam == 1 else "B", hist=h, lab=lab))
    return out, truth


def score(tracklets, truth):
    links, groups = link_all(tracklets, log=lambda m: None)
    got = {id(a): id(b) for a, b, _ in links}
    correct = sum(1 for a, b in got.items() if truth.get(a) == b)
    precision = correct / len(got) if got else 1.0      # linking nothing is not a precision failure
    return len(got), correct, precision, correct / max(len(truth), 1), groups


class EndToEndTests(unittest.TestCase):
    def test_mixed_traffic_is_matched_accurately(self):
        n, correct, prec, rec, _ = score(*traffic(seed=0))
        self.assertGreaterEqual(prec, 0.95, f"only {correct}/{n} links correct")
        self.assertGreaterEqual(rec, 0.85)

    def test_it_is_stable_across_seeds(self):
        for seed in range(3):
            n, correct, prec, rec, _ = score(*traffic(seed=seed))
            self.assertGreaterEqual(prec, 0.95, f"seed {seed}: {correct}/{n}")

    def test_a_vehicle_seen_twice_becomes_one_identity(self):
        tracklets, truth = traffic(seed=1)
        _, groups = link_all(tracklets, log=lambda m: None)
        self.assertTrue(any(len(g) > 1 for g in groups.values()))
        for g in groups.values():                       # never two tracklets of one camera in one identity
            self.assertEqual(len(g), len({t.camera_id for t in g}))

    def test_strays_are_left_alone(self):
        """Vehicles only one camera saw must stay single: most traffic never reaches the other camera."""
        tracklets, truth = traffic(seed=2, n=40, strays=40)
        _, groups = link_all(tracklets, log=lambda m: None)
        singles = sum(1 for g in groups.values() if len(g) == 1)
        self.assertGreater(singles, 20)


class RefusalTests(unittest.TestCase):
    """The safeguards. Each was written against a measured failure of the matcher, not anticipated."""

    def test_an_indistinguishable_fleet_is_refused_not_guessed(self):
        """Every vehicle the same class and the exact same colour: nothing can separate them, so the only
        correct output is no links. Measured before the guard: 25 links, none of them right."""
        for headway in (1.0, 0.5, 0.25):
            tracklets, truth = traffic(seed=0, n=60, headway=headway, classes=["CNG"], identical=True)
            n, correct, prec, rec, _ = score(tracklets, truth)
            self.assertEqual(n, 0, f"headway {headway}s: linked {n} indistinguishable vehicles")

    def test_colour_rescues_a_dense_single_class_stream(self):
        """Same class, dense, but the colours differ: now it may and should link."""
        tracklets, truth = traffic(seed=0, n=60, headway=0.5, classes=["CNG"], identical=False)
        n, correct, prec, rec, _ = score(tracklets, truth)
        self.assertGreater(n, 10)
        self.assertGreaterEqual(prec, 0.9)

    def test_a_flow_is_never_discovered_in_both_directions(self):
        """Traffic cannot run both ways between the same two groups. Taking the reverse peak as well made
        every link backwards."""
        tracklets, _ = traffic(seed=0, n=60)
        flows, _ = reid.discover_flows(tracklets, log=lambda m: None)
        seen = set()
        for f in flows:
            pair = frozenset((f.src, f.dst))
            self.assertNotIn(pair, seen, "both directions of one flow were accepted")
            seen.add(pair)

    def test_cameras_cannot_be_implausibly_close(self):
        """A peak a few metres away is not a route between two cameras; it is the same traffic correlating
        with itself, and it used to outrank the real flow."""
        tracklets, _ = traffic(seed=0, n=60)
        flows, _ = reid.discover_flows(tracklets, log=lambda m: None)
        for f in flows:
            self.assertGreaterEqual(f.gap_m, reid.MIN_GAP_M)

    def test_unidentifiable_fires_only_when_colour_is_useless(self):
        off = reid.LLR()
        on = reid.LLR(pos=np.full(20, 0.9), neg=np.full(40, 0.2))
        src = [Tracklet(1, i, "CNG", i * 0.3, 2.0, 8.0, "outgoing") for i in range(20)]
        P = reid.default_params(120.0)
        self.assertIn("CNG", reid.unidentifiable(src, P, {("hist", "other"): off, ("lab", "other"): off}))
        self.assertNotIn("CNG", reid.unidentifiable(src, P, {("hist", "other"): on, ("lab", "other"): off}))

    def test_well_spaced_traffic_is_not_refused(self):
        src = [Tracklet(1, i, "Car", i * 30.0, 2.0, 8.0, "outgoing") for i in range(20)]
        off = reid.LLR()
        P = reid.default_params(120.0)
        self.assertEqual(reid.unidentifiable(src, P, {("hist", "other"): off, ("lab", "other"): off}), set())


class SingleClassTests(unittest.TestCase):
    """Dhaka traffic is dominated by one class, so a matcher that only works on a varied fleet is no use
    here. The flow gate used to be the sharpness of the transit-distance peak, which is biased against
    exactly this: when every vehicle is the same class every pair is a candidate, the histogram background
    rises, and a correct route scored 4.1 sigma and was discarded while explaining 138 pairs against 68."""

    def test_a_single_class_stream_is_still_matched(self):
        tracklets, truth = traffic(seed=0, n=90, classes=["CNG"])
        n, correct, prec, rec, _ = score(tracklets, truth)
        self.assertGreater(n, 30, "a one-class stream produced almost no links")
        self.assertGreaterEqual(prec, 0.9)

    def test_the_route_is_found_at_the_right_distance(self):
        tracklets, _ = traffic(seed=0, n=90, classes=["CNG"], gap=120.0)
        flows, _ = reid.discover_flows(tracklets, log=lambda m: None)
        self.assertTrue(flows, "no route discovered in single-class traffic")
        self.assertAlmostEqual(flows[0].gap_m, 120.0, delta=25.0)

    def test_a_fixed_distance_explains_a_real_route_better_than_a_fixed_delay(self):
        """The physics the route test rests on: vehicles that really travel between two cameras all cover
        the same distance, so a fast one arrives sooner. Streams that merely overlap in time share a delay
        instead. Built both ways from the same speeds, the evidence must point opposite ways."""
        rng = np.random.default_rng(0)
        t0 = np.cumsum(rng.normal(1.0, 0.3, 80))
        speeds = rng.uniform(4.0, 16.0, 80)         # a wide speed spread is what separates the two models
        src = [Tracklet(1, i, "Car", float(t), 2.0, float(v), "outgoing")
               for i, (t, v) in enumerate(zip(t0, speeds))]

        # a real route: 120 m covered, so the delay is 120/v and differs vehicle by vehicle
        route_dst = [Tracklet(2, 100 + i, "Car", a.t_cross + 120.0 / a.speed, 8.0, a.speed, "incoming")
                     for i, a in enumerate(src)]
        # a coincidence: everything reappears 8 s later however fast it was going
        delay_dst = [Tracklet(2, 200 + i, "Car", a.t_cross + 8.0, 8.0, a.speed, "incoming")
                     for i, a in enumerate(src)]

        g1, _ = reid.gap_peak(src, route_dst)
        _, route_r, delay_r = reid.is_a_route(src, route_dst, g1)
        g2, _ = reid.gap_peak(src, delay_dst)
        _, route_d, delay_d = reid.is_a_route(src, delay_dst, g2)
        self.assertGreater(route_r / max(delay_r, 1), route_d / max(delay_d, 1),
                           "a genuine route must look more like a fixed distance than a coincidence does")


class EmbeddingTests(unittest.TestCase):
    """The appearance vector exists for one case: a fleet whose colour is identical. Without it that fleet
    is refused outright (correctly); with it, the vehicles can be told apart again."""

    def fleet(self, seed, noise=None, headway=1.2, n=90):
        tracklets, truth = traffic(seed=seed, n=n, headway=headway, classes=["CNG"], identical=True)
        if noise is None:
            return tracklets, truth
        rng = np.random.default_rng(seed)
        vecs = {}
        for a, b in truth.items():
            v = rng.normal(size=576)
            v /= np.linalg.norm(v)
            w = v + rng.normal(0, noise, 576)
            w /= np.linalg.norm(w)
            vecs[a], vecs[b] = v.astype(np.float32), w.astype(np.float32)
        for t in tracklets:
            if id(t) in vecs:
                t.embed = vecs[id(t)]
            else:
                v = rng.normal(size=576)
                t.embed = (v / np.linalg.norm(v)).astype(np.float32)
        return tracklets, truth

    def test_identical_colour_is_refused_without_a_vector(self):
        n, _, _, _, _ = score(*self.fleet(0, noise=None))
        self.assertEqual(n, 0)

    def test_a_vector_recovers_an_identical_colour_fleet(self):
        n, correct, prec, rec, _ = score(*self.fleet(0, noise=0.35))
        self.assertGreater(n, 40, "the appearance vector bought nothing")
        self.assertGreaterEqual(prec, 0.9)

    def test_the_vector_is_carried_through_the_cost_model(self):
        a = Tracklet(1, 1, "Car", 0.0, 2.0, 8.0, "outgoing", 1.7)
        b = Tracklet(2, 2, "Car", 15.0, 8.0, 8.0, "incoming", 1.7)
        v = np.zeros(576, np.float32)
        v[0] = 1.0
        a.embed, b.embed = v, v
        P = dict(reid.default_params(120.0), l_a=10.0)
        self.assertAlmostEqual(reid.pair_terms(a, b, P)["embed"], 1.0, places=5)

    def test_matching_still_works_when_only_some_have_vectors(self):
        tracklets, truth = self.fleet(0, noise=0.35)
        for t in tracklets[::3]:
            t.embed = None                       # a camera that could not embed, mixed with ones that could
        n, correct, prec, _, _ = score(tracklets, truth)
        self.assertGreaterEqual(prec, 0.85)


class CostTests(unittest.TestCase):
    def tl(self, **kw):
        base = dict(camera_id=1, track_id=1, cls="Car", t_cross=0.0, lateral=2.0, speed=8.0,
                    direction="outgoing", width=1.7)
        base.update(kw)
        return Tracklet(**base)

    def test_a_different_class_is_never_matched(self):
        P = reid.default_params(120.0)
        a = self.tl(cls="Car")
        b = self.tl(camera_id=2, cls="Bus", t_cross=15.0)
        self.assertIsNone(reid.pair_terms(a, b, P))

    def test_time_must_run_forwards(self):
        P = reid.default_params(120.0)
        self.assertIsNone(reid.pair_terms(self.tl(t_cross=10.0), self.tl(camera_id=2, t_cross=5.0), P))

    def test_a_plausible_transit_beats_an_implausible_one(self):
        P = dict(reid.default_params(120.0), l_a=10.0)   # second camera mirrors: l2 = 10 - l1
        a = self.tl(lateral=2.0)
        good = reid.pair_terms(a, self.tl(camera_id=2, t_cross=15.0, lateral=8.0), P)
        far = reid.pair_terms(a, self.tl(camera_id=2, t_cross=17.0, lateral=8.0), P)
        self.assertIsNotNone(good, "a transit of gap/speed exactly should be allowed")
        self.assertLess(good["cost"], far["cost"])

    def test_the_lateral_mapping_is_enforced(self):
        """Crossing in the wrong lane on the far camera is rejected even when the timing is perfect."""
        P = dict(reid.default_params(120.0), l_a=10.0)
        a = self.tl(lateral=2.0)
        self.assertIsNotNone(reid.pair_terms(a, self.tl(camera_id=2, t_cross=15.0, lateral=8.0), P))
        self.assertIsNone(reid.pair_terms(a, self.tl(camera_id=2, t_cross=15.0, lateral=0.5), P))

    def test_a_wildly_wrong_transit_time_is_rejected_outright(self):
        P = reid.default_params(120.0)
        self.assertIsNone(reid.pair_terms(self.tl(), self.tl(camera_id=2, t_cross=400.0), P))

    def test_llr_switches_itself_off_when_a_cue_does_not_separate(self):
        rng = np.random.default_rng(0)
        same = reid.LLR(pos=rng.normal(0.5, 0.1, 30), neg=rng.normal(0.5, 0.1, 60))
        self.assertFalse(same.ok)
        self.assertEqual(same.cost(0.9), 0.0)
        real = reid.LLR(pos=rng.normal(0.9, 0.05, 30), neg=rng.normal(0.3, 0.1, 60))
        self.assertTrue(real.ok)
        self.assertLess(real.cost(0.9), real.cost(0.3))

    def test_assignment_is_one_to_one(self):
        P = reid.default_params(120.0)
        src = [self.tl(track_id=i, t_cross=i * 20.0, lateral=1.0 + i) for i in range(5)]
        dst = [self.tl(camera_id=2, track_id=100 + i, t_cross=i * 20.0 + 15.0, lateral=9.0 - i)
               for i in range(5)]
        matches, _ = reid.assign(src, dst, P)
        self.assertEqual(len({i for i, _, _ in matches}), len(matches))
        self.assertEqual(len({j for _, j, _ in matches}), len(matches))

    def test_an_empty_side_does_not_crash(self):
        self.assertEqual(reid.assign([], [], reid.default_params(120.0))[0], [])


class UnionTests(unittest.TestCase):
    def test_links_merge_transitively_across_three_cameras(self):
        u = reid.Union()
        u.union("a1", "b1")
        u.union("b1", "c1")
        u.find("d1")
        groups = u.groups()
        sizes = sorted(len(v) for v in groups.values())
        self.assertEqual(sizes, [1, 3], "a vehicle seen by three cameras must become one identity")


class SignatureTests(unittest.TestCase):
    def test_two_wheelers_are_read_from_the_rider_not_the_machine(self):
        import cv2
        crop = np.zeros((100, 60, 3), np.uint8)
        crop[:50] = (40, 200, 40)       # rider's jacket
        crop[50:] = (20, 20, 20)        # the machine
        self.assertGreater(reid.colour_region(crop, True)[..., 1].mean(),
                           reid.colour_region(crop, False)[..., 1].mean())

    def test_signature_of_similar_crops_scores_higher_than_of_different_ones(self):
        rng = np.random.default_rng(0)
        red = [np.full((60, 40, 3), (30, 30, 200), np.uint8) + rng.integers(0, 12, (60, 40, 3), dtype=np.uint8)
               for _ in range(3)]
        blue = [np.full((60, 40, 3), (200, 30, 30), np.uint8) for _ in range(3)]
        a = Tracklet(1, 1, "Car", 0, 0, 8, "outgoing")
        b = Tracklet(2, 2, "Car", 0, 0, 8, "incoming")
        c = Tracklet(2, 3, "Car", 0, 0, 8, "incoming")
        a.hist, a.lab = reid.signature(red, False)
        b.hist, b.lab = reid.signature(red, False)
        c.hist, c.lab = reid.signature(blue, False)
        self.assertGreater(reid.colour_similarity(a, b)[0], reid.colour_similarity(a, c)[0])
        self.assertGreater(reid.colour_similarity(a, b)[1], reid.colour_similarity(a, c)[1])

    def test_an_embedding_separates_vehicles_when_present(self):
        from surveillance.embed import Embedder, similarity
        red = [np.full((70, 50, 3), (30, 30, 200), np.uint8) for _ in range(3)]
        blue = [np.full((70, 50, 3), (200, 30, 30), np.uint8) for _ in range(3)]
        a, b, c = Embedder.encode(red), Embedder.encode(red), Embedder.encode(blue)
        if a is None:
            self.skipTest("embedding weights not available offline")
        self.assertEqual(a.shape, (Embedder.dim,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=3)
        self.assertGreater(similarity(a, b), similarity(a, c))

    def test_a_missing_embedding_is_just_a_missing_cue(self):
        from surveillance.embed import similarity
        self.assertTrue(np.isnan(similarity(None, np.zeros(576, np.float32))))

    def test_missing_signatures_are_not_fatal(self):
        a = Tracklet(1, 1, "Car", 0, 0, 8, "outgoing")
        b = Tracklet(2, 2, "Car", 0, 0, 8, "incoming")
        h, lab, emb = reid.colour_similarity(a, b)
        self.assertTrue(np.isnan(h) and np.isnan(lab) and np.isnan(emb))

    def test_a_tiny_crop_yields_no_signature(self):
        self.assertIsNone(reid.hsv_hist(np.zeros((3, 3, 3), np.uint8)))
        self.assertIsNone(reid.lab_stats(np.zeros((3, 3, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
