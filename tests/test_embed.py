"""The appearance vector: the team's ReID model, the MobileNet fallback, and the rules that keep them apart.

What these pin down is the behaviour that must never regress when the model behind `Embedder` changes:
the trained weights fit the architecture key for key (a quietly half-loaded network gives plausible-looking
but worse vectors, which no downstream test would notice), a vector is unit length and repeatable, the
service degrades to MobileNet instead of failing when the weights are absent or broken, and vectors made
by different models are never compared - tracklets already in the store hold 576-number MobileNet vectors,
and a cosine between those and a 2048-number ReID vector would be arithmetic on unrelated number systems.

The ReID tests need `models/reid/*.pth`, which is not in git, and are skipped without it; each class loads
the model once. Nothing here says the ReID model is *better* than MobileNet on Dhaka footage - that is
measured on real video by `tools/reid_eval.py --embedder reid`, not asserted on synthetic crops.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import embed  # noqa: E402
from surveillance.embed import CameraBias, Embedder, similarity  # noqa: E402

HAVE_WEIGHTS = embed.weights_path() is not None


def vehicle(seed, n=2, size=(90, 60)):
    """`n` noisy crops of one synthetic vehicle (BGR): a body colour, a pale window band, a dark wheel line."""
    rng = np.random.default_rng(seed)
    body = rng.integers(40, 220, 3)
    out = []
    for _ in range(n):
        img = np.empty((size[0], size[1], 3), np.float32)
        img[:] = body
        img[int(.15 * size[0]):int(.40 * size[0])] = 235 - body // 4
        img[int(.80 * size[0]):int(.90 * size[0])] = 25
        out.append(np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8))
    return out


def quietly(fn, *a, **kw):
    """(result, what it printed)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **kw)
    return r, buf.getvalue()


@unittest.skipUnless(HAVE_WEIGHTS, "ReID weights are not in models/reid/")
class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from surveillance import reid_backbone
        cls.torch, cls.rb = torch, reid_backbone
        cls.path = embed.weights_path()
        cls.ck = torch.load(cls.path, map_location="cpu", weights_only=True)
        cls.net = reid_backbone.load(cls.path)
        cls.crops = vehicle(1, 3)
        cls.vec = reid_backbone.embed(cls.net, cls.crops)

    def test_the_trained_weights_fit_the_architecture_key_for_key(self):
        theirs, mine = self.ck["state_dict"], self.net.state_dict()
        self.assertEqual(set(theirs), set(mine))
        self.assertEqual(len(mine), 350)
        self.assertTrue(all(theirs[k].shape == mine[k].shape for k in mine))

    def test_the_checkpoint_records_what_it_was_trained_for(self):
        self.assertEqual((self.net.meta["backbone"], self.net.size), ("resnet50_ibn_a", 320))
        self.assertEqual(self.net.meta["scenes_held_out"], "S02")

    def test_a_checkpoint_missing_a_layer_is_refused(self):
        sd = dict(self.ck["state_dict"])
        del sd["neck.running_var"]
        with mock.patch.object(self.torch, "load", return_value=dict(self.ck, state_dict=sd)):
            with self.assertRaises(RuntimeError):
                self.rb.load("any.pth")

    def test_a_checkpoint_with_a_renamed_layer_is_refused(self):
        sd = {("neck.gamma" if k == "neck.weight" else k): v for k, v in self.ck["state_dict"].items()}
        with mock.patch.object(self.torch, "load", return_value=dict(self.ck, state_dict=sd)):
            with self.assertRaises(RuntimeError):
                self.rb.load("any.pth")

    def test_a_checkpoint_of_another_backbone_is_refused(self):
        with mock.patch.object(self.torch, "load", return_value=dict(self.ck, backbone="resnet101")):
            with self.assertRaises(ValueError):
                self.rb.load("any.pth")

    def test_a_checkpoint_with_non_finite_weights_is_refused(self):
        sd = {k: v.clone() for k, v in self.ck["state_dict"].items()}
        sd["neck.weight"][0] = float("nan")
        with mock.patch.object(self.torch, "load", return_value=dict(self.ck, state_dict=sd)):
            with self.assertRaises(ValueError):
                self.rb.load("any.pth")

    def test_every_crop_gets_a_2048_number_vector_of_unit_length(self):
        self.assertEqual(self.vec.shape, (3, 2048))
        self.assertEqual(self.vec.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(self.vec, axis=1), 1.0, atol=1e-5)

    def test_preprocessing_is_the_notebooks_test_transform(self):
        import cv2
        import torchvision.transforms as T
        from PIL import Image
        tf = T.Compose([T.Resize((320, 320)), T.ToTensor(), T.Normalize(self.rb.MEAN.tolist(), self.rb.STD.tolist())])
        rng = np.random.default_rng(0)
        for shape in ((57, 83, 3), (410, 233, 3)):          # one enlarged, one reduced
            crop = rng.integers(0, 256, shape, dtype=np.uint8)
            theirs = tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).numpy()
            np.testing.assert_allclose(self.rb.prepare([crop])[0], theirs, atol=1e-6)

    def test_a_blue_channel_is_read_as_blue(self):
        crop = np.zeros((40, 40, 3), np.uint8)
        crop[..., 0] = 255                                    # BGR: pure blue
        x = self.rb.prepare([crop])[0]
        self.assertGreater(x[2].mean(), 2.0)                  # RGB channel 2 is blue, normalised
        self.assertLess(x[0].mean(), 0.0)

    def test_the_same_crops_give_the_same_vectors(self):
        again = self.rb.embed(self.net, self.crops)
        np.testing.assert_array_equal(self.vec, again)

    def test_the_batch_size_does_not_change_the_vectors(self):
        single = self.rb.embed(self.net, self.crops, batch=1)
        np.testing.assert_allclose(single, self.vec, atol=1e-4)

    def test_flip_averaging_leaves_a_mirror_symmetric_crop_unchanged(self):
        half = self.crops[0][:, :30]
        sym = np.ascontiguousarray(np.hstack([half, half[:, ::-1]]))
        plain, mirrored = (self.rb.embed(self.net, [sym], flip=f)[0] for f in (False, True))
        self.assertGreater(float(plain @ mirrored), 0.999)

    def test_flip_averaging_gives_a_unit_vector_that_differs_from_the_plain_one(self):
        plain = self.rb.embed(self.net, self.crops[:1], flip=False)[0]
        flipped = self.rb.embed(self.net, self.crops[:1], flip=True)[0]
        self.assertAlmostEqual(float(np.linalg.norm(flipped)), 1.0, places=5)
        self.assertLess(float(plain @ flipped), 1.0 - 1e-6)


@unittest.skipUnless(HAVE_WEIGHTS, "ReID weights are not in models/reid/")
class ReIDEmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Embedder.reset()
        Embedder.use("reid")
        cls.a, cls.b = vehicle(1), vehicle(2)
        cls.va, cls.va2, cls.vb = Embedder.encode(cls.a), Embedder.encode(cls.a), Embedder.encode(cls.b)

    @classmethod
    def tearDownClass(cls):
        Embedder.reset()

    def test_the_team_model_is_what_was_loaded(self):
        self.assertEqual((Embedder.name, Embedder.dim), ("reid", 2048))
        self.assertTrue(Embedder.tag.startswith("reid:"))

    def test_a_vehicle_gets_one_2048_number_vector_of_unit_length(self):
        self.assertEqual(self.va.shape, (2048,))
        self.assertEqual(self.va.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(self.va)), 1.0, places=5)

    def test_encoding_is_deterministic(self):
        np.testing.assert_array_equal(self.va, self.va2)

    def test_an_identical_crop_scores_one_against_itself(self):
        self.assertAlmostEqual(similarity(self.va, self.va2), 1.0, places=5)

    def test_a_different_vehicle_scores_lower(self):
        self.assertLess(similarity(self.va, self.vb), 0.999)

    def test_the_vector_is_the_mean_of_the_crops_renormalised(self):
        from surveillance import reid_backbone
        net = Embedder._backends["reid"].net
        each = reid_backbone.embed(net, self.a, flip=embed.REID_FLIP)
        mean = each.mean(0) / np.linalg.norm(each.mean(0))
        np.testing.assert_allclose(self.va, mean, atol=1e-6)


class FallbackTests(unittest.TestCase):
    """The appearance cue is optional: whatever goes wrong with the ReID model, matching goes on."""

    def setUp(self):
        Embedder.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(Embedder.reset)
        patch = mock.patch.object(embed, "WEIGHTS_DIR", Path(self.tmp.name))
        patch.start()
        self.addCleanup(patch.stop)

    def mobilenet_or_skip(self, v):
        if v is None:
            self.skipTest("MobileNet weights are not cached and there is no network")

    def test_mobilenet_is_used_when_there_are_no_reid_weights(self):
        for policy in ("auto", "reid", None):
            Embedder.reset()
            Embedder.use(policy)
            v, _ = quietly(Embedder.encode, vehicle(1))
            self.mobilenet_or_skip(v)
            self.assertEqual((Embedder.name, Embedder.dim, v.shape), ("mobilenet", 576, (576,)), policy)
            self.assertEqual(Embedder.tag, embed.MOBILENET_TAG)           # what the store labels old rows with

    def test_asking_for_reid_without_weights_says_so_once(self):
        Embedder.use("reid")
        v, out = quietly(Embedder.encode, vehicle(1))
        self.mobilenet_or_skip(v)
        _, out2 = quietly(Embedder.encode, vehicle(1))
        self.assertIn("'reid' unavailable", out)
        self.assertEqual(out2, "")

    def test_a_file_that_is_not_a_checkpoint_falls_back_to_mobilenet(self):
        (Path(self.tmp.name) / "broken.pth").write_bytes(b"this is not a torch file")
        Embedder.use("auto")
        v, out = quietly(Embedder.encode, vehicle(1))
        self.mobilenet_or_skip(v)
        self.assertEqual(Embedder.name, "mobilenet")
        self.assertIn("'reid' unavailable", out)

    def test_a_checkpoint_that_does_not_fit_the_architecture_falls_back_to_mobilenet(self):
        import torch
        torch.save({"backbone": "resnet50_ibn_a", "size": 320, "state_dict": {"body.0.weight": torch.zeros(1)}},
                   Path(self.tmp.name) / "partial.pth")
        Embedder.use("auto")
        v, out = quietly(Embedder.encode, vehicle(1))
        self.mobilenet_or_skip(v)
        self.assertEqual(Embedder.name, "mobilenet")
        self.assertIn("'reid' unavailable", out)

    def test_choosing_mobilenet_never_touches_the_reid_weights(self):
        (Path(self.tmp.name) / "broken.pth").write_bytes(b"this is not a torch file")
        Embedder.use("mobilenet")
        v, out = quietly(Embedder.encode, vehicle(1))
        self.mobilenet_or_skip(v)
        self.assertEqual((Embedder.name, out), ("mobilenet", ""))

    def test_the_environment_can_choose_the_model(self):
        (Path(self.tmp.name) / "model.pth").write_bytes(b"stands in for weights that exist")
        with mock.patch.dict("os.environ", {"EMBED_MODEL": "mobilenet"}):
            self.assertEqual(Embedder._order(), ["mobilenet"])
        with mock.patch.dict("os.environ", {"EMBED_MODEL": "reid"}):
            self.assertEqual(Embedder._order(), ["reid", "mobilenet"])
        with mock.patch.dict("os.environ", {"EMBED_MODEL": "auto"}):
            self.assertEqual(Embedder._order(), ["reid", "mobilenet"])
        with mock.patch.dict("os.environ", {"EMBED_MODEL": "nonsense"}):
            unknown = Embedder._order()
        with mock.patch.dict("os.environ", {"EMBED_MODEL": ""}):
            self.assertEqual(unknown, Embedder._order())               # an unknown value changes nothing

    def test_use_overrides_the_environment(self):
        with mock.patch.dict("os.environ", {"EMBED_MODEL": "reid"}):
            Embedder.use("mobilenet")
            self.assertEqual(Embedder._order(), ["mobilenet"])
            Embedder.use(None)
            self.assertEqual(Embedder._order(), ["reid", "mobilenet"])

    def test_an_unknown_model_name_is_refused(self):
        with self.assertRaises(ValueError):
            Embedder.use("resnet152")

    def test_nothing_to_encode_gives_none(self):
        Embedder.use("mobilenet")
        self.assertIsNone(Embedder.encode([]))
        self.assertIsNone(Embedder.encode([None]))
        self.assertIsNone(Embedder.encode([np.zeros((5, 5, 3), np.uint8)]))       # too small to mean anything


class SimilarityTests(unittest.TestCase):
    @staticmethod
    def unit(width):
        v = np.random.default_rng(width).normal(size=width).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_vectors_of_different_widths_are_not_compared(self):
        mobilenet, reid = self.unit(576), self.unit(2048)
        self.assertTrue(np.isnan(similarity(mobilenet, reid)))
        self.assertTrue(np.isnan(similarity(reid, mobilenet)))

    def test_vectors_of_the_same_width_from_different_weights_are_not_compared(self):
        v = self.unit(2048)
        self.assertTrue(np.isnan(similarity(v, v, model_a="reid:old", model_b="reid:retrained")))

    def test_vectors_from_the_same_model_are_compared(self):
        v = self.unit(2048)
        self.assertAlmostEqual(similarity(v, v, model_a="reid:x", model_b="reid:x"), 1.0, places=5)
        self.assertAlmostEqual(similarity(v, v), 1.0, places=5)
        self.assertAlmostEqual(similarity(v, -v), -1.0, places=5)

    def test_a_vector_with_no_label_is_not_refused(self):
        v = self.unit(576)
        self.assertAlmostEqual(similarity(v, v, model_a="mobilenet:x", model_b=None), 1.0, places=5)

    def test_a_missing_vector_is_just_a_missing_cue(self):
        v = self.unit(576)
        self.assertTrue(np.isnan(similarity(None, v)))
        self.assertTrue(np.isnan(similarity(v, None)))


class CameraBiasTests(unittest.TestCase):
    def cameras(self, seed=0, n=60, d=64, shared=1.5):
        """Vehicles seen by two cameras; every vector in a camera carries that camera's own offset."""
        rng = np.random.default_rng(seed)
        unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
        own = unit(rng.normal(size=(n, d)))
        offset = unit(rng.normal(size=(2, d)))
        return [unit(own + shared * offset[c] + 0.1 * rng.normal(size=(n, d))).astype(np.float32) for c in (0, 1)]

    def test_the_cameras_shared_component_is_removed(self):
        for cam, vecs in enumerate(self.cameras()):
            bias = CameraBias(min_count=2)
            for v in vecs:
                bias.update(cam, v)
            mean = vecs.mean(0) / np.linalg.norm(vecs.mean(0))
            before = np.mean([v @ mean for v in vecs])
            after = np.mean([bias.centre(cam, v) @ mean for v in vecs])
            self.assertGreater(before, 0.5)
            self.assertLess(abs(after), 0.05)

    def test_centred_vectors_have_unit_length(self):
        vecs = self.cameras()[0]
        bias = CameraBias(min_count=2)
        for v in vecs:
            bias.update("a", v)
        out = np.stack([bias.centre("a", v) for v in vecs])
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        self.assertEqual(out.dtype, np.float32)

    def test_with_two_as_the_minimum_it_is_the_notebooks_formula(self):
        vecs = self.cameras()[0]
        theirs = vecs - vecs.mean(0)
        theirs /= np.maximum(np.linalg.norm(theirs, axis=1, keepdims=True), 1e-12)
        bias = CameraBias(min_count=2)
        for v in vecs:
            bias.update(0, v)
        np.testing.assert_allclose(np.stack([bias.centre(0, v) for v in vecs]), theirs, atol=1e-5)

    def test_a_camera_with_few_vectors_is_left_alone(self):
        vecs = self.cameras()[0]
        bias = CameraBias()                                       # default minimum
        for v in vecs[:5]:
            bias.update(0, v)
        np.testing.assert_array_equal(bias.centre(0, vecs[0]), vecs[0])
        self.assertIsNone(bias.mean(0, 64))

    def test_the_mean_is_kept_per_camera(self):
        a, b = self.cameras()
        bias = CameraBias(min_count=2)
        for v in a:
            bias.update("a", v)
        self.assertEqual((bias.count("a", 64), bias.count("b", 64)), (len(a), 0))
        np.testing.assert_array_equal(bias.centre("b", b[0]), b[0])        # camera b has no mean yet

    def test_vectors_of_different_widths_never_share_a_mean(self):
        bias = CameraBias(min_count=2)
        for v in self.cameras(d=64)[0]:
            bias.update(0, v)
        other = np.ones(576, np.float32) / 24
        np.testing.assert_array_equal(bias.centre(0, other), other)
        self.assertEqual(bias.count(0, 576), 0)

    def test_vectors_that_cannot_be_averaged_are_ignored(self):
        bias = CameraBias(min_count=1)
        self.assertFalse(bias.update(0, None))
        self.assertFalse(bias.update(0, np.array([1.0, np.nan], np.float32)))
        self.assertTrue(bias.update(0, np.array([1.0, 0.0], np.float32)))
        self.assertEqual(bias.count(0, 2), 1)
        self.assertIsNone(bias.centre(0, None))

    def test_centring_a_vector_equal_to_the_mean_does_not_divide_by_zero(self):
        bias = CameraBias(min_count=1)
        v = np.array([0.6, 0.8], np.float32)
        bias.update(0, v)
        np.testing.assert_array_equal(bias.centre(0, v), v)


if __name__ == "__main__":
    unittest.main()
