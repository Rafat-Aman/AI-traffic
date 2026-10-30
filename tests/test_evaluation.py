"""Metric tests on tiny hand-computed cases.   ./.venv/bin/python -m unittest tests.test_evaluation -v"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import detection, events, retrieval, stats, tracking  # noqa: E402
from evaluation.assign import min_cost_assign  # noqa: E402


def box(x, y=0, w=10, h=10):
    return (x, y, x + w, y + h)


class AssignTests(unittest.TestCase):
    def test_global_optimum_beats_greedy(self):
        # greedy would take (0,0)=1 then be forced into (1,1)=100; the optimum is (0,1)+(1,0)=2+2
        pairs = sorted(min_cost_assign([[1, 2], [2, 100]]))
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_limit_leaves_expensive_pairs_unmatched(self):
        self.assertEqual(min_cost_assign([[0.2, 0.9], [0.8, 0.95]], limit=0.5), [(0, 0)])


class DetectionTests(unittest.TestCase):
    def test_ap_hand_computed(self):
        # 2 GT; preds by conf: TP, FP, TP -> recall .5,.5,1  precision 1,.5,2/3 -> envelope 1,2/3,2/3
        # AP = .5*1 + .5*2/3 = 5/6
        gt = {"a": [box(0)], "b": [box(0)]}
        preds = [("a", .9, box(0)), ("a", .8, box(50)), ("b", .7, box(1))]
        self.assertAlmostEqual(detection.average_precision(gt, preds), 5 / 6, places=9)

    def test_edge_cases(self):
        self.assertEqual(detection.average_precision({"a": [box(0)]}, []), 0.0)
        self.assertIsNone(detection.average_precision({}, [("a", .9, box(0))]))

    def test_duplicate_prediction_is_a_false_positive(self):
        gt = {"a": [box(0)]}
        preds = [("a", .9, box(0)), ("a", .8, box(0))]   # second matches the already-used GT -> FP, AP still 1
        self.assertAlmostEqual(detection.average_precision(gt, preds), 1.0)

    def test_iou_threshold(self):
        # boxes shifted by 5 of width 10: inter 50, union 150 -> IoU 1/3 < .5
        self.assertAlmostEqual(detection.iou_xyxy(box(0), box(5)), 1 / 3)
        self.assertEqual(detection.average_precision({"a": [box(0)]}, [("a", .9, box(5))]), 0.0)

    def test_map_skips_class_without_gt(self):
        gt = [("a", 0, box(0)), ("a", 1, box(30))]
        pr = [("a", 0, .9, box(0)), ("a", 1, .9, box(80)), ("a", 2, .9, box(0))]
        r = detection.evaluate(gt, pr)
        self.assertAlmostEqual(r["map"], 0.5)        # class0 AP 1, class1 AP 0, class2 has no GT -> excluded
        self.assertIsNone(r["per_class"][2]["ap"])

    def test_yolo_loader(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "i.txt").write_text("1 0.5 0.5 0.2 0.4 0.77\n")
            out = detection.load_yolo_dir(d, with_conf=True)
            self.assertEqual(out[0][:2], ("i", 1))
            self.assertAlmostEqual(out[0][2], .77)
            self.assertEqual([round(v, 3) for v in out[0][3]], [.4, .3, .6, .7])


class TrackingTests(unittest.TestCase):
    def rows(self, ident_by_frame, frames=None):
        return [(f, i, box(0)) for f, i in ident_by_frame.items() if i is not None]

    def test_perfect(self):
        g = self.rows({1: 1, 2: 1, 3: 1})
        r = tracking.evaluate(g, g)
        self.assertEqual((r["mota"], r["id_switches"], r["fragmentations"], r["idf1"]), (1.0, 0, 0, 1.0))
        self.assertAlmostEqual(r["motp_iou"], 1.0)

    def test_id_switch(self):
        # GT id 1 for 4 frames; tracker id 1 then id 2 -> IDSW 1; MOTA = 1-1/4; IDTP 2 -> IDF1 = 4/(4+2+2) = .5
        r = tracking.evaluate(self.rows({1: 1, 2: 1, 3: 1, 4: 1}), self.rows({1: 1, 2: 1, 3: 2, 4: 2}))
        self.assertEqual(r["id_switches"], 1)
        self.assertAlmostEqual(r["mota"], 0.75)
        self.assertAlmostEqual(r["idf1"], 0.5)
        self.assertEqual(r["fragmentations"], 0)

    def test_fragmentation_and_miss(self):
        # tracker misses frame 3 only: FN 1, one fragmentation, same id -> no switch. MOTA = 1-1/5; IDTP 4 -> 8/9
        r = tracking.evaluate(self.rows({f: 1 for f in range(1, 6)}), self.rows({1: 1, 2: 1, 4: 1, 5: 1}))
        self.assertEqual((r["fn"], r["fp"], r["id_switches"], r["fragmentations"]), (1, 0, 0, 1))
        self.assertAlmostEqual(r["mota"], 0.8)
        self.assertAlmostEqual(r["idf1"], 8 / 9)
        self.assertAlmostEqual(r["idp"], 1.0)
        self.assertAlmostEqual(r["idr"], 0.8)

    def test_false_positive_and_leading_miss_is_not_fragmentation(self):
        g = [(2, 1, box(0)), (3, 1, box(0))]
        h = [(1, 9, box(100)), (3, 1, box(0))]     # frame 1: FP only; GT starts untracked at frame 2 -> FN, then tracked
        r = tracking.evaluate(g, h)
        self.assertEqual((r["fp"], r["fn"], r["fragmentations"]), (1, 1, 0))
        self.assertAlmostEqual(r["mota"], 0.0)

    def test_idf1_global_matching_not_per_frame(self):
        # two GT objects swap which hypothesis covers them half way; global matching keeps the best 1:1 pairing
        g = [(f, 1, box(0)) for f in range(1, 5)] + [(f, 2, box(100)) for f in range(1, 5)]
        h = [(1, 7, box(0)), (2, 7, box(0)), (3, 8, box(0)), (4, 8, box(0)),
             (1, 8, box(100)), (2, 8, box(100)), (3, 7, box(100)), (4, 7, box(100))]
        r = tracking.idf1(g, h)
        self.assertEqual(r["idtp"], 4)             # best 1:1: (1->7,2->8) gives 2+2 = 4 of 8
        self.assertAlmostEqual(r["idf1"], 0.5)
        self.assertEqual(tracking.clear_mot(g, h)["id_switches"], 2)

    def test_load_mot(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1,3,10,20,30,40,1,-1,-1,-1\n2,3,11,20,30,40,1\n")
        rows = tracking.load_mot(f.name)
        self.assertEqual(rows[0], (1, 3, (10.0, 20.0, 40.0, 60.0)))
        self.assertEqual(len(rows), 2)


class EventTests(unittest.TestCase):
    GT = [{"code": "V1", "start": 10, "end": 20}, {"code": "V2", "start": 30, "end": 32}]
    PR = [{"code": "V1", "start": 12, "end": 22}, {"code": "V1", "start": 100, "end": 105},
          {"code": "V2", "start": 50, "end": 51}]

    def test_tolerance_mode(self):
        r = events.evaluate(self.GT, self.PR, mode="tolerance", tol=3)
        m = r["micro"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 2, 1))
        self.assertAlmostEqual(m["precision"], 1 / 3)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.4)
        self.assertEqual(r["per_code"]["V2"]["tp"], 0)

    def test_iou_mode(self):
        # [10,20] vs [12,22]: inter 8, union 12 -> 2/3
        self.assertEqual(events.evaluate(self.GT, self.PR, mode="iou", thr=0.6)["micro"]["tp"], 1)
        self.assertEqual(events.evaluate(self.GT, self.PR, mode="iou", thr=0.7)["micro"]["tp"], 0)

    def test_one_to_one_and_class_must_agree(self):
        gt = [{"code": "V1", "start": 10, "end": 10, "cls": "Bus"}]
        pr = [{"code": "V1", "start": 10, "end": 10, "cls": "Car"}, {"code": "V1", "start": 11, "end": 11, "cls": "Bus"},
              {"code": "V1", "start": 11.5, "end": 11.5, "cls": "Bus"}]
        r = events.evaluate(gt, pr, tol=3)["micro"]
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 2, 0))

    def test_empty_prediction_has_no_precision(self):
        r = events.evaluate(self.GT, [])["micro"]
        self.assertIsNone(r["precision"])
        self.assertEqual(r["recall"], 0.0)

    def test_csv_loader_accepts_system_export(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("id,code,class,video_time_s\n5,v2,Car,74.5\n")
        e = events.load_events(f.name)
        self.assertEqual((e[0]["code"], e[0]["start"], e[0]["end"], e[0]["cls"]), ("V2", 74.5, 74.5, "Car"))


class RetrievalTests(unittest.TestCase):
    R = {"q1": ["a", "b", "c"], "q2": ["x", "y"], "q3": ["m", "n"]}
    REL = {"q1": {"b"}, "q2": {"x"}, "q3": {"z"}}

    def test_mrr_and_recall(self):
        self.assertAlmostEqual(retrieval.mrr(self.R, self.REL), (0.5 + 1 + 0) / 3)
        self.assertAlmostEqual(retrieval.recall_at_k(self.R, self.REL, 1), 1 / 3)
        self.assertAlmostEqual(retrieval.recall_at_k(self.R, self.REL, 2), 2 / 3)

    def test_recall_fraction_multi_answer(self):
        self.assertAlmostEqual(retrieval.recall_at_k_fraction({"q": ["a", "b", "c"]}, {"q": {"a", "c", "d", "e"}}, 3), 0.5)

    def test_query_without_relevant_items_skipped(self):
        self.assertAlmostEqual(retrieval.mrr({"q": ["a"], "r": ["a"]}, {"q": {"a"}}), 1.0)
        self.assertIsNone(retrieval.mrr({"r": ["a"]}, {}))


class CrossCameraTests(unittest.TestCase):
    def test_pair_precision_recall(self):
        gt = {("c1", 1): "A", ("c1", 2): "B", ("c2", 1): "A", ("c2", 2): "B"}
        pr = {("c1", 1): "A", ("c1", 2): "B", ("c2", 1): "A", ("c2", 2): "A"}
        r = retrieval.pair_precision_recall(pr, gt)
        # cross-camera pairs: (c1.1,c2.1) TP; (c1.1,c2.2) FP; (c1.2,c2.1) neither; (c1.2,c2.2) FN
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 0.5)

    def test_mtmc_idf1(self):
        gt = [(c, f, 1, box(0)) for c in ("c1", "c2") for f in (1, 2)]
        perfect = retrieval.mtmc_idf1(gt, gt)
        self.assertAlmostEqual(perfect["idf1"], 1.0)
        split = [(c, f, 10 if c == "c1" else 20, box(0)) for c in ("c1", "c2") for f in (1, 2)]   # one id per camera
        r = retrieval.mtmc_idf1(gt, split)
        self.assertEqual(r["idtp"], 2)                    # global id 1 can follow only one camera's id
        self.assertAlmostEqual(r["idf1"], 0.5)            # 2*2 / (2*2 + 2 + 2)


class StatsTests(unittest.TestCase):
    def test_spearman_ties_hand_computed(self):
        # same case as tests/test_impact.py: rho = 4.5 / sqrt(4.5 * 5)
        self.assertAlmostEqual(stats.spearman([1, 2, 2, 3], [1, 3, 2, 4]), 4.5 / math.sqrt(22.5), places=9)

    def test_undefined_is_none(self):
        self.assertIsNone(stats.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(stats.spearman([1], [1]))


if __name__ == "__main__":
    unittest.main()
