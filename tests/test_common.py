import unittest

from visionmetrics.common.utils import precision_recall_f1_scalar


class TestPrecisionRecallF1(unittest.TestCase):
    def test_perfect(self):
        report = precision_recall_f1_scalar(tp=5, fp=0, fn=0)
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_perfect_precision_partial_recall(self):
        report = precision_recall_f1_scalar(tp=5, fp=0, fn=3)
        self.assertEqual(report["Precision"], 1.0)
        self.assertEqual(report["Recall"], 0.625)
        self.assertAlmostEqual(report["F1"], 0.7692307692307692)

    def test_perfect_recall_partial_precision(self):
        report = precision_recall_f1_scalar(tp=5, fp=3, fn=0)
        self.assertEqual(report["Precision"], 0.625)
        self.assertEqual(report["Recall"], 1.0)
        self.assertAlmostEqual(report["F1"], 0.7692307692307692)

    def test_imperfect_equal_precision_recall(self):
        report = precision_recall_f1_scalar(tp=5, fp=3, fn=3)
        self.assertEqual(report, {
            "Precision": 0.625,
            "Recall": 0.625,
            "F1": 0.625
        })

    def test_imperfect_unequal_precision_recall(self):
        report = precision_recall_f1_scalar(tp=5, fp=3, fn=5)
        self.assertEqual(report, {
            "Precision": 0.625,
            "Recall": 0.5,
            "F1": 0.5555555555555556
        })

    def test_zero_recall_no_precision_samples(self):
        report = precision_recall_f1_scalar(tp=0, fp=0, fn=1)
        self.assertEqual(report, {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0
        })

    def test_zero_precision_no_recall_samples(self):
        report = precision_recall_f1_scalar(tp=0, fp=1, fn=0)
        self.assertEqual(report, {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0
        })

    def test_no_samples(self):
        report = precision_recall_f1_scalar(tp=0, fp=0, fn=0)
        self.assertEqual(report, {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0
        })


if __name__ == '__main__':
    unittest.main()
