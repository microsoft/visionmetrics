import unittest

from visionmetrics.grounding import Recall


class TestObjectGrounding(unittest.TestCase):

    def test_recall_top1_simple(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 80, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 0.6666666666666666)

    def test_number_predict_target_not_equal_raise_error(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]]), (['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]]])]
        metric = Recall()
        with self.assertRaises(ValueError):
            metric.update(PREDICTIONS, TARGETS)
            metric.compute()

    def test_number_phrase_bbox_not_equal_raise_error(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]]])]
        TARGETS = [(['a', 'b'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        with self.assertRaises(ValueError):
            metric.update(PREDICTIONS, TARGETS)
            metric.compute()
