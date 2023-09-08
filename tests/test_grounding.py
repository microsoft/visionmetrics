import unittest

from visionmetrics.grounding import Recall


class TestObjectGrounding(unittest.TestCase):
    def test_recall_value(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 1.0)
