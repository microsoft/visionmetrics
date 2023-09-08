import pathlib
import unittest

import numpy as np
from visionmetrics.grounding import Recall


class TestObjectGrounding(unittest.TestCase):
    predictions_file = pathlib.Path(__file__).resolve().parent / 'data' / 'flickr30k.npz'
    data = np.load(predictions_file, allow_pickle=True)
    PREDICTIONS = data['predictions'].tolist()
    TARGETS = data['targets'].tolist()

    def test_recall_top1_simple(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 1.0)

    def test_recall_real_data(self):
        metric = Recall()
        metric.update(self.PREDICTIONS, self.TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 0.7632508833922261)

    def test_recall_top5_real_data(self):
        metric = Recall(k=5)
        metric.update(self.PREDICTIONS, self.TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@5'], 0.8208272708376637)

    def test_number_predict_target_not_equal_raise_error(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]]), (['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]]])]
        metric = Recall()
        with self.assertRaises(AssertionError):
            metric.update(PREDICTIONS, TARGETS)
            metric.compute()

    def test_number_phrase_bbox_not_equal_raise_error(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]]])]
        TARGETS = [(['a', 'b'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        with self.assertRaises(AssertionError):
            metric.update(PREDICTIONS, TARGETS)
            metric.compute()
