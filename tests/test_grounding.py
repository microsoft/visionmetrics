import pathlib
import unittest

import numpy as np
from visionmetrics.grounding import Recall


class TestObjectGrounding(unittest.TestCase):
    def test_recall_simple(self):
        PREDICTIONS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        TARGETS = [(['a', 'b', 'c'], [[[0, 0, 100, 100]], [[50, 50, 100, 100]], [[10, 10, 50, 50]]])]
        metric = Recall()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 1.0)

    def test_recall_real_data(self):
        predictions_file = pathlib.Path(__file__).resolve().parent / 'data' / 'flickr30k.npz'
        data = np.load(predictions_file, allow_pickle=True)
        PREDICTIONS = data['predictions'].tolist()
        TARGETS = data['targets'].tolist()
        metric = Recall()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['recall@1'], 0.7632508833922261)
