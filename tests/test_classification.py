import unittest

import torch
from visionmetrics.classification import ThresholdAccuracy


class TestClassification(unittest.TestCase):

    TARGETS = [torch.tensor([[0, 1, 0], [1, 0, 1]]),
               torch.tensor([[1, 1, 0], [1, 0, 1]])]

    PREDICTIONS = [torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.8, 0.6]]),
                   torch.tensor([[0.2, 0.6, 0.7], [0.1, 0.3, 0.7]])]

    def test_threshold_accuracy(self):
        gts = [[0.333, 0.333, 0.333], [0.5, 0.5, 0.333]]

        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            for j, threshold in enumerate([0.3, 0.5, 0.7]):
                metric = ThresholdAccuracy(num_labels=3, threshold=threshold)
                metric.update(predictions, targets)
                metric_output = metric.compute()[f"accuracy_thres={threshold}"].item()
                self.assertAlmostEqual(metric_output, gts[i][j], places=3)


if __name__ == '__main__':
    unittest.main()
