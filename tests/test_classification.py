import unittest

import torch
from visionmetrics.classification import ThresholdAccuracy


class TestClassification(unittest.TestCase):

    TARGETS = [
        torch.tensor([1, 0, 0, 0, 1, 1, 0, 0, 0, 1]),
        torch.tensor([1, 0, 2, 0, 1, 2, 0, 0, 0, 1, 2, 1, 2, 2, 0])]
    PREDICTIONS = [
        torch.tensor([[1, 0],
                      [0, 1],
                      [0.5, 0.5],
                      [0.1, 0.9],
                      [0.44, 0.56],
                      [0.09, 0.91],
                      [0.91, 0.09],
                      [0.37, 0.63],
                      [0.34, 0.66],
                      [0.89, 0.11]]),
        torch.tensor([[0.99, 0.01, 0],
                      [0, 0.99, 0.01],
                      [0.51, 0.49, 0.0],
                      [0.09, 0.8, 0.11],
                      [0.34, 0.36, 0.3],
                      [0.09, 0.90, 0.01],
                      [0.91, 0.06, 0.03],
                      [0.37, 0.60, 0.03],
                      [0.34, 0.46, 0.2],
                      [0.79, 0.11, 0.1],
                      [0.34, 0.16, 0.5],
                      [0.04, 0.56, 0.4],
                      [0.04, 0.36, 0.6],
                      [0.04, 0.36, 0.6],
                      [0.99, 0.01, 0.0]])]

    def test_threshold_accuracy(self):
        gts = [[0.4, 0.35, 0.2], [0.355555, 0.4, 0.133333]]
        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            for j, threshold in enumerate(['0.3', '0.5', '0.7']):
                metric = ThresholdAccuracy(float(threshold))
                metric.update(predictions, targets)
                metric_output = metric.compute()[f"accuracy_thres={threshold}"].item()
                self.assertAlmostEqual(metric_output, gts[i][j], places=5)


if __name__ == '__main__':
    unittest.main()
