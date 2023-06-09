import unittest

import torch

from visionmetrics.classification import MultilabelPrecision


class TestMultilabelPrecision(unittest.TestCase):

    TARGETS = torch.tensor([[1, 0, 0],
                            [0, 1, 1],
                            [1, 1, 1]])

    PROB_PREDS = torch.tensor([[1, 0.3, 0],
                               [0, 1, 0.5],
                               [0.5, 0.6, 0.5]])

    INDEX_PREDS = torch.tensor([[0, 1, 2],
                                [1, 2, 0],
                                [1, 0, 2]])

    def test_threshold_based(self):
        THRESHOLD = [0.0, 0.3, 0.6, 0.7]
        EXPECTED_RESULT = [0.88888, 1.0, 0.66666, 0.66666]

        for i, t in enumerate(THRESHOLD):
            metric = MultilabelPrecision(threshold=THRESHOLD[i], num_labels=3, average='macro')
            metric.update(self.PROB_PREDS, self.TARGETS)
            self.assertAlmostEqual(metric.compute().item(), EXPECTED_RESULT[i], places=4)

    def test_topk_based(self):
        TOP_K = [1, 2, 3]
        EXPECTED_RESULT = [0.66666, 0.88888, 0.66666]

        for i, k in enumerate(TOP_K):
            metric = MultilabelPrecision(top_k=TOP_K[i], num_labels=3, average='macro')
            metric.update(self.PROB_PREDS, self.TARGETS)
            self.assertAlmostEqual(metric.compute().item(), EXPECTED_RESULT[i], places=4)

        # indices mode
        for i, k in enumerate(TOP_K):
            metric = MultilabelPrecision(top_k=TOP_K[i], prediction_mode='indices', num_labels=3, average='macro')
            metric.update(self.INDEX_PREDS, self.TARGETS)
            self.assertAlmostEqual(metric.compute().item(), EXPECTED_RESULT[i], places=4)


if __name__ == '__main__':
    unittest.main()
