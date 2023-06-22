import unittest

import torch

from visionmetrics.retrieval import (RetrievalMAP, RetrievalPrecision,
                                     RetrievalRecall)


class TestRetrievalPrecision(unittest.TestCase):

    PREDICTIONS = [torch.tensor([[5, 4, 3, 2, 1]]),
                   torch.tensor([[5, 4, 3, 2, 1]]),
                   torch.tensor([[1, 2, 3, 4, 5]]),
                   torch.tensor([[5, 4, 3, 2, 1]]),
                   torch.tensor([[5, 4, 3, 2, 1],
                                 [5, 4, 3, 2, 1]]),
                   torch.tensor([[5, 4, 3, 2, 1],
                                 [5, 4, 3, 2, 1]]),
                   torch.tensor([[1]]),
                   torch.tensor([[2],
                                 [3]])]
    TARGETS = [torch.tensor([[1, 1, 0, 0, 1]]),
               torch.tensor([[1, 1, 0, 0, 1]]),
               torch.tensor([[1, 0, 0, 1, 1]]),
               torch.tensor([[0, 0, 0, 0, 1]]),
               torch.tensor([[0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1]]),
               torch.tensor([[1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1]]),
               torch.tensor([[1]]),
               torch.tensor([[1],
                             [0]])]

    def test_recall_at_k(self):
        ks = [6, 8, 6, 6, 6, 6, 4, 4]
        expectations = [[0.33333, 0.66666, 0.66666, 0.66666, 1.0],
                        [0.33333, 0.66666, 0.66666, 0.66666, 1.0, 1.0, 1.0],
                        [0.33333, 0.66666, 0.66666, 0.66666, 1.0],
                        [0, 0, 0, 0, 1.0],
                        [0, 0, 0, 0, 1.0],
                        [0.25, 0.25, 0.25, 0.25, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 0.5]]

        for preds, target, exps, k in zip(self.PREDICTIONS, self.TARGETS, expectations, ks):
            for i in range(1, k):
                metric = RetrievalRecall(k=i, adaptive_k=True)
                metric.update(preds.float(), target)
                result = metric.compute()
                self.assertAlmostEqual(result.item(), exps[i - 1], places=4)

    def test_precision_at_k(self):
        ks = [6, 8, 6, 6, 6, 6, 4, 4]
        expectations = [[1.0, 1.0, 0.66666, 0.5, 0.6],
                        [1.0, 1.0, 0.66666, 0.5, 0.6, 0.6, 0.6],
                        [1.0, 1.0, 0.66666, 0.5, 0.6],
                        [0, 0, 0, 0, 0.2],
                        [0, 0, 0, 0, 0.2],
                        [0.5, 0.25, 0.16666, 0.125, 0.3],
                        [1.0, 1.0, 1.0],
                        [0.5, 0.5, 0.5]]

        for preds, target, exps, k in zip(self.PREDICTIONS, self.TARGETS, expectations, ks):
            for i in range(1, k):
                metric = RetrievalPrecision(k=i, adaptive_k=True)
                metric.update(preds.float(), target)
                result = metric.compute()
                self.assertAlmostEqual(result.item(), exps[i - 1], places=4)

    def test_mean_average_precision_at_k(self):
        targets = [torch.tensor([[1, 0, 1, 1],
                                 [1, 0, 0, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 0, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[1, 0, 1, 1]]),
                   torch.tensor([[0, 0, 0, 0]]),
                   torch.tensor([[1, 0, 1]]),
                   torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])]
        predictions = [torch.tensor([[5, 4, 3, 2],
                                     [5, 4, 3, 2]]),
                       torch.tensor([[5, 4, 3, 2]]),
                       torch.tensor([[5, 4, 3, 2]]),
                       torch.tensor([[2, 3, 5, 4]]),
                       torch.tensor([[4, 2, 3, 5]]),
                       torch.tensor([[4, 2, 3, 5]]),
                       torch.tensor([[2, 3, 5, 4]]),
                       torch.tensor([[2, 3, 5, 4]]),
                       torch.tensor([[2, 3, 5, 4]]),
                       torch.tensor([[2, 3, 5, 4]]),
                       torch.tensor([[2, 3, 5]]),
                       torch.tensor([[2, 3, 5, 4, 2, 3, 5, 4, 2, 3, 5, 4, 2, 3, 5, 7, 3, 4]])]
        rank = [4, 4, 4, 4, 4, 4, 3, 3, 5, 2, 4, 4, 8]
        expectations = [0.77777, 0.80555, 0.75, 0.91666, 1.0, 1.0, 0.91666, 0.91666, 0.91666, 0.0, 0.83333, 0.8220]

        for preds, target, exps, k in zip(predictions, targets, expectations, rank):
            if preds.numel() == 0:
                continue
            metric = RetrievalMAP(k=k, adaptive_k=True)
            metric.update(preds.float(), target)
            result = metric.compute()
            self.assertAlmostEqual(result.item(), exps, places=4)


if __name__ == '__main__':
    unittest.main()