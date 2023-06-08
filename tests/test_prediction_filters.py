import unittest
import torch
from visionmetrics.prediction_filters import TopKPredictionFilter


class TestTopKPredictionFilter(unittest.TestCase):

    PROB_PREDS = torch.tensor([[1, 0.3, 0],
                               [0, 1, 0.5],
                               [0.5, 0.6, 0.5]])

    INDEX_PREDS = torch.tensor([[0, 1, 2],
                                [1, 2, 0],
                                [1, 0, 2]])

    EXPECTED_TOPK_PREDS = [torch.tensor([[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]]),
                           torch.tensor([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 1, 0]]),
                           torch.tensor([[1, 1, 0],
                                         [0, 1, 1],
                                         [1, 1, 0]]),
                           torch.tensor([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])]
    K = [0, 1, 2, 3]

    def test_topk_prediction_filter_prob_mode(self):
        for i, k in enumerate(self.K):
            pred_filter = TopKPredictionFilter(k=k, prediction_mode='prob')
            topk_preds = pred_filter.filter(self.PROB_PREDS, return_mode='vec')
            self.assertTrue(torch.equal(topk_preds, self.EXPECTED_TOPK_PREDS[i]))

    def test_topk_prediction_filter_indices_mode(self):
        for i, k in enumerate(self.K):
            pred_filter = TopKPredictionFilter(k=k, prediction_mode='indices')
            topk_preds = pred_filter.filter(self.INDEX_PREDS, return_mode='vec')
            self.assertTrue(torch.equal(topk_preds, self.EXPECTED_TOPK_PREDS[i]))


if __name__ == '__main__':
    unittest.main()
