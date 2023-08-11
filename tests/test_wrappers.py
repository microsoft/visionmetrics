import unittest

import torch

from visionmetrics.wrappers import MetricCollection
from visionmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall


class TestMetricCollection(unittest.TestCase):
    def test_multiclass(self):
        targets = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([[1, 0],
                                    [0, 1],
                                    [0.5, 0.5],
                                    [0.1, 0.9]])
        metric = MetricCollection([MetricCollection({'MulticlassAccuracy': MulticlassAccuracy(num_classes=2, average='micro'),
                                                     'TagwiseMulticlassAccuracy': MulticlassAccuracy(num_classes=2, average='none')}),
                                   MulticlassPrecision(num_classes=2, average='macro'),
                                   MulticlassRecall(num_classes=2, average='macro'),
                                   MulticlassAUROC(num_classes=2, average='macro')])
        metric.update(predictions, targets)
        result = metric.compute()
        self.assertAlmostEqual(result['MulticlassAccuracy'].tolist(), 1.0)
        self.assertAlmostEqual(result['TagwiseMulticlassAccuracy'].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(result['MulticlassPrecision'].tolist(), 1.0)
        self.assertAlmostEqual(result['MulticlassRecall'].tolist(), 1.0)
        self.assertAlmostEqual(result['MulticlassAUROC'].tolist(), 1.0)

    def test_multilabel(self):
        targets = torch.tensor([[1, 0, 0],
                                [0, 1, 1],
                                [1, 1, 1]])
        predictions = torch.tensor([[1, 0.31, 0.1],
                                    [0.1, 1, 0.51],
                                    [0.51, 0.61, 0.51]])
        metric = MetricCollection(([MultilabelAccuracy(num_labels=3, average='micro', threshold=0.5),
                                    MultilabelPrecision(top_k=1, num_labels=3, average='macro'),
                                    MultilabelRecall(num_labels=3, average='macro', threshold=0.6)]))
        metric.update(predictions, targets)
        result = metric.compute()
        self.assertAlmostEqual(result['MultilabelAccuracy'].tolist(), 1.0, places=4)
        self.assertAlmostEqual(result['MultilabelPrecision'].tolist(), 0.6667, places=4)
        self.assertAlmostEqual(result['MultilabelRecall'].tolist(), 0.500, places=4)


if __name__ == '__main__':
    unittest.main()
