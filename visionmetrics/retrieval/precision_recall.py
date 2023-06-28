from typing import Any

import torch
import torchmetrics
from torchmetrics.functional.classification import binary_precision_recall_curve


class RetrievalExtend:
    def update(self, predictions, targets):
        """ torchmetrics implementation of Retrieval* metrics expects an indexes tensor corresponding to each query.
        For e.g. preds = torch.tensor([[0.6, 0.3],[0.2, 0.7])) --> indexes = torch.tensor([[0, 0], [1, 1]])
        """
        indexes = self._create_indexes(predictions)
        super().update(predictions, targets, indexes)

    def _create_indexes(self, predictions):
        if len(predictions.shape) == 1:
            indexes = torch.zeros_like(predictions)
        else:
            indexes = torch.arange(predictions.shape[0]).unsqueeze(1).repeat(1, predictions.shape[1])
        return indexes.long()


class RetrievalPrecision(RetrievalExtend, torchmetrics.retrieval.RetrievalPrecision):
    pass


class RetrievalRecall(RetrievalExtend, torchmetrics.retrieval.RetrievalRecall):
    pass


class RetrievalMAP(RetrievalExtend, torchmetrics.retrieval.RetrievalMAP):
    pass


class RetrievalPrecisionRecallCurveNPoints(torchmetrics.Metric):
    def __init__(self, n_points, **kwargs: Any) -> None:
        self.n_points = n_points
        super().__init__(**kwargs)

        self.add_state("recall_thresholds", default=torch.linspace(1, 0, self.n_points), dist_reduce_fx=None)
        self.add_state("precision_averaged", default=torch.zeros(self.n_points), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        n_samples = predictions.shape[0]
        self.total_samples += n_samples
        for i in range(n_samples):
            precision_interp = self._compute_precision_recall_interp(predictions[i], targets[i], self.recall_thresholds)
            self.precision_averaged += precision_interp

    def compute(self):
        return self.precision_averaged / self.total_samples, self.recall_thresholds

    def _compute_precision_recall_interp(self, prediction, target, recall_thresholds):
        assert len(prediction) == len(target)
        assert len(target.shape) == 1

        # NOTE: thresholds are calculated in this way to achieve parity with vision-evaluation
        # that uses sklearn-based precision_recall_curve which uses thresholds in this way
        thresholds = torch.unique(prediction)

        precision, recall, _ = binary_precision_recall_curve(prediction, target, thresholds)
        precision_interp = torch.zeros_like(recall_thresholds)
        mask = recall_thresholds.unsqueeze(-1) <= recall
        precision_interp = (precision * mask).max(dim=-1).values
        return precision_interp
