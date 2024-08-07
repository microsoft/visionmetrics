import torch
from torchmetrics import Metric


class MeanAbsoluteErrorF1Score(Metric):
    """
    Calculates and returns a discretized version of the MeanAbsoluteError metric.
    Args:
        threshold: float indicating the exclusive threshold below which the mean absolute error is considered a true positive.
    """
    def __init__(self, threshold=1.):
        super().__init__()
        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        super().update(predictions, targets)

        n_obs = targets.numel()
        tp = torch.sum(torch.where(torch.abs(predictions - targets) < self.threshold, 1, 0))
        fp = n_obs - tp
        fn = n_obs - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        # Note that we currently do not distinguish false positives vs false negatives (as in, false positive and false negative are the same),
        # we effectively have that precision == recall == F1 == accuracy.
        try:
            precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            precision = 0.
        try:
            recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            recall = 0.
        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
