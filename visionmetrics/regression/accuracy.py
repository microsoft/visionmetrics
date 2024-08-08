import torch
from torchmetrics import Metric


class MeanAbsoluteErrorF1Score(Metric):
    """
    Calculates and returns a discretized version of the MeanAbsoluteError metric.
    Args:
        error_threshold: float indicating the exclusive threshold below which the mean absolute error is considered a true positive.
    """
    def __init__(self, error_threshold=1.):
        super().__init__()
        self.error_threshold = error_threshold

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        n_obs = targets.numel()
        tp = torch.sum(torch.where(torch.abs(predictions - targets) < self.error_threshold, 1, 0))
        fp = n_obs - tp
        fn = n_obs - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        # Note that we currently do not distinguish false positives vs false negatives (as in, false positive and false negative are the same),
        # we effectively have that precision == recall == F1 == accuracy.
        tp = self.tp.item()
        fp = self.fp.item()
        fn = self.fn.item()
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.
        try:
            recall = tp / (tp + fn)
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
