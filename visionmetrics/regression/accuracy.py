import torch
from torchmetrics import Metric

from visionmetrics.common.utils import precision_recall_f1_scalar


class MeanAbsoluteErrorF1Score(Metric):
    """
    Calculates and returns a discretized version of the MeanAbsoluteError metric.
    Args:
        error_threshold: float indicating the exclusive threshold below which the mean absolute error is considered a true positive.
    """
    def __init__(self, error_threshold=0.0):
        super().__init__()
        if error_threshold < 0:
            raise ValueError(f"'error_threshold' must be >= 0; got error_threshold={error_threshold}.")
        self.error_threshold = error_threshold

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        if not torch.is_tensor(predictions):
            raise ValueError("'predictions' must be a float or integer torch.tensor.")
        if not torch.is_tensor(targets):
            raise ValueError("'targets' must be a float or integer torch.tensor.")
        if predictions.shape != targets.shape:
            raise ValueError(f"'predictions' and 'targets' must have the same shape; got predictions of shape {list(predictions.shape)} and targets of shape {list(targets.shape)}.")
        n_obs = targets.numel()
        tp = torch.sum(torch.where(torch.abs(predictions - targets) <= self.error_threshold, 1, 0))
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
        return precision_recall_f1_scalar(tp=tp, fp=fp, fn=fn)
