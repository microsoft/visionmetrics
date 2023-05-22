import torch
from torchmetrics import Metric

from visionmetrics import utils


class ThresholdAccuracy(Metric):
    """
    Threshold-based accuracy evaluator for multilabel classification, calculated in a sample-based flavor
    Note that
        1. this could be used for multi-class classification, but does not make much sense
        2. sklearn.metrics.accuracy_score actually is computing exact match ratio for multi-label classification, which is too harsh
    """

    def __init__(self, threshold) -> None:
        super().__init__()
        self.prediction_filter = ThresholdPredictionFilter(threshold)

        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sample_accuracy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """ Add a batch of predictions for evaluation.
        Args:
            predictions (torch.tensor): the model output array. Shape (N, num_class)
            targets: the ground truths (torch.tensor). Shape (N, num_class) for multi-label (or (N,) for multi-class)
        """

        assert len(predictions) == len(targets)

        num_samples = len(predictions)
        target_mat = utils.targets_to_mat(targets, predictions.shape[1])

        prediction_over_threshold = self.prediction_filter.filter(predictions, 'vec')
        n_correct_predictions = torch.mul(prediction_over_threshold, target_mat).sum(1)  # shape (N,)
        n_total = (torch.add(prediction_over_threshold, target_mat) >= 1).sum(1)  # shape (N,)
        n_total[n_total == 0] = 1  # To avoid zero-division. If n_total==0, num should be zero as well.

        self.sample_accuracy_sum += (n_correct_predictions / n_total).sum()
        self.num_samples += num_samples

    def compute(self) -> dict:
        return {f'accuracy_{self.prediction_filter.identifier}': float(self.sample_accuracy_sum) / self.num_samples if self.num_samples else 0.0}


class ThresholdPredictionFilter:
    def __init__(self, threshold: float):
        """
        Args:
            threshold: confidence threshold
        """

        self.threshold = threshold

    def filter(self, predictions, return_mode):
        """ Return predictions over confidence over threshold
        Args:
            predictions (torch.tensor): the model output array. Shape (N, num_class)
            return_mode: can be 'indices' or 'vec', indicating whether return value is a set of class indices or 0-1 vector

        Returns:
            (list): labels with probabilities over threshold, for each sample
        """
        if return_mode == 'indices':
            preds_over_thres = [[] for _ in range(len(predictions))]
            for indices in torch.argwhere(predictions >= self.threshold):
                preds_over_thres[indices[0]].append(indices[1])

            return preds_over_thres
        else:
            return predictions >= self.threshold

    @property
    def identifier(self):
        return f'thres={self.threshold}'
