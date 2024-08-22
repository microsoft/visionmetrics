from collections import Counter
import torch
import torchmetrics

from visionmetrics.classification.common import MulticlassMixin, MultilabelMixin
from visionmetrics.common.utils import precision_recall_f1_scalar


class MulticlassPrecision(MulticlassMixin, torchmetrics.classification.MulticlassPrecision):
    pass


class MulticlassRecall(MulticlassMixin, torchmetrics.classification.MulticlassRecall):
    pass


class MulticlassF1Score(MulticlassMixin, torchmetrics.classification.MulticlassF1Score):
    pass


class MultilabelPrecision(MultilabelMixin, torchmetrics.classification.MultilabelPrecision):
    pass


class MultilabelRecall(MultilabelMixin, torchmetrics.classification.MultilabelRecall):
    pass


class MultilabelF1Score(MultilabelMixin, torchmetrics.classification.MultilabelF1Score):
    pass


class MultilabelF1ScoreWithDuplicates(torchmetrics.Metric):
    """
    Calculates and returns a variant of multilabel F1 Score where, instead of considering sets (no duplicate values) of tags, lists of tags with potential duplicates are permitted.
    """
    def __init__(self):
        super().__init__()

        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: list, targets: list):
        if len(predictions) != len(targets):
            raise ValueError(f"Number of predictions and targets should be the same, but got predictions of length {len(predictions)} and targets of length {len(targets)}.")
        for prediction, target in zip(predictions, targets):
            pred_counts = Counter(prediction)
            target_counts = Counter(target)
            self.tp += sum([min(pred_counts[k], target_counts[k]) for k in pred_counts if k in target_counts])
            self.fp += sum([max(pred_counts[k] - target_counts[k], 0) for k in pred_counts if k in target_counts] + [pred_counts[k] for k in pred_counts if k not in target_counts])
            self.fn += sum([max(target_counts[k] - pred_counts[k], 0) for k in target_counts if k in pred_counts] + [target_counts[k] for k in target_counts if k not in pred_counts])

    def compute(self):
        tp = self.tp.item()
        fp = self.fp.item()
        fn = self.fn.item()
        return precision_recall_f1_scalar(tp=tp, fp=fp, fn=fn)
