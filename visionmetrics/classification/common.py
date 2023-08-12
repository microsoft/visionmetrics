from typing import Literal

from visionmetrics.prediction_filters import TopKPredictionFilter


class MulticlassMixin:
    """
    Extend torchmetrics.classification.Multiclass* metrics to accept top_k > num_classes.
    """

    def __init__(self, num_classes, top_k=1, *args, **kwargs):
        self.top_k = min(num_classes, top_k)
        super().__init__(num_classes, top_k=self.top_k, *args, **kwargs)


class MultilabelMixin:
    """
    Extend torchmetrics.classification.Multilabel* metrics to support:
        1. top_k filtering
        2. 'indices' prediction_mode
    """

    def __init__(self, top_k: int = None, prediction_mode: Literal['prob', 'indices'] = 'prob', *args, **kwargs):
        self.top_k = top_k
        self.prediction_mode = prediction_mode
        super().__init__(*args, **kwargs)

    def update(self, predictions, targets):
        # torchmetrics only supports prob mode
        if self.top_k is None and self.prediction_mode == 'prob':
            super().update(predictions, targets)
        else:
            self.top_k = predictions.shape[1] if self.top_k is None else self.top_k
            pred_filter = TopKPredictionFilter(k=self.top_k, prediction_mode=self.prediction_mode)
            topk_preds = pred_filter.filter(predictions)
            super().update(topk_preds, targets)
