import torchmetrics
from typing import Literal

from visionmetrics.prediction_filters import TopKPredictionFilter


class MultilabelExtend:
    """
    Extend torchmetrics.classification.Multilabel* metrics to support:
        1. top_k filtering
        2. 'indices' prediction_mode
    """

    def __init__(self, top_k: int = None, prediction_mode: Literal['prob', 'indices'] = 'prob'):
        self.top_k = top_k
        self.prediction_mode = prediction_mode

    def update(self, predictions, targets):
        # torchmetrics only supports prob mode
        if self.top_k is None and self.prediction_mode == 'prob':
            super().update(predictions, targets)
        else:
            self.top_k = predictions.shape[1] if self.top_k is None else self.top_k
            pred_filter = TopKPredictionFilter(k=self.top_k, prediction_mode=self.prediction_mode)
            topk_preds = pred_filter.filter(predictions)
            super().update(topk_preds, targets)


class MultilabelPrecision(MultilabelExtend, torchmetrics.classification.MultilabelPrecision):
    def __init__(self, top_k=None, prediction_mode='prob', *args, **kwargs):
        MultilabelExtend.__init__(self, top_k=top_k, prediction_mode=prediction_mode)
        torchmetrics.classification.MultilabelPrecision.__init__(self, *args, **kwargs)


class MultilabelRecall(MultilabelExtend, torchmetrics.classification.MultilabelRecall):
    def __init__(self, top_k=None, prediction_mode='prob', *args, **kwargs):
        MultilabelExtend.__init__(self, top_k=top_k, prediction_mode=prediction_mode)
        torchmetrics.classification.MultilabelRecall.__init__(self, *args, **kwargs)


class MultilabelF1Score(MultilabelExtend, torchmetrics.classification.MultilabelF1Score):
    def __init__(self, top_k=None, prediction_mode='prob', *args, **kwargs):
        MultilabelExtend.__init__(self, top_k=top_k, prediction_mode=prediction_mode)
        torchmetrics.classification.MultilabelF1Score.__init__(self, *args, **kwargs)
