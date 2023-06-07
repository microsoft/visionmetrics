import torchmetrics
from visionmetrics.prediction_filters import TopKPredictionFilter


class MultilabelPrecision(torchmetrics.classification.MultilabelPrecision):
    def __init__(self, top_k=None, prediction_mode='prob', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.prediction_mode = prediction_mode

    def update(self, predictions, targets):
        if self.top_k is None:
            self.top_k = predictions.shape[1]

        if self.prediction_mode == 'prob':
            super().update(predictions, targets)
        else:
            pred_filter = TopKPredictionFilter(k=self.top_k, prediction_mode=self.prediction_mode)
            topk_preds = pred_filter.filter(predictions)
            super().update(topk_preds, targets)
