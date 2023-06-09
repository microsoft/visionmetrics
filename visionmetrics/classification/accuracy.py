import torchmetrics

from visionmetrics.common import targets_to_mat


class MultilabelAccuracy(torchmetrics.classification.MultilabelAccuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, predictions, targets):
        targets = targets_to_mat(targets, n_class=predictions.shape[1])
        super().update(predictions, targets)
