from torch import Tensor
from torchmetrics.classification import MultilabelAccuracy


class ThresholdAccuracy(MultilabelAccuracy):
    """
    Calculates the accuracy for multi-label classification tasks given a threshold.
    Note: this could be used for multi-class classification, but does not make much sense
    """

    def _identifier(self):
        return f'thres={self.threshold}'

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """ Add a batch of predictions for evaluation.
        Args:
            predictions (torch.tensor): the model output array. Shape (N, num_class)
            targets (torch.tensor): the ground truths. Shape (N, num_class) for multi-label or (N,1) for multi-class
        """
        return super().update(predictions, targets)

    def compute(self) -> dict:
        return {f'accuracy_{self._identifier()}': super().compute()}
