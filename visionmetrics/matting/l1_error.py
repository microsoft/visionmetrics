import numpy as np
import torch
from visionmetrics.matting.matting_eval_base import MattingEvaluatorBase


class L1Error(MattingEvaluatorBase):
    """
    L1 error evaluator
    """

    def __init__(self):
        super().__init__(metric='L1Err')

    def update(self, predictions, targets):
        """ Adding predictions and ground truth of images for image matting task
        Args:
            predictions: list of image matting predictions, [matting1, matting2, ...]. Shape: (N, ), type: PIL image object
            targets: list of image matting ground truth, [gt1, gt2, ...]. Shape: (N, ), type: PIL image object
        """

        assert len(predictions) == len(targets)

        self._num_samples += len(predictions)
        for pred_mask, gt_mask in zip(predictions, targets):
            pred_mask = torch.tensor(np.array(pred_mask))
            gt_mask = torch.tensor(np.array(gt_mask))
            mean_l1 = torch.abs(pred_mask.to(torch.float)-gt_mask.to(torch.float)).mean()
            self._metric_sum += mean_l1.numpy()
