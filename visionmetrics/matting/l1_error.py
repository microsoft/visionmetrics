import numpy as np
from .matting_eval_base import MattingEvaluatorBase


class L1ErrorEvaluator(MattingEvaluatorBase):
    """
    L1 error evaluator
    """

    def __init__(self):
        super(L1ErrorEvaluator, self).__init__(metric='L1Err')

    def update(self, predictions, targets):
        """ Adding predictions and ground truth of images for image matting task
        Args:
            predictions: list of image matting predictions, [matting1, matting2, ...]. Shape: (N, ), type: PIL image object
            targets: list of image matting ground truth, [gt1, gt2, ...]. Shape: (N, ), type: PIL image object
        """

        assert len(predictions) == len(targets)

        self._num_samples += len(predictions)
        for pred_mask, gt_mask in zip(predictions, targets):
            pred_mask = np.asarray(pred_mask)
            gt_mask = np.asarray(gt_mask)
            mean_l1 = np.abs(pred_mask.astype(float)-gt_mask.astype(float)).mean()
            self._metric_sum += mean_l1
