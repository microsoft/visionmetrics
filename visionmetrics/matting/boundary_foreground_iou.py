import numpy as np
import torch
from visionmetrics.matting.matting_eval_base import MattingEvaluatorBase


class BoundaryForegroundIOU(MattingEvaluatorBase):
    """
    Boundary foreground intersection-over-union evaluator
    """

    def __init__(self):
        super().__init__(metric='b_fgIOU')

    def _preprocess(self, pred_mask, gt_mask):
        pred_mask = torch.tensor(np.array(pred_mask))
        gt_mask = torch.tensor(np.array(gt_mask))

        pred_binmask = self._convert2binary(pred_mask)
        gt_binmask = self._convert2binary(gt_mask)
        gt_boundary_mask, pred_boundary_mask = self._create_contour_mask(gt_binmask, pred_binmask)

        return pred_boundary_mask.to(torch.int64), gt_boundary_mask.to(torch.int64)

    def update(self, predictions, targets):
        """ Adding predictions and ground truth of images for image matting task
        Args:
            predictions: list of image matting predictions, [matting1, matting2, ...]. Shape: (N, ), type: PIL image object
            targets: list of image matting ground truth, [gt1, gt2, ...]. Shape: (N, ), type: PIL image object
        """

        assert len(predictions) == len(targets)

        num_class = 2
        self._num_samples += len(predictions)
        for pred_mask, gt_mask in zip(predictions, targets):
            pred_mask = torch.tensor(np.array(pred_mask))
            gt_mask = torch.tensor(np.array(gt_mask))
            pred_binmask, gt_binmask = self._preprocess(pred_mask, gt_mask)
            if torch.all(gt_binmask == 0):
                res = 1 if torch.all(pred_binmask == 0) else 0
                self.metric_sum += res
                continue

            label = num_class * gt_binmask + pred_binmask
            count = torch.bincount(label.flatten(), minlength=num_class**2)
            confusion_matrix = count.reshape(num_class, num_class)
            iou = torch.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - torch.diag(confusion_matrix) + 1e-10)
            self._metric_sum += iou[1]
