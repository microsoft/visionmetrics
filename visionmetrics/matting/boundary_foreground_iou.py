import numpy as np
from .foreground_iou import ForegroundIOUEvaluator


class BoundaryForegroundIOUEvaluator(ForegroundIOUEvaluator):
    """
    Boundary foreground intersection-over-union evaluator
    """

    def __init__(self):
        super(BoundaryForegroundIOUEvaluator, self).__init__(metric='b_fgIOU')

    def _preprocess(self, pred_mask, gt_mask):
        pred_mask = np.asarray(pred_mask)
        gt_mask = np.asarray(gt_mask)
        pred_binmask = self._convert2binary(pred_mask)
        gt_binmask = self._convert2binary(gt_mask)
        gt_boundary_mask, pred_boundary_mask = self._create_contour_mask(gt_binmask, pred_binmask)

        return pred_boundary_mask.astype(np.int64), gt_boundary_mask.astype(np.int64)
