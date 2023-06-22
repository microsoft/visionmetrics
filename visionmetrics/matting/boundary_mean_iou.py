import numpy as np
import torch
from visionmetrics.matting.mean_iou import MeanIOU


class BoundaryMeanIOU(MeanIOU):
    """
    Boundary mean intersection-over-union evaluator
    """

    def __init__(self):
        super().__init__(metric='b_mIOU')

    def _preprocess(self, pred_mask, gt_mask):
        pred_mask = torch.tensor(np.array(pred_mask))
        gt_mask = torch.tensor(np.array(gt_mask))
        pred_binmask = self._convert2binary(pred_mask)
        gt_binmask = self._convert2binary(gt_mask)
        gt_boundary_mask, pred_boundary_mask = self._create_contour_mask(gt_binmask, pred_binmask)

        return pred_boundary_mask.to(torch.int64), gt_boundary_mask.to(torch.int64)
