import cv2
import numpy as np
from torchmetrics import Metric


class MattingEvaluatorBase(Metric):
    """
    Base class for image matting evaluator
    """

    def __init__(self, metric):
        super(MattingEvaluatorBase, self).__init__()
        self._metric = metric
        self._num_samples = 0
        self._metric_sum = 0

    def reset(self):
        super(MattingEvaluatorBase, self).reset()
        self._num_samples = 0
        self._metric_sum = 0

    def _convert2binary(self, mask, threshold=128):
        bin_mask = mask >= threshold
        return bin_mask.astype(mask.dtype)

    def _preprocess(self, pred_mask, gt_mask):
        pred_mask = np.asarray(pred_mask)
        gt_mask = np.asarray(gt_mask)
        pred_binmask = self._convert2binary(pred_mask)
        gt_binmask = self._convert2binary(gt_mask)
        return pred_binmask, gt_binmask

    def _find_contours(self, matting, thickness=10):
        matting = np.copy(matting)
        opencv_major_version = int(cv2.__version__.split('.')[0])
        if opencv_major_version >= 4:
            contours, _ = cv2.findContours(matting, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(matting, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(matting.shape, np.uint8)

        cv2.drawContours(mask, contours, -1, 255, thickness)
        return mask

    def _create_contour_mask(self, gt_mask, pred_mask, line_width=10):
        contour_mask = self._find_contours((gt_mask * 255).astype('uint8'), thickness=line_width) / 255.0
        gt_contour_mask = gt_mask * contour_mask
        pred_contour_mask = pred_mask * contour_mask
        return gt_contour_mask, pred_contour_mask

    def compute(self):
        return {self._metric: self._metric_sum / self._num_samples if self._num_samples else 0.0}
