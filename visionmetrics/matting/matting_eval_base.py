import cv2
import torch
from torchmetrics import Metric


class MattingEvaluatorBase(Metric):
    """
    Base class for image matting evaluator
    """

    def __init__(self, metric):
        super().__init__()
        self._metric = metric
        self.add_state("_num_samples", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("_metric_sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def _convert2binary(self, mask, threshold=128):
        bin_mask = mask >= threshold
        return bin_mask

    def _preprocess(self, pred_mask: torch.tensor, gt_mask: torch.tensor):

        pred_binmask = self._convert2binary(pred_mask)
        gt_binmask = self._convert2binary(gt_mask)
        return pred_binmask, gt_binmask

    def _find_contours(self, matting, thickness=10):
        matting = matting.clone().detach()
        opencv_major_version = int(cv2.__version__.split('.')[0])
        if opencv_major_version >= 4:
            contours, _ = cv2.findContours(matting.numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(matting.numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = torch.zeros(matting.shape, dtype=torch.uint8)

        cv2.drawContours(mask.numpy(), contours, -1, 255, thickness)
        return mask

    def _create_contour_mask(self, gt_mask, pred_mask, line_width=10):
        contour_mask = self._find_contours((gt_mask * 255).to(torch.uint8), thickness=line_width) / 255.0
        gt_contour_mask = gt_mask * contour_mask
        pred_contour_mask = pred_mask * contour_mask
        return gt_contour_mask, pred_contour_mask

    def compute(self):
        return {self._metric: self._metric_sum / self._num_samples if self._num_samples else 0.0}
