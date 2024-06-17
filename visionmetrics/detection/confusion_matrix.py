import torch
from torchmetrics import Metric


class DetectionConfusionMatrix(Metric):
    """
        Calcuates confusion matrix in the context of object detection.

        Parameters
        ----------
        predictions : list of lists
            Each list contains list of bounding box predictions for an image in the format [class_id, score, xmin, ymin, xmax, ymax]
        targets : list of lists
            Each list contains ground truth bounding boxes for an image in the format [class_id, xmin, ymin, xmax, ymax]
        iou_threshold : float
            IoU threshold to consider a detection as True Positive

        Example:
        predictions = [[[0, 0.9, 10, 20, 50, 100], [1, 0.8, 30, 40, 80, 120]], [[1, 0.7, 20, 30, 60, 90]]]
        targets = [[[0, 10, 20, 50, 100], [1, 30, 40, 80, 120]], [[1, 20, 30, 60, 90]]]

        Returns
        -------
        confusion_matrix : dict
            Dictionary containing the counts of TP, FP, and FN, as well as details for FP reasons

            TP: When a predicted bounding box has IoU greater than the threshold with a ground truth box of the same class
            FN: When a ground truth bounding box has no corresponding predicted bounding box
            FP:
                1. FP_due_to_wrong_class: IOU >= threshold but predicted class is different from the ground truth class. 
                2. FP_due_to_low_iou: Predicted bbox IOU < threshold (including no overlap)
                3. fp_due_to_extra_pred_boxes: Excess predicted bboxes when all ground truth boxes have been matched

        """

    def __init__(self, iou_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_wrong_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_low_iou", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_extra_pred_boxes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        if len(predictions) != len(targets):
            raise ValueError("Number of predictions and targets should be the same.")
        if not isinstance(predictions[0][0], list):
            raise ValueError(f"Expected predictions to be a list of lists, got a list of {type(predictions[0][0])}")

        self._update_confusion_matrix(predictions, targets, self.iou_threshold)

    def compute(self):
        return {
            'TP': self.tp.item(),
            'FP': self.fp.item(),
            'FN': self.fn.item(),
            'FP_due_to_wrong_class': self.fp_due_to_wrong_class.item(),
            'FP_due_to_low_iou': self.fp_due_to_low_iou.item(),
            'FP_due_to_extra_pred_boxes': self.fp_due_to_extra_pred_boxes.item()
        }

    def _update_confusion_matrix(self, predictions, targets, iou_threshold):
        for preds, gts in zip(predictions, targets):
            gt_used = [False] * len(gts)

            # Empty prediction list: count all GT boxes as FN
            if self._is_empty(preds):
                self.fn += len(gts)
                continue

            # Empty GT list: count all prediction boxes as FP
            if self._is_empty(gts):
                self.fp += len(preds)
                self.fp_due_to_extra_pred_boxes += len(preds)
                continue

            preds = sorted(preds, key=lambda x: x[1], reverse=True)

            # Iterate over predictions and ground truth boxes for each image
            for pred in preds:
                pred_class_id, pred_box = pred[0], pred[2:]  # [xmin, ymin, xmax, ymax]

                best_iou = 0
                best_gt_index = -1

                # If all GT boxes for this image have been matched
                # with a prediction bbox then all remaining predictions are FPs
                if all(gt_used):
                    self.fp += 1
                    self.fp_due_to_extra_pred_boxes += 1
                    continue

                for gt_index, gt in enumerate(gts):
                    if gt_used[gt_index]:
                        continue

                    gt_class_id, gt_box = gt[0], gt[1:]
                    current_iou = self.iou(pred_box, gt_box)

                    if current_iou >= best_iou:
                        best_iou = current_iou
                        best_gt_index = gt_index

                    if current_iou >= iou_threshold:
                        if pred_class_id == gt_class_id:
                            self.tp += 1  # TP
                            gt_used[best_gt_index] = True
                        else:
                            self.fp += 1  # FP
                            self.fp_due_to_wrong_class += 1
                    else:
                        self.fp += 1
                        self.fp_due_to_low_iou += 1

            # Count unused GT boxes as FNs
            for gt_index, gt in enumerate(gts):
                if not gt_used[gt_index]:
                    self.fn += 1

    @staticmethod
    def _is_empty(preds):
        if len(preds) == 0:
            return True
        elif all([len(pred) == 0 for pred in preds]):
            return True
        else:
            return False

    @staticmethod
    def iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        box1 : list, tuple
            [xmin, ymin, xmax, ymax]
        box2 : list, tuple
            [xmin, ymin, xmax, ymax]

        Returns
        -------
        float
            in [0, 1]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_min_inter = max(x1_min, x2_min)
        y_min_inter = max(y1_min, y2_min)
        x_max_inter = min(x1_max, x2_max)
        y_max_inter = min(y1_max, y2_max)

        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection = inter_width * inter_height

        area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area_box1 + area_box2 - intersection

        iou_value = intersection / union if union != 0 else 0

        return iou_value
