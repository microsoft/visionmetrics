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
        """

    def __init__(self, iou_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_wrong_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_low_iou", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp_due_to_no_corresponding_gt_box", default=torch.tensor(0), dist_reduce_fx="sum")

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
            'FP_due_to_no_corresponding_gt_box': self.fp_due_to_no_corresponding_gt_box.item()
        }

    def _update_confusion_matrix(self, predictions, targets, iou_threshold):
        for preds, gts in zip(predictions, targets):
            gt_used = [False] * len(gts)

            for pred in preds:
                pred_class_id, score, pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred
                pred_box = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]

                best_iou = 0
                best_gt_index = -1

                for gt_index, gt in enumerate(gts):
                    if gt_used[gt_index]:
                        continue

                    gt_class_id, gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt
                    gt_box = [gt_xmin, gt_ymin, gt_xmax, gt_ymax]

                    if pred_class_id == gt_class_id:
                        current_iou = self.iou(pred_box, gt_box)
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_gt_index = gt_index

                if best_gt_index != -1:
                    if best_iou >= iou_threshold:
                        self.tp += 1
                        gt_used[best_gt_index] = True
                    else:

                        self.fp += 1
                        self.fp_due_to_low_iou += 1
                else:
                    self.fp += 1
                    if all(gt_used):
                        self.fp_due_to_no_corresponding_gt_box += 1
                    else:
                        self.fp_due_to_wrong_class += 1

            for gt_index, gt in enumerate(gts):
                if not gt_used[gt_index]:
                    self.fn += 1

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
