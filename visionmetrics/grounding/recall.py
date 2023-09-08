import torch
from torchmetrics import Metric


class Recall(Metric):
    """
    Compute Recall@k for object grounding task.
    """

    def __init__(self, iou_thresh=0.5, k=1):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.topk = k
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def _box_area(self, boxes):
        assert boxes.ndim == 2 and boxes.shape[-1] == 4
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _box_iou(self, boxes1, boxes2):
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clip(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou

    def update(self, predictions, targets):
        """
        Args:
            predictions: list of tuple predictions (pred_phrases, pred_bboxes) for image object grounding task:
                         [([pred_phrase1, pred_phrase2, ...], [[pred_bbox, pred_bbox, ...], [pred_bbox, pred_bbox, ...], ...]), ...], type: list of tuple.
                         "pred_bboxes" is a list of list of bbox (top, left, bottom, right) in absolute scale.
            targets: list of tuple ground truth (target_phrases, target_bboxes) for image object grounding task:
                     [([target_phrase1, target_phrase2, ...], [[target_bbox, target_bbox, ...], [target_bbox, target_bbox, ...], ...]), ...], type: list of tuple.
                     "target_bboxes" is a list of list of bbox (top, left, bottom, right) in absolute scale.
        """
        self.targets += targets
        self.predictions += predictions

    def compute(self, **kwargs):
        """
        Returns:
            recall@k: top-k recall score
        """
        total_prediction = 0
        true_positive = 0

        assert len(self.predictions) == len(self.targets), "Number of predictions and targets must be equal."

        for pred, target in zip(self.predictions, self.targets):
            pred_phrases, pred_bboxes = pred
            target_phrases, target_bboxes = target

            assert len(pred_phrases) == len(pred_bboxes), "Number of predicted phrases and predicted bboxes must be equal."
            assert len(target_phrases) == len(target_bboxes), "Number of target phrases and target bboxes must be equal."

            for target_phrase, target_bbox in zip(target_phrases, target_bboxes):
                total_prediction += 1
                if target_phrase in pred_phrases:
                    # Only consider the first matched prediction, need to be adjusted for multiple prediction
                    cur_boxes = pred_bboxes[pred_phrases.index(target_phrase)]
                    if len(cur_boxes) > 0 and len(target_bbox) > 0:
                        ious = self._box_iou(torch.tensor(cur_boxes), torch.tensor(target_bbox))
                        if ious[:self.topk].max() >= self.iou_thresh:
                            true_positive += 1
        return {f'recall@{self.topk}': true_positive / total_prediction}
