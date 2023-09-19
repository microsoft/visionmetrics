import torch
from torchmetrics import Metric


class Recall(Metric):
    """
    Compute Recall@k for object grounding task.
    """

    def __init__(self, iou_thresh=0.5, k=1):
        super().__init__()
        if not (0 <= iou_thresh <= 1):
            raise ValueError(f"iou_thresh must be in [0, 1], got {iou_thresh}.")

        if not isinstance(k, int) or not (k >= 1):
            raise ValueError(f"k must be a postive integer, got {k}.")

        self.iou_thresh = iou_thresh
        self.topk = k
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def _box_area(self, boxes):
        assert boxes.ndim == 2 and boxes.shape[-1] == 4
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _box_iou(self, boxes1, boxes2):
        """
        Compute intersection-over-union of input boxes. Both sets of boxes are expected to be in (top, left, bottom, right) format.

        Args:
            boxes1 (Tensor[N, 4])
            boxes2 (Tensor[M, 4])

        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        """

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
        if len(predictions) != len(targets):
            raise ValueError(f"Number of predictions and targets must be equal, got {len(predictions)} and {len(targets)}.")

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            pred_phrases, pred_bboxes = pred
            target_phrases, target_bboxes = target
            if len(pred_phrases) != len(pred_bboxes):
                raise ValueError(f"Number of predicted phrases and predicted bboxes must be equal, got {len(pred_phrases)} and {len(pred_bboxes)}, index: {i}.")

            if len(target_phrases) != len(target_bboxes):
                raise ValueError(f"Number of target phrases and target bboxes must be equal, got {len(target_phrases)} and {len(target_bboxes)}, index: {i}.")

        self.targets += targets
        self.predictions += predictions

    def compute(self, **kwargs):
        """
        Returns:
            recall@k: top-k recall score
        """
        phrase_num = 0
        true_positive = 0

        for pred, target in zip(self.predictions, self.targets):
            pred_phrases, pred_bboxes = pred
            target_phrases, target_bboxes = target
            for target_phrase, target_bbox in zip(target_phrases, target_bboxes):
                phrase_num += 1
                pred_phrases = [phrase.lower() for phrase in pred_phrases]
                target_phrase = target_phrase.lower()
                if target_phrase in pred_phrases:
                    # Only consider the first matched prediction, need to be adjusted for multiple prediction
                    cur_boxes = pred_bboxes[pred_phrases.index(target_phrase)]
                    if len(cur_boxes) > 0 and len(target_bbox) > 0:
                        ious = self._box_iou(torch.tensor(cur_boxes), torch.tensor(target_bbox))
                        if ious[:self.topk].max() >= self.iou_thresh:
                            true_positive += 1
        return {f'recall@{self.topk}': true_positive / phrase_num}
