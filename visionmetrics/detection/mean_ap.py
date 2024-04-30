from typing import Dict, List, Tuple

import torch
from torchmetrics import detection


class MeanAveragePrecision(detection.mean_ap.MeanAveragePrecision):
    """
    Mean Average Precision (mAP) metric for object detection task.

    This implementation extends the `torchmetrics` implementation of mAP to:

    1. Accept predictions and targets in a different format. The `update` method expects predictions and targets to be a list of lists of lists, where each inner list corresponds to a single image.
    The innermost list contains the predicted or ground truth boxes for that image, where each box is represented as a list of 6 (or 5 for target with no score) elements:
    [label, score, L, T, R, B], where L, T, R, B are the coordinates of the box in the format 'xyxy' (xmin, ymin, xmax, ymax).

    2. Accept both relative and absolute coordinates.
    NOTE: torchmetrics.*.MeanAveragePrecision only expects absolute coordinates so it is possible after some version update MeanAveragePrecision might break suddenly.

    3. Only compute the following metrics: map, map_50, map_75, map_per_class (which are independent of the coordinate format).

    Example:
    ```
    predictions = [[[0, 0.9, 10, 20, 50, 100], [1, 0.8, 30, 40, 80, 120]], [[1, 0.7, 20, 30, 60, 90]]]
    targets = [[[0, 10, 20, 50, 100], [1, 30, 40, 80, 120]], [[1, 20, 30, 60, 90]]]

    metric = MeanAveragePrecision()
    metric.update(predictions, targets)
    ap = metric.compute()
    ```
    """

    def __init__(self, box_format='xyxy', coords='relative', **kwargs):
        if coords not in ['relative', 'absolute']:
            raise ValueError(f'Expected coordinates to be "relative" or "absolute", got {coords}')
        self.coords = coords
        # TODO: add support for other box formats
        if box_format != 'xyxy':
            raise ValueError(f'Expected box format to be "xyxy", got {box_format}')
        super().__init__(box_format=box_format, **kwargs)

    def update(self, predictions: List[List[List[float]]], targets: List[List[List[float]]]) -> None:
        predictions, targets = self._preprocess(predictions, targets)
        super().update(predictions, targets)

    def compute(self):
        result = super().compute()
        common_keys = ['map', 'map_50', 'map_75', 'map_per_class', 'classes']
        if self.coords == 'relative':
            keys = common_keys
        elif self.coords == 'absolute':
            keys = common_keys + ['map_small', 'map_medium', 'map_large']
        return {k: v for k, v in result.items() if k in keys}

    def _preprocess(self, predictions: List[List[List[float]]], targets: List[List[List[float]]]) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        predictions = [self._convert_to_dict(p) for p in predictions]
        targets = [self._convert_to_dict(t, scores=False) for t in targets]
        return predictions, targets

    @staticmethod
    def _convert_to_dict(boxes: List[List[float]], scores: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            boxes (list): list of boxes. Each box is a list of 6 (or 5 when no score) elements: [label, score, L, T, R, B]
            score (bool): whether to include scores in the output dictionary
        """
        if not boxes:
            boxes = torch.empty(0, 6) if scores else torch.empty(0, 5)
        else:
            boxes = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes

        if scores:
            return {'boxes': boxes[:, -4:], 'labels': boxes[:, 0].to(torch.int), 'scores': boxes[:, 1].to(torch.float)}
        else:
            return {'boxes': boxes[:, -4:], 'labels': boxes[:, 0].to(torch.int)}


class ClassAgnosticAveragePrecision(MeanAveragePrecision):
    """
    Calculates the average precision (AP) for object detection tasks in a class-agnostic manner.
    It treats all classes as a single class (-1) and evaluates average precision for this class.
    """

    def update(self, predictions: List[List[List[float]]], targets: List[List[List[float]]]) -> None:
        predictions, targets = self._make_class_agnostic(predictions), self._make_class_agnostic(targets)
        return super().update(predictions, targets)

    def _make_class_agnostic(self, preds_or_targets: List[List[List[float]]]) -> List[List[List[float]]]:
        # Replace the class labels with -1
        preds_or_targets = [[[-1] + box[1:] for box in boxes_per_img] for boxes_per_img in preds_or_targets]
        return preds_or_targets
