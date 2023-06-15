import torch
from torchmetrics import detection


class MeanAveragePrecision(detection.mean_ap.MeanAveragePrecision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, predictions, targets):
        predictions, targets = self._preprocess(predictions, targets)
        super().update(predictions, targets)

    def _preprocess(self, predictions, targets):
        """torchmetrics implementation of MeanAveragePrecision expects predictions and targets to be a list of dictionaries. Each dictionary corresponds to a single image.
            Default box format is 'xyxy' (xmin, ymin, xmax, ymax).
        """

        predictions = [self._convert_to_dict(p) for p in predictions]
        targets = [self._convert_to_dict(t, scores=False) for t in targets]
        return predictions, targets

    @staticmethod
    def _convert_to_dict(boxes, scores=True):
        """
        Args:
            boxes (list): list of boxes. Each box is a list of 6 (or 5 when no score) elements: [label, score, x1, y1, x2, y2]
        """
        if not boxes:
            boxes = torch.empty(0, 6) if scores else torch.empty(0, 5)
        else:
            boxes = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
            boxes = boxes.unsqueeze(0) if len(boxes.shape) == 1 else boxes

        if scores:
            return {'boxes': boxes[:, -4:].reshape(-1, 4), 'labels': boxes[:, 0].reshape(-1), 'scores': boxes[:, 1].reshape(-1)}
        else:
            return {'boxes': boxes[:, -4:].reshape(-1, 4), 'labels': boxes[:, 0].reshape(-1)}
