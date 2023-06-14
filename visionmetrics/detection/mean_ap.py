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
        _predictions = []
        _targets = []
        for prediction, target in zip(predictions, targets):
            _predictions.append(self._convert_to_dict(prediction))
            _targets.append(self._convert_to_dict(target, scores=False))

        return _predictions, _targets

    @staticmethod
    def _convert_to_dict(boxes, scores=True):
        boxes_per_image = {
            'boxes': torch.empty(0, 4),
            'labels': torch.empty(0),
            'scores': torch.empty(0),
        }
        for box in boxes:
            assert len(box) >= 5  # e.g. [label, score, xmin, ymin, xmax, ymax] or [label, xmin, ymin, xmax, ymax]
            boxes_per_image['boxes'] = torch.tensor(box[-4:]) if boxes_per_image['boxes'].numel() == 0 else torch.vstack((boxes_per_image['boxes'], torch.tensor(box[-4:])))
            boxes_per_image['labels'] = torch.tensor(box[0]) if boxes_per_image['labels'].numel() == 0 else torch.vstack((boxes_per_image['labels'], torch.tensor(box[0])))
            if scores:
                boxes_per_image['scores'] = torch.tensor(box[1]) if boxes_per_image['scores'].numel() == 0 else torch.vstack((boxes_per_image['scores'], torch.tensor(box[1])))

        boxes_per_image['boxes'] = boxes_per_image['boxes'].reshape(-1, 4)  # [N, 4]
        boxes_per_image['labels'] = boxes_per_image['labels'].reshape(-1)  # [N]
        boxes_per_image['scores'] = boxes_per_image['scores'].reshape(-1)  # [N]

        if not scores:
            boxes_per_image.pop('scores')

        return boxes_per_image
