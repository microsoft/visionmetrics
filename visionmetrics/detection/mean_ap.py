import torch
from torchmetrics import detection


class MeanAveragePrecision(detection.mean_ap.MeanAveragePrecision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, predictions, targets):
        predictions, targets = self._preprocess(predictions, targets)
        super().update(predictions, targets)

    def _preprocess(self, predictions, targets):
        """torchmetrics implementation of MeanAveragePrecision expects predictions and targets to be a list of dictionaries
        """
        _predictions = []
        _targets = []
        for prediction, target in zip(predictions, targets):
            _predictions.append(self._convert_prediction_to_dict(prediction))
            _targets.append(self._convert_target_to_dict(target))

        return _predictions, _targets

    @staticmethod
    def _convert_prediction_to_dict(prediction):
        _prediction = {
            'boxes': torch.empty(0, 4),
            'scores': torch.empty(0),
            'labels': torch.empty(0),
        }
        for single_prediction in prediction:
            assert len(single_prediction) == 6  # [label, score, x1, y1, x2, y2]
            _prediction['boxes'] = torch.tensor(single_prediction[-4:]) if _prediction['boxes'].numel() == 0 else torch.vstack((_prediction['boxes'], torch.tensor(single_prediction[-4:])))
            _prediction['scores'] = torch.tensor(single_prediction[1]) if _prediction['scores'].numel() == 0 else torch.vstack((_prediction['scores'], torch.tensor(single_prediction[1])))
            _prediction['labels'] = torch.tensor(single_prediction[0]) if _prediction['labels'].numel() == 0 else torch.vstack((_prediction['labels'], torch.tensor(single_prediction[0])))

        _prediction['boxes'] = _prediction['boxes'].reshape(-1, 4)
        _prediction['labels'] = _prediction['labels'].reshape(-1)
        _prediction['scores'] = _prediction['scores'].reshape(-1)
        return _prediction

    @staticmethod
    def _convert_target_to_dict(target):
        _targets = {
            'boxes': torch.empty(0, 4),
            'labels': torch.empty(0),
        }
        for single_target in target:
            assert len(single_target) == 5  # [label, x1, y1, x2, y2]
            _targets['boxes'] = torch.tensor(single_target[-4:]) if _targets['boxes'].numel() == 0 else torch.vstack((_targets['boxes'], torch.tensor(single_target[-4:])))
            _targets['labels'] = torch.tensor(single_target[0]) if _targets['labels'].numel() == 0 else torch.vstack((_targets['labels'], torch.tensor(single_target[0])))

        _targets['boxes'] = _targets['boxes'].reshape(-1, 4)
        _targets['labels'] = _targets['labels'].reshape(-1)
        return _targets


# if __name__ == '__main__':

#     metric = MeanAveragePrecision()

#     predictions = [[[0, 1.0, 0, 0, 1, 1],
#                     [1, 1.0, 0.5, 0.5, 1, 1],
#                     [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

#     targets = [[[0, 0, 0, 1, 1],
#                 [1, 0.5, 0.5, 1, 1],
#                 [2, 0.1, 0.1, 0.5, 0.5]]]

#     metric.update(predictions, targets)
#     print(metric.compute())
