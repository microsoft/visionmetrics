import torch


class TopKPredictionFilter:

    def __init__(self, k: int, prediction_mode='prob'):
        """
        Args:
            k: k predictions with highest confidence
            prediction_mode: can be 'indices' or 'prob', indicating whether the predictions are a set of class indices or predicted probabilities.
        """
        assert k >= 0
        assert prediction_mode == 'prob' or prediction_mode == 'indices', f"Prediction mode {prediction_mode} is not supported!"

        self.prediction_mode = prediction_mode
        self.k = k

    def filter(self, predictions, return_mode='vec'):
        """ Return k class predictions with highest confidence.
        Args:
            predictions:
                when 'prediction_mode' is 'prob', refers to predicted probabilities of N samples belonging to C classes. Shape (N, C)
                when 'prediction_mode' is 'indices', refers to indices of M highest confidence of C classes in descending order, for each of the N samples. Shape (N, M)
            return_mode: can be 'indices' or 'vec', indicating whether return value is a set of class indices or 0-1 vector

        Returns:
            k labels with highest probabilities, for each sample
        """

        k = min(predictions.shape[1], self.k)

        if self.prediction_mode == 'prob':
            if k == 0:
                top_k_pred_indices = torch.tensor([[] for i in range(predictions.shape[1])], dtype=int)
            elif k == 1:
                top_k_pred_indices = torch.argmax(predictions, axis=1)
                top_k_pred_indices = top_k_pred_indices.reshape((-1, 1))
            else:
                top_k_pred_indices = torch.topk(predictions, k=k, dim=1).indices
        else:
            top_k_pred_indices = predictions[:, :k]

        if return_mode == 'indices':
            return list(top_k_pred_indices)
        else:
            preds = torch.zeros_like(predictions, dtype=int)
            row_index = torch.arange(predictions.shape[0]).repeat_interleave(k)
            col_index = top_k_pred_indices.reshape((1, -1))
            preds[row_index, col_index] = 1

            return preds
