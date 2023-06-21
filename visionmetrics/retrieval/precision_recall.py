import torch
import torchmetrics


class RetrievalExtend:
    def update(self, predictions, targets):
        """ torchmetrics implementation of Retrieval* metrics expects an indexes tensor corresponding to each query.
        For e.g. preds = torch.tensor([[0.6, 0.3],[0.2, 0.7])) --> indexes = torch.tensor([[0, 0], [1, 1]])
        """
        indexes = self._create_indexes(predictions)
        super().update(predictions, targets, indexes)

    def _create_indexes(self, predictions):
        if len(predictions.shape) == 1:
            indexes = torch.zeros_like(predictions)
        else:
            indexes = torch.arange(predictions.shape[0]).unsqueeze(1).repeat(1, predictions.shape[1])
        return indexes.long()


class RetrievalPrecision(RetrievalExtend, torchmetrics.retrieval.RetrievalPrecision):
    pass


class RetrievalRecall(RetrievalExtend, torchmetrics.retrieval.RetrievalRecall):
    pass


class RetrievalMAP(RetrievalExtend, torchmetrics.retrieval.RetrievalMAP):
    pass


class RetrievalPrecisionRecallCurve(RetrievalExtend, torchmetrics.retrieval.RetrievalPrecisionRecallCurve):
    pass
