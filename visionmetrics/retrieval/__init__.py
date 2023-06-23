# Import metrics directly from torchmetrics
# from torchmetrics.retrieval import RetrievalRecall, RetrievalPrecision

# Import custom metrics from visionmetrics
from visionmetrics.retrieval.precision_recall import (
    RetrievalMAP, RetrievalPrecision, RetrievalPrecisionRecallCurveNPoints,
    RetrievalRecall)
