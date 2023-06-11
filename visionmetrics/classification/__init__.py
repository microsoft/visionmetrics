# Import metrics directly from torchmetrics
from torchmetrics.classification import (Accuracy, AveragePrecision,
                                         CalibrationError, MulticlassAccuracy,
                                         MulticlassPrecision,
                                         MultilabelAccuracy)

# Import custom metrics from visionmetrics
from visionmetrics.classification.precision_recall import (MultilabelF1Score,
                                                           MultilabelPrecision,
                                                           MultilabelRecall)
