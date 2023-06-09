# Import metrics directly from torchmetrics
from torchmetrics.classification import (AveragePrecision, CalibrationError,
                                         MulticlassAccuracy,
                                         MulticlassPrecision)

# Import custom metrics from visionmetrics
from visionmetrics.classification.accuracy import MultilabelAccuracy
from visionmetrics.classification.precision_recall import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
