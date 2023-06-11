# Import metrics directly from torchmetrics
from torchmetrics.classification import Accuracy

# Import custom metrics from visionmetrics
from visionmetrics.classification.precision_recall import (MultilabelF1Score,
                                                           MultilabelPrecision,
                                                           MultilabelRecall)
