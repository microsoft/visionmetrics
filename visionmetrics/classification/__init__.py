# Import metrics directly from torchmetrics
from torchmetrics.classification import (BinaryAUROC, MulticlassAccuracy,
                                         MulticlassAUROC,
                                         MulticlassAveragePrecision,
                                         MulticlassCalibrationError,
                                         MulticlassConfusionMatrix,
                                         MulticlassPrecision, MulticlassRecall,
                                         MultilabelAccuracy, MultilabelAUROC,
                                         MultilabelAveragePrecision,
                                         MultilabelConfusionMatrix)

# Import custom metrics from visionmetrics
from visionmetrics.classification.precision_recall import (MultilabelF1Score,
                                                           MultilabelPrecision,
                                                           MultilabelRecall)
