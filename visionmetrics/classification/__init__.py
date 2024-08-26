# Import metrics directly from torchmetrics
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryAveragePrecision,
                                         BinaryConfusionMatrix, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall,
                                         MulticlassAUROC,
                                         MulticlassAveragePrecision,
                                         MulticlassCalibrationError,
                                         MulticlassConfusionMatrix,
                                         MulticlassExactMatch,
                                         MultilabelAccuracy, MultilabelAUROC,
                                         MultilabelAveragePrecision,
                                         MultilabelConfusionMatrix,
                                         MultilabelExactMatch)

# Import custom metrics from visionmetrics
from visionmetrics.classification.accuracy import MulticlassAccuracy
from visionmetrics.classification.precision_recall import (MulticlassF1Score,
                                                           MulticlassPrecision,
                                                           MulticlassRecall,
                                                           MultilabelF1Score,
                                                           MultilabelF1ScoreWithDuplicates,
                                                           MultilabelPrecision,
                                                           MultilabelRecall)
