import torchmetrics

from visionmetrics.classification.common import MulticlassMixin


class MulticlassAccuracy(MulticlassMixin, torchmetrics.classification.MulticlassAccuracy):
    pass
