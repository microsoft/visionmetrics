import torchmetrics

from visionmetrics.classification.common import MulticlassMixin, MultilabelMixin


class MulticlassPrecision(MulticlassMixin, torchmetrics.classification.MulticlassPrecision):
    pass


class MulticlassRecall(MulticlassMixin, torchmetrics.classification.MulticlassRecall):
    pass


class MulticlassF1Score(MulticlassMixin, torchmetrics.classification.MulticlassF1Score):
    pass


class MultilabelPrecision(MultilabelMixin, torchmetrics.classification.MultilabelPrecision):
    pass


class MultilabelRecall(MultilabelMixin, torchmetrics.classification.MultilabelRecall):
    pass


class MultilabelF1Score(MultilabelMixin, torchmetrics.classification.MultilabelF1Score):
    pass
