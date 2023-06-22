from torchmetrics import Metric


class ImageCaptionEvaluatorBase(Metric):
    """
    Base class for image caption metric evaluator
    """

    def __init__(self, metric):
        self.targets = []
        self.predictions = []
        super().__init__()
        self.metric = metric

    def update(self, predictions, targets):
        """ Evaluate list of image with image caption results using pycocoimcap tools.
        Args:
            predictions: list of string predictions [caption1, caption2, ...], shape: (N, ), type: string
            targets: list of string ground truth for image caption task: [[gt1, gt2, ...], [gt1, gt2, ...], ...], type: string
        """
        self.targets += targets
        self.predictions += predictions

    def reset(self):
        super().reset()
        self.targets = []
        self.predictions = []

    def compute(self, **kwargs):
        from .coco_evalcap_utils import ImageCaptionCOCOEval, ImageCaptionCOCO, ImageCaptionWrapper
        imcap_predictions, imcap_targets = ImageCaptionWrapper.convert(self.predictions, self.targets)
        coco = ImageCaptionCOCO(imcap_targets)
        cocoRes = coco.loadRes(imcap_predictions)
        cocoEval = ImageCaptionCOCOEval(coco, cocoRes, self.metric)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        result = cocoEval.eval
        return result
