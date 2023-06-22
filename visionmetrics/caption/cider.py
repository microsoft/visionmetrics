from visionmetrics.caption.caption_eval_base import ImageCaptionEvaluatorBase


class CIDErScore(ImageCaptionEvaluatorBase):
    """
    CIDEr score evaluator for image caption task. For more details, refer to http://arxiv.org/pdf/1411.5726.pdf.
    """

    def __init__(self):
        super().__init__(metric='CIDEr')
