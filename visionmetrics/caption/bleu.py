from .caption_eval_base import ImageCaptionEvaluatorBase


class BleuScore(ImageCaptionEvaluatorBase):
    """
    BLEU score evaluator for image caption task. For more details, refer to http://www.aclweb.org/anthology/P02-1040.pdf.
    """

    def __init__(self):
        super().__init__(metric='Bleu')
        self.predictions = []
        self.targets = []
