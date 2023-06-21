from .caption_eval_base import ImageCaptionEvaluatorBase


class ROUGELScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    ROUGE_L score evaluator for image caption task. For more details, refer to http://anthology.aclweb.org/W/W04/W04-1013.pdf
    """

    def __init__(self):
        super().__init__(metric='ROUGE_L')
        self.predictions = []
        self.targets = []
