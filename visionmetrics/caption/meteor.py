from .caption_eval_base import ImageCaptionEvaluatorBase


class METEORScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    METEOR score evaluator for image caption task. For more details, refer to http://www.cs.cmu.edu/~alavie/METEOR/.
    """

    def __init__(self):
        super().__init__(metric='METEOR')
