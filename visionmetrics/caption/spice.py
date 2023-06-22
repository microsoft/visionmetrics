from visionmetrics.caption.caption_eval_base import ImageCaptionEvaluatorBase


class SPICEScoreEvaluator(ImageCaptionEvaluatorBase):
    """
    SPICE score evaluator for image caption task. For more details, refer to https://arxiv.org/abs/1607.08822.
    """

    def __init__(self):
        super().__init__(metric='SPICE')
