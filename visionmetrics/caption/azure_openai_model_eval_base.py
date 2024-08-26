import torch
from collections import Counter
from enum import Enum
from irisml.tasks.create_azure_openai_chat_model import OpenAITextChatModel
import logging
from torchmetrics import Metric

from visionmetrics.common.utils import precision_recall_f1_scalar

logger = logging.getLogger(__name__)


# Prompt template placeholder constants
PREDICTION_PLACEHOLDER = "<|prediction|>"
TARGET_PLACEHOLDER = "<|target|>"

# Response parsing constants
MULTIPLE_RESPONSES_DELIMITER = "<|delimiter|>"

# Evaluator separator constants for joining multiple ground truth values
OR_SEPARATOR = " <|OR|> "
AND_SEPARATOR = " <|AND|> "


# Enum for per-value result status type
class ResultStatusType(Enum):
    TruePositive = 1
    TrueNegative = 2
    FalseNegative = 3
    FalsePositiveGtNotNull = 4
    FalsePositiveGtNull = 5


class AzureOpenAITextModelCategoricalEvaluatorBase(Metric):
    """
    Evaluator that uses single-turn text-only calls to an Azure OpenAI model for evaluation and returns
    standard categorical metrics (precision, recall, F1, accuracy) based on the scores returned.

    Args:
        prompt_template: string template for the prompt instructing how to score the prediction. Must have two placeholders: <|prediction|> and <|target|>.
        positive_threshold: determines the threshold (inclusive) at which the score assigned by the model to the prediction means the prediction is correct. Range: [0.0, 1.0].
        negative_value: indicates the value (e.g., None, 'null', 0, '', []) for which predicted and target values are considered null. Note that the evaluator strictly requires
                        an exact match for either a prediction or target to be considered negative. TODO: Implement fuzzy negative matching. Scores are always computed so they
                        can be used for average_score computation.

        Other parameters follow the standards of irisml.tasks.create_azure_openai_chat_model.OpenAITextChatModel;
        see https://github.com/microsoft/irisml-tasks-azure-openai/blob/main/irisml/tasks/create_azure_openai_chat_model.py.
    """
    def __init__(self, endpoint: str, deployment_name: str, system_message: str, prompt_template: str,
                 temperature=0.0, max_tokens=50, requests_interval=30, num_responses=1, positive_threshold=0.5, negative_value=''):
        super().__init__()
        if PREDICTION_PLACEHOLDER not in prompt_template or TARGET_PLACEHOLDER not in prompt_template:
            raise ValueError("Both the predicted placeholder {PREDICTION_PLACEHOLDER} and target placeholder {TARGET_PLACEHOLDER} must be present in prompt_template.")
        if positive_threshold < 0.0 or positive_threshold > 1.0:
            raise ValueError("Parameter positive_threshold should be between [0.0, 1.0], inclusive.")
        logger.info(f"Initializing evaluator with positive_threshold={positive_threshold}, negative_value={negative_value}, temperature={temperature}, max_tokens={max_tokens}, "
                    "system_message=\"{system_message}\", prompt_template=\"{prompt_template}\"")
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.positive_threshold = positive_threshold
        self.negative_value = negative_value

        if num_responses != 1:
            raise NotImplementedError("This metric currently only supports single-response scoring.")
        self.num_responses = num_responses

        self.model = OpenAITextChatModel(endpoint=endpoint, deployment_name=deployment_name, api_key=None, temperature=temperature, max_tokens=max_tokens,
                                         requests_inverval=requests_interval, num_responses=num_responses, delimiter=MULTIPLE_RESPONSES_DELIMITER, system_message=system_message)

        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("raw_scores", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("score_parse_failures", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("result_status_types", default=[], dist_reduce_fx="cat")

    def _get_numeric_score(self, raw_score: str):
        """
        Parses the numeric score value from the string `raw_score`; if not parseable, returns 0.
        """
        score = 0.
        try:
            score = float(raw_score)
        except ValueError:
            self.score_parse_failures += 1
            logger.debug(f"Failed to parse raw_score \"{raw_score}\" to numeric value; returning 0.")
        return score

    def update(self, predictions, targets, multi_target_separator=AND_SEPARATOR):
        """
        Evaluate list of predicted results using Azure OpenAI text model.
        Args:
            predictions: list of string predictions [text1, text2, ...], shape: (N, ), type: string
            targets: list of one or more string ground truth values: [[gt1, gt2, ...], [gt1, gt2, ...], ...], type: string
                Note: with multiple targets, a good match with any of them is considered correct.
            multiple_target_join_method: string specifying the joining method to join multiple reference ground truth values in the prompt, either 'any' or 'all'
        """
        if len(predictions) != len(targets):
            raise ValueError("Please update with an equal number of predictions and targets.")

        self.predictions += predictions
        self.targets += targets

        for prediction, target in zip(predictions, targets):
            prompt = self.prompt_template.replace(PREDICTION_PLACEHOLDER, prediction)
            final_target = multi_target_separator.join(target)
            prompt = prompt.replace(TARGET_PLACEHOLDER, final_target)

            if prediction == target:
                result = "1.0"
            else:
                result = self.model(([prompt], [[]]))[0]
            if self.num_responses == 1:
                self.raw_scores.append(result)
                numeric_score = self._get_numeric_score(result)
                self.scores.append(numeric_score)
            else:
                # TODO: Implement multiple-response score aggregation
                pass

            # Assign ResultStatusTypes
            target_is_negative = not target or all([t == self.negative_value for t in target])
            prediction_match = numeric_score >= self.positive_threshold
            if prediction == self.negative_value:
                # If any target value is negative, it is a true negative
                if target_is_negative:
                    self.result_status_types.append(ResultStatusType.TrueNegative)
                else:
                    self.result_status_types.append(ResultStatusType.FalseNegative)
            else:
                if target_is_negative:
                    self.result_status_types.append(ResultStatusType.FalsePositiveGtNull)
                else:
                    self.result_status_types.append(ResultStatusType.TruePositive if prediction_match else ResultStatusType.FalsePositiveGtNotNull)

    def compute(self):
        result_counts = Counter(self.result_status_types)
        tp, tn, fn, fp_gt_null, fp_gt_not_null = result_counts[ResultStatusType.TruePositive], \
            result_counts[ResultStatusType.TrueNegative], \
            result_counts[ResultStatusType.FalseNegative], \
            result_counts[ResultStatusType.FalsePositiveGtNull], \
            result_counts[ResultStatusType.FalsePositiveGtNotNull]
        precision_recall_f1 = precision_recall_f1_scalar(tp=tp, fp=fp_gt_null + fp_gt_not_null, fn=fn + fp_gt_not_null)
        try:
            accuracy = (tp + tn) / (tp + tn + fn + fp_gt_null + fp_gt_not_null)
        except ZeroDivisionError:
            accuracy = 0.
        try:
            average_score = sum(self.scores) / len(self.scores)
        except ZeroDivisionError:
            average_score = 0.
        return {
            # Summary statistics
            "Precision": precision_recall_f1["Precision"],
            "Recall": precision_recall_f1["Recall"],
            "F1": precision_recall_f1["F1"],
            "Accuracy": accuracy,
            "AverageScore": average_score,
            # Raw statistic counts
            "TP": tp,
            "TN": tn,
            "FN": fn,
            "FP_GTNull": fp_gt_null,
            "FP_GTNotNull": fp_gt_not_null,
            "ScoreParseFailures": self.score_parse_failures.item()
        }
