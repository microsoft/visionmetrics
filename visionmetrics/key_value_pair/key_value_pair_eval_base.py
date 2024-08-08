from enum import Enum
from functools import reduce
import logging
import operator
import torch
from torchmetrics import Metric

# Import relevant visionmetrics modules; even though they appear to be unused to flake8, they are needed for dynamic metric instantiation at runtime.
from visionmetrics import caption, classification, detection, grounding, regression  # noqa: F401

logger = logging.getLogger(__name__)


class SupportedKeyWiseMetric(str, Enum):
    Caption_AzureOpenAITextModelCategoricalScore = "caption.AzureOpenAITextModelCategoricalScore"
    Classification_MulticlassAccuracy = "classification.MulticlassAccuracy"
    Classification_MultilabelAccuracy = "classification.MultilabelAccuracy"
    Classification_MulticlassF1 = "classification.MulticlassF1Score"
    Classification_MultilabelF1 = "classification.MultilabelF1Score"
    Detection_MeanAveragePrecision = "detection.MeanAveragePrecision"
    Detection_MicroPrecisionRecallF1 = "detection.DetectionMicroPrecisionRecallF1"
    Regression_MeanAbsoluteError = "regression.MeanAbsoluteError"
    Regression_MeanAbsoluteErrorF1Score = "regression.MeanAbsoluteErrorF1Score"


SUPPORTED_KEY_WISE_METRICS = [m.value for m in SupportedKeyWiseMetric]
SUPPORTED_F1_METRICS = [
    SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
    SupportedKeyWiseMetric.Classification_MulticlassF1,
    SupportedKeyWiseMetric.Classification_MultilabelF1,
    SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
    SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score
]


class KeyValuePairEvaluatorBase(Metric):
    """
    Evaluator for evaluating key-value pair datasets, where each key in the labels represents an extracted field with a consistent type (string, number, etc.).
    It accepts a dictionary mapping the keys to corresponding metric information, and returns the corresponding key-wise set of metrics.
    Each key can have a different metric for evaluation. The metrics supported are specified in SupportedKeyWiseMetric.

    Args:
        key_metric_map: dictionary from keys (extracted field names) to a dictionary with four required fields:
        1. metric_name: string of the metric name as defined in visionmetrics (e.g., 'classification.MulticlassAccuracy'), which should be among the ones specified in SupportedKeyWiseMetric.
        2. metric_args: dictionary of args to pass in as keyword arguments to the initialization function of the metric.
        3. preprocessor: function object that can be called with (prediction, target) values for a single instance to preprocess them into the desired format for the corresponding metric.
        4. key_trace: list of strings of key names that traces the path to the current key in the key-value pair prediction/target object (not in the schema).
        Examples (corresponding to the examples in key_value_pair_eval.py):
        defect_detection_key_metric_map = {
            "defect_types": {
                "metric_name": SupportedKeyWiseMetric.Classification_MultilabelF1,
                "metric_args": {"num_labels": 5, "average": "micro"},
                "preprocessor": <reference to multilabel classification preprocessing function; see example implementation in key_value_pair_eval.py>,
                "key_trace": ["defect_types"]
            },
            "defect_locations": {
                "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                "metric_args": {"box_format": "xyxy", "coords": "absolute"},
                "preprocessor": <reference to detection preprocessing function; see example implementation in key_value_pair_eval.py>,
                "key_trace": ["defect_locations"]
            },
            "rationale": {
                "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
                "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
                "preprocessor": <reference to captioning preprocessing function; see example implementations in key_value_pair_eval.py>,
                "key_trace": ["rationale"]
            }
        }
        brand_sentiment_key_metric_map = {
            "brand_sentiment_has_non_contoso_brands": {
                "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
                "metric_args": {"num_classes": 3, "average": "micro"},
                "preprocessor": <reference to multiclass classification preprocessing function; see example implementation in key_value_pair_eval.py>,
                "key_trace": ["brand_sentiment", "has_non_contoso_brands"]
            },
            "brand_sentiment_contoso_specific_sentiment": {
                "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
                "metric_args": {"num_classes": 6, "average": "micro"},
                "preprocessor": <reference to multiclass classification preprocessing function; see example implementation in key_value_pair_eval.py>,
                "key_trace": ["brand_sentiment", "contoso_specific", "sentiment"]
            },
            "brand_sentiment_contoso_specific_logo_bounding_box": {
                "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                "metric_args": {"box_format": "xyxy", "coords": "absolute"},
                "preprocessor": <reference to detection preprocessing function; see example implementation in key_value_pair_eval.py>,
                "key_trace": ["brand_sentiment", "contoso_specific", "logo_bounding_box"]
            }
        }
    }
    """
    def __init__(self, key_metric_map: dict):
        super().__init__()
        if not isinstance(key_metric_map, dict):
            raise ValueError("key_metric_map must be a dictionary mapping keys to their corresponding evaluation metrics.")
        self.key_metric_map = key_metric_map

        self.key_evaluator_map = {}
        for key in key_metric_map:
            # Validate metric_name and metric_args
            try:
                metric_name = key_metric_map[key]["metric_name"]
                metric_args = key_metric_map[key]["metric_args"]
            except KeyError:
                raise ValueError(f"Each value in key_metric_map must be a dictionary with a 'metric_name' key for a supported metric in visionmetrics of the form <task_name>.<metric_name>"
                                 f" and a 'metric_args' key for a dictionary of arguments to pass to the initialization function of that metric. Key '{key}' does not satisfy this rule.")
            if metric_name not in SUPPORTED_KEY_WISE_METRICS:
                raise ValueError(f"Metric '{key_metric_map[key]['metric_name']}' is not supported. "
                                 f"Each key's metric must be among the supported metrics: {', '.join(SUPPORTED_KEY_WISE_METRICS)}.")

            # Instantiate metric
            try:
                self.key_evaluator_map[key] = eval(f"{metric_name}(**metric_args)")
            except Exception as e:
                raise ValueError(f"Encountered error '{repr(e)}' when instantiating metric '{metric_name}' for key '{key}' with arguments '{metric_args}'.")

    def update(self, predictions, targets):
        """
        Updates metrics for each key using all samples in predictions and targets.
        Both predictions and targets should be dictionaries of the form {'<key>': <value>}, where <value> is in the format expected for the respective metric for that key.
        Each sample in predictions and targets must have the same keys (though they do not have to have all the keys in the dataset).
        """
        for key in self.key_evaluator_map:
            metric = self.key_evaluator_map[key]
            metric_name = self.key_metric_map[key]["metric_name"]
            key_predictions = []
            key_targets = []
            for prediction, target in zip(predictions, targets):
                # Use the key trace to traverse the prediction and target objects to get the values for these keys
                try:
                    key_prediction = reduce(operator.getitem, self.key_metric_map[key]["key_trace"], prediction)
                except KeyError:
                    logger.debug(f"No prediction exists in this sample for key '{key}'.")
                    continue
                try:
                    key_target = reduce(operator.getitem, self.key_metric_map[key]["key_trace"], target)
                except KeyError:
                    logger.debug(f"No target exists in this sample for key '{key}'.")
                    continue

                # Construct expected evaluation metric update format for the current key
                try:
                    preprocessor = self.key_metric_map[key]["preprocessor"]
                    key_prediction_formatted, key_target_formatted = preprocessor(key_prediction, key_target)
                    key_predictions.append(key_prediction_formatted)
                    key_targets.append(key_target_formatted)
                except ValueError as e:
                    logger.debug(f"Encountered error {repr(e)} when preprocessing prediction '{key_prediction}' and target '{key_target}' for key '{key}' to the metric's expected format.")

            # Convert lists to tensors for metrics that expect torch tensors
            if metric_name in [SupportedKeyWiseMetric.Classification_MulticlassAccuracy, SupportedKeyWiseMetric.Classification_MulticlassF1,
                               SupportedKeyWiseMetric.Classification_MultilabelAccuracy, SupportedKeyWiseMetric.Classification_MultilabelF1,
                               SupportedKeyWiseMetric.Regression_MeanAbsoluteError, SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score]:
                key_predictions = torch.tensor(key_predictions)
                key_targets = torch.tensor(key_targets)

            try:
                metric.update(key_predictions, key_targets)
            except Exception as e:
                raise ValueError(f"Encountered error '{repr(e)}' when updating metric '{self.key_metric_map[key]['metric_name']}' for key '{key}'.")

    def compute(self):
        """
        Computes key-wise metrics and returns a dictionary mapping keys to verbatim results from the evaluator for the corresponding key.
        """
        key_wise_scores = {k: None for k in self.key_evaluator_map}
        for key in self.key_evaluator_map:
            try:
                metric = self.key_evaluator_map[key]
                key_wise_scores[key] = metric.compute()
            except Exception as e:
                raise ValueError(f"Encountered error '{repr(e)}' when computing metric '{self.key_metric_map[key]['metric_name']}' for key '{key}'.")

        # If all keys support F1 scores, compute total micro- and macro-averaged F1 scores
        if all([self.key_metric_map[key]["metric_name"] in SUPPORTED_F1_METRICS]):
            macro_f1 = 0.
            total_tp, total_fp, total_fn = 0, 0, 0
            for key in self.key_metric_map:
                metric_name = self.key_metric_map[key]["metric_name"]
                if metric_name == SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore:
                    macro_f1 += key_wise_scores[key]["F1"]
                    total_tp += key_wise_scores[key]["TP"]
                    total_fp += key_wise_scores[key]["FP_GTNull"]
                    total_fp += key_wise_scores[key]["FP_GTNotNull"]
                    total_fn += key_wise_scores[key]["FN"]
                elif metric_name == SupportedKeyWiseMetric.Classification_MulticlassF1 or metric_name == SupportedKeyWiseMetric.Classification_MultilabelF1:
                    macro_f1 += key_wise_scores[key].item()
                    total_tp += self.key_evaluator_map[key].tp.sum().item()
                    total_fp += self.key_evaluator_map[key].fp.sum().item()
                    total_fn += self.key_evaluator_map[key].fn.sum().item()
                elif metric_name == SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1 or metric_name == SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score:
                    macro_f1 += key_wise_scores[key]["F1"]
                    total_tp += self.key_evaluator_map[key].tp.item()
                    total_fp += self.key_evaluator_map[key].fp.item()
                    total_fn += self.key_evaluator_map[key].fn.item()
            macro_f1 = macro_f1 / len(self.key_metric_map)
            try:
                total_precision = total_tp / (total_tp + total_fp)
            except ZeroDivisionError:
                total_precision = 0.
            try:
                total_recall = total_tp / (total_tp + total_fn)
            except ZeroDivisionError:
                total_recall = 0.
            try:
                micro_f1 = (2 * total_precision * total_recall) / (total_precision + total_recall)
            except ZeroDivisionError:
                micro_f1 = 0.

        return {
            "MicroF1": micro_f1,
            "MacroF1": macro_f1,
            "KeyWiseScores": key_wise_scores
        }