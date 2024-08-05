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
    Regression_MeanAbsoluteError = "regression.MeanAbsoluteError"


class KeyValuePairEvaluatorBase(Metric):
    """
    Evaluator for evaluating key-value pair datasets, where each key in the labels represents an extracted field with a consistent type (string, number, etc.).
    It accepts a dictionary mapping the keys to corresponding metric information, and returns the corresponding key-wise set of metrics.
    Each key can have a different metric for evaluation. The metrics supported are specified in SupportedKeyWiseMetric.

    Args:
        key_metric_map: dictionary from keys (extracted field names) to a dictionary with three required fields:
        1. metric_name: string of the metric name as defined in visionmetrics (e.g., 'classification.MulticlassAccuracy'), which should be among the ones specified in SupportedKeyWiseMetric.
        2. metric_args: dictionary of args to pass in as keyword arguments to the initialization function of the metric.
        3. preprocessor: function object that can be called with (prediction, target) values for a single instance to preprocess them into the desired format for the corresponding metric.
        4. key_trace: list of strings of key names that traces the path to the current key in the key-value pair prediction/target object (not in the schema).
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
            if metric_name not in [m.value for m in SupportedKeyWiseMetric]:
                raise ValueError(f"Metric '{key_metric_map[key]['metric_name']}' is not supported. "
                                 f"Each key's metric must be among the supported metrics: {', '.join([m.value for m in SupportedKeyWiseMetric])}.")

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
                               SupportedKeyWiseMetric.Regression_MeanAbsoluteError]:
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
        key_wise_results = {k: None for k in self.key_evaluator_map}
        for key in self.key_evaluator_map:
            try:
                metric = self.key_evaluator_map[key]
                key_wise_results[key] = metric.compute()
            except Exception as e:
                raise ValueError(f"Encountered error '{repr(e)}' when computing metric '{self.key_metric_map[key]['metric_name']}' for key '{key}'.")
        return key_wise_results
