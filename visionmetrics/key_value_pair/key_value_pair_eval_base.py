from enum import Enum
import logging
import torch
from torchmetrics import Metric

# Import relevant visionmetrics modules; even though they appear to be unused, they are needed for dynamic metric instantiation at runtime.
from visionmetrics import caption, classification, detection, grounding, regression

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
                raise ValueError(f"Metric '{key_metric_map[key]['metric_name']}' is not supported. Each key's metric must be among the supported metrics: {', '.join([m.value for m in SupportedKeyWiseMetric])}.")

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
                try:
                    key_prediction = prediction[key]
                except KeyError:
                    logger.debug(f"No prediction exists in this sample for key '{key}'.")
                try:
                    key_target = target[key]
                except KeyError:
                    logger.debug(f"No target exists in this sample for key '{key}'.")
                if (key_prediction and not key_target) or (not key_prediction and key_target):
                    raise ValueError(f"Prediction and target for this sample have different keys; key '{key}' is present in one, but not the other.")

                # Construct expected evaluation metric update format for the current key
                try:
                    preprocessor = self.key_metric_map[key]["preprocessor"]
                    key_prediction_formatted = preprocessor(key_prediction)
                    key_target_formatted = preprocessor(key_target)
                    key_predictions.append(key_prediction_formatted)
                    key_targets.append(key_target_formatted)
                except KeyError:
                    if not isinstance(key_prediction, dict) or not isinstance(key_target, dict):
                        logger.debug(f"Skipping prediction and target for key '{key}' since the key does not have a preprocessor, but either prediction or target is not a dictionary. "
                                     "This means they do not have potential subkeys to be evaluated either.")
                        continue
                    subkey_full_name = f"{key}_{subkey}"
                    try:
                        preprocessor = self.key_metric_map[subkey_full_name]["preprocessor"]
                    except KeyError:
                        logger.debug(f"Skipping prediction '{key_prediction}' and target '{key_target}' for key '{key}' -> subkey '{subkey}' since preprocessor does not exist.")
                        continue
                    for subkey in key_prediction:
                        try:
                            key_prediction_formatted = preprocessor(key_prediction[subkey])
                            key_predictions.append(key_prediction_formatted)
                        except KeyError:
                            logger.debug(f"Skipping prediction '{key_prediction}' for key '{key}' -> subkey '{subkey}' since predicted value does not exist.")
                    for subkey in key_target:
                        try:
                            key_target_formatted = preprocessor(key_target[subkey])
                            key_targets.append(key_target_formatted)
                        except KeyError:
                            logger.debug(f"Skipping target '{key_target}' for key '{key}' -> subkey '{subkey}' since target value does not exist.")
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
