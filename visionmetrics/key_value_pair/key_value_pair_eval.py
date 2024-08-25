from enum import Enum
import logging

from visionmetrics.key_value_pair.key_value_pair_eval_base import KeyValuePairEvaluatorBase, SupportedKeyWiseMetric

logger = logging.getLogger(__name__)


class JSONSchemaKeyType(str, Enum):
    String = "string"
    Number = "number"
    Integer = "integer"
    Boolean = "boolean"
    Array = "array"
    Object = "object"


# Constants for schemas
SIMPLE_KEY_TYPES = [JSONSchemaKeyType.String, JSONSchemaKeyType.Number, JSONSchemaKeyType.Integer, JSONSchemaKeyType.Boolean]
OUT_OF_DISTRIBUTION_ENUM_KEY = "<|other|>"

# Constants for predictions and targets
VALUE_SUBKEY = "value"
GROUNDINGS_SUBKEY = "groundings"


class KeyValuePairExtractionScore(KeyValuePairEvaluatorBase):
    """
    Key-value pair extraction refers to the task of arbitrary schema-based structured field extraction. Each schema is an adapted version of a
    JSON Schema (https://json-schema.org/understanding-json-schema/reference)-formatted dictionary that contains keys, each of which specifies
    the standard JSON Schema type of the key and a string description, whether to perform grounding on the key, classes for closed-vocabulary
    values, and additional information describing list items and object properties (sub-keys).

    Exceptions to the standard JSON Schema format are:
    - Usage of an "includeGrounding" boolean for each field, indicating whether the field's annotations are grounded (whether each has a list of bounding boxes) or not.

    Based on the properties defined in the JSON Schema, this class infers the best evaluation metric for each key's data type, and defaults to text-based evaluation
    for cases that have no clear inferrable metric. It then constructs the key_metric_map specifying the mapping between each key of such a schema and the corresponding
    visionmetrics metric, metric arguments, preprocessors for predictions and targets, and a list trace describing the path to the key through the prediction and target objects.
    Evaluation and metric computation are done through the parent class, KeyValuePairEvaluatorBase. Supported key-wise metrics are enumerated in KeyValuePairEvaluatorBase.SupportedKeyWiseMetric.

    Args:
        key_value_pair_schema: dictionary in JSON Schema format indicating each key and expected value type for each extracted field.
        Examples (corresponding to the examples in key_value_pair_eval_base.py):
        defect_detection_schema = {
            "defects": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["scratch", "dent", "discoloration", "crack"],
                    "includeGrounding": True
                }
            },
            "rationale": {
                "type": "string"
            }
        }
        "brand_sentiment": {
            "type": "object",
            "properties": {
                "has_non_contoso_brands": {
                    "type": "boolean"
                },
                "contoso_specific": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["very positive", "somewhat positive", "neutral", "somewhat negative", "negative"]
                        },
                        "logos": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["text", "grayscale", "rgb"],
                                "includeGrounding": True
                            }
                        }
                    }
                }
            }
        }
        Note: In the above brand_sentiment_schema example, since the nested objects are all objects (not arrays), the object is recursively traversed to assign metrics to the
        innermost object fields ("sentiment" and "logos").

        endpoint: string of the Azure OpenAI endpoint to be used as the default text evaluator.
        deployment_name: string of the Azure OpenAI deployment name to be used for the default text evaluator.

        The latter two arguments follow the standards of irisml.tasks.create_azure_openai_chat_model.OpenAITextChatModel;
        see https://github.com/microsoft/irisml-tasks-azure-openai/blob/main/irisml/tasks/create_azure_openai_chat_model.py.
    """
    def __init__(self, key_value_pair_schema: dict, endpoint: str, deployment_name: str):
        if not isinstance(key_value_pair_schema, dict):
            raise ValueError("key_value_pair_schema must be a dictionary in JSON Schema format specifying the schema for the dataset.")
        self.key_value_pair_schema = key_value_pair_schema
        self.endpoint = endpoint
        self.deployment_name = deployment_name

        # Parse the schema and map each key to a metric
        self.key_metric_map = {}
        for key in key_value_pair_schema:
            key_schema = key_value_pair_schema[key]
            self._populate_key_metric_map(key=key, key_schema=key_schema, key_trace=[key])

        super().__init__(key_metric_map=self.key_metric_map)

    def _get_enum_class_map(self, classes: list):
        """
        Constructs and returns an enum class map from string class names to integer class indices,
        used in preprocessing enum-type values that are evaluated using classification and detection metrics.
        Args:
            classes: list of class names. Class names can be strings, integers, or floats.
        """
        class_map = {str(class_name): i for i, class_name in enumerate(classes)}
        # Reserve one class to catch cases where predictions are not in the set of expected classes
        class_map[OUT_OF_DISTRIBUTION_ENUM_KEY] = len(class_map)
        return class_map

    def _assign_key_metric_map_values(self, key: str, metric_name: str, metric_args: dict, class_map: dict = None):
        """
        Assigns values for the metric_name, metric_args, and class_map for a given key in the object-level key_metric_map dictionary.
        Args:
            key: string of the key name, which should be identical to the key that will be evaluated in the prediction and target dictionaries.
            metric_name: string of the metric name as defined in visionmetrics (e.g., 'classification.MulticlassAccuracy'), which
                         should be among those specified in key_value_pair_eval_base.SupportedKeyWiseMetric.
            metric_args: dictionary of args to pass in as keyword arguments to the initialization function of the metric.
            class_map (optional): dictionary mapping string class names to integer class indices for this key.
        """
        if key not in self.key_metric_map:
            self.key_metric_map[key] = {}
        self.key_metric_map[key]["metric_name"] = metric_name
        self.key_metric_map[key]["metric_args"] = metric_args
        if class_map:
            self.key_metric_map[key]["class_map"] = class_map

    def _populate_key_metric_map(self, key: str, key_schema: dict, key_trace: list):
        """
        Recursive function that populates the object-level key_metric_map dictionary. For a given key and key_schema in JSON Schema format,
        the function populates the metric_name, metric_args, and preprocessor values in the dictionary.
        Note: The recursive behavior currently only occurs for JSONSchemaKeyType.Object key types, not arrays.
        Args:
            key: string of the key name, which should be identical to the key that will be evaluated in the prediction and target dictionaries.
            key_schema: dictionary in JSON schema format specifying the expected schema for the prediction and target dictionaries to be used in evaluation.
            key_trace: list of strings of key names that traces the path to the current key in the key-value pair prediction/target object (not in the schema).
        """
        if key_schema["type"] in [JSONSchemaKeyType.String, JSONSchemaKeyType.Number, JSONSchemaKeyType.Integer]:
            if "enum" in key_schema:
                class_map = self._get_enum_class_map(key_schema["enum"])
                if "includeGrounding" in key_schema:
                    self._assign_key_metric_map_values(key=key,
                                                       metric_name=SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                                                       metric_args={"iou_threshold": 0.5, "box_format": "xyxy", "coords": "absolute"},
                                                       class_map=class_map)
                else:
                    self._assign_key_metric_map_values(key=key,
                                                       metric_name=SupportedKeyWiseMetric.Classification_MulticlassF1,
                                                       metric_args={"num_classes": len(class_map), "average": "micro"},
                                                       class_map=class_map)
            elif key_schema["type"] in [JSONSchemaKeyType.Number, JSONSchemaKeyType.Integer]:
                self._assign_key_metric_map_values(key=key,
                                                   metric_name=SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score,
                                                   metric_args={"error_threshold": 0.0})
        elif key_schema["type"] == JSONSchemaKeyType.Boolean:
            class_map = self._get_enum_class_map([True, False])
            if "includeGrounding" in key_schema:
                self._assign_key_metric_map_values(key=key,
                                                   metric_name=SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                                                   metric_args={"iou_threshold": 0.5, "box_format": "xyxy", "coords": "absolute"},
                                                   class_map=class_map)
            else:
                self._assign_key_metric_map_values(key=key,
                                                   metric_name=SupportedKeyWiseMetric.Classification_MulticlassF1,
                                                   metric_args={"num_classes": len(class_map), "average": "micro"},
                                                   class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.Array:
            # For more complex arrays, we default to the caption evaluator
            if key_schema["items"]["type"] in SIMPLE_KEY_TYPES:
                if "enum" in key_schema["items"]:
                    class_map = self._get_enum_class_map(key_schema["items"]["enum"])
                    if "includeGrounding" in key_schema["items"]:
                        self._assign_key_metric_map_values(key=key,
                                                           metric_name=SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                                                           metric_args={"iou_threshold": 0.5, "box_format": "xyxy", "coords": "absolute"},
                                                           class_map=class_map)
                    else:
                        self._assign_key_metric_map_values(key=key,
                                                           metric_name=SupportedKeyWiseMetric.Classification_MultilabelF1WithDuplicates,
                                                           metric_args={},
                                                           class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.Object:
            for subkey in key_schema["properties"]:
                subkey_name = f"{key}_{subkey}"
                self._populate_key_metric_map(key=subkey_name, key_schema=key_schema["properties"][subkey], key_trace=key_trace + [VALUE_SUBKEY, subkey])

        if key not in self.key_metric_map and key_schema["type"] != JSONSchemaKeyType.Object:
            # Use text as the default metric for all keys; 'object' key type should not have its own key
            self._assign_key_metric_map_values(key=key,
                                               metric_name=SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
                                               metric_args={"endpoint": self.endpoint, "deployment_name": self.deployment_name})
            logger.debug(f"Using default metric '{SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore.value}' for key '{key}'.")

        if key in self.key_metric_map:
            self._populate_key_preprocessor(key=key, type=key_schema["type"])
            self.key_metric_map[key]["key_trace"] = key_trace

    def _detection_preprocess_single_prediction(self, key: str, value: dict):
        """
        Preprocessing function for a single detection prediction of the form {"value": <class_name>, "groundings": [[[optional: <score>], <x1>, <y1>, <x2>, <y2>], [...], [...]]}.
        Args:
            key: string of the key name.
            value: dictionary with a "value" key of type string, integer, or float, and a "groundings" list of lists of bounding boxes, optionally with scores.
        """
        class_map = self.key_metric_map[key]["class_map"]
        class_index = class_map.get(str(value[VALUE_SUBKEY]), class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY))
        groundings = value[GROUNDINGS_SUBKEY]
        if not isinstance(groundings, list):
            return_pred = [[class_index] + [0.0] + [0., 0., 0., 0.]]
        else:
            return_pred = []
            for grounding in groundings:
                if not isinstance(grounding, list) or (isinstance(grounding, list) and len(grounding) < 4):
                    return_pred.append([class_index] + [0.0] + [0., 0., 0., 0.])
                else:
                    return_pred.append([class_index] + grounding if len(grounding) == 5 else [class_index] + [1.0] + grounding)
        return return_pred

    def _detection_preprocess_single_target(self, key: str, value: dict):
        """
        Preprocessing function for a single detection target of the form {"value": <class_name>, "groundings": [[<x1>, <y1>, <x2>, <y2>], [...], [...]]}.
        Args:
            key: string of the key name.
            value: dictionary with a "value" key of type string, integer, or float, and a "groundings" list of lists of bounding boxes.
        """
        class_map = self.key_metric_map[key]["class_map"]
        class_index = class_map.get(str(value[VALUE_SUBKEY]), class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY))
        groundings = value[GROUNDINGS_SUBKEY]
        if not isinstance(groundings, list):
            return_gt = [[0, 0., 0., 0., 0.]]
        else:
            return_gt = []
            for grounding in groundings:
                if not isinstance(grounding, list) or (isinstance(grounding, list) and len(grounding) < 4):
                    return_gt.append([class_index] + [0., 0., 0., 0.])
                else:
                    return_gt.append([class_index] + grounding)
        return return_gt

    def _populate_key_preprocessor(self, key: str, type: JSONSchemaKeyType):
        """
        Given a key and the specified JSON Schema type of that key, constructs the preprocessor function corresponding to the key
        and populates the object-level key_metric_map dictionary with the preprocessor function.
        Args:
            key: string of the key name, which should be identical to the key that will be evaluated in the prediction and target dictionaries.
            type: string specifying the JSON Schema type of the key, which should be among those in JSONSchemaKeyType.
        """
        metric_name = self.key_metric_map[key]["metric_name"]
        if metric_name == SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore:
            # Expects list of strings for predictions, list of list of strings for targets
            self.key_metric_map[key]["prediction_preprocessor"] = lambda value: str(value[VALUE_SUBKEY])
            self.key_metric_map[key]["target_preprocessor"] = lambda value: [str(value[VALUE_SUBKEY])]
        elif metric_name == SupportedKeyWiseMetric.Classification_MulticlassAccuracy or metric_name == SupportedKeyWiseMetric.Classification_MulticlassF1:
            # Expects torch int or float tensor of shape (N, ...) or (N, C, ...) for predictions, torch tensor of shape (N, ...) for targets
            class_map = self.key_metric_map[key]["class_map"]
            self.key_metric_map[key]["prediction_preprocessor"] = self.key_metric_map[key]["target_preprocessor"] = \
                lambda value: class_map.get(str(value[VALUE_SUBKEY]), class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY))
        elif metric_name == SupportedKeyWiseMetric.Classification_MultilabelAccuracy or metric_name == SupportedKeyWiseMetric.Classification_MultilabelF1:
            # Expects torch int or float tensor of shape (N, C, ...)
            class_map = self.key_metric_map[key]["class_map"]

            def multilabel_preprocess(value: list):
                class_indices = [class_map.get(v[VALUE_SUBKEY], class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)) for v in value[VALUE_SUBKEY]]
                one_hot_list = [1 if class_index in class_indices else 0 for class_index in range(0, len(class_map))]
                return one_hot_list
            self.key_metric_map[key]["prediction_preprocessor"] = self.key_metric_map[key]["target_preprocessor"] = multilabel_preprocess
        elif metric_name == SupportedKeyWiseMetric.Classification_MultilabelF1WithDuplicates:
            class_map = self.key_metric_map[key]["class_map"]
            self.key_metric_map[key]["prediction_preprocessor"] = self.key_metric_map[key]["target_preprocessor"] = \
                lambda value: [class_map.get(v[VALUE_SUBKEY], class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)) for v in value[VALUE_SUBKEY]]
        elif metric_name == SupportedKeyWiseMetric.Detection_MeanAveragePrecision or metric_name == SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1:
            # Expects list of list of list of classes, [optionally scores], and bounding box coordinates, e.g., [[[0, 1.0, 0, 0, 10, 10]]]
            if type in SIMPLE_KEY_TYPES:
                self.key_metric_map[key]["prediction_preprocessor"] = lambda value: self._detection_preprocess_single_prediction(key, value)
                self.key_metric_map[key]["target_preprocessor"] = lambda value: self._detection_preprocess_single_target(key, value)
            elif type == JSONSchemaKeyType.Array:
                def detection_list_preprocess_prediction(value):
                    value_list = value[VALUE_SUBKEY]
                    return_pred = []
                    for v in value_list:
                        return_pred.extend(self._detection_preprocess_single_prediction(key, v))
                    return return_pred

                def detection_list_preprocess_target(value):
                    value_list = value[VALUE_SUBKEY]
                    return_gt = []
                    for v in value_list:
                        return_gt.extend(self._detection_preprocess_single_target(key, v))
                    return return_gt
                self.key_metric_map[key]["prediction_preprocessor"] = detection_list_preprocess_prediction
                self.key_metric_map[key]["target_preprocessor"] = detection_list_preprocess_target
        elif metric_name == SupportedKeyWiseMetric.Regression_MeanAbsoluteError or metric_name == SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score:
            # Expects torch int or float tensor of shape (N)
            self.key_metric_map[key]["prediction_preprocessor"] = self.key_metric_map[key]["target_preprocessor"] = lambda value: float(value[VALUE_SUBKEY])
