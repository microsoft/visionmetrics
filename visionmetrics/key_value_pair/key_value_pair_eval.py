from enum import Enum

from visionmetrics.key_value_pair.key_value_pair_eval_base import KeyValuePairEvaluatorBase, SupportedKeyWiseMetric


OUT_OF_DISTRIBUTION_ENUM_KEY = "<|other|>"


class JSONSchemaKeyType(str, Enum):
    String = "string"
    Number = "number"
    Integer = "integer"
    Boolean = "boolean"
    BoundingBox = "bbox"
    Array = "array"
    Object = "object"


SIMPLE_KEY_TYPES = [JSONSchemaKeyType.String, JSONSchemaKeyType.Number, JSONSchemaKeyType.Integer, JSONSchemaKeyType.Boolean, JSONSchemaKeyType.BoundingBox]


class KeyValuePairExtractionScore(KeyValuePairEvaluatorBase):
    """
    Evaluator for a key-value pair dataset. The default evaluator (if the key does not have values of a particular form) is the caption.AzureOpenAITextModelCategoricalScore due to its flexibility.
    Note: Currently, detection metrics are class-agnostic; bounding boxes are evaluated separately from classes. TODO: Consider supporting multiclass detection metrics, which may complicate logic.
    Args:
        key_value_pair_schema: dictionary in JSON Schema format indicating each key and expected value type for each extracted field.
        Examples (corresponding to the examples in key_value_pair_eval_base.py):
        defect_detection_schema = {
            "defect_types": {
                "type": "array",
                "description": "The defect types present in the image.",
                "items": {
                    "type": "string",
                    "description": "The type of defect detected",
                    "enum": ["scratch", "dent", "discoloration", "crack"]
                }
            },
            "defect_locations": {
                "type": "array",
                "description": "The defect bounding boxes corresponding to each of the identified types.",
                "items": {
                    "type": "bbox",
                    "description": "Bounding box indicating the location of the defect."
                }
            },
            "rationale": {
                "type": "string",
                "description": "Rationale for the identified defects."
            }
        }
        Note: In the above defect_detection_schema example, if "defect_types" and "defect_locations" were both items of a larger "defects" array rather than separate top-level arrays,
        then the whole "defects" field would be evaluated as text. We currently do not support non-text evaluation for arrays with complex item types.
        brand_sentiment_schema = {
            "brand_sentiment": {
                "type": "object",
                "description": "Attributes of sentiment toward brands depicted in the image.",
                "properties": {
                    "has_non_contoso_brands": {
                        "type": "boolean",
                        "description": "Whether the image depicts or contains anything about non-Contoso brands."
                    },
                    "contoso_specific": {
                        "type": "object",
                        "description": "Sentiment related specifically to the company Contoso.",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "description": "Sentiment toward the brand as depicted in the image.",
                                "enum": ["very positive", "somewhat positive", "neutral", "somewhat negative", "negative"]
                            },
                            "logo_bounding_box": {
                                "type": "bbox",
                                "description": "The bounding box around the Contoso logo in the image, if applicable."
                            }
                        }
                    }
                }
            }
        }
        Note: In the above brand_sentiment_schema example, since the nested objects are all objects (not arrays), the object is recursively traversed to assign metrics to the
        innermost object fields ("sentiment" and "logo_bounding_box").

        endpoint: string of the Azure OpenAI endpoint to be used as the default text evaluator.
        deployment_name: string of the Azure OpenAI deployment name to be used for the default text evaluator.
        The latter two arguments follow the standards of irisml.tasks.create_azure_openai_chat_model.OpenAITextChatModel.
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
            classes: list of class names, of any type compatible with the str() function.
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
        # Use text as the default metric for all keys
        self._assign_key_metric_map_values(key=key,
                                           metric_name=SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
                                           metric_args={"endpoint": self.endpoint, "deployment_name": self.deployment_name})
        if key_schema["type"] == JSONSchemaKeyType.String:
            if "enum" in key_schema:
                class_map = self._get_enum_class_map(key_schema["enum"])
                self._assign_key_metric_map_values(key=key,
                                                   metric_name=SupportedKeyWiseMetric.Classification_MulticlassF1,
                                                   metric_args={"num_classes": len(class_map), "average": "micro"},
                                                   class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.Number or key_schema["type"] == JSONSchemaKeyType.Integer:
            self._assign_key_metric_map_values(key=key,
                                               metric_name=SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score,
                                               metric_args={"error_threshold": 1.0})
            if "enum" in key_schema:
                class_map = self._get_enum_class_map(key_schema["enum"])
                self._assign_key_metric_map_values(key=key,
                                                   metric_name=SupportedKeyWiseMetric.Classification_MulticlassF1,
                                                   metric_args={"num_classes": len(class_map), "average": "micro"},
                                                   class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.Boolean:
            class_map = self._get_enum_class_map([True, False])
            self._assign_key_metric_map_values(key=key,
                                               metric_name=SupportedKeyWiseMetric.Classification_MulticlassF1,
                                               metric_args={"num_classes": len(class_map), "average": "micro"},
                                               class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.BoundingBox:
            # Currently only supports class-agnostic detection metrics
            self._assign_key_metric_map_values(key=key,
                                               metric_name=SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                                               metric_args={"iou_threshold": 0.5, "box_format": "xyxy", "coords": "absolute"},
                                               class_map={"single_class": 0})
        elif key_schema["type"] == JSONSchemaKeyType.Array:
            # For more complex arrays, we default to the caption evaluator
            if key_schema["items"]["type"] in SIMPLE_KEY_TYPES:
                if key_schema["items"]["type"] == JSONSchemaKeyType.BoundingBox:
                    # Currently only supports class-agnostic detection metrics
                    self._assign_key_metric_map_values(key=key,
                                                       metric_name=SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
                                                       metric_args={"iou_threshold": 0.5, "box_format": "xyxy", "coords": "absolute"},
                                                       class_map={"single_class": 0})
                else:
                    if "enum" in key_schema["items"]:
                        class_map = self._get_enum_class_map(key_schema["items"]["enum"])
                        self._assign_key_metric_map_values(key=key,
                                                           metric_name=SupportedKeyWiseMetric.Classification_MultilabelF1,
                                                           metric_args={"num_labels": len(class_map), "average": "micro"},
                                                           class_map=class_map)
        elif key_schema["type"] == JSONSchemaKeyType.Object:
            for subkey in key_schema["properties"]:
                subkey_name = f"{key}_{subkey}"
                self._populate_key_metric_map(key=subkey_name, key_schema=key_schema["properties"][subkey], key_trace=key_trace + [subkey])
            del self.key_metric_map[key]

        if key in self.key_metric_map:
            self._populate_key_preprocessor(key=key, type=key_schema["type"])
            self.key_metric_map[key]["key_trace"] = key_trace

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
            self.key_metric_map[key]["preprocessor"] = lambda pred, gt: (str(pred), [str(gt)])
        elif metric_name == SupportedKeyWiseMetric.Classification_MulticlassAccuracy or metric_name == SupportedKeyWiseMetric.Classification_MulticlassF1:
            # Expects torch int or float tensor of shape (N, ...) or (N, C, ...) for predictions, torch tensor of shape (N, ...) for targets
            class_map = self.key_metric_map[key]["class_map"]
            self.key_metric_map[key]["preprocessor"] = lambda pred, gt: (class_map.get(str(pred), class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)),
                                                                         class_map.get(str(gt), class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)))
        elif metric_name == SupportedKeyWiseMetric.Classification_MultilabelAccuracy or metric_name == SupportedKeyWiseMetric.Classification_MultilabelF1:
            # Expects torch int or float tensor of shape (N, C, ...)
            class_map = self.key_metric_map[key]["class_map"]

            def multilabel_preprocess(pred, gt):
                class_indices_pred = [class_map.get(p, class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)) for p in pred]
                one_hot_pred = [1 if class_index in class_indices_pred else 0 for class_index in range(0, len(class_map))]
                class_indices_gt = [class_map.get(g, class_map.get(OUT_OF_DISTRIBUTION_ENUM_KEY)) for g in gt]
                one_hot_gt = [1 if class_index in class_indices_gt else 0 for class_index in range(0, len(class_map))]
                return (one_hot_pred, one_hot_gt)
            self.key_metric_map[key]["preprocessor"] = multilabel_preprocess
        elif metric_name == SupportedKeyWiseMetric.Detection_MeanAveragePrecision or metric_name == SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1:
            # Expects list of list of list of classes, [optionally scores], and bounding box coordinates, e.g., [[[0, 1.0, 0, 0, 10, 10]]]
            class_map = self.key_metric_map[key]["class_map"]
            if type == JSONSchemaKeyType.BoundingBox:
                def detection_preprocess(pred, gt):
                    if not isinstance(pred, list) or (isinstance(pred, list) and len(pred) < 4):
                        return_pred = [[class_map["single_class"]] + [1.0] + [0., 0., 0., 0.]]
                    else:
                        return_pred = [[class_map["single_class"]] + pred] if len(pred) == 5 else [[class_map["single_class"]] + [1.0] + pred]
                    if not isinstance(gt, list) or (isinstance(gt, list) and len(gt) < 4):
                        return_gt = [[class_map["single_class"]] + [0., 0., 0., 0.]]
                    else:
                        return_gt = [[class_map["single_class"]] + gt]
                    return (return_pred, return_gt)
                self.key_metric_map[key]["preprocessor"] = detection_preprocess
            elif type == JSONSchemaKeyType.Array:
                def detection_preprocess(pred, gt):
                    return_pred = []
                    return_gt = []
                    for p in pred:
                        if not isinstance(p, list) or (isinstance(p, list) and len(p) < 4):
                            return_pred.append([class_map["single_class"]] + [1.0] + [0., 0., 0., 0.])
                        else:
                            return_pred.append([class_map["single_class"]] + p if len(p) == 5 else [class_map["single_class"]] + [1.0] + p)
                    for g in gt:
                        if not isinstance(g, list) or (isinstance(g, list) and len(g) < 4):
                            return_gt.append([class_map["single_class"]] + [0., 0., 0., 0.])
                        else:
                            return_gt.append([class_map["single_class"]] + g)
                    return (return_pred, return_gt)
                self.key_metric_map[key]["preprocessor"] = detection_preprocess
        elif metric_name == SupportedKeyWiseMetric.Regression_MeanAbsoluteError or metric_name == SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score:
            # Expects torch int or float tensor of shape (N)
            self.key_metric_map[key]["preprocessor"] = lambda pred, gt: (float(pred), float(gt))