from irisml.tasks.create_azure_openai_chat_model import OpenAITextChatModel
import unittest

from visionmetrics.key_value_pair.key_value_pair_eval import KeyValuePairExtractionScore, SupportedKeyWiseMetric


class TestKeyValuePairExtractionEvaluator(unittest.TestCase):
    ENDPOINT = "https://endpoint-name.openai.azure.com/"
    DEPLOYMENT = "gpt-4o"
    simple_schema = {
        "image_description": {
            "type": "string",
            "description": "Description of the image in a few sentences, with attention to detail."
        },
        "number_of_chinchillas": {
            "type": "integer",
            "description": "Number of chinchillas visible in the image."
        },
        "estimated_temperature": {
            "type": "number",
            "description": "Estimated temperature of the environment in the image, in degrees Celsius."
        },
        "escaped": {
            "type": "boolean",
            "description": "Whether any chinchillas appear to have escaped the cage."
        },
        "activity": {
            "type": "string",
            "description": "The most salient activity of the chinchillas in the image.",
            "enum": ["sleeping", "running", "playing", "fighting", "eating", "drinking"]
        },
        "cage_number": {
            "type": "integer",
            "description": "The cage number being shown in the scene.",
            "enum": [1, 2, 3, 4, 5]
        },
        "cage_bounding_box": {
            "type": "bbox",
            "description": "The bounding box of the cage in the image."
        }
    }
    simple_key_metric_map = {
        "image_description": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["image_description"]
        },
        "number_of_chinchillas": {
            "metric_name": SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score,
            "metric_args": {"error_threshold": 1.0},
            "key_trace": ["number_of_chinchillas"]
        },
        "estimated_temperature": {
            "metric_name": SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score,
            "metric_args": {"error_threshold": 1.0},
            "key_trace": ["estimated_temperature"]
        },
        "escaped": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 3, "average": "micro"},
            "class_map": {"True": 0, "False": 1, "<|other|>": 2},
            "key_trace": ["escaped"]
        },
        "activity": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 7, "average": "micro"},
            "class_map": {"sleeping": 0, "running": 1, "playing": 2, "fighting": 3, "eating": 4, "drinking": 5, "<|other|>": 6},
            "key_trace": ["activity"]
        },
        "cage_number": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 6, "average": "micro"},
            "class_map": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "<|other|>": 5},
            "key_trace": ["cage_number"]
        },
        "cage_bounding_box": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"single_class": 0},
            "key_trace": ["cage_bounding_box"]
        }
    }

    simple_list_schema = {
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
    simple_list_key_metric_map = {
        "defect_types": {
            "metric_name": SupportedKeyWiseMetric.Classification_MultilabelF1,
            "metric_args": {"num_labels": 5, "average": "micro"},
            "class_map": {"scratch": 0, "dent": 1, "discoloration": 2, "crack": 3, "<|other|>": 4},
            "key_trace": ["defect_types"]
        },
        "defect_locations": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"single_class": 0},
            "key_trace": ["defect_locations"]
        },
        "rationale": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["rationale"]
        }
    }

    complex_list_schema = {
        "defect_types": {
            "type": "array",
            "description": "The defect types with bounding boxes present in the image.",
            "items": {
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "The type of defect detected",
                        "enum": ["scratch", "dent", "discoloration", "crack"]
                    },
                    "defect_location": {
                        "type": "bbox",
                        "description": "Bounding box indicating the location of the defect."
                    }
                }
            }
        }
    }
    complex_list_key_metric_map = {
        "defect_types": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["defect_types"]
        }
    }

    simple_object_schema = {
        "chart_type": {
            "type": "string",
            "description": "The type of chart shown in the image.",
            "enum": ["line", "bar", "pie", "scatter", "waterfall", "histogram", "gantt", "heatmap"]
        },
        "chart_title": {
            "type": "string",
            "description": "The title of the chart, exactly as it appears in the image."
        },
        "chart_axes": {
            "type": "object",
            "description": "Information about the axes of the chart in the image.",
            "properties": {
                "x_axis_title": {
                    "type": "string",
                    "description": "The title of the x-axis."
                },
                "y_axis_title": {
                    "type": "string",
                    "description": "The title of the y-axis."
                },
                "x_axis_units": {
                    "type": "string",
                    "description": "The units of the x-axis."
                },
                "y_axis_units": {
                    "type": "string",
                    "description": "The units of the y-axis."
                }
            }
        }
    }
    simple_object_key_metric_map = {
        "chart_type": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 9, "average": "micro"},
            "class_map": {"line": 0, "bar": 1, "pie": 2, "scatter": 3, "waterfall": 4, "histogram": 5, "gantt": 6, "heatmap": 7, "<|other|>": 8},
            "key_trace": ["chart_type"]
        },
        "chart_title": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_title"]
        },
        "chart_axes_x_axis_title": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_axes", "x_axis_title"]
        },
        "chart_axes_y_axis_title": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_axes", "y_axis_title"]
        },
        "chart_axes_x_axis_units": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_axes", "x_axis_units"]
        }
    }

    complex_object_schema = {
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
    complex_object_key_metric_map = {
        "brand_sentiment_has_non_contoso_brands": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 3, "average": "micro"},
            "class_map": {"True": 0, "False": 1, "<|other|>": 2},
            "key_trace": ["brand_sentiment", "has_non_contoso_brands"]
        },
        "brand_sentiment_contoso_specific_sentiment": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 6, "average": "micro"},
            "class_map": {"very positive": 0, "somewhat positive": 1, "neutral": 2, "somewhat negative": 3, "negative": 4, "<|other|>": 5},
            "key_trace": ["brand_sentiment", "contoso_specific", "sentiment"]
        },
        "brand_sentiment_contoso_specific_logo_bounding_box": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"single_class": 0},
            "key_trace": ["brand_sentiment", "contoso_specific", "logo_bounding_box"]
        }
    }

    # In each unit test, we don"t unit test the preprocessor name explicitly because we will test it implicitly via the "update" function call tests.
    def test_key_value_pair_extraction_evaluator_simple_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_key_metric_map:
                for field in self.simple_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "image_description": "Two chinchillas eating.",
                                "number_of_chinchillas": 2,
                                "estimated_temperature": 21,
                                "escaped": False,
                                "activity": "eating",
                                "cage_number": 3,
                                "cage_bounding_box": [0, 0, 2000, 3000]
                            }],
                            targets=[{
                                "image_description": "Two chinchillas are eating in a cage.",
                                "number_of_chinchillas": 2,
                                "estimated_temperature": 24,
                                "escaped": False,
                                "activity": "eating",
                                "cage_number": 2,
                                "cage_bounding_box": [0, 0, 2100, 2500]
                            }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["image_description"]["F1"], 1.0)
        self.assertEqual(report["KeyWiseScores"]["number_of_chinchillas"]["F1"], 1.)
        self.assertEqual(report["KeyWiseScores"]["estimated_temperature"]["F1"], 0.)
        self.assertEqual(report["KeyWiseScores"]["escaped"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["activity"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["cage_number"].item(), 0.)
        self.assertEqual(report["KeyWiseScores"]["cage_bounding_box"]["F1"], 1.)

        self.assertAlmostEqual(report["MicroF1"], 0.7142857142857143)
        self.assertAlmostEqual(report["MacroF1"], 0.7142857142857143)

    def test_key_value_pair_extraction_evaluator_simple_list_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_list_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_list_key_metric_map:
                for field in self.simple_list_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_list_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "defect_types": ["scratch", "crack"],
                                "defect_locations": [[0, 0, 10, 10], [10, 10, 20, 20]],
                                "rationale": "There are two small shadows in the upper left corner of the image, which appear to be abnormal."
                            }],
                            targets=[{
                                "defect_types": ["scratch", "dent"],
                                "defect_locations": [[0, 0, 10, 15], [0, 10, 20, 30]]
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["defect_types"].item(), 0.5000)
            self.assertEqual(report["KeyWiseScores"]["defect_locations"]["F1"], 0.5)
            self.assertEqual(report["KeyWiseScores"]["rationale"]["F1"], 0)

            self.assertEqual(report["MicroF1"], 0.5)
            self.assertAlmostEqual(report["MacroF1"], 0.3333333333333333)

    def test_key_value_pair_extraction_evaluator_complex_list_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_list_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.0"]):
            for key in self.complex_list_key_metric_map:
                for field in self.complex_list_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.complex_list_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "defect_types": [{"defect_type": "scratch", "defect_location": [0, 0, 10, 10]}]
                            }],
                            targets=[{
                                "defect_types": [{"defect_type": "dent", "defect_location": [0, 0, 20, 20]}]
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["defect_types"]["F1"], 0.0)

            self.assertEqual(report["MicroF1"], 0.0)
            self.assertEqual(report["MacroF1"], 0.0)

    def test_key_value_pair_extraction_evaluator_simple_object_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_object_key_metric_map:
                for field in self.simple_object_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_object_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "chart_type": "bar",
                                "chart_title": "Number of groundhogs that saw their shadow from 2000-2024",
                                "chart_axes": {
                                    "x_axis_title": "Year",
                                    "y_axis_title": "Groundhogs",
                                    "x_axis_units": "Year",
                                    "y_axis_units": "Count"
                                }
                            }],
                            targets=[{
                                "chart_type": "bar",
                                "chart_title": "Count of groundhogs that saw their shadow from 2000-2024",
                                "chart_axes": {
                                    "x_axis_title": "Year",
                                    "y_axis_title": "Groundhog",
                                    "x_axis_units": "Year",
                                    "y_axis_units": "Count"
                                }
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["chart_type"].item(), 1.0)
            self.assertEqual(report["KeyWiseScores"]["chart_title"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["chart_axes_x_axis_title"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["chart_axes_y_axis_title"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["chart_axes_x_axis_units"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["chart_axes_y_axis_units"]["F1"], 1.0)

            self.assertEqual(report["MicroF1"], 1.0)
            self.assertEqual(report["MacroF1"], 1.0)

    def test_key_value_pair_extraction_evaluator_nested_object_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        for key in self.complex_object_key_metric_map:
            for field in self.complex_object_key_metric_map[key]:
                self.assertEqual(evaluator.key_metric_map[key][field], self.complex_object_key_metric_map[key][field])
        evaluator.update(predictions=[{
                            "brand_sentiment": {
                                "has_non_contoso_brands": True,
                                "contoso_specific": {
                                    "sentiment": "somewhat positive",
                                    "logo_bounding_box": None
                                }
                            }
                        }],
                        targets=[{
                            "brand_sentiment": {
                                "has_non_contoso_brands": True,
                                "contoso_specific": {
                                    "sentiment": "very positive",
                                    "logo_bounding_box": [0, 0, 100, 100]
                                }
                            }
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 0.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logo_bounding_box"]["F1"], 0.)

        self.assertAlmostEqual(report["MicroF1"], 0.3333333333333333)
        self.assertAlmostEqual(report["MacroF1"], 0.3333333333333333)
