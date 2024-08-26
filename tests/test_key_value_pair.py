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
            "enum": ["sleeping", "running", "playing", "fighting", "eating", "drinking", "none"],
            "includeGrounding": True
        },
        "cage_number": {
            "type": "integer",
            "description": "The cage number being shown in the scene.",
            "enum": [1, 2, 3, 4, 5]
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
            "metric_args": {"error_threshold": 0.0},
            "key_trace": ["number_of_chinchillas"]
        },
        "estimated_temperature": {
            "metric_name": SupportedKeyWiseMetric.Regression_MeanAbsoluteErrorF1Score,
            "metric_args": {"error_threshold": 0.0},
            "key_trace": ["estimated_temperature"]
        },
        "escaped": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 3, "average": "micro"},
            "class_map": {"True": 0, "False": 1, "<|other|>": 2},
            "key_trace": ["escaped"]
        },
        "activity": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"sleeping": 0, "running": 1, "playing": 2, "fighting": 3, "eating": 4, "drinking": 5, "none": 6, "<|other|>": 7},
            "key_trace": ["activity"]
        },
        "cage_number": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 6, "average": "micro"},
            "class_map": {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "<|other|>": 5},
            "key_trace": ["cage_number"]
        }
    }

    simple_list_schema = {
        "defects": {
            "type": "array",
            "description": "The defect types present in the image.",
            "items": {
                "type": "string",
                "description": "The type of defect detected",
                "enum": ["scratch", "dent", "discoloration", "crack"]
            }
        }
    }
    simple_list_key_metric_map = {
        "defects": {
            "metric_name": SupportedKeyWiseMetric.Classification_MultilabelF1WithDuplicates,
            "metric_args": {},
            "class_map": {"scratch": 0, "dent": 1, "discoloration": 2, "crack": 3, "<|other|>": 4},
            "key_trace": ["defects"]
        }
    }

    simple_list_grounding_schema = {
        "defects": {
            "type": "array",
            "description": "The defect types present in the image.",
            "items": {
                "type": "string",
                "description": "The type of defect detected",
                "enum": ["scratch", "dent", "discoloration", "crack"],
                "includeGrounding": True
            }
        }
    }
    simple_list_grounding_key_metric_map = {
        "defects": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"scratch": 0, "dent": 1, "discoloration": 2, "crack": 3, "<|other|>": 4},
            "key_trace": ["defects"]
        }
    }

    complex_list_schema = {
        "defects": {
            "type": "array",
            "description": "The defect types present in the image.",
            "items": {
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "The type of defect detected",
                        "enum": ["scratch", "dent", "discoloration", "crack"],
                        "includeGrounding": True
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Rationale for the defects identified."
                    }
                }
            }
        }
    }
    complex_list_key_metric_map = {
        "defects": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["defects"]
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
            "key_trace": ["chart_axes", "value", "x_axis_title"]
        },
        "chart_axes_y_axis_title": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_axes", "value", "y_axis_title"]
        },
        "chart_axes_x_axis_units": {
            "metric_name": SupportedKeyWiseMetric.Caption_AzureOpenAITextModelCategoricalScore,
            "metric_args": {"endpoint": ENDPOINT, "deployment_name": DEPLOYMENT},
            "key_trace": ["chart_axes", "value", "x_axis_units"]
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
                            "enum": ["very positive", "somewhat positive", "neutral", "somewhat negative", "very negative"]
                        },
                        "logos": {
                            "type": "array",
                            "description": "The types of Contoso logos present in the image.",
                            "items": {
                                "type": "string",
                                "description": "The type of Contoso logo in the image.",
                                "enum": ["text", "grayscale", "rgb"],
                                "includeGrounding": True
                            }
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
            "key_trace": ["brand_sentiment", "value", "has_non_contoso_brands"]
        },
        "brand_sentiment_contoso_specific_sentiment": {
            "metric_name": SupportedKeyWiseMetric.Classification_MulticlassF1,
            "metric_args": {"num_classes": 6, "average": "micro"},
            "class_map": {"very positive": 0, "somewhat positive": 1, "neutral": 2, "somewhat negative": 3, "very negative": 4, "<|other|>": 5},
            "key_trace": ["brand_sentiment", "value", "contoso_specific", "value", "sentiment"]
        },
        "brand_sentiment_contoso_specific_logos": {
            "metric_name": SupportedKeyWiseMetric.Detection_MicroPrecisionRecallF1,
            "metric_args": {"box_format": "xyxy", "coords": "absolute", "iou_threshold": 0.5},
            "class_map": {"text": 0, "grayscale": 1, "rgb": 2, "<|other|>": 3},
            "key_trace": ["brand_sentiment", "value", "contoso_specific", "value", "logos"]
        }
    }

    # In each unit test, we don"t unit test the preprocessor name explicitly because we will test it implicitly via the "update" function call tests.
    def test_two_images_simple_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_key_metric_map:
                for field in self.simple_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "image_description": {"value": "Two chinchillas eating."},
                                "number_of_chinchillas": {"value": 2},
                                "estimated_temperature": {"value": 21.0},
                                "escaped": {"value": False},
                                "activity": {"value": "eating", "groundings": [[0, 0, 200, 300]]},
                                "cage_number": {"value": 3}
                            }, {
                                "image_description": {"value": "There is possibly one chinchilla in the cage."},
                                "number_of_chinchillas": {"value": 0},
                                "estimated_temperature": {"value": 20.0},
                                "escaped": {"value": False},
                                "activity": {"value": "none", "groundings": [[0, 100, 200, 200]]},
                                "cage_number": {"value": 2}
                            }],
                            targets=[{
                                "image_description": {"value": "Two chinchillas are eating in a cage."},
                                "number_of_chinchillas": {"value": 2},
                                "estimated_temperature": {"value": 21.0},
                                "escaped": {"value": False},
                                "activity": {"value": "eating", "groundings": [[0, 0, 200, 300]]},
                                "cage_number": {"value": 2}
                            }, {
                                "image_description": {"value": "There is one sleeping chinchilla in the far corner of the cage."},
                                "number_of_chinchillas": {"value": 1},
                                "estimated_temperature": {"value": 20.5},
                                "escaped": {"value": False},
                                "activity": {"value": "snoozing", "groundings": [[0, 100, 250, 250]]},
                                "cage_number": {"value": 1}
                            }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["image_description"]["F1"], 1.0)
        self.assertEqual(report["KeyWiseScores"]["number_of_chinchillas"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["estimated_temperature"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["escaped"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["activity"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["cage_number"].item(), 0.)

        self.assertAlmostEqual(report["MicroF1"], 0.5833333333333333)
        self.assertAlmostEqual(report["MacroF1"], 0.5833333333333333)

    def test_simple_list_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_list_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_list_key_metric_map:
                for field in self.simple_list_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_list_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "defects": {"value": [
                                    {"value": "scratch"},
                                    {"value": "crack"},
                                    {"value": "scratch"}
                                ]}
                            }],
                            targets=[{
                                "defects": {"value": [
                                    {"value": "scratch"},
                                    {"value": "dent"}
                                ]}
                            }])
            report = evaluator.compute()
            self.assertAlmostEqual(report["KeyWiseScores"]["defects"]["Precision"], 0.3333333333333333)
            self.assertEqual(report["KeyWiseScores"]["defects"]["Recall"], 0.5)
            self.assertEqual(report["KeyWiseScores"]["defects"]["F1"], 0.4)

            self.assertEqual(report["MicroF1"], 0.4)
            self.assertAlmostEqual(report["MacroF1"], 0.4)

    def test_simple_list_grounding_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_list_grounding_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_list_grounding_key_metric_map:
                for field in self.simple_list_grounding_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_list_grounding_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "defects": {"value": [
                                    {"value": "scratch", "groundings": [[0, 0, 10, 10]]},
                                    {"value": "crack", "groundings": [[10, 10, 20, 20]]}
                                ]}
                            }],
                            targets=[{
                                "defects": {"value": [
                                    {"value": "scratch", "groundings": [[0, 0, 10, 15]]},
                                    {"value": "dent", "groundings": [[10, 10, 20, 20]]}
                                ]}
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["defects"]["F1"], 0.5)

            self.assertEqual(report["MicroF1"], 0.5)
            self.assertAlmostEqual(report["MacroF1"], 0.5)

    def test_complex_list_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_list_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.0"]):
            for key in self.complex_list_key_metric_map:
                for field in self.complex_list_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.complex_list_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "defects": {"value": [{
                                    "defect_type": {"value": "scratch", "groundings": [[0, 0, 10, 10]]},
                                    "explanation": {"value": "There is a line on the otherwise smooth metal."}
                                }]}
                            }],
                            targets=[{
                                "defects": {"value": [{
                                    "defect_type": {"value": "dent", "groundings": [[0, 0, 10, 10]]},
                                    "explanation": {"value": "There is a divot in the upper left."}
                                }]}
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["defects"]["F1"], 0.0)

            self.assertEqual(report["MicroF1"], 0.0)
            self.assertEqual(report["MacroF1"], 0.0)

    def test_simple_object_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_object_key_metric_map:
                for field in self.simple_object_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_object_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "chart_type": {"value": "bar"},
                                "chart_title": {"value": "Number of groundhogs that saw their shadow from 2000-2024"},
                                "chart_axes": {"value": {
                                    "x_axis_title": {"value": "Year"},
                                    "y_axis_title": {"value": "Groundhogs"},
                                    "x_axis_units": {"value": "Year"},
                                    "y_axis_units": {"value": "Count"}
                                }}
                            }],
                            targets=[{
                                "chart_type": {"value": "bar"},
                                "chart_title": {"value": "Count of groundhogs that saw their shadow from 2000-2024"},
                                "chart_axes": {"value": {
                                    "x_axis_title": {"value": "Year"},
                                    "y_axis_title": {"value": "Groundhog"},
                                    "x_axis_units": {"value": "Year"},
                                    "y_axis_units": {"value": "Count"}
                                }}
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

    def test_complex_object_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        for key in self.complex_object_key_metric_map:
            for field in self.complex_object_key_metric_map[key]:
                self.assertEqual(evaluator.key_metric_map[key][field], self.complex_object_key_metric_map[key][field])
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]},
                                        {"value": "something_else", "groundings": [[100, 5, 50, 50]]}
                                    ]}
                                }}
                            }}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 90, 90]]},
                                        {"value": "rgb", "groundings": [[100, 5, 50, 50]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 0.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 0.5)

        self.assertAlmostEqual(report["MicroF1"], 0.5)
        self.assertAlmostEqual(report["MacroF1"], 0.5)

    def test_prediction_missing_key(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 0.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 1.)

        self.assertAlmostEqual(report["MicroF1"], 0.8)
        self.assertAlmostEqual(report["MacroF1"], 0.6666666666666666)

    def test_target_missing_key(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        self.assertRaisesRegex(ValueError,
                               r"The key 'brand_sentiment_contoso_specific_logos' does not exist in the target sample "
                               r"'{'brand_sentiment': {'value': {'has_non_contoso_brands': {'value': True}, "
                               r"'contoso_specific': {'value': {'sentiment': {'value': 'very positive'}}}}}}'.",
                               evaluator.update,
                               [{
                                   "brand_sentiment": {"value": {
                                       "has_non_contoso_brands": {"value": True},
                                       "contoso_specific": {"value": {
                                           "sentiment": {"value": "very positive"},
                                           "logos": {"value": [
                                               {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                           ]}
                                        }}
                                    }}
                                }],
                               [{
                                   "brand_sentiment": {"value": {
                                       "has_non_contoso_brands": {"value": True},
                                       "contoso_specific": {"value": {
                                           "sentiment": {"value": "very positive"}
                                        }}
                                    }}
                                }]
                               )

    def test_prediction_invalid_keys(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]},
                                    "invalid_key": {"value": None}
                                }},
                                "another_invalid_key": {"value": None}
                            }}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 1.)

        self.assertAlmostEqual(report["MicroF1"], 0.75)
        self.assertAlmostEqual(report["MacroF1"], 1.0)

    def test_target_invalid_keys(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        self.assertRaisesRegex(ValueError,
                               r"The target sample '{'brand_sentiment': {'value': {'has_non_contoso_brands': {'value': True}, 'contoso_specific': "
                               r"{'value': {'sentiment': {'value': 'very positive'}, 'logos': {'value': \[{'value': 'text', 'groundings': \[\[0, 0, 100, 100\]\]}\]}, "
                               r"'invalid_key': {'value': 2}}}, 'another_invalid_key': {'value': None}}}}' has at least one invalid key "
                               r"not present in the schema: brand_sentiment_contoso_specific_invalid_key, brand_sentiment_another_invalid_key.",
                               evaluator.update,
                               [{
                                   "brand_sentiment": {"value": {
                                       "has_non_contoso_brands": {"value": True},
                                       "contoso_specific": {"value": {
                                           "sentiment": {"value": "very positive"},
                                           "logos": {"value": [
                                               {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                           ]}
                                        }}
                                    }}
                                }],
                               [{
                                   "brand_sentiment": {"value": {
                                       "has_non_contoso_brands": {"value": True},
                                       "contoso_specific": {"value": {
                                           "sentiment": {"value": "very positive"},
                                           "logos": {"value": [
                                               {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                           ]},
                                           "invalid_key": {"value": 2}
                                        }},
                                       "another_invalid_key": {"value": None}
                                    }}
                                }]
                               )

    def test_two_images_nested_object_schema(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        for key in self.complex_object_key_metric_map:
            for field in self.complex_object_key_metric_map[key]:
                self.assertEqual(evaluator.key_metric_map[key][field], self.complex_object_key_metric_map[key][field])
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        },
                        {
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": False},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[20, 20, 20, 20]]}
                                    ]}
                                }}
                            }}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        },
                        {
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": False},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[20, 20, 20, 20]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 1.)

        self.assertAlmostEqual(report["MicroF1"], 1.0)
        self.assertAlmostEqual(report["MacroF1"], 1.0)

    def test_two_images_prediction_invalid_keys(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        for key in self.complex_object_key_metric_map:
            for field in self.complex_object_key_metric_map[key]:
                self.assertEqual(evaluator.key_metric_map[key][field], self.complex_object_key_metric_map[key][field])
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "invalid_key_1": {"value": None},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }},
                                "invalid_key_2": {"value": None}
                            }}
                        }, {
                            "brand_sentiment": {"value": {
                                "invalid_key_3": {"value": None},
                                "has_non_contoso_brands": {"value": False},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[20, 20, 20, 20]]}
                                    ]},
                                    "invalid_key_4": {"value": None}
                                }}
                            }},
                            "invalid_key_5": {"value": None}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        }, {
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": False},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[20, 20, 20, 20]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 1.)

        self.assertAlmostEqual(report["MicroF1"], 0.7058823529411765)
        self.assertAlmostEqual(report["MacroF1"], 1.0)

    def test_two_images_prediction_wrong_invalid_keys(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.complex_object_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        for key in self.complex_object_key_metric_map:
            for field in self.complex_object_key_metric_map[key]:
                self.assertEqual(evaluator.key_metric_map[key][field], self.complex_object_key_metric_map[key][field])
        evaluator.update(predictions=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "invalid_key_1": {"value": None},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }},
                                "invalid_key_2": {"value": None}
                            }}
                        }, {
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "somewhat positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[20, 20, 40, 40]]}
                                    ]}
                                }}
                            }}
                        }],
                        targets=[{
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": True},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[0, 0, 100, 100]]}
                                    ]}
                                }}
                            }}
                        }, {
                            "brand_sentiment": {"value": {
                                "has_non_contoso_brands": {"value": False},
                                "contoso_specific": {"value": {
                                    "sentiment": {"value": "very positive"},
                                    "logos": {"value": [
                                        {"value": "text", "groundings": [[21, 22, 45, 40]]}
                                    ]}
                                }}
                            }}
                        }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_has_non_contoso_brands"].item(), 0.5)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_sentiment"].item(), 0.0)
        self.assertEqual(report["KeyWiseScores"]["brand_sentiment_contoso_specific_logos"]["F1"], 1.0)

        self.assertAlmostEqual(report["MicroF1"], 0.42857142857142855)
        self.assertAlmostEqual(report["MacroF1"], 0.5)

    def test_simple_schema_batch_update(self):
        evaluator = KeyValuePairExtractionScore(key_value_pair_schema=self.simple_schema,
                                                endpoint=self.ENDPOINT,
                                                deployment_name=self.DEPLOYMENT)
        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            for key in self.simple_key_metric_map:
                for field in self.simple_key_metric_map[key]:
                    self.assertEqual(evaluator.key_metric_map[key][field], self.simple_key_metric_map[key][field])
            evaluator.update(predictions=[{
                                "image_description": {"value": "Two chinchillas eating."},
                                "number_of_chinchillas": {"value": 2},
                                "estimated_temperature": {"value": 21},
                                "escaped": {"value": False},
                                "activity": {"value": "eating", "groundings": [[0, 0, 200, 300]]},
                                "cage_number": {"value": 3}
                            }],
                            targets=[{
                                "image_description": {"value": "Two chinchillas are eating in a cage."},
                                "number_of_chinchillas": {"value": 2},
                                "estimated_temperature": {"value": 24},
                                "escaped": {"value": False},
                                "activity": {"value": "eating", "groundings": [[0, 0, 210, 250]]},
                                "cage_number": {"value": 2}
                            }])
            report = evaluator.compute()
            self.assertEqual(report["KeyWiseScores"]["image_description"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["number_of_chinchillas"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["estimated_temperature"]["F1"], 0.0)
            self.assertEqual(report["KeyWiseScores"]["escaped"].item(), 1.0)
            self.assertEqual(report["KeyWiseScores"]["activity"]["F1"], 1.0)
            self.assertEqual(report["KeyWiseScores"]["cage_number"].item(), 0.0)

            self.assertAlmostEqual(report["MicroF1"], 0.6666666666666666)
            self.assertAlmostEqual(report["MacroF1"], 0.6666666666666666)

            evaluator.update(predictions=[{
                                "image_description": {"value": "There is possibly one chinchilla in the cage."},
                                "number_of_chinchillas": {"value": 0},
                                "estimated_temperature": {"value": 20},
                                "escaped": {"value": False},
                                "activity": {"value": "none", "groundings": [[]]},
                                "cage_number": {"value": 2}
                            }],
                            targets=[{
                                "image_description": {"value": "There is one sleeping chinchilla in the far corner of the cage."},
                                "number_of_chinchillas": {"value": 1},
                                "estimated_temperature": {"value": 20},
                                "escaped": {"value": False},
                                "activity": {"value": "sleeping", "groundings": [[0, 0, 400, 400]]},
                                "cage_number": {"value": 1}
                            }])
        report = evaluator.compute()
        self.assertEqual(report["KeyWiseScores"]["image_description"]["F1"], 1.0)
        self.assertEqual(report["KeyWiseScores"]["number_of_chinchillas"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["estimated_temperature"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["escaped"].item(), 1.)
        self.assertEqual(report["KeyWiseScores"]["activity"]["F1"], 0.5)
        self.assertEqual(report["KeyWiseScores"]["cage_number"].item(), 0.)

        self.assertAlmostEqual(report["MicroF1"], 0.5833333333333333)
        self.assertAlmostEqual(report["MacroF1"], 0.5833333333333333)


if __name__ == '__main__':
    unittest.main()
