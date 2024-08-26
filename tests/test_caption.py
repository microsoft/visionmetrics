from irisml.tasks.create_azure_openai_chat_model import OpenAITextChatModel
import json
import pathlib
import unittest

from visionmetrics.caption import BleuScore, CIDErScore, METEORScore, ROUGELScore, AzureOpenAITextModelCategoricalScore
from visionmetrics.caption.azure_openai_model_eval_base import ResultStatusType


class TestImageCaptionEvaluator(unittest.TestCase):
    predictions_file = pathlib.Path(__file__).resolve().parent / 'data' / 'image_caption_prediction.json'
    ground_truth_file = pathlib.Path(__file__).resolve().parent / 'data' / 'image_caption_gt.json'
    imcap_predictions, imcap_targets = [], []
    predictions_dict = json.loads(predictions_file.read_text())
    ground_truth_dict = json.loads(ground_truth_file.read_text())

    gts_by_id = {}
    predictions_by_id = {pred['image_id']: pred['caption'] for pred in predictions_dict}

    for gt in ground_truth_dict['annotations']:
        if not gt['image_id'] in gts_by_id:
            gts_by_id[gt['image_id']] = []
        gts_by_id[gt['image_id']].append(gt['caption'])
    for key, value in predictions_by_id.items():
        imcap_predictions.append(value)
        imcap_targets.append(gts_by_id[key])

    def test_image_caption_bleu_score_evaluator(self):
        evaluator = BleuScore()
        evaluator.update(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["Bleu_1"], 0.783228681385441)
        self.assertAlmostEqual(report["Bleu_2"], 0.6226378540059051)
        self.assertAlmostEqual(report["Bleu_3"], 0.47542636331846966)
        self.assertAlmostEqual(report["Bleu_4"], 0.3573567238999926)

    def test_image_caption_meteor_score_evaluator(self):
        evaluator = METEORScore()
        evaluator.update(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["METEOR"], 0.2878681068021112)

    def test_image_caption_rouge_l_score_evaluator(self):
        evaluator = ROUGELScore()
        evaluator.update(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["ROUGE_L"], 0.5774238052522583)

    def test_image_caption_cider_score_evaluator(self):
        evaluator = CIDErScore()
        evaluator.update(predictions=self.imcap_predictions, targets=self.imcap_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["CIDEr"], 1.2346054374217474)


class TestAzureOpenAITextModelCategoricalEvaluator(unittest.TestCase):
    tp_exact_match_testcase = {
        "predictions": ["This is an image of a dog."],
        "targets": [["This is an image of a dog."]]
    }
    tp_semantic_match_testcase = {
        "predictions": ["In this image, there is a dog."],
        "targets": [["This is an image of a dog."]]
    }
    tn_testcase = {
        "predictions": [""],
        "targets": [""]
    }
    fn_testcase = {
        "predictions": [""],
        "targets": [["This is an image of a dog."]]
    }
    fp_gt_null_testcase = {
        "predictions": ["In this image, there is a dog."],
        "targets": [[""]]
    }
    fp_gt_not_null_testcase = {
        "predictions": ["In this image, there is a dog."],
        "targets": [["There is a cat on a bench."]]
    }

    # Since AverageScore and ScoreParseFailures are not deterministic, omitting these from the testcase
    expected_report = {
        # Summary statistics
        "Precision": 0.5,
        "Recall": 0.5,
        "F1": 0.5,
        "Accuracy": 0.5,
        # Raw statistic counts
        "TP": 2,
        "TN": 1,
        "FN": 1,
        "FP_GTNull": 1,
        "FP_GTNotNull": 1
    }

    def test_azure_openai_text_model_categorical_evaluator(self):
        # Update with each test case and check intermediate states for correctness, followed by overall metrics.
        evaluator = AzureOpenAITextModelCategoricalScore(endpoint="https://endpoint-name.openai.azure.com/", deployment_name="gpt-4o")

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            evaluator.update(predictions=self.tp_exact_match_testcase["predictions"], targets=self.tp_exact_match_testcase["targets"])
            self.assertEqual(evaluator.raw_scores[-1], "1.0")
            self.assertEqual(evaluator.scores[-1], 1.0)
            self.assertEqual(evaluator.score_parse_failures.item(), 0)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.TruePositive)

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.9"]):
            evaluator.update(predictions=self.tp_semantic_match_testcase["predictions"], targets=self.tp_semantic_match_testcase["targets"])
            self.assertGreater(evaluator.scores[-1], evaluator.positive_threshold)
            self.assertEqual(evaluator.score_parse_failures.item(), 0)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.TruePositive)

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["1.0"]):
            evaluator.update(predictions=self.tn_testcase["predictions"], targets=self.tn_testcase["targets"])
            self.assertEqual(evaluator.raw_scores[-1], "1.0")
            self.assertEqual(evaluator.scores[-1], 1.0)
            self.assertEqual(evaluator.score_parse_failures.item(), 0)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.TrueNegative)

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.0"]):
            evaluator.update(predictions=self.fn_testcase["predictions"], targets=self.fn_testcase["targets"])
            self.assertLess(evaluator.scores[-1], evaluator.positive_threshold)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.FalseNegative)

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.0"]):
            evaluator.update(predictions=self.fp_gt_null_testcase["predictions"], targets=self.fp_gt_null_testcase["targets"])
            self.assertLess(evaluator.scores[-1], evaluator.positive_threshold)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.FalsePositiveGtNull)

        with unittest.mock.patch.object(OpenAITextChatModel, "forward", return_value=["0.0"]):
            evaluator.update(predictions=self.fp_gt_not_null_testcase["predictions"], targets=self.fp_gt_not_null_testcase["targets"])
            self.assertLess(evaluator.scores[-1], evaluator.positive_threshold)
            self.assertEqual(evaluator.result_status_types[-1], ResultStatusType.FalsePositiveGtNotNull)

        report = evaluator.compute()
        self.assertAlmostEqual(report["Precision"], self.expected_report["Precision"])
        self.assertAlmostEqual(report["Recall"], self.expected_report["Recall"])
        self.assertAlmostEqual(report["F1"], self.expected_report["F1"])
        self.assertAlmostEqual(report["Accuracy"], self.expected_report["Accuracy"])
        self.assertEqual(report["TP"], self.expected_report["TP"])
        self.assertEqual(report["TN"], self.expected_report["TN"])
        self.assertEqual(report["FN"], self.expected_report["FN"])
        self.assertEqual(report["FP_GTNull"], self.expected_report["FP_GTNull"])
        self.assertEqual(report["FP_GTNotNull"], self.expected_report["FP_GTNotNull"])
