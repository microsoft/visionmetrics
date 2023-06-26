import json
import pathlib
import unittest

from visionmetrics.caption import BleuScore, CIDErScore, METEORScore, ROUGELScore


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

    def test_image_caption_blue_score_evaluator(self):
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
