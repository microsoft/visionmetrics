import pathlib
import unittest

from PIL import Image
from visionmetrics.matting import L1Error, MeanIOU,  BoundaryMeanIOU, ForegroundIOU, BoundaryForegroundIOU


class TestImageMattingEvaluator(unittest.TestCase):

    image_matting_predictions = []
    image_matting_targets = []

    image_matting_predictions.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'test_0.png'))
    image_matting_predictions.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'test_1.png'))
    image_matting_targets.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'gt_0.png'))
    image_matting_targets.append(Image.open(pathlib.Path(__file__).resolve().parent / 'data' / 'gt_1.png'))

    def test_image_matting_mean_iou_evaluator(self):
        evaluator = MeanIOU()
        evaluator.update(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["mIOU"], 0.4530012134867148)

    def test_image_matting_foreground_iou_evaluator(self):
        evaluator = ForegroundIOU()
        evaluator.update(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["fgIOU"], 0.23992256209190865)

    def test_image_matting_l1_error_evaluator(self):
        evaluator = L1Error()
        evaluator.update(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["L1Err"], 77.07374954223633)

    def test_image_matting_boundary_mean_iou_evaluator(self):
        evaluator = BoundaryMeanIOU()
        evaluator.update(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["b_mIOU"], 0.6022812)

    def test_image_matting_boundary_foreground_iou_evaluator(self):
        evaluator = BoundaryForegroundIOU()
        evaluator.update(predictions=self.image_matting_predictions, targets=self.image_matting_targets)
        report = evaluator.compute()
        self.assertAlmostEqual(report["b_fgIOU"], 0.2460145344436508)
