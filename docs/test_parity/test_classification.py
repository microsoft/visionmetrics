import unittest
from cv2 import threshold

import numpy as np
import torch
from vision_evaluation import TopKAccuracyEvaluator, PrecisionEvaluator, AveragePrecisionEvaluator, TagWiseAccuracyEvaluator, TagWiseAveragePrecisionEvaluator
from vision_evaluation.prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter

from visionmetrics.classification import Accuracy, Precision, AveragePrecision, MultilabelAccuracy


class TestClassificationEvaluator(unittest.TestCase):
    TARGETS = [
        np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1]),
        np.array([1, 0, 2, 0, 1, 2, 0, 0, 0, 1, 2, 1, 2, 2, 0])]
    PREDICTIONS = [
        np.array([[1, 0],
                  [0, 1],
                  [0.5, 0.5],
                  [0.1, 0.9],
                  [0.44, 0.56],
                  [0.09, 0.91],
                  [0.91, 0.09],
                  [0.37, 0.63],
                  [0.34, 0.66],
                  [0.89, 0.11]]),
        np.array([[0.99, 0.01, 0],
                  [0, 0.99, 0.01],
                  [0.51, 0.49, 0.0],
                  [0.09, 0.8, 0.11],
                  [0.34, 0.36, 0.3],
                  [0.09, 0.90, 0.01],
                  [0.91, 0.06, 0.03],
                  [0.37, 0.60, 0.03],
                  [0.34, 0.46, 0.2],
                  [0.79, 0.11, 0.1],
                  [0.34, 0.16, 0.5],
                  [0.04, 0.56, 0.4],
                  [0.04, 0.36, 0.6],
                  [0.04, 0.36, 0.6],
                  [0.99, 0.01, 0.0]])]
    NUM_CLASSES = [2, 3]

    def test_top_k_accuracy_evaluator(self):
        gts = [[0.4, 1.0, 1.0], [0.4666666, 0.7333333, 1.0]]
        for k_idx, top_k in enumerate([1, 2]):
            for i, (targets, predictions, num_classes) in enumerate(zip(self.TARGETS, self.PREDICTIONS, self.NUM_CLASSES)):
                eval = TopKAccuracyEvaluator(top_k)
                eval.add_predictions(predictions, targets)
                top_k_acc = eval.get_report()[f"accuracy_top{top_k}"]
                self.assertAlmostEqual(top_k_acc, gts[i][k_idx], places=5)

                # visionmetrics
                predictions, targets = torch.from_numpy(predictions), torch.from_numpy(targets)
                vmetric_eval = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k, average='micro')
                vmetric_eval.update(predictions, targets)
                vmetric_top_k_acc = vmetric_eval.compute()
                self.assertAlmostEqual(top_k_acc, vmetric_top_k_acc, places=5)

    def test_top_1_accuracy_evaluator_equivalent_to_top1_precision_eval(self):
        for i, (targets, predictions, num_classes) in enumerate(zip(self.TARGETS, self.PREDICTIONS, self.NUM_CLASSES)):
            top1_acc_evaluator = TopKAccuracyEvaluator(1)
            top1_acc_evaluator.add_predictions(predictions, targets)

            top1_prec_evaluator = PrecisionEvaluator(TopKPredictionFilter(1))
            top1_prec_evaluator.add_predictions(predictions, targets)

            self.assertEqual(top1_acc_evaluator.get_report()["accuracy_top1"], top1_prec_evaluator.get_report(average='samples')['precision_top1'])

            # visionmetrics
            predictions, targets = torch.from_numpy(predictions), torch.from_numpy(targets)
            vmetric_eval = Precision(task='multiclass', num_classes=num_classes, average='micro', top_k=1)
            vmetric_eval.update(predictions, targets)
            vmetric_top1_prec = vmetric_eval.compute()
            self.assertAlmostEqual(vmetric_top1_prec, top1_prec_evaluator.get_report(average='samples')['precision_top1'])

    def test_average_precision_evaluator(self):
        gts = [[0.447682, 0.475744, 0.490476190, 0.65], [0.384352058, 0.485592, 0.50326599, 0.6888888]]
        for i, (targets, predictions, num_classes) in enumerate(zip(self.TARGETS, self.PREDICTIONS, self.NUM_CLASSES)):
            evaluator = AveragePrecisionEvaluator()
            evaluator.add_predictions(predictions, targets)
            for fl_i, flavor in enumerate(['micro', 'macro', 'weighted', 'samples']):
                self.assertAlmostEqual(evaluator.get_report(average=flavor)['average_precision'], gts[i][fl_i], places=5)

            # visionmetrics
            # doesnot support micro and samples-based average for multiclass
            if flavor in ['macro', 'weighted']:
                predictions, targets = torch.from_numpy(predictions), torch.from_numpy(targets)
                vmetric_eval = AveragePrecision(task='multiclass', num_classes=num_classes, average='micro')
                vmetric_eval.update(predictions, targets)
                vmetric_avg_prec = vmetric_eval.compute()
                self.assertAlmostEqual(vmetric_avg_prec, gts[i][fl_i], places=5)

    def test_tagwise_accuracy_evaluator(self):
        evaluator = TagWiseAccuracyEvaluator()
        evaluator.add_predictions(self.PREDICTIONS[0], self.TARGETS[0])
        result = evaluator.get_report()
        self.assertAlmostEqual(result['tag_wise_accuracy'][0], 0.33333, 5)
        self.assertEqual(result['tag_wise_accuracy'][1], 0.5)

        # visionmetrics
        predictions, targets = torch.from_numpy(self.PREDICTIONS[0]), torch.from_numpy(self.TARGETS[0])
        vmetric_eval = Accuracy(task='multiclass', num_classes=2, average=None)
        vmetric_eval.update(predictions, targets)
        vmetric_tag_wise_acc = vmetric_eval.compute()
        self.assertAlmostEquals(vmetric_tag_wise_acc[0].item(), 0.33333, 5)
        self.assertAlmostEquals(vmetric_tag_wise_acc[1].item(), 0.5, 5)

    def test_perclass_accuracy_evaluator_with_missing_class(self):
        target_missing_class = np.array([0, 1, 0, 0])
        predicitons_missing_class = np.array([[1, 0, 0],
                                              [0, 1, 0],
                                              [0.5, 0.5, 0],
                                              [0.1, 0.9, 0]])
        evaluator = TagWiseAccuracyEvaluator()
        evaluator.add_predictions(predicitons_missing_class, target_missing_class)
        result = evaluator.get_report()
        self.assertEqual(len(result['tag_wise_accuracy']), 3)
        self.assertAlmostEqual(result['tag_wise_accuracy'][0], 0.666666, 5)
        self.assertEqual(result['tag_wise_accuracy'][1], 1.0)
        self.assertEqual(result['tag_wise_accuracy'][2], 0.0)

        # visionmetrics
        target_missing_class, predicitons_missing_class = torch.from_numpy(target_missing_class), torch.from_numpy(predicitons_missing_class)
        vmetric_eval = Accuracy(task='multiclass', num_classes=3, average=None)
        vmetric_eval.update(predicitons_missing_class, target_missing_class)
        vmetric_tag_wise_acc = vmetric_eval.compute()
        self.assertAlmostEqual(vmetric_tag_wise_acc[0].item(), 0.666666, 5)
        self.assertEqual(vmetric_tag_wise_acc[1].item(), 1.0, 5)
        self.assertEqual(vmetric_tag_wise_acc[2].item(), 0.0, 5)

    def test_perclass_average_precision_evaluator(self):
        evaluator = TagWiseAveragePrecisionEvaluator()
        evaluator.add_predictions(self.PREDICTIONS[0], self.TARGETS[0])
        result = evaluator.get_report()
        self.assertAlmostEqual(result['tag_wise_average_precision'][0], 0.54940, 5)
        self.assertAlmostEqual(result['tag_wise_average_precision'][1], 0.40208, 5)

        # visionmetrics
        predictions, targets = torch.from_numpy(self.PREDICTIONS[0]), torch.from_numpy(self.TARGETS[0])
        vmetric_eval = AveragePrecision(task='multiclass', num_classes=2, average=None)
        vmetric_eval.update(predictions, targets)
        vmetric_tag_wise_avg_prec = vmetric_eval.compute()
        self.assertAlmostEqual(vmetric_tag_wise_avg_prec[0].item(), 0.54940, 5)
        self.assertAlmostEqual(vmetric_tag_wise_avg_prec[1].item(), 0.40208, 5)


class TestMultilabelClassificationEvaluator(unittest.TestCase):
    TARGETS = np.array([[1, 0, 0],
                        [0, 1, 1],
                        [1, 1, 1]])
    PROB_PREDICTIONS = np.array([[1, 0.3, 0],
                                 [0, 1, 0.5],
                                 [0.5, 0.6, 0.5]])
    INDEX_PREDICTIONS = np.array([[0, 1, 2],
                                  [1, 2, 0],
                                  [1, 0, 2]])

    def test_precision_evaluator(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.88888, 0.66666, 0.66666]
        for i in range(len(thresholds)):
            prec_eval = PrecisionEvaluator(ThresholdPredictionFilter(thresholds[i]))
            prec_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            # print(prec_eval.get_report(average='macro'))
            self.assertAlmostEqual(prec_eval.get_report(average='macro')[f"precision_thres={thresholds[i]}"], expectations[i], places=4)

            # visionmetrics
            # predictions, targets = torch.from_numpy(self.PROB_PREDICTIONS), torch.from_numpy(self.TARGETS)
            # # predictions, targets = predictions.unsqueeze(0), targets.unsqueeze(0)
            # vmetric_eval = Precision(task='multilabel', num_labels=3, average='macro', threshold=thresholds[i])
            # vmetric_eval.update(predictions, targets)
            # vmetric_prec = vmetric_eval.compute()
            # self.assertAlmostEqual(vmetric_prec.item(), expectations[i], places=4)

    def test_precision_evaluator_samplewise(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.83333, 1.0, 0.66666]
        for i in range(len(thresholds)):
            prec_eval = PrecisionEvaluator(ThresholdPredictionFilter(thresholds[i]))
            prec_eval.add_predictions(self.PROB_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(prec_eval.get_report(average='samples')[f"precision_thres={thresholds[i]}"], expectations[i], places=4)

            # visionmetrics
            predictions, targets = torch.from_numpy(self.PROB_PREDICTIONS), torch.from_numpy(self.TARGETS)
            predictions, targets = predictions.unsqueeze(0), targets.unsqueeze(0)
            vmetric_eval = Precision(task='multilabel', num_labels=3, threshold=thresholds[i], multidim_average='samplewise', average='macro')
            vmetric_eval.update(predictions, targets)
            vmetric_prec = vmetric_eval.compute()
            self.assertAlmostEqual(vmetric_prec.item(), expectations[i], places=4)


if __name__ == "__main__":
    unittest.main()
