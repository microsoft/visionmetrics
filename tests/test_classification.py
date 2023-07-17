import unittest

import torch

from visionmetrics.classification import (BinaryAUROC, MulticlassAccuracy,
                                          MulticlassAUROC,
                                          MulticlassAveragePrecision,
                                          MulticlassCalibrationError,
                                          MulticlassPrecision,
                                          MultilabelAccuracy, MultilabelAUROC,
                                          MultilabelAveragePrecision,
                                          MultilabelF1Score,
                                          MultilabelPrecision,
                                          MultilabelRecall)


class TestMulticlassClassification(unittest.TestCase):
    TARGETS = [
        torch.tensor([1, 0, 0, 0, 1, 1, 0, 0, 0, 1]),
        torch.tensor([1, 0, 2, 0, 1, 2, 0, 0, 0, 1, 2, 1, 2, 2, 0])]
    PREDICTIONS = [
        torch.tensor([[1, 0],
                      [0, 1],
                      [0.5, 0.5],
                      [0.1, 0.9],
                      [0.44, 0.56],
                      [0.09, 0.91],
                      [0.91, 0.09],
                      [0.37, 0.63],
                      [0.34, 0.66],
                      [0.89, 0.11]]),
        torch.tensor([[0.99, 0.01, 0],
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
                metric_topk_acc = MulticlassAccuracy(num_classes=num_classes, top_k=top_k, average='micro')
                metric_topk_acc.update(predictions, targets)
                metric_top_k_acc = metric_topk_acc.compute()
                self.assertAlmostEqual(metric_top_k_acc.item(), gts[i][k_idx], places=5)

    def test_top_1_accuracy_evaluator_equivalent_to_top1_precision_eval(self):
        for i, (targets, predictions, num_classes) in enumerate(zip(self.TARGETS, self.PREDICTIONS, self.NUM_CLASSES)):
            metric_top1_acc = MulticlassAccuracy(num_classes=num_classes, average='micro', top_k=1)
            metric_top1_acc.update(predictions, targets)
            metric_top1_acc = metric_top1_acc.compute()

            metric_top1_prec = MulticlassPrecision(num_classes=num_classes, average='micro', top_k=1)
            metric_top1_prec.update(predictions, targets)
            metric_top1_prec = metric_top1_prec.compute()
            self.assertAlmostEqual(metric_top1_prec, metric_top1_acc, places=5)

    def test_average_precision(self):
        gts = [[0.475744, 0.490476190], [0.485592, 0.50326599]]
        for i, (targets, predictions, num_classes) in enumerate(zip(self.TARGETS, self.PREDICTIONS, self.NUM_CLASSES)):
            for fl_i, flavor in enumerate(['macro', 'weighted']):
                # NOTE: doesnot support samples-based average for multiclass
                metric = MulticlassAveragePrecision(num_classes=num_classes, average=flavor)
                metric.update(predictions, targets)
                metric_avg_prec = metric.compute()
                self.assertAlmostEqual(metric_avg_prec.item(), gts[i][fl_i], places=5)

    def test_ece_loss(self):
        gts = [0.584, 0.40800000]
        for i, (targets, predictions) in enumerate(zip(self.TARGETS, self.PREDICTIONS)):
            metric = MulticlassCalibrationError(num_classes=self.NUM_CLASSES[i])
            metric.update(predictions, targets)
            self.assertAlmostEqual(metric.compute().item(), gts[i], places=5)

    def test_tagwise_accuracy(self):
        metric = MulticlassAccuracy(num_classes=self.NUM_CLASSES[0], average=None)
        metric.update(self.PREDICTIONS[0], self.TARGETS[0])
        metric_tag_wise_acc = metric.compute()
        self.assertAlmostEqual(metric_tag_wise_acc[0].item(), 0.33333, 5)
        self.assertAlmostEqual(metric_tag_wise_acc[1].item(), 0.5, 5)

    def test_perclass_accuracy_with_missing_class(self):
        target_missing_class = torch.tensor([0, 1, 0, 0])
        predicitons_missing_class = torch.tensor([[1, 0, 0],
                                                  [0, 1, 0],
                                                  [0.5, 0.5, 0],
                                                  [0.1, 0.9, 0]])

        metric = MulticlassAccuracy(num_classes=3, average=None)
        metric.update(predicitons_missing_class, target_missing_class)
        metric_tag_wise_acc = metric.compute()
        self.assertAlmostEqual(metric_tag_wise_acc[0].item(), 0.666666, 5)
        self.assertEqual(metric_tag_wise_acc[1].item(), 1.0, 5)
        self.assertEqual(metric_tag_wise_acc[2].item(), 0.0, 5)

    def test_perclass_average_precision(self):
        metric = MulticlassAveragePrecision(num_classes=2, average=None)
        metric.update(self.PREDICTIONS[0], self.TARGETS[0])
        metric_tag_wise_avg_prec = metric.compute()
        self.assertAlmostEqual(metric_tag_wise_avg_prec[0].item(), 0.54940, 5)
        self.assertAlmostEqual(metric_tag_wise_avg_prec[1].item(), 0.40208, 5)


class TestMultilabelClassification(unittest.TestCase):
    TARGETS = torch.tensor([[1, 0, 0],
                            [0, 1, 1],
                            [1, 1, 1]])
    PROB_PREDICTIONS = torch.tensor([[1, 0.31, 0.1],
                                     [0.1, 1, 0.51],
                                     [0.51, 0.61, 0.51]])
    INDEX_PREDICTIONS = torch.tensor([[0, 1, 2],
                                      [1, 2, 0],
                                      [1, 0, 2]])

    def test_multilabel_accuracy(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.88888, 0.66666, 0.55555]
        for i in range(len(thresholds)):
            metric = MultilabelAccuracy(num_labels=3, threshold=thresholds[i])
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric_acc = metric.compute()
            self.assertAlmostEqual(metric_acc.item(), expectations[i], places=4)

    def test_threshold_based_precision(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.66666, 0.88888, 0.66666, 0.66666]
        for i in range(len(thresholds)):
            metric = MultilabelPrecision(num_labels=3, average='macro', threshold=thresholds[i])
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric_prec = metric.compute()
            self.assertAlmostEqual(metric_prec.item(), expectations[i], places=4)

    def test_topk_based_precision(self):
        ks = [1, 2, 3]
        expectations = [0.66666, 0.88888, 0.66666]
        for i in range(len(ks)):
            metric = MultilabelPrecision(top_k=ks[i], num_labels=3, average='macro')
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric = metric.compute()
            self.assertAlmostEqual(metric.item(), expectations[i], places=4)

        # indices mode
        for i in range(len(ks)):
            metric = MultilabelPrecision(top_k=ks[i], prediction_mode='indices', num_labels=3, average='macro')
            metric.update(self.INDEX_PREDICTIONS, self.TARGETS)
            self.assertAlmostEqual(metric.compute().item(), expectations[i], places=4)

    def test_threshold_based_recall(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [1.0, 1.0, 0.5, 0.3333]
        for i in range(len(thresholds)):
            metric = MultilabelRecall(num_labels=3, average='macro', threshold=thresholds[i])
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric_prec = metric.compute()
            self.assertAlmostEqual(metric_prec.item(), expectations[i], places=4)

    def test_topk_based_recall(self):
        ks = [0, 1, 2, 3]
        expectations = [0, 0.5, 0.83333, 1.0]
        for i in range(len(ks)):
            metric = MultilabelRecall(top_k=ks[i], num_labels=3, average='macro')
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric_prec = metric.compute()
            self.assertAlmostEqual(metric_prec.item(), expectations[i], places=4)

        # index based predictions
        for i in range(len(ks)):
            metric = MultilabelRecall(top_k=ks[i], prediction_mode='indices', num_labels=3, average='macro')
            metric.update(self.INDEX_PREDICTIONS, self.TARGETS)
            metric_prec = metric.compute()
            self.assertAlmostEqual(metric_prec.item(), expectations[i], places=4)

    def test_threshold_based_f1_score(self):
        thresholds = [0.0, 0.3, 0.6, 0.7]
        expectations = [0.8, 0.93333, 0.55555, 0.44444]
        for i in range(len(thresholds)):
            metric = MultilabelF1Score(num_labels=3, average='macro', threshold=thresholds[i])
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            vmetric_f1 = metric.compute()
            self.assertAlmostEqual(vmetric_f1.item(), expectations[i], places=4)

    def test_topk_based_f1_score(self):
        ks = [0, 1, 2, 3]
        expectations = [0.0, 0.55555, 0.82222, 0.8]
        for i in range(len(ks)):
            metric = MultilabelF1Score(top_k=ks[i], num_labels=3, average='macro')
            metric.update(self.PROB_PREDICTIONS, self.TARGETS)
            metric_f1 = metric.compute()
            self.assertAlmostEqual(metric_f1.item(), expectations[i], places=4)

    def test_average_precision(self):
        targets = torch.tensor([[1, 0, 0, 0],
                                [0, 1, 1, 1],
                                [0, 0, 1, 1],
                                [1, 1, 1, 0]])
        predictions = torch.tensor([[0, 0.3, 0.7, 0],
                                    [0, 1, 0.5, 0],
                                    [0, 0, 0.5, 0],
                                    [0.5, 0.6, 0, 0.5]])
        gts = [0.67328, 0.73611, 0.731481]
        for fl_i, flavor in enumerate(['micro', 'macro', 'weighted']):
            # NOTE: doesnot support samples-based average for multilabel
            vmetric_eval = MultilabelAveragePrecision(num_labels=4, average=flavor)
            vmetric_eval.update(predictions, targets)
            vmetric_avg_prec = vmetric_eval.compute()
            self.assertAlmostEqual(vmetric_avg_prec.item(), gts[fl_i], places=5)


class TestROCAUC(unittest.TestCase):
    @staticmethod
    def _get_metric(predictions, targets, task='multiclass', num_classes=None, average='macro'):
        if task == 'binary':
            metric = BinaryAUROC()
        elif task == 'multiclass':
            metric = MulticlassAUROC(num_classes=num_classes, average=average)
        else:
            metric = MultilabelAUROC(num_labels=num_classes, average=average)

        metric.update(predictions, targets)
        roc_auc = metric.compute()
        return roc_auc

    def test_perfect_predictions(self):
        predictions = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        targets = torch.tensor([0, 0, 0, 1, 1, 1])
        roc_auc = self._get_metric(predictions, targets, task='binary', num_classes=2)
        self.assertAlmostEqual(roc_auc.item(), 1.0, places=4)

    def test_abysmal_predictions(self):
        predictions = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        targets = torch.tensor([1, 1, 1, 0, 0, 0])
        roc_auc = self._get_metric(predictions, targets, task='binary', num_classes=2)
        self.assertAlmostEqual(roc_auc.item(), 0.0, places=4)

    def test_imperfect_predictions(self):
        predictions = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        targets = torch.tensor([0, 0, 0, 1, 0, 1])
        roc_auc = self._get_metric(predictions, targets, task='binary', num_classes=2)
        self.assertAlmostEqual(roc_auc.item(), 0.875, places=4)

    def test_multiclass_perfect_predictions(self):
        predictions = torch.tensor([[0.8, 0.2, 0.0], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        roc_auc = self._get_metric(predictions, targets, task='multiclass', num_classes=3)
        self.assertAlmostEqual(roc_auc.item(), 1.0, places=4)

    def test_multilabel_perfect_predictions(self):
        predictions = torch.tensor([[0.8, 0.2, 0.0], [0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
        targets = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]])
        roc_auc = self._get_metric(predictions, targets, task='multilabel', num_classes=3)
        self.assertAlmostEqual(roc_auc.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
