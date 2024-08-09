import unittest
import torch


from visionmetrics.detection import (ClassAgnosticAveragePrecision,
                                     DetectionConfusionMatrix,
                                     MeanAveragePrecision,
                                     DetectionMicroPrecisionRecallF1)


class TestMeanAveragePrecision(unittest.TestCase):
    def test_perfect_one_image_absolute_coordinates(self):

        PREDICTIONS = [[[0, 1.0, 0, 0, 10, 10],
                        [1, 1.0, 5, 5, 10, 10],
                        [2, 1.0, 1, 1, 5, 5]]]

        TARGETS = [[[0, 0, 0, 10, 10],
                    [1, 5, 5, 10, 10],
                    [2, 1, 1, 5, 5]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)

    def test_perfect_one_image_relative_coordinates(self):

        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)

    def test_wrong_one_image(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [0, 1.0, 0.5, 0.5, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)

    def test_perfect_two_images(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)

    def test_two_batches(self):

        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)

        PREDICTIONS = [[[0, 1.0, 0.9, 0.9, 1, 1],  # Wrong
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]],
                   [[2, 0.1, 0.1, 0.5, 0.5]]]

        metric.update(PREDICTIONS, TARGETS)

        result = metric.compute()
        self.assertAlmostEqual(result['map_50'].item(), 0.834983, places=5)

    def test_iou_thresholds(self):
        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        PREDICTIONS = [[[0, 1.0, 0.5, 0.5, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertAlmostEqual(result['map_50'].item(), 0.5, places=5)

        metric = MeanAveragePrecision(iou_thresholds=[0.2])
        PREDICTIONS = [[[0, 1.0, 0.5, 0.5, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1]]]

        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertAlmostEqual(result['map'].item(), 1.0, places=5)

    def test_no_predictions(self):
        PREDICTIONS = [[]]
        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision()
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertAlmostEqual(result['map_50'].item(), 0.0, places=5)

    def test_no_targets(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]
        TARGETS = [[]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertAlmostEqual(result['map_50'].item(), -1, places=5)

    def test_cat_id_remap(self):
        PREDICTIONS = [[(0, 1.0, 0, 0, 1, 1),
                        [1, 1.0, 0.5, 0.5, 1, 1],
                        [2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [0, 0.5, 0.5, 1, 1],
                    [2, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()

        PREDICTIONS_REMAP_CAT_ID = [[[0, 1.0, 0, 0, 1, 1],
                                     [2, 1.0, 0.5, 0.5, 1, 1],
                                     [1, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS_REMAP_CAT_ID = [[[0, 0, 0, 1, 1],
                                 [0, 0.5, 0.5, 1, 1],
                                 [1, 0.1, 0.1, 0.5, 0.5]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        metric.update(PREDICTIONS_REMAP_CAT_ID, TARGETS_REMAP_CAT_ID)
        result_remap_cat_id = metric.compute()

        for k in result.keys():
            self.assertTrue(torch.allclose(result[k], result_remap_cat_id[k]))

    def test_map_small_medium_large(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 32, 32],  # <= 32**2
                        [1, 1.0, 0, 0, 96, 96],  # >= 32**2, <= 96**2
                        [2, 1.0, 0, 0, 100, 100]]]  # > 96**2

        TARGETS = [[[0, 0, 0, 32, 32],
                    [1, 5, 5, 96, 96],
                    [2, 1, 1, 100, 100]]]

        metric = MeanAveragePrecision(coords='absolute', iou_thresholds=[0.5])
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()

        self.assertAlmostEqual(result['map_small'].item(), 1.0, places=5)
        self.assertAlmostEqual(result['map_medium'].item(), 1.0, places=5)
        self.assertAlmostEqual(result['map_large'].item(), 1.0, places=5)


class TestClassAgnosticAveragePrecision(unittest.TestCase):
    def test_class_agnostic_ap_correct(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[1, 0, 0, 1, 1]]]

        metric = ClassAgnosticAveragePrecision(iou_thresholds=[0.5], class_metrics=False)
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)
        self.assertEqual(result['classes'], -1)
        self.assertEqual(result['map_per_class'], -1)

        metric = ClassAgnosticAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['classes'], -1)
        self.assertEqual(result['map_per_class'], 1)

    def test_class_agnostic_ap_incorrect(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[1, 0, 0, 0.1, 0.1]]]
        metric = ClassAgnosticAveragePrecision(iou_thresholds=[0.5])
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 0.0)
        self.assertEqual(result['classes'], -1)
        self.assertEqual(result['map_per_class'], -1)

    def test_two_images(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0.5, 0.5, 1, 1]],
                       [[2, 1.0, 0.1, 0.1, 0.5, 0.5]]]

        TARGETS = [[[1, 0, 0, 1, 1],
                    [2, 0.5, 0.5, 1, 1]],
                   [[1, 0.1, 0.1, 0.5, 0.5]]]

        metric = ClassAgnosticAveragePrecision(iou_thresholds=[0.5])
        metric.update(PREDICTIONS, TARGETS)
        result = metric.compute()
        self.assertEqual(result['map_50'], 1.0)
        self.assertEqual(result['classes'], -1)


class TestDetectionConfusionMatrix(unittest.TestCase):
    def test_true_positive(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_wrong_class(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[0, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_low_iou_correct_class(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 0.5, 0.5]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_low_iou_wrong_class(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 0.5, 0.5]]]
        targets = [[[0, 0, 0, 1, 1]], [[0, 0, 0, 1, 1]]]  # wrong class

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 1)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_no_overlap_correct_class(self):
        predictions = [[[1, 1.0, 0, 0, 0.1, 0.1]]]
        targets = [[[1, 0.2, 0.2, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_no_overlap_wrong_class(self):
        predictions = [[[0, 1.0, 0, 0, 10, 10]]]
        targets = [[[1, 90, 90, 100, 100]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 1)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_false_positive_extra_pred_boxes(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1], [1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 1)

    def test_false_negative(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_batch_update(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[[1, 0, 0, 1, 1]]]
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_empty_predictions(self):
        predictions = [[[]]]
        targets = [[[0, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_empty_targets(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 1)

    def test_two_images(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_two_images_one_wrong_class_one_low_iou(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 0.5, 0.5]]]
        targets = [[[1, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 0)
        self.assertEqual(result['FP'], 2)
        self.assertEqual(result['FN'], 2)
        self.assertEqual(result['FP_due_to_wrong_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 1)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_two_images_extra_preds(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1], [1, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 1)

    def test_two_images_extra_targets(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)

    def test_two_images_zero_area_box(self):
        predictions = [[[1, 1.0, 0, 0, 100, 100]], [[1, 1.0, 20, 20, 20, 20]]]
        targets = [[[1, 0, 0, 100, 100]], [[1, 20, 20, 20, 20]]]

        metric = DetectionConfusionMatrix(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['TP'], 2)
        self.assertEqual(result['FP'], 0)
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['FP_due_to_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_correct_class'], 0)
        self.assertEqual(result['FP_due_to_low_iou_wrong_class'], 0)
        self.assertEqual(result['FP_due_to_extra_pred_boxes'], 0)


class TestDetectionMicroPrecisionRecallF1(unittest.TestCase):
    # We include a subset of cases here since the underlying logic is inherited from DetectionConfusionMatrix.
    def test_true_positive(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 1.0)
        self.assertEqual(result['Recall'], 1.0)
        self.assertEqual(result['F1'], 1.0)

    def test_false_positive_wrong_class(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[0, 0, 0, 1, 1]]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 0.5)
        self.assertEqual(result['Recall'], 0.5)
        self.assertEqual(result['F1'], 0.5)

    def test_false_negative(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 1.0)
        self.assertEqual(result['Recall'], 0.5)
        self.assertEqual(result['F1'], 0.6666666666666666)

    def test_empty_predictions(self):
        predictions = [[[]]]
        targets = [[[0, 0, 0, 1, 1]]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 0.0)
        self.assertEqual(result['Recall'], 0.0)
        self.assertEqual(result['F1'], 0.0)

    def test_empty_targets(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1]]]
        targets = [[]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 0.0)
        self.assertEqual(result['Recall'], 0.0)
        self.assertEqual(result['F1'], 0.0)

    def test_two_images_extra_preds(self):
        predictions = [[[0, 1.0, 0, 0, 1, 1], [1, 1.0, 0, 0, 1, 1]], [[1, 1.0, 0, 0, 1, 1]]]
        targets = [[[0, 0, 0, 1, 1]], [[1, 0, 0, 1, 1]]]

        metric = DetectionMicroPrecisionRecallF1(iou_threshold=0.5)
        metric.update(predictions, targets)
        result = metric.compute()

        self.assertEqual(result['Precision'], 0.6666666666666666)
        self.assertEqual(result['Recall'], 1.0)
        self.assertEqual(result['F1'], 0.8)


if __name__ == '__main__':
    unittest.main()
