import unittest
from sklearn import metrics

import torch

from visionmetrics.detection import MeanAveragePrecision


class TestDetection(unittest.TestCase):
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

    def test_class_wise_metrics(self):
        PREDICTIONS = [[[0, 1.0, 0, 0, 1, 1],
                        [1, 1.0, 0, 0, 1, 1]]]

        TARGETS = [[[0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        results = metric.compute()
        self.assertAlmostEqual(results['map_per_class'].get(0), 1.0, places=5)
        self.assertAlmostEqual(results['map_per_class'].get(1), 0.0, places=5)

    def test_class_wise_with_missing_class_in_predictions(self):
        PREDICTIONS = [[[1, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[1, 0, 0, 1, 1],
                    [2, 0, 0, 1, 1]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        results = metric.compute()
        self.assertAlmostEqual(results['map_per_class'].get(1), 1.0, places=5)
        self.assertAlmostEqual(results['map_per_class'].get(2), 0.0, places=5)

    def test_class_wise_with_missing_class_in_targets(self):
        PREDICTIONS = [[[1, 1.0, 0, 0, 1, 1],
                        [2, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[2, 0, 0, 1, 1]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        results = metric.compute()
        self.assertAlmostEqual(results['map_per_class'].get(1), -1.0, places=5)
        self.assertAlmostEqual(results['map_per_class'].get(2), 1.0, places=5)

    def test_class_wise_after_reset(self):
        PREDICTIONS = [[[1, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[1, 0, 0, 1, 1]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        metric.reset()

        PREDICTIONS = [[[2, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[2, 0, 0, 1, 1]]]

        metric.update(PREDICTIONS, TARGETS)
        results = metric.compute()

        metric2 = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric2.update(PREDICTIONS, TARGETS)
        results2 = metric2.compute()

        self.assertEqual(results['map_per_class'].get(1, "Does not exist"), "Does not exist")
        self.assertAlmostEqual(results['map_per_class'].get(2), 1.0, places=5)
        self.assertEqual(results, results2)

    def test_class_wise_after_clone(self):
        PREDICTIONS = [[[2, 1.0, 0, 0, 1, 1]]]
        TARGETS = [[[2, 0, 0, 1, 1]]]

        metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric.update(PREDICTIONS, TARGETS)
        results = metric.compute()

        metric2 = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        metric2.update(PREDICTIONS, TARGETS)
        results2 = metric2.compute()

        self.assertEqual(results, results2)


if __name__ == '__main__':
    unittest.main()
