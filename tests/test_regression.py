import unittest
from torch import tensor

from visionmetrics.regression.accuracy import MeanAbsoluteErrorF1Score


class TestDetectionMicroPrecisionRecallF1(unittest.TestCase):
    def test_all_perfect(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.0)
        evaluator.update(predictions=tensor([1, 2, 3, 4]), targets=tensor([1, 2, 3, 4]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_all_right(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.5)
        evaluator.update(predictions=tensor([1, 2, 3, 4]), targets=tensor([1.4, 2.4, 3.4, 3.6]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_partially_right(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.5)
        evaluator.update(predictions=tensor([1, 2, 3, 4]), targets=tensor([1, 2.6, 3.5, 4.6]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 0.5,
            "Recall": 0.5,
            "F1": 0.5
        })

    def test_all_right_equal_to_threshold(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        evaluator.update(predictions=tensor([1, 2, 3, 4]), targets=tensor([2, 3, 4, 5]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_all_wrong_greater_than_threshold(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        evaluator.update(predictions=tensor([1, 2, 3, 4]), targets=tensor([3, 4, 5, 6]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0
        })

    def test_all_perfect_matrix(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.0)
        evaluator.update(predictions=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]),
                         targets=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_all_right_matrix(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        evaluator.update(predictions=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]),
                         targets=tensor([[1.1, 2.4, 3.6, 4.5], [0.1, 1, 1, 2.2]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_partially_right_matrix(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.5)
        evaluator.update(predictions=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]),
                         targets=tensor([[1.1, 2.4, 3.6, 4.6], [0.1, 1.1, 1, 2.2]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 0.625,
            "Recall": 0.625,
            "F1": 0.625
        })

    def test_all_right_matrix_equal_to_threshold(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        evaluator.update(predictions=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]),
                         targets=tensor([[2, 3, 4, 5], [-1, -0.5, 0, 1]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 1.0,
            "Recall": 1.0,
            "F1": 1.0
        })

    def test_all_wrong_matrix_greater_than_threshold(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        evaluator.update(predictions=tensor([[1, 2, 3, 4], [0, 0.5, 1, 2]]),
                         targets=tensor([[2.5, 3.5, 4.5, 5.5], [-1.5, -1, -0.5, 0.5]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0
        })

    def test_partially_right_nested_array(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=0.5)
        evaluator.update(predictions=tensor([[[1, 2, 3, 4]]]), targets=tensor([[[1, 2.6, 3.6, 4]]]))
        report = evaluator.compute()
        self.assertEqual(report, {
            "Precision": 0.5,
            "Recall": 0.5,
            "F1": 0.5
        })

    def test_predictions_not_numeric_tensor(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        self.assertRaisesRegex(ValueError,
                               r"'predictions' must be a float or integer torch.tensor.",
                               evaluator.update,
                               [1, 2, 3, 4],
                               tensor([2.5, 3.5, 4.5, 5.5])
                               )

    def test_targets_not_numeric_tensor(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        self.assertRaisesRegex(ValueError,
                               r"'targets' must be a float or integer torch.tensor.",
                               evaluator.update,
                               tensor([2.5, 3.5, 4.5, 5.5]),
                               [1, 2, 3, 4]
                               )

    def test_predictions_not_same_shape(self):
        evaluator = MeanAbsoluteErrorF1Score(error_threshold=1.0)
        self.assertRaisesRegex(ValueError,
                               r"'predictions' and 'targets' must have the same shape; got predictions of shape \[4\] and targets of shape \[1, 4\].",
                               evaluator.update,
                               tensor([2.5, 3.5, 4.5, 5.5]),
                               tensor([[1, 2, 3, 4]])
                               )

    def test_initialize_with_negative_threshold(self):
        self.assertRaisesRegex(ValueError,
                               r"'error_threshold' must be >= 0; got error_threshold=-1.0.",
                               MeanAbsoluteErrorF1Score,
                               error_threshold=-1.0)


if __name__ == '__main__':
    unittest.main()
