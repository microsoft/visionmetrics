# Imports


def precision_recall_f1_scalar(tp: int, fp: int, fn: int):
    """
    Computes precision, recall, and F1 from scalar values, defined as zero when there is division by zero.
    Args:
        tp: int, the number of true positives.
        fp: int, the number of false positives.
        fn: int, the number of false negatives.
    """
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
