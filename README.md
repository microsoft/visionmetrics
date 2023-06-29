# visionmetrics

This repo contains evaluation metrics for vision tasks such as classification, object detection, image caption, and image matting. It uses [torchmetrics](https://github.com/Lightning-AI/torchmetrics) as a base library and extends it to support custom vision tasks as necessary.

## Available Metrics

### Image Classification:
  - `Accuracy`: computes the top-k accuracy for a classification problem. A prediction is considered correct, if the ground truth label is within the labels with top k confidences.
  - `PrecisionEvaluator`: computes precision.
  - `RecallEvaluator`: computes recall.
  - `AveragePrecisionEvaluator`: computes the average precision, i.e., precision averaged across different confidence thresholds. 
  - `AUCROC`: computes Area under the Receiver Operating Characteristic Curve.
  - `F1Score`: computes f1-score.
  - `CalibrationLoss`<sup>**</sup>: computes the [ECE loss](https://arxiv.org/pdf/1706.04599.pdf), i.e., the expected calibration error, given the model confidence and true labels for a set of data points.
  - `ConfusionMatrix`: computes the confusion matrix of a classification. By definition a confusion matrix C is such that Cij is equal to the number of observations known to be in group i and predicted to be in group j (https://en.wikipedia.org/wiki/Confusion_matrix).

The above metrics are available for Binary, Multiclass, and Multilabel classification tasks. For example, `BinaryAccuracy` is the binary version of `Accuracy` and `MultilabelAccuracy` is the multilabel version of `Accuracy`. Please refer to the example usage below for more details.

<sup>**</sup> The `CalibrationLoss` metric is only for binary and multiclass classification tasks.

### Object Detection:
- `MeanAveragePrecision`: Coco mean average precision (mAP) computation across different classes, under multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).

### Image Caption:
  - `BleuScore`: computes the Bleu score. For more details, refer to [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
  - `METEORScore`: computes the Meteor score. For more details, refer to [Project page](http://www.cs.cmu.edu/~alavie/METEOR/). We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor).
  - `ROUGELScore`: computes the Rouge-L score. Refer to [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf) for more details.
  - `CIDErScore`:  computes the CIDEr score. Refer to [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf) for more details.
  - `SPICEScore`:  computes the SPICE score. Refer to [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) for more details.

### Image Matting:
  - `MeanIOU`: computes the mean intersection-over-union score. 
  - `ForegroundIOU`: computes the foreground intersection-over-union evaluator score.
  - `BoundaryMeanIOU`: computes the boundary mean intersection-over-union score. 
  - `BoundaryForegroundIOU`:  computes the boundary foreground intersection-over-union score.
  - `L1Error`:  computes the L1 error.

### Regression:
  - `MeanSquaredError`: computes the mean squared error. 
  - `MeanAbsoluteError`: computes the mean absolute error.

### Retrieval:
  - `RetrievalRecall`: computes Recall@k, which is the percentage of relevant items in top-k among all relevant items
  - `RetrievalPrecision`: computes Precision@k, which is the percentage of TP among all items classified as P in top-k.
  - `RetrievalMAP`: computes [Mean Average Precision@k](https://stackoverflow.com/questions/54966320/mapk-computation), an information retrieval metric.
  - `RetrievalPrecisionRecallCurveNPoints`: computes a Precision-Recall Curve, interpolated at k points and averaged over all samples. 


## Example Usage

```python
import torch
from visionmetrics.classification import MulticlassAccuracy

preds = torch.rand(10, 10)
target = torch.randint(0, 10, (10,))

# Initialize metric
metric = MulticlassAccuracy(num_classes=10, top_k=1, average='macro')

# Add batch of predictions and targets
metric.update(preds, target)

# Compute metric
result = metric.compute()
```

## Implementing Custom Metrics
Please refer to [torchmetrics](https://github.com/Lightning-AI/torchmetrics#implementing-your-own-module-metric) for more details on how to implement custom metrics.


## Additional Requirements

The image caption metric calculation requires Jave Runtime Environment (JRE) (Java 1.8.0) and some extra dependencies which can be installed with `pip install visionmetrics[caption]`. This is not required for other evaluators. If you do not need image caption metrics, JRE is not required.