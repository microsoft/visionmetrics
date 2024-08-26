# visionmetrics

This repo contains evaluation metrics for vision tasks such as classification, object detection, image caption, and image matting. It uses [torchmetrics](https://github.com/Lightning-AI/torchmetrics) as a base library and extends it to support custom vision tasks as necessary.

## Available Metrics

### Image Classification:
  - `Accuracy`: Computes the top-k accuracy for a classification problem. A prediction is considered correct, if the ground truth label is within the labels with top k confidences.
  - `PrecisionEvaluator`: Computes precision.
  - `RecallEvaluator`: Computes recall.
  - `AveragePrecisionEvaluator`: Computes the average precision, i.e., precision averaged across different confidence thresholds. 
  - `AUCROC`: Computes Area under the Receiver Operating Characteristic Curve.
  - `F1Score`: Computes f1-score.
  - `CalibrationLoss`<sup>**</sup>: Computes the [ECE loss](https://arxiv.org/pdf/1706.04599.pdf), i.e., the expected calibration error, given the model confidence and true labels for a set of data points.
  - `ConfusionMatrix`: Computes the confusion matrix of a classification. By definition a confusion matrix C is such that Cij is equal to the number of observations known to be in group i and predicted to be in group j (https://en.wikipedia.org/wiki/Confusion_matrix).
  - `ExactMatch`: Computes the exact match score, i.e., the percentage of samples where the predicted label is exactly the same as the ground truth label.
  - `MultilabelF1ScoreWithDuplicates`: Computes a variant of the MultilabelF1Score to perform evaluation over lists of predictions that may contain duplicates, where the number of each value is also factored into the score and contributes to true positives, false positives, and false negatives. Returns micro precision, recall, and F1 in a dictionary.

The above metrics are available for Binary, Multiclass, and Multilabel classification tasks. For example, `BinaryAccuracy` is the binary version of `Accuracy` and `MultilabelAccuracy` is the multilabel version of `Accuracy`. Please refer to the example usage below for more details.

<sup>**</sup> The `CalibrationLoss` metric is only for binary and multiclass classification tasks.

### Object Detection:
- `MeanAveragePrecision`: Coco mean average precision (mAP) computation across different classes, under multiple [IoU(s)](https://en.wikipedia.org/wiki/Jaccard_index).
- `ClassAgnosticAveragePrecision`: Coco mean average prevision (mAP) calculated in a class-agnostic manner. Considers all classes as one class.
- `DetectionConfusionMatrix`: Similar to classification confusion matrix, but for object detection tasks.
- `DetectionMicroPrecisionRecallF1`: Computes the micro precision, recall, and F1 scores based on the true positive, false positive, and false negative values computed by `DetectionConfusionMatrix`. Returns the three values in a dictionary.

### Image Caption:
  - `BleuScore`: Computes the Bleu score. For more details, refer to [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
  - `METEORScore`: Computes the Meteor score. For more details, refer to [Project page](http://www.cs.cmu.edu/~alavie/METEOR/). We use the latest version (1.5) of the [Code](https://github.com/mjdenkowski/meteor).
  - `ROUGELScore`: Computes the Rouge-L score. Refer to [ROUGE: A Package for Automatic Evaluation of Summaries](http://anthology.aclweb.org/W/W04/W04-1013.pdf) for more details.
  - `CIDErScore`:  Computes the CIDEr score. Refer to [CIDEr: Consensus-based Image Description Evaluation](http://arxiv.org/pdf/1411.5726.pdf) for more details.
  - `SPICEScore`:  Computes the SPICE score. Refer to [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) for more details.
  - `AzureOpenAITextModelCategoricalScore`: Computes micro precision, recall, F1, and accuracy scores, and an average model score, based on scores generated from a specified prompt to an Azure OpenAI model. Returns the results in a dictionary.

### Image Matting:
  - `MeanIOU`: Computes the mean intersection-over-union score. 
  - `ForegroundIOU`: Computes the foreground intersection-over-union evaluator score.
  - `BoundaryMeanIOU`: Computes the boundary mean intersection-over-union score. 
  - `BoundaryForegroundIOU`: Computes the boundary foreground intersection-over-union score.
  - `L1Error`: Computes the L1 error.

### Regression:
  - `MeanSquaredError`: Computes the mean squared error. 
  - `MeanAbsoluteError`: Computes the mean absolute error.
  - `MeanAbsoluteErrorF1Score`: Computes the micro precision, recall, and F1 scores based on the true positive, false positive, and false negative values determined by a provided error threshold. Returns the three values in a dictionary.

### Retrieval:
  - `RetrievalRecall`: Computes Recall@k, which is the percentage of relevant items in top-k among all relevant items
  - `RetrievalPrecision`: Computes Precision@k, which is the percentage of TP among all items classified as P in top-k.
  - `RetrievalMAP`: Computes [Mean Average Precision@k](https://stackoverflow.com/questions/54966320/mapk-computation), an information retrieval metric.
  - `RetrievalPrecisionRecallCurveNPoints`: Computes a Precision-Recall Curve, interpolated at k points and averaged over all samples.

### Grounding:
  - `Recall`: Computes Recall@k, which is the percentage of correct grounding in top-k among all relevant items.

### Key-Value Pair Extraction:
  - `KeyValuePairExtractionScore`: Evaluates methods that perform arbitrary schema-based structured field extraction. Each schema is an adapted version of a [JSON Schema](https://json-schema.org/understanding-json-schema/reference)-formatted dictionary that contains keys, each of which specifies the standard JSON Schema type of the key and a string description, whether to perform grounding on the key, classes for closed-vocabulary values, and additional information describing list items and object properties (sub-keys). Based on the properties defined in the JSON Schema, infers the best evaluation metric for each key's data type, and defaults to text-based evaluation for cases that have no clear definition. For each key, definitions of true positive, false positive, and false negative are inherited from the corresponding metric. In addition to metric-specific definitions, missing keys in predictions are counted as false negatives, and invalid keys in predictions are counted as false positives. Computes the key-wise metrics for each key in the schema and returns the overall micro F1, macro F1, and key-wise scores in their raw format for each key.

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