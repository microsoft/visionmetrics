# visionmetrics [WIP :construction:]

This repo contains evaluation metrics for vision tasks such as classification, object detection, image caption, and image matting. It uses [torchmetrics](https://github.com/Lightning-AI/torchmetrics) as a base library and extends it to support custom vision tasks as necessary.

## Available Metrics

### Image Classification:


| Metric                         |  Binary | Multiclass |Multilabel |                                                           
| ------------------------------ | --------|------------|------------|
| Accuracy                       |         |             |           |     
| Precision                      |         |             |           |       
| Recall                         |         |             |           |                             
| F1Score                        |         |             |           |                             
| CalibrationError               |         |             |❌        |                                                               
| AveragePrecision               |         |             |           |                                                              
| AUCROC                         |         |             |           |                                                             
| ConfusionMatrix                |         |             |           |                                                            

❌: Not available for that task. 


### Object Detection:

| Metric                                   |                                                                
| ---------------------------------------- | 
| MeanAveragePrecision                     |          


### Image Regression:

| Metric                                   |                                                                
| ---------------------------------------- | 
| MeanAbsoluteError                        |     
| MeanSquaredError                         |


### Image Retrieval:

| Metric                                   |                                                                
| ---------------------------------------- | 
| RetrievalRecall                          |     
| RetrievalPrecision                       |
| RetrievalMAP                             |


### Image Caption
| Metric                                   |                                                       
| ---------------------------------------- | 
| BleuScore                                |     
| METEORScore                              |
| ROUGELScore                              |
| CIDErScore                               |
| SPICEScore                               |

### Image Matting
| Metric                                   |                                                                
| ---------------------------------------- | 
| MeanIOU                                  |     
| ForegroundIOU                            |
| BoundaryMeanIOU                          |
| BoundaryForegroundIOU                    |
| L1Error                                  |

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