# visionmetrics [WIP :construction:]

This repo contains evaluation metrics for vision tasks such as classification, object detection, image caption, and image matting. It uses `torchmetrics` as a base and extends it to support custom vision tasks as necessary.

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
| RetrievalPrecisio                        |
| RetrievalMAP                             |

