# visionmetrics [WIP :construction:]

This repo contains evaluation metrics for vision tasks such as classification, object detection, image caption, and image matting. It uses `torchmetrics` as a base and extends it to support custom vision tasks as necessary.

## Available Metrics

### Image Classification:

| Metric                                   |                                                                
| ---------------------------------------- | 
| MulticlassAccuracy                       |                                         
| MulticlassPrecision                      |                                          
| MulticlassRecall                         |                                                                
| MulticlassF1Score                        |                                                                
| MulticlassCalibrationError               |                                                                
| MulticlassAveragePrecision               |                                                                
| MulticlassAUCROC                         |                                                                
| MulticlassConfusionMatrix                |                                                                

**Note**: The corresponding metrics for binary and multilabel classification are available as Binary* and Multilabel*.


### Object Detection:

| Metric                                   |                                                                
| ---------------------------------------- | 
| MeanAveragePrecision                     |                                                                