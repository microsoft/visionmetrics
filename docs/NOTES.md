- `visionmetrics` uses a strict thresholding. For e.g., if thres=0.3 and pred=0.3, it will be counted as a negative prediction. This is different from the behavior of `vision-evaluation` repo which counts it as a positive prediction. Note that the `azure_openai_*` evaluators are an exception to this rule since the scores are generated on a discrete scale; these evaluators are thresholded inclusively (0.3 would be considered a positive prediction).

- For the IC Accuracy metric, `visionmetrics` does not support samples-based averaging like `sklearn`. Supported averaging methods are `macro`, `micro`, `weighted`, and `none`.

- For metrics like Recall:

```
targets = torch.tensor([1, 2, 3, 4, 5]) 
predictions = torch.tensor([[0, 1, 0, 0, 0, 0], 
                      [0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, 1, 0, 0], 
                      [0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1]])

# Class-wise Recall
visionmetrics -> [0,1,1,1,1,1] (avg: 0.8333)
vision-evaluation -> [1,1,1,1,1] (avg: 1.0) # does not report the class with no positives

```
