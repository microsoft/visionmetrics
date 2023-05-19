from typing import Any
import torch
from torchmetrics import Metric


class ThresholdAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()

        # add state variables
        # self.add_state("state_variable_name", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # update state variables
        # self.state_variable_name += ...
        pass

    def compute(self) -> Any:
        # compute and return metric
        # return ...
        pass
