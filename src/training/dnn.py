#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


#Defining the structure of the custom Dense Neural Network
class CustomDNN1(nn.Module) :
    """Rajpurkar et al. custom DNN 1

    Args:
        nn (type): Base class for all neural network modules
    """

    def __init__(
        self
        ) -> None:
        """Initialization of network modules"""

        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(150528, 20)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(20, 10)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(10, 1))
        ]))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Defining network structure

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        h0 = self.flatten(x)
        h1 = self.linear_relu_stack(h0)
        return h1
