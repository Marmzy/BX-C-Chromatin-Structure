#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


#Defining the layers that constitute the basic structure of the base block
class BaseBlock(nn.Module):
    """Basic block that consitute the Rajpurkar et al. custom CNN

    Args:
        nn (type): Base class for all neural network modules
    """

    def __init__(self, input, channels, pad):
        super().__init__()

        self.conv2 = nn.Conv2d(input, channels, kernel_size=7, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bnorm2(self.conv2(x)))


#Defining the structure of the custom Convolutional Neural Network
class CustomCNN(nn.Module) :
    """Rajpurkar et al. custom CNN

    Args:
        nn (type): Base class for all neural network modules
    """

    def __init__(
        self
    ) -> None:
        """Initialization of network modules"""

        super().__init__()

        self.baseblock0 = BaseBlock(3, 32, 4)
        self.baseblock1 = BaseBlock(32, 32, 0)
        self.l1 = nn.Linear(32, 8)

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

        h1 = self.baseblock0(x)
        h2 = F.max_pool2d(h1, 2)
        h3 = self.baseblock1(h2)
        h4 = F.max_pool2d(h3, 2)
        h5 = torch.flatten(h4, 1)
        h6 = self.l1(h5)
        return h6
