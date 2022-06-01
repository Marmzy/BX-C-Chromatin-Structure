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

    def __init__(
        self,
        input: int,
        channels: int,
        pad: str
    ) -> None:
        """Initialization of network modules

        Args:
            input (int): Number of input channels (image channels)
            channels (int): Number of channels after convolution
            pad (str): Kernel padding
        """

        super().__init__()

        self.conv2 = nn.Conv2d(input, channels, kernel_size=7, stride=1, padding=pad)
        self.bnorm2 = nn.BatchNorm2d(channels)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Defining network structure

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Convolved, normalised and activated tensor
        """

        return F.relu(self.bnorm2(self.conv2(x)))


#Defining the structure of the custom Convolutional Neural Network
class CustomCNN1(nn.Module) :
    """Rajpurkar et al. custom CNN

    Args:
        nn (type): Base class for all neural network modules
    """

    def __init__(
        self
    ) -> None:
        """Initialization of network modules"""

        super().__init__()


        self.baseblock0 = BaseBlock(3, 32, "same")
        self.baseblock1 = BaseBlock(32, 32, "valid")
        self.l1 = nn.Linear(96800, 1)

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

        h1 = F.pad(x, pad=(4, 4, 4, 4))     # [batch_size, 3, 232, 232]
        h2 = self.baseblock0(h1)            # [batch_size, 32, 232, 232]
        h3 = F.max_pool2d(h2, 2)            # [batch_size, 32, 116, 116]
        h4 = self.baseblock1(h3)            # [batch_size, 32, 110, 110]
        h5 = F.max_pool2d(h4, 2)            # [batch_size, 32, 55, 55]
        h6 = torch.flatten(h5, 1)           # [batch_size, 96800])
        h7 = self.l1(h6)                    # [batch_size, 1])
        return h7
