#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Any, List, Tuple


def create_sequence(
    layers: List[int]
) -> List[Tuple[str, Any]]:
    """Creating the sequence of layers

    Args:
        layers (List[int]): List of hidden layer sizes

    Returns:
        List[Tuple[str, Any]]: List of layer tuples
    """

    sequence = [('fc1', nn.Linear(150528, layers[0])), ('relu1', nn.ReLU())]
    
    for i in range(len(layers)-1):
        sequence.append((f'fc{i+2}', nn.Linear(layers[i], layers[i+1])))
        sequence.append((f'relu{i+2}', nn.ReLU()))

    sequence.append((f'fc{len(layers)+1}', nn.Linear(layers[-1], 1)))

    return sequence


#Defining the structure of the custom Dense Neural Network
class CustomDNN(nn.Module) :
    """Rajpurkar et al. custom DNN

    Args:
        nn (type): Base class for all neural network modules
    """

    def __init__(
        self,
        layers: List[int]
    ) -> None:
        """Initialization of network modules

        Args:
            layers (List[int]): Hidden layer sizes
        """

        super().__init__()

        sequence = create_sequence(layers)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(OrderedDict(sequence))

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
