#!/usr/bin/env python

import torch

from typing import Any, Dict, List


def get_config_val(
    config: Dict[str, Any],
    keys: List[str]
) -> Any:
    """Extract value from config file

    Args:
        config (Dict[str, Any]): Configuration (.yaml) file
        keys (List[str]): List of key names

    Raises:
        KeyError: If key cannot be loaded

    Returns:
        Any: Value given the input keys
    """

    #Loading keys from the config dict
    for key in keys:
        if config.get(key) is not None:
            config = config.get(key)
        else:
            msg = "Failed to load parameter '{}' from yaml file".format(key)
            raise KeyError(msg)

    return config
    

def get_device() -> torch.device:
    """Device on which PyTorch will run

    Returns:
        torch.device: CUDA if available, otherwise CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")