#!/usr/bin/env python

from typing import Any, Dict, List

def get_config_val(
    config: Dict[str, Any],
    keys: List[str]
) -> Any:
    """Extract value from config file

    Args:
        config (Dict[str, Any]): Configuration (.yaml) file
        keys (List[str]): List of key names

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