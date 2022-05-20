#!/usr/bin/env python

import errno
import os
import yaml

from typing import Any, Dict


def check_file(
    filename: str
) -> str:
    """Checks if file exists

    Args:
        filename (str): Complete path to file

    Raises:
        FileNotFoundError: If file does not exist

    Returns:
        str: Original file name
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    return filename


def check_path(
    filename: str
) -> str:
    """Checks the path of a given file and creates new directories if necessary

    Args:
        filename (str): Complete path to file

    Returns:
        str: Original file name
    """
    
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    return filename


def get_path() -> str:
    """Get project path

    Returns:
        str: Full path to the project
    """

    idx = os.getcwd().split("/").index("BX-C_Chromatin_Structure")
    path = "/".join(os.getcwd().split("/")[:idx+1])

    return path


def read_yaml(
    config: str
) -> Dict[str, Any]:
    """Read yaml file into dict

    Args:
        config (str): yaml file path

    Raises:
        YAMLError: If yaml file cannot be read

    Returns:
        Dict[str, Any]: yaml file contents
    """

    with open(check_file(config), "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data