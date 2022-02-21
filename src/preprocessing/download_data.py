#!/usr/bin/env python

import os
import wget

def download_raw(
    path: str,
    data_dir: str
) -> None:
    """Downloading the raw data

    Args:
        path (str): Project path
        data_dir (str): Output directory name
    """

    #Initialising variables
    out_dir = os.path.join(path, data_dir, "raw")

    #Downloading the data
    wget.download("https://zenodo.org/record/4741214/files/dnaData_exp1.csv", out=out_dir)
    wget.download("https://zenodo.org/record/4741214/files/dnaData_exp2.csv", out=out_dir)
    wget.download("https://zenodo.org/record/4741214/files/rnaData_exp1.csv", out=out_dir)
    wget.download("https://zenodo.org/record/4741214/files/rnaData_exp2.csv", out=out_dir)