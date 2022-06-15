#!/usr/bin/env python

import numpy as np
import pandas as pd

from sys import modules

def rna_overview(
    df: pd.DataFrame,
    target: str
) -> np.ndarray:
    """Get expression state of target gene

    Args:
        df (pd.DataFrame): RNA expression data
        target (str): Target gene

    Returns:
        np.ndarray: Expression state of target gene
    """

    #Getting the number of molecules per cell
    cols = [col for col in df.columns if target in col]
    cols = [c for c in cols if "intron" not in c.lower()]
    molecules = df[cols].sum(axis=1)

    #Convert the number of molecules to expression states
    return np.array(list(molecules.where(molecules < 1, 1)))