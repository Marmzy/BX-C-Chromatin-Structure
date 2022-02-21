#!/usr/bin/env python

import numpy as np
import pandas as pd

def rna_overview(
    df: pd.DataFrame
) -> np.ndarray:
    """Get expression states of target genes

    Args:
        df (pd.DataFrame): RNA expression data

    Returns:
        np.ndarray: Genes Abd-A, Abd-B and Ubx expression states
    """

    #Initialising variables
    states = {}

    #Looping over the three genes
    for gene in ["Abd-A", "Abd-B", "Ubx"]:
        cols = ["cellNumber", "embNumber", "segNumber"]
        cols.extend([col for col in df.columns if gene in col])

        #Getting the number of molecules per cell
        mini_df = df[cols]
        molecules = mini_df.iloc[:, 3:].sum(axis=1)

        #Convert the number of molecules to expression states
        states[gene] = list(molecules.where(molecules < 1, 1))

    #Combining the data for all three genes
    exp_states = [",".join([str(a), str(b), str(u)]) for a, b, u in zip(states["Abd-A"], states["Abd-B"], states["Ubx"])]

    return np.array(exp_states)