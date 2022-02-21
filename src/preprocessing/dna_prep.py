#!/usr/bin/env python

import itertools
import numpy as np
import pandas as pd

from scipy import interpolate
from typing import Dict, List


def interpolate_coords(
    coords: np.ndarray
) -> np.ndarray:
    """Interpolate missing coordinates of a single axis

    Args:
        coords (np.ndarray): Axis coordinates

    Returns:
        np.ndarray: Numpy array with interpolated values
    """

    #Seperate the mising values from the actual coordinates
    nan_idx = np.argwhere(np.isnan(coords))
    coord_idx = np.argwhere(~np.isnan(coords))

    #Interpolating the missing vales
    f = interpolate.interp1d(coord_idx.squeeze(), coords[coord_idx].squeeze(), fill_value="extrapolate")

    #Replacing the missing values with the interpolated ones
    y_new = f(nan_idx)
    coords[nan_idx] = y_new

    return coords


def pairwise_distances(
    df: pd.DataFrame,
    missing: List[int],
    interpolate: bool
) -> np.ndarray:
    """Calculates pairwise distances between all positions

    Args:
        df (pd.DataFrame): cellNumber data
        missing (List[int]): List of missing coordinates
        interpolate (bool): Whether to interpolate missing coordinates or not

    Returns:
        np.ndarray: Array of all pairwaise distances
    """

    #Extracting the (x,y,z) coordinates from the cell dataframe and adding missing data
    xyz_data = [np.array([row["x"], row["y"], row["z"]]) for _, row in df.iterrows()]
    for num in missing:
        xyz_data.insert(num-1, np.array([np.nan, np.nan, np.nan]))

    #Interpolate missing values by linear interpolation between adjacent coordinates
    if interpolate:
        x_coords = interpolate_coords(np.array([arr[0] for arr in xyz_data]))
        y_coords = interpolate_coords(np.array([arr[1] for arr in xyz_data]))
        z_coords = interpolate_coords(np.array([arr[2] for arr in xyz_data]))

        xyz_data = [np.array([x, y, z]) for x, y, z in zip(x_coords, y_coords, z_coords)]

    #Calculating pairwise distances between the coordinates of the barcodes
    xyz_product = itertools.product(xyz_data, repeat=2)
    pairwise_xyz = [np.linalg.norm(arr1-arr2) for arr1, arr2 in xyz_product]
    pairwise_xyz = np.reshape(pairwise_xyz, (52, 52)).T

    return pairwise_xyz


def dna_overview(
    df: pd.DataFrame,
    id: str,
    cut_off: float,
    learn: str,
    interpolate: bool,
    verbose: bool
) -> Dict[str, np.ndarray]:
    """Get pairwise distances from cells that pass filtering

    Args:
        df (pd.DataFrame): 3D ORCA data
        id (str): Dataset name
        cut_off (float): NaN filter
        learn (str): Machine or Deep to decide the output
        interpolate (bool): Whether to interpolate missing coordinates or not
        verbose (bool): Print detailed information

    Returns:
        Dict[str, np.ndarray]: Dictionary with cellNumber as keys and pairwise distances as values 
    """

    #Initialising variables
    cell_dict = {}

    if verbose:
        if interpolate:
            print("Interpolating missing coordinates and calculating pairwise distances for cells that contain at least {} barcodes...".format(str(cut_off)))
        else:
            print("Calculating pairwise distances for cells that contain at least {} barcodes...".format(str(cut_off)))

    #Looping over the cells in the dataset
    for cn in df["cellNumber"].unique():
        mini_df = df[df["cellNumber"] == cn]
        missing = list(set(range(1,53)) - set(mini_df["barcode"].values))

        #Continue data prep only with cells that contain enough barcode data
        if len(mini_df) >= cut_off:

            #Calculate pairwise distances for all barcodes in the cell
            pairwise_xyz = pairwise_distances(mini_df, missing, interpolate)

            #Adding pairwise data to the dictionary
            if learn == "deep":
                cell_dict[id + "_" + str(cn)] = pairwise_xyz
            else:
                cell_dict[id + "_" + str(cn)] = pairwise_xyz[np.triu_indices(52, k=1)]

    return cell_dict