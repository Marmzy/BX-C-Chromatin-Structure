#!/usr/bin/env python

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.preprocessing.dna_prep import dna_overview
from src.preprocessing.rna_prep import rna_overview
from src.preprocessing.download_data import download_raw
from src.utils.file_helper import get_path, check_file, check_path
from src.utils.general import get_config_val
from typing import Any, Dict

 
def main_prep(
    conf_dict: Dict[str, Any]
) -> None:
    """Prepare data for training

    Args:
        conf_dict (Dict[str, Any]): Yaml file contents
    """

    #Initialising variables
    cut_off = 52 * get_config_val(conf_dict, ["data", "cutoff"])
    data_dir = get_config_val(conf_dict, ["data", "dirname"])
    download = get_config_val(conf_dict, ["data", "download"])
    visualise = get_config_val(conf_dict, ["data", "visualise"])
    interpolate = get_config_val(conf_dict, ["data", "interpolate"])
    model_type = get_config_val(conf_dict, ["model", "type"])
    pairwise = {}
    path = get_path()
    target = get_config_val(conf_dict, ["model", "target"])
    verbose =  get_config_val(conf_dict, ["verbose"])

    #Downloading the raw data
    if download:
        if verbose:
            print("Downloading the raw data...\n")
        download_raw(path, data_dir)

    #Reading the input DNA data as a dataframe
    dna1_df = pd.read_csv(check_file(os.path.join(path, data_dir, "raw/dnaData_exp1.csv")))
    dna2_df = pd.read_csv(check_file(os.path.join(path, data_dir, "raw/dnaData_exp2.csv")))

    #Getting pairwise distances
    pairwise_dict1 = dna_overview(dna1_df, "A", cut_off, model_type, interpolate, verbose)
    pairwise_dict2 = dna_overview(dna2_df, "B", cut_off, model_type, interpolate, verbose)
    pairwise.update(pairwise_dict1)
    pairwise.update(pairwise_dict2)

    #Getting the cell IDs
    cells1 = [int(cell.split("_")[1]) for cell in pairwise_dict1.keys()]
    cells2 = [int(cell.split("_")[1]) for cell in pairwise_dict2.keys()]

    #Reading the input RNA data as a dataframe
    rna1_df = pd.read_csv(check_file(os.path.join(path, data_dir, "raw/rnaData_exp1.csv")))
    rna2_df = pd.read_csv(check_file(os.path.join(path, data_dir, "raw/rnaData_exp2.csv")))

    if verbose:
        print(f"\nCalculating single cell expression state of {target}...")

    #Getting the expression states of the genes Abd-A, Abd-B and Ubx
    exp_states1 = rna_overview(rna1_df, target)
    exp_states2 = rna_overview(rna2_df, target)

    #Filtering the expression states to only get those of cells with sufficient barcodes
    exp_states_filtered1 = exp_states1[cells1]
    exp_states_filtered2 = exp_states2[cells2]
    exp_states_combined = np.concatenate([exp_states_filtered1, exp_states_filtered2])

    if verbose:
        print("Cell expression states:")
        for key in [0, 1]:
            print("{}: {}".format(key, Counter(exp_states_combined)[key]))
        print("Total: {}".format(str(exp_states_combined.shape[0])))

    #Defining the explanatory and response data
    if model_type == "machine":
        X = pd.DataFrame.from_dict(pairwise, orient="index", columns=[str(c1) + "_" + str(c2) for c1, c2 in itertools.combinations(list(range(1, 53)), 2)])
        X = X.fillna(-1)
    else:
        X = list(pairwise.keys())
    y = exp_states_combined.reshape(-1, 1)

    #Outputting the pairwise distances as images
    if interpolate:
        image_dir = "interpolate"
    else:
        image_dir = "missing"

    if visualise:
        for key in X:
            heatmap = sns.heatmap(pairwise[key], cbar=False, xticklabels=False, yticklabels=False)
            heatmap.set_facecolor("white")
            fig = heatmap.get_figure()
            fig.savefig(check_path(os.path.join(path, data_dir, "raw/images/{}/{}.png".format(image_dir, key))))
            fig.clf()

    #Splitting the data into (temporary) train and test
    test = get_config_val(conf_dict, ["pipeline", "split", "test"])
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test, stratify=y)

    if verbose:
        print("\nSplitting the dataset and into train ({}%) and test ({}%)...".format(str(int((1-float(test))*100)), str(int(float(test)*100))))

    #Splitting the temporary training dataset into K training and validation datasets
    kfold = get_config_val(conf_dict, ["pipeline", "cv"])
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)

    if verbose:
        print("\nSplitting the temporary training dataset into {} folds...".format(kfold))

    if interpolate:
        suffix = "_interpolate"
    else:
        suffix = ""

    #Looping over the k folds
    for idx, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        if verbose:
            print("TRAIN:", train_index, "VAL:", val_index)

        X_train, X_val = np.array(X_temp)[train_index], np.array(X_temp)[val_index]
        y_train, y_val = np.array(y_temp)[train_index], np.array(y_temp)[val_index]

        #Saving the train and validation datasets
        if model_type == "machine":
            train_idx = [X_temp.index.values.tolist()[i] for i in train_index]
            val_idx = [X_temp.index.values.tolist()[i] for i in val_index]

            X_train = pd.DataFrame(X_train, index=train_idx, columns=X_temp.columns.values.tolist())
            X_train.to_csv(check_path(os.path.join(path, data_dir, "train/{}/{}/X_train{}_{}.txt".format(model_type, target, suffix, idx))))
            np.savetxt(check_path(os.path.join(path, data_dir, "train/{}/{}/y_train{}_{}.txt".format(model_type, target, suffix, idx))), y_train, fmt='%s')

            X_val = pd.DataFrame(X_val, index=val_idx, columns=X_temp.columns.values.tolist())
            X_val.to_csv(check_path(os.path.join(path, data_dir, "val/{}/{}/X_val{}_{}.txt".format(model_type, target, suffix, idx))))
            np.savetxt(check_path(os.path.join(path, data_dir, "val/{}/{}/y_val{}_{}.txt".format(model_type, target, suffix, idx))), y_val, fmt='%s')
        else:
            train_paths = [os.path.join(path, f"data/raw/images/{image_dir}/{key}.png") for key in X_train]
            val_paths = [os.path.join(path, f"data/raw/images/{image_dir}/{key}.png") for key in X_val]
            
            np.savetxt(check_path(os.path.join(path, data_dir, "train/{}/{}/X_train{}_{}.txt".format(model_type, target, suffix, idx))), np.array(train_paths), fmt='%s')
            np.savetxt(check_path(os.path.join(path, data_dir, "train/{}/{}/y_train{}_{}.txt".format(model_type, target, suffix, idx))), y_train, fmt='%s')

            np.savetxt(check_path(os.path.join(path, data_dir, "val/{}/{}/X_val{}_{}.txt".format(model_type, target, suffix, idx))), np.array(val_paths), fmt='%s')
            np.savetxt(check_path(os.path.join(path, data_dir, "val/{}/{}/y_val{}_{}.txt".format(model_type, target, suffix, idx))), y_val, fmt='%s')

    #Saving the test dataset
    if model_type == "machine":
        X_test.to_csv(check_path(os.path.join(path, data_dir, "test/{}/{}/X_test{}.txt".format(model_type, target, suffix))))
        np.savetxt(check_path(os.path.join(path, data_dir, "test/{}/{}/y_test{}.txt".format(model_type, target, suffix))), y_test, fmt='%s')
    else:
        test_paths = [os.path.join(path, f"data/raw/images/{image_dir}/{key}.png") for key in X_test]
        np.savetxt(check_path(os.path.join(path, data_dir, "test/{}/{}/X_test{}.txt".format(model_type, target, suffix, idx))), np.array(test_paths), fmt='%s')
        np.savetxt(check_path(os.path.join(path, data_dir, "test/{}/{}/y_test{}.test".format(model_type, target, suffix))), y_test, fmt='%s')

    if verbose:
        print("\nSaved the training datasets to: {}".format(os.path.join(path, data_dir, "train")))
        print("Saved the validation datasets to: {}".format(os.path.join(path, data_dir, "val")))
        print("Saved the test dataset to: {}".format(os.path.join(path, data_dir, "test")))
