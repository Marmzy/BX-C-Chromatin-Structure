#!/usr/bin/env python

import numpy as np
import os
import pandas as pd

from src.training.bxc_model import BXCModel
from src.utils.file_helper import check_file, check_path, get_path
from src.utils.general import get_config_val, get_device


def main_train(conf_dict):

    #Initialsing variables
    data_dir = get_config_val(conf_dict, ["data", "dirname"])
    device = get_device()
    kfold = get_config_val(conf_dict, ["pipeline", "cv"])
    model_type = get_config_val(conf_dict, ["model", "type"])
    model_name = get_config_val(conf_dict, ["model", "name"])
    path = get_path()
    verbose =  get_config_val(conf_dict, ["verbose"])

    #Setting up the model
    model = BXCModel(path, data_dir, model_name, model_type, device)
    model.create_clf()

    #Looping over the K folds
    for k in range(kfold):

        if model_type == "machine":
            if verbose:
                print("Loading data for fold {}...".format(str(k)))

            #Loading the training and validation datasets
            X_train_path = check_file(os.path.join(path, data_dir, "train/X_train_{}_{}.txt".format(model_type, str(k))))
            y_train_path = check_file(os.path.join(path, data_dir, "train/y_train_{}_{}.txt".format(model_type, str(k))))

            X_val_path = check_file(os.path.join(path, data_dir, "val/X_val_{}_{}.txt".format(model_type, str(k))))
            y_val_path = check_file(os.path.join(path, data_dir, "val/y_val_{}_{}.txt".format(model_type, str(k))))

            model.get_data(X_train_path, y_train_path, X_val_path, y_val_path)

            #Creating output dir and file
            metric = get_config_val(conf_dict, ["model", "metric"])
            f_name = check_path(os.path.join(path, data_dir, "output/randomforest_{}/randomforest_{}_fold{}.pkl".format(
                metric, metric, str(k)
            )))

            #Training the model
            model.train(f_name, k, verbose)
