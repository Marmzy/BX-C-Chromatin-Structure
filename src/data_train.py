#!/usr/bin/env python

import joblib
import os

from src.training.bxc_model import BXCModel
from src.utils.file_helper import check_path, get_path
from src.utils.general import get_config_val, get_device
from typing import Any, Dict


def main_train(
    conf_dict: Dict[str, Any]
) -> None:
    """Training the model and optimising hyperparameters

    Args:
        conf_dict (Dict[str, Any]): Yaml file contents
    """

    #Initialising variables
    data_dir = get_config_val(conf_dict, ["data", "dirname"])
    device = get_device()
    interpol = get_config_val(conf_dict, ["data", "interpolate"])
    kfold = get_config_val(conf_dict, ["pipeline", "cv"])
    model_name = get_config_val(conf_dict, ["model", "name"])
    path = get_path()
    target = get_config_val(conf_dict, ["model", "target"])

    #Setting up the model
    model = BXCModel(path, conf_dict, device)
    model.create_clf()

    #Looping over the K folds
    for k in range(kfold):
        
        #Loading the training and validation datasets
        model.load_train_val(k)
        quit()

        break

        # #Training the model
        # model.train(k)
        # if interpol:
        #     joblib.dump(model, check_path(os.path.join(path, data_dir, f"output/{model_name.lower()}_interpolate/{target}/{model_name.lower()}_fold{str(k)}.pkl")))
        # else:
        #     joblib.dump(model, check_path(os.path.join(path, data_dir, f"output/{model_name.lower()}/{target}/{model_name.lower()}_fold{str(k)}.pkl")))
