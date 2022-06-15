#!/usr/bin/env python

import joblib
import glob
import os
import pandas as pd

from natsort import natsort_keygen
from src.utils.file_helper import get_path
from src.utils.general import get_config_val
from typing import Any, Dict


def main_eval(
    conf_dict: Dict[str, Any]
) -> None:
    """Evaluating the trained model

    Args:
        conf_dict (Dict[str, Any]): Yaml file contents
    """

    #Initialising variables
    batch_size = get_config_val(conf_dict, ["model", "params", "batch"])
    data_dir = get_config_val(conf_dict, ["data", "dirname"])
    decay = get_config_val(conf_dict, ["model", "params", "decay"])
    epochs = get_config_val(conf_dict, ["model", "params", "epochs"])
    interpol = get_config_val(conf_dict, ["data", "interpolate"])
    lr = get_config_val(conf_dict, ["model", "params","lr"])
    model_name = get_config_val(conf_dict, ["model", "name"])
    model_type = get_config_val(conf_dict, ["model", "type"])
    path = get_path()
    target = get_config_val(conf_dict, ["model", "target"])

    #Initialising variables
    pred_dict = {}
    suffix = ""
    if interpol:
        suffix = "_interpolate"
    if model_type == "machine":
        out_f = f"{os.path.join(path, data_dir)}/output/{model_name.lower()}{suffix}/{target}/{model_name.lower()}_predictions.txt"
    else:
        out_f = f"{os.path.join(path, data_dir)}/output/{model_name.lower()}{suffix}/{target}/" + \
        model_name + suffix + f"_lr{lr}_decay{decay}_epochs{epochs}_batch{batch_size}_predictions.txt"

    #Making predictions
    for k, model in enumerate(sorted(glob.glob(f"{os.path.join(path, data_dir)}/output/{model_name.lower()}{suffix}/{target}/*.pkl"))):

        #Loading the test datasets
        model = joblib.load(model)
        model.load_test(k)

        #Making predictions
        key = "PredFold_" + str(k)
        y_true, y_pred = model.predict(k)
        pred_dict[key] = y_pred.values

    #Outputting the predictions
    pred_dict["GroundTruth"] = y_true.values
    pred_dict["Sample"] = y_true.index
    df = pd.DataFrame(pred_dict).sort_values(by="Sample", key=natsort_keygen())
    df.to_csv(out_f, sep="\t", index=False)