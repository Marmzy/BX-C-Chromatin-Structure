#!/usr/bin/env python

import joblib
import glob
import os
import pandas as pd

from natsort import natsort_keygen
from src.utils.file_helper import get_path
from src.utils.general import get_config_val


def main_score(conf_dict):

    #Initialising variables
    data_dir = get_config_val(conf_dict, ["data", "dirname"])
    interpol = get_config_val(conf_dict, ["data", "interpolate"])
    model_name = get_config_val(conf_dict, ["model", "name"])
    model_type = get_config_val(conf_dict, ["model", "type"])
    path = get_path()
    target = get_config_val(conf_dict, ["model", "target"])

    #Loading the trained models
    if model_type == "machine":

        #Initialising variables
        pred_dict = {}
        suffix = ""

        if interpol:
            suffix = "_interpolate"
        
        #Making predictions
        for k, model in enumerate(glob.glob(f"{os.path.join(path, data_dir)}/output/{model_name.lower()}{suffix}/{target.lower()}/*.pkl")):

            #Loading the test datasets
            model = joblib.load(model)
            model.load_test()

            #Making predictions
            key = "PredFold_" + str(k)
            y_true, y_pred = model.predict(k)
            pred_dict[key] = y_pred.values

        #Outputting the predictions
        pred_dict["GroundTruth"] = y_true.values
        pred_dict["Sample"] = y_true.index
        df = pd.DataFrame(pred_dict).sort_values(by="Sample", key=natsort_keygen())
        out_f = f"{os.path.join(path, data_dir)}/output/{model_name.lower()}{suffix}/{target.lower()}/{model_name.lower()}_predictions.txt"
        df.to_csv(out_f, sep="\t", index=False)
