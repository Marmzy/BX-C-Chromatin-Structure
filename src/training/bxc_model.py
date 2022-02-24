#!/usr/bin/env python

import joblib
import numpy as np
import pandas as pd
import os
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
from src.training.cnn import CustomCNN
from src.utils.file_helper import check_path


class BXCModel():
    def __init__(
        self,
        path: str,
        data_dir: str,
        model_name: str,
        model_type: str,
        device: torch.device
    ) -> None:
        """init

        Args:
            path (str): Project path
            data_dir (str): Data directory name
            model_name (str): Classifier name
            model_type (str): Machine or Deep depending on classifier
            device (torch.device): Device on which to train
        """
        
        self.path = check_path(os.path.join(path, data_dir))
        self.model = model_name
        self.type = model_type
        self.device = device

    def create_clf(
        self
    ) -> None:
        """Load model classifier"""

        if self.model == "RandomForest":
            self.clf = RandomForestClassifier()
        else:
            self.clf = CustomCNN().to(self.device)

    def get_data(
        self,
        x_train_path: str,
        y_train_path: str,
        x_val_path: str,
        y_val_path: str
    ) -> None:
        """Load training and validation datasets

        Args:
            x_train_path (str): Training explanatory data path
            y_train_path (str): Training response data path
            x_val_path (str): Validation explanatory data path
            y_val_path (str): Validation response data path
        """

        if self.type == "machine":

            #Loading the training dataset
            X_train = pd.read_csv(x_train_path, index_col=0)
            y_train = pd.read_csv(y_train_path, sep=" ", names=["0,0,0", "0,0,1", "0,1,0", "0,1,1",
                                                                "1,0,0", "1,0,1", "1,1,0", "1,1,1"])

            #Loading the validation dataset
            X_val = pd.read_csv(x_val_path, index_col=0)
            y_val = pd.read_csv(y_val_path, sep=" ", names=["0,0,0", "0,0,1", "0,1,0", "0,1,1",
                                                            "1,0,0", "1,0,1", "1,1,0", "1,1,1"])

            #Combining the train and val datasets
            self.X = pd.concat([X_train, X_val])
            self.y = pd.concat([y_train, y_val])

            #Indicating the validation records
            self.predefined = [-1] * len(X_train) + [0] * len(X_val)

    def train(
        self,
        f_name: str,
        fold: int,
        verbose: bool
    ) -> None:
        """Train and save best model

        Args:
            f_name (str): Output file name
            fold (int): Fold number
            verbose (bool): Print detailed information
        """

        if self.type == "machine":

            #Initialising variables
            param_grid = {
                "n_estimators": [50, 100, 250, 500],
                "max_depth": [None, 5, 10, 25, 50],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 5, 10, 15],
            }
            ps = PredefinedSplit(self.predefined)

            #Tuning hyperparameters on the validation dataset
            print("Training the model and optimizing hyperparameters...")
            opt = BayesSearchCV(self.clf, param_grid, scoring=make_scorer(roc_auc_score, average="macro", multi_class="ovo"), cv=ps, n_jobs=-1, random_state=0)
            opt.fit(self.X, self.y)

            if verbose:
                print("Optimal hyperparameters for fold {}:".format(str(fold)))
                for key, val in opt.best_params_.items():
                    print("{}: {}".format(key, str(val)))

            #Saving the classifier with best hyperparameters
            joblib.dump(opt.best_estimator_, os.path.join(self.path, f_name))
        