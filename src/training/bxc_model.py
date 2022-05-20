#!/usr/bin/env python

import pandas as pd
import os
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
from src.training.cnn import CustomCNN
from src.utils.file_helper import check_file, check_path
from typing import Tuple


class BXCModel():
    def __init__(
        self,
        path: str,
        data_dir: str,
        model_name: str,
        model_type: str,
        target: str,
        interpol: bool,
        device: torch.device,
        verbose: bool
    ) -> None:
        """init

        Args:
            path (str): Project path
            data_dir (str): Data directory name
            model_name (str): Classifier name
            model_type (str): Machine or Deep depending on classifier
            target (str): Target gene name
            interpol (bool): Missing values interpolated boolean
            device (torch.device): Device on which to train
            verbose (bool): Detailed log 
        """
        
        self.path = check_path(os.path.join(path, data_dir))
        self.model = model_name
        self.type = model_type
        self.target = target
        self.interpol = interpol
        self.device = device
        self.verbose = verbose

    def create_clf(
        self
    ) -> None:
        """Load model classifier"""

        if self.model == "RandomForest":
            self.clf = RandomForestClassifier(class_weight="balanced")
        else:
            self.clf = CustomCNN().to(self.device)

    def load_test(
        self
    ) -> None:
        """Load test datasets"""

        if self.verbose:
                print(f"Loading test data...")

        if self.interpol:
            suffix = "_interpolate"
        else:
            suffix = ""

        #Loading the test datasets
        if self.type == "machine":
            X_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/X_test{suffix}.txt"))
            y_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/y_test{suffix}.txt"))
            self.X_test = pd.read_csv(X_test_path, index_col=0)
            with open(y_test_path) as f:
                self.y_test = pd.Series([line.rstrip() for line in f], index=self.X_test.index)
        else:
            X_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/X_test{suffix}.npy"))
            y_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/y_test{suffix}.npy"))

    def load_train_val(
        self,
        fold: int
    ) -> None:
        """Load training and validation datasets

        Args:
            fold (int): K-fold number
        """

        if self.verbose:
                print(f"Loading data for fold {fold}...")

        if self.interpol:
            suffix = "_interpolate"
        else:
            suffix = ""

        if self.type == "machine":

            #Loading the training datasets
            X_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/X_train{suffix}_{str(fold)}.txt"))
            y_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/y_train{suffix}_{str(fold)}.txt"))
            X_train = pd.read_csv(X_train_path, index_col=0)
            with open(y_train_path) as f:
                y_train = pd.Series([line.rstrip() for line in f], index=X_train.index)

            #Loading the validation datasets
            X_val_path = check_file(os.path.join(self.path, f"val/{self.type}/{self.target}/X_val{suffix}_{str(fold)}.txt"))
            y_val_path = check_file(os.path.join(self.path, f"val/{self.type}/{self.target}/y_val{suffix}_{str(fold)}.txt"))
            X_val = pd.read_csv(X_val_path, index_col=0)
            with open(y_val_path) as f:
                y_val = pd.Series([line.rstrip() for line in f], index=X_val.index)

            #Combining the train and val datasets
            self.X = pd.concat([X_train, X_val])
            self.y = pd.concat([y_train, y_val])

            #Indicating the validation records
            self.predefined = [-1] * len(X_train) + [0] * len(X_val)

    def predict(
        self,
        fold: int,
    ) -> Tuple[pd.Series, pd.Series]:
        """Make predictions with the trained model
        
        Args:
            fold (int): K-fold number
        """

        if self.interpol:
            suffix = "_interpolate"
        else:
            suffix = ""

        if self.type == "machine":

            #Making predictions
            y_pred = self.trained_model.predict(self.X_test)

            #Reporting and returning results
            with open(check_path(os.path.join(self.path, f"output/{self.model.lower()}{suffix}/{self.target}/{self.model.lower()}_scores.txt")), "a") as out_f:
                print("ROC-AUC score for fold {}: {:.6f}".format(fold, roc_auc_score(self.y_test, y_pred, average="macro")))
                print("ROC-AUC score for fold {}: {:.6f}".format(fold, roc_auc_score(self.y_test, y_pred, average="macro")), file=out_f)
                return (self.y_test, pd.Series(y_pred, index=self.y_test.index))

    def train(
        self,
        fold: int,
    ) -> None:
        """Train and save best model

        Args:
            fold (int): Fold number
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
            print("Training the model and optimising hyperparameters...")
            opt = BayesSearchCV(self.clf, param_grid, scoring="roc_auc", cv=ps, n_jobs=-1, random_state=0)
            opt.fit(self.X, self.y)

            if self.verbose:
                print(f"Optimal hyperparameters for fold {str(fold)}:")
                for key, val in opt.best_params_.items():
                    print("\t{}: {}".format(key, str(val)))
                print("")

            #Saving the classifier with best hyperparameters
            self.trained_model = opt.best_estimator_