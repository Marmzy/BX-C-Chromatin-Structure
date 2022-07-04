#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
from src.evaluating.predict import predict
from src.training.cnn import CustomCNN1, CustomCNN2
from src.training.dnn import CustomDNN
from src.training.image_data import BXCDataset, get_image_mean, get_weights
from src.training.model_training import train_model
from src.utils.file_helper import check_file, check_path
from src.utils.general import get_config_val
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple


class BXCModel():
    def __init__(
        self,
        path: str,
        conf_dict: Dict[str, Any],
        device: torch.device,
    ) -> None:
        """init
        Args:
            path (str): Project path
            conf_dict (Dict[str, Any]): Yaml file contents
            device (torch.device): Device on which to train
        """
        
        self.yaml = conf_dict
        self.device = device

        data_dir = get_config_val(self.yaml, ["data", "dirname"])
        self.path = check_path(os.path.join(path, data_dir))

        self.model = get_config_val(self.yaml, ["model", "name"])
        self.type = get_config_val(self.yaml, ["model", "type"])
        self.target = get_config_val(self.yaml, ["model", "target"])
        self.interpol = get_config_val(self.yaml, ["data", "interpolate"])
        self.rs = get_config_val(self.yaml, ["random_state"])
        self.verbose = get_config_val(self.yaml, ["verbose"])

    def create_clf(
        self
    ) -> None:
        """Load model classifier"""

        if self.model == "RandomForest":
            self.clf = RandomForestClassifier(class_weight="balanced")
        elif self.model == "CustomCNN1":
            self.clf = CustomCNN1().to(self.device)
        elif self.model == "CustomCNN2":
            self.clf = CustomCNN2().to(self.device)
        elif "CustomDNN" in self.model:
            self.clf = CustomDNN(get_config_val(self.yaml, ["model", "layers"])).to(self.device)

    def load_test(
        self,
        fold: int
    ) -> None:
        """Load test datasets
        
        Args:
            fold (int): K-fold number
        """

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
            #Specifying the training datasets
            X_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/X_train{suffix}_{str(fold)}.txt"))
            y_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/y_train{suffix}_{str(fold)}.txt"))

            #Specifying the training datasets
            X_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/X_test{suffix}.txt"))
            y_test_path = check_file(os.path.join(self.path, f"test/{self.type}/{self.target}/y_test{suffix}.txt"))

            #Getting the mean and standard deviation of our dataset
            batch_size = get_config_val(self.yaml, ["model", "params", "batch"])
            img_mean, img_std = get_image_mean(X_train_path, y_train_path, batch_size)
            
            #Defining image transformation techniques to apply
            image_transform = {
                "test": transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((img_mean), (img_std)),
                ])
            }

            #Creating the test dataset
            bxc_test = BXCDataset(X_test_path, y_test_path, image_transform["test"])
            self.data_loader_test = {"test": DataLoader(bxc_test, batch_size=batch_size, shuffle=True, pin_memory=True)}

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

        #Specifying the training datasets
        X_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/X_train{suffix}_{str(fold)}.txt"))
        y_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/y_train{suffix}_{str(fold)}.txt"))

        #Specifying the validation datasets
        X_val_path = check_file(os.path.join(self.path, f"val/{self.type}/{self.target}/X_val{suffix}_{str(fold)}.txt"))
        y_val_path = check_file(os.path.join(self.path, f"val/{self.type}/{self.target}/y_val{suffix}_{str(fold)}.txt"))

        if self.type == "machine":

            #Loading the training datasets
            X_train = pd.read_csv(X_train_path, index_col=0)
            with open(y_train_path) as f:
                y_train = pd.Series([line.rstrip() for line in f], index=X_train.index)

            #Loading the validation datasets
            X_val = pd.read_csv(X_val_path, index_col=0)
            with open(y_val_path) as f:
                y_val = pd.Series([line.rstrip() for line in f], index=X_val.index)

            #Combining the train and val datasets
            self.X_train_val = pd.concat([X_train, X_val])
            self.y_train_val = pd.concat([y_train, y_val])

            #Indicating the validation records
            self.predefined = [-1] * len(X_train) + [0] * len(X_val)

        else:

            #Getting the mean and standard deviation of our dataset
            batch_size = get_config_val(self.yaml, ["model", "params", "batch"])
            img_mean, img_std = get_image_mean(X_train_path, y_train_path, batch_size)
            
            #Defining image transformation techniques to apply
            image_transform = {
                "train": transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                    transforms.Normalize((img_mean), (img_std)),
                ]),
                "val": transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((img_mean), (img_std)),
                ])
            }

            #Creating the training and validation datasets
            bxc_train = BXCDataset(X_train_path, y_train_path, image_transform["train"])
            bxc_val = BXCDataset(X_val_path, y_val_path, image_transform["val"])

            self.data_loader = {"train": DataLoader(bxc_train, batch_size=batch_size, shuffle=True, pin_memory=True),
                                "val": DataLoader(bxc_val, batch_size=batch_size, shuffle=True, pin_memory=True)}

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

        else:

            #Initialising variables
            batch_size = get_config_val(self.yaml, ["model", "params", "batch"])
            decay = get_config_val(self.yaml, ["model", "params", "decay"])
            epochs = get_config_val(self.yaml, ["model", "params", "epochs"])
            lr = get_config_val(self.yaml, ["model", "params","lr"])
            suffix = ""
            if self.interpol:
                suffix = "_interpolate"

            #Settings for training the model
            fout = self.model + suffix + f"_lr{lr}_decay{decay}_epochs{epochs}_batch{batch_size}_scores.txt"
            fout = open(check_path(os.path.join(self.path, f"output/{self.model.lower()}{suffix}/{self.target}/{fout}")), "a")

            #Making predictions with the trained model on the test dataset
            y_true, y_pred = predict(self.trained_model, self.data_loader_test, fout, fold, self.device, self.rs, self.verbose)
            fout.close()
            return y_true, y_pred

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
            opt = BayesSearchCV(self.clf, param_grid, scoring="roc_auc", cv=ps, n_jobs=-1, random_state=self.rs)
            opt.fit(self.X_train_val, self.y_train_val)

            if self.verbose:
                print(f"Optimal hyperparameters for fold {str(fold)}:")
                for key, val in opt.best_params_.items():
                    print("\t{}: {}".format(key, str(val)))
                print("")

            #Saving the classifier with best hyperparameters
            self.trained_model = opt.best_estimator_

        else:

            #Initialising variables
            batch_size = get_config_val(self.yaml, ["model", "params", "batch"])
            decay = get_config_val(self.yaml, ["model", "params", "decay"])
            early_stop = get_config_val(self.yaml, ["model", "early_stop"])
            epochs = get_config_val(self.yaml, ["model", "params", "epochs"])
            lr = get_config_val(self.yaml, ["model", "params","lr"])

            if self.interpol:
                suffix = "_interpolate"
            else:
                suffix = ""
            y_train_path = check_file(os.path.join(self.path, f"train/{self.type}/{self.target}/y_train{suffix}_{str(fold)}.txt"))

            #Checking class imbalance
            weights = get_weights(y_train_path, self.verbose)

            #Settings for training the model
            logfile = self.model + suffix + f"_lr{lr}_decay{decay}_epochs{epochs}_batch{batch_size}_f{fold}"
            logfile = check_path(os.path.join(self.path, f"output/{self.model.lower()}{suffix}/{self.target}/{logfile}"))
            fout = open(logfile + ".log", "w")
            loss = nn.BCEWithLogitsLoss(weight=weights)
            optimizer = optim.AdamW(self.clf.parameters(), lr=lr, weight_decay=decay)

            #Training the model
            self.trained_model, self.epoch_name = train_model(self.clf, loss, optimizer, epochs, early_stop, self.data_loader, fold, fout, self.device, self.rs, self.verbose)
            fout.close()