#!/usr/bin/env python

import joblib
import numpy as np
import pandas as pd
import os

from cnn import CustomCNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV


class BXCModel():
    def __init__(self, path, data_dir, learn, device):
        self.path = os.path.join(path, data_dir)
        self.learn = learn
        self.device = device

    def create_clf(self):
        if self.learn == "machine":
            self.clf = RandomForestClassifier()
        else:
            self.clf = CustomCNN().to(self.device)

    def get_data(self, x_train_path, y_train_path, x_val_path, y_val_path):
        if self.learn == "machine":

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

    def train(self, f_name):
        if self.learn == "machine":

            #Initialising variables
            param_grid = {
                "n_estimators": [50, 100, 250, 500],
                "max_depth": [None, 5, 10, 25, 50],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 5, 10, 15],
            }
            ps = PredefinedSplit(self.predefined)

            #Tuning hyperparameters on the validation dataset
            print("Optimizing hyperparameters...")
            opt = BayesSearchCV(self.clf, param_grid, scoring=make_scorer(roc_auc_score, average="macro", multi_class="ovo"), cv=ps, n_jobs=-1, random_state=0)
            opt.fit(self.X, self.y)

            #Saving the classifier with best hyperparameters
            joblib.dump(opt.best_estimator_, os.path.join(self.path, f_name))



