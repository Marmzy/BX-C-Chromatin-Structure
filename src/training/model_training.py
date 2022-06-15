#!/usr/bin/env python

import _io
import copy
import numpy as np
import os
import time
import torch
import torch.nn as nn

from collections import deque
from sklearn.metrics import roc_auc_score
from typing import Any, Dict, Tuple, Union


def train_model(
    model: Any,
    loss: nn.modules.loss,
    optimizer: torch.optim,
    epochs: int,
    early_stop: Union[int, bool],
    data_loader: Dict[str, torch.utils.data.dataloader.DataLoader],
    k: int,
    f: _io.TextIOWrapper,
    device: torch.device,
    rs: int,
    verbose: bool
) -> Tuple[Any, deque]:
    """Train model on training dataset and save model state

    Args:
        model (Any): _description_
        loss (nn.modules.loss): Loss function
        optimizer (torch.optim): Optimizer function
        epochs (int): Number of epochs to train model
        early_stop (Union[int, bool]): Apply early stopping boolean
        data_loader (Dict[str, torch.utils.data.dataloader.DataLoader]): Train and val datasets DataLoader
        k (int): Fold number
        f (_io.TextIOWrapper): Opened output file
        device (torch.device): Device to train model on
        rs (int): Random state number
        verbose (bool): Detailed output boolean

    Returns:
        Tuple[Any, collections.deque]: Trained model and best model name tuple
    """

    #Initialising variables
    pkl_queue = deque()
    best_metric = -1.0
    best_loss = 100.0
    best_model_weights = model.state_dict()
    sigmoid = nn.Sigmoid()
    since = time.time()
    end = time.time()
    no_improve = 0

    torch.manual_seed(rs)

    if not early_stop:
        early_stop = epochs

    if verbose:
        print(f"\n{model}\n")
        print(f"\n{model}\n", file=f)

    print("Analysing cross-validation fold {}...".format(k))
    print("Analysing cross-validation fold {}...".format(k), file=f)

    #Looping over the epochs
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch+1, epochs), end="")
        print("Epoch:{}/{}".format(epoch+1, epochs), end="", file=f)
        improvement = False

        #Making sure training and validating occurs seperately
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            #Setting the data
            data = data_loader[phase]

            #Initialising more variables
            running_loss = 0
            truths, preds = [], []

            #Looping over the minibatches
            for data_train, target_train, _ in data:
                optimizer.zero_grad()
                x, y = data_train.to(device), target_train.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    logits = model(x)
                    y_pred = sigmoid(logits)
                    l = loss(logits, y.unsqueeze(1))

                    #Keeping track of predictions and truth labels
                    preds.append(y_pred.cpu().detach().numpy())
                    truths.append(y.unsqueeze(1).cpu().detach().numpy())

                    if phase == "train":
                        l.backward()
                        optimizer.step()

                #Calculating statistics
                running_loss += l.item()

            #Scoring predictions through various metrics
            truths, preds = np.vstack(truths), np.vstack(preds)
            auc = roc_auc_score(truths, preds, average="weighted")
            epoch_loss = running_loss / len(x)

            print("\t{} Loss: {:.4f} AUC: {:.4f} Time: {:.4f}".format(phase, epoch_loss, auc, time.time()-end), end="")
            print("\t{} Loss: {:.4f} AUC: {:.4f} Time: {:.4f}".format(phase, epoch_loss, auc, time.time()-end), end="", file=f)

            #Validation phase only
            if phase == "val":
                print("\n", end="")
                print("\n", end="", file=f)
                model_score = auc
                m = "AUC"

                #Saving the model with the highest AUC for the validation data
                if (model_score > best_metric) or (model_score == best_metric and best_loss > epoch_loss):
                    improvement = True
                    no_improve = 0
                    best_metric = model_score
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    pkl_queue.append("{}_epoch{}.pkl".format(f.name.split(".")[0], epoch+1))
                    if len(pkl_queue) > 1:
                        pkl_queue.popleft()
                else:
                    improvement = False
                
            end = time.time()

        #Stop the model early, if no improvement is seen
        if not improvement:
            no_improve += 1
            if no_improve >= early_stop:

                #Print overall training information
                time_elapsed = time.time() - since
                print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), file=f)
                print("Best val {}: {:.4f}".format(m, best_metric))
                print("Best val {}: {:.4f}".format(m, best_metric), file=f)

                #Loading the model with the best weights
                model.load_state_dict(best_model_weights)
                return model, pkl_queue.popleft()

    #Print overall training information
    time_elapsed = time.time() - since
    print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), file=f)
    print("Best val {}: {:.4f}".format(m, best_metric))
    print("Best val {}: {:.4f}".format(m, best_metric), file=f)
    
    #Loading the model with the best weights
    model.load_state_dict(best_model_weights)
    return model, pkl_queue