#!/usr/bin/env python

import _io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from typing import Any, Dict, Tuple


def predict(
    model: Any,
    data_loader: Dict[str, torch.utils.data.dataloader.DataLoader],
    fout: _io.TextIOWrapper,
    fold: int,
    device: torch.device,
    rs: int,
    verbose: bool
) -> Tuple[pd.Series, pd.Series]:
    """Make predictions on the test dataset with a trained model

    Args:
        model (Any): Trained model
        data_loader (Dict[str, torch.utils.data.dataloader.DataLoader]): Test dataset DataLoader
        fout (_io.TextIOWrapper): Opened output file
        fold (int): Cross-validation fold
        device (torch.device): Device to run calculations on
        rs (int): Random state number
        verbose (bool): Detailed output boolean

    Returns:
        Tuple[pd.Series, pd.Series]: Tuple of ground truths and predictions
    """

    #Initialising variables
    sigmoid = nn.Sigmoid()
    truths, preds = [], []
    out_dict = {"Truth": [], "Conf": [], "Name": []}
    
    torch.manual_seed(rs)

    if verbose:
        print(f"\nEvaluating the model trained on training dataset fold {fold}...")
    
    #Setting the data and model
    data = data_loader["test"]
    model.eval()

    #Looping over the minibatches
    for data_test, target_test, name in data:
        x, y = data_test.to(device), target_test.to(device)

        with torch.set_grad_enabled(False):
            logits = model(x)
            y_pred = sigmoid(logits)

            #Keeping track of predictions and truth labels
            preds.append(y_pred.cpu().detach().numpy())
            truths.append(y.unsqueeze(1).cpu().detach().numpy())

            #Add the ground truths and predictions to the output dictionary
            for i in range(len(y_pred)):
                if int(y.unsqueeze(1)[i].item()):
                    conf = y_pred[i].item()
                else:
                    conf = 1 - y_pred[i].item()
                
                out_dict["Truth"].append(int(y.unsqueeze(1)[i].item()))
                out_dict["Conf"].append(conf)
                out_dict["Name"].append(name[i])

    #Scoring predictions through various metrics
    truths, preds = np.vstack(truths), np.vstack(preds)
    auc = roc_auc_score(truths, preds, average="weighted")
    print("ROC-AUC score for fold {}: {:.6f}".format(fold, auc))
    print("ROC-AUC score for fold {}: {:.6f}".format(fold, auc), file=fout)

    return pd.Series(out_dict["Truth"], index=out_dict["Name"]), pd.Series(out_dict["Conf"], index=out_dict["Name"])