#!/usr/bin/env python

import copy
import numpy as np
import os
import time
import torch
import torch.nn as nn

from collections import deque
from sklearn.metrics import roc_auc_score


def train_model(model, loss, optimizer, epochs, early_stop, data_loader, k, f, device, verbose):

    #Initialising variables
    pkl_queue = deque()
    best_metric = -1.0
    best_loss = 100.0
    best_model_weights = model.state_dict()
    sigmoid = nn.Sigmoid()
    since = time.time()
    end = time.time()
    no_improve = 0

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
            for idx, (data_train, target_train, name) in enumerate(data):
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
                    best_metric = model_score
                    best_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    # torch.save(model.state_dict(), "{}_epoch{}.pkl".format(fout, epoch+1))
                    pkl_queue.append("{}_epoch{}.pkl".format(f.name, epoch+1))
                    if len(pkl_queue) > 1:
                        pkl_queue.popleft()
                        # pkl_file = pkl_queue.popleft()
                        # os.remove(pkl_file)
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
                break

                # #Loading the model with the best weights
                # model.load_state(best_model_weights)
                # return model, pkl_queue

    #Print overall training information
    time_elapsed = time.time() - since
    print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), file=f)
    print("Best val {}: {:.4f}".format(m, best_metric))
    print("Best val {}: {:.4f}".format(m, best_metric), file=f)
    
    