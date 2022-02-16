#!/usr/bin/env python

import argparse
import numpy as np
import os
import pandas as pd
import torch

from bxc_model import BXCModel
from utils.file_helper import check_file, check_path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')
    parser.add_argument('--data', type=str, help='Path to the data directory')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')

    #Options for the optimizer
    parser.add_argument('--decay', type=float, default=0.0, nargs='?', help='ADAM weight decay (default: 0.0)')
    parser.add_argument('--lr', type=float, help='ADAM gradient descent optimizer learning rate')

    #Options for training
    parser.add_argument('--learn', type=str, help='Prepare data for type of algorith: machine learning (numbers); deep learning (images)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch', type=int, help='Minibatch size')
    parser.add_argument('--metric', type=str, default='accuracy', help='Evaluation metric of the model (default: accuracy)')

    #Printing arguments to the command line
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    return args


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Setting up the model
    model = BXCModel(path, args.data, args.learn, device)
    model.create_clf()

    #Looping over the K folds
    for k in range(args.kfold):

        #Getting the mean and standard deviation of our dataset
        if args.learn == "machine":
            X_train_path = check_file(os.path.join(path, args.data, "train/X_train_{}_{}.txt".format(args.learn, str(k))))
            y_train_path = check_file( os.path.join(path, args.data, "train/y_train_{}_{}.txt".format(args.learn, str(k))))

            X_val_path = check_file(os.path.join(path, args.data, "val/X_val_{}_{}.txt".format(args.learn, str(k))))
            y_val_path = check_file(os.path.join(path, args.data, "val/y_val_{}_{}.txt".format(args.learn, str(k))))

            f_name = check_path(os.path.join(path, args.data, "output/randomforest_lr{}_{}/randomforest_lr{}_{}_fold{}.pkl".format(
                args.lr, args.metric, args.lr, args.metric, str(k)
            )))
            model.get_data(X_train_path, y_train_path, X_val_path, y_val_path)
            model.train(f_name)


if __name__ == '__main__':
    main()
