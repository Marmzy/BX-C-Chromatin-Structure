import argparse
import glob
import itertools
import numpy as np
import os
import pandas as pd

from collections import Counter
from itertools import combinations
from scipy import interpolate
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('-o', '--output', type=str, help='Name of the output data directory')
    parser.add_argument('-v', '--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')

    #Options for preprocessing
    parser.add_argument('--test', type=float, help='Ratio of samples that is included in the test dataset')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')
    parser.add_argument('--learn', type=str, help='Prepare data for type of algorith: machine learning (numbers); deep learning (images)')
    parser.add_argument('--interpol', type=str2bool, help='Interpolate missing values')
    parser.add_argument('--perc', type=float, help='Minimum percentage of barcodes a cell needs to have to select it for the analysis')

    #Printing arguments to the command line
    args = parser.parse_args()

    #Checking arguments
    assert args.learn in ['machine', 'deep'], 'Please choose a valid argument: "machine" | "deep"'

    print('Called with args:')
    print(args)

    return args


def interpolate_coords(coords):

    #Seperate the mising values from the actual coordinates
    nan_idx = np.argwhere(np.isnan(coords))
    coord_idx = np.argwhere(~np.isnan(coords))

    #Interpolating the missing vales
    f = interpolate.interp1d(coord_idx.squeeze(), coords[coord_idx].squeeze(), fill_value="extrapolate")

    #Replacing the missing values with the interpolated ones
    y_new = f(nan_idx)
    coords[nan_idx] = y_new

    return coords


def pairwise_distances(df, missing, interpolate):

    #Extracting the (x,y,z) coordinates from the cell dataframe and adding missing data
    xyz_data = [np.array([row["x"], row["y"], row["z"]]) for _, row in df.iterrows()]
    for num in missing:
        xyz_data.insert(num-1, np.array([np.nan, np.nan, np.nan]))

    #Interpolate missing values by linear interpolation between adjacent coordinates
    if interpolate:
        x_coords = interpolate_coords(np.array([arr[0] for arr in xyz_data]))
        y_coords = interpolate_coords(np.array([arr[1] for arr in xyz_data]))
        z_coords = interpolate_coords(np.array([arr[2] for arr in xyz_data]))

        xyz_data = [np.array([x, y, z]) for x, y, z in zip(x_coords, y_coords, z_coords)]

    #Calculating pairwise distances between the coordinates of the barcodes
    xyz_product = itertools.product(xyz_data, repeat=2)
    pairwise_xyz = [np.linalg.norm(arr1-arr2) for arr1, arr2 in xyz_product]
    pairwise_xyz = np.reshape(pairwise_xyz, (52, 52)).T

    return pairwise_xyz


def dna_overview(df, id, cut_off, learn, interpolate, verbose):

    #Initialising variables
    cell_dict = {}

    #Looping over the cells in the dataset
    for cn in df["cellNumber"].unique():
        mini_df = df[df["cellNumber"] == cn]
        missing = list(set(range(1,53)) - set(mini_df["barcode"].values))

        #Continue data prep only with cells that contain enough barcode data
        if len(mini_df) >= cut_off:

            #Calculate pairwise distances for all barcodes in the cell
            pairwise_xyz = pairwise_distances(mini_df, missing, interpolate)

            #Adding pairwise data to the dictionary
            if learn == "deep":
                cell_dict[id + "_" + str(cn)] = pairwise_xyz
            else:
                cell_dict[id + "_" + str(cn)] = pairwise_xyz[np.triu_indices(52, k=1)]

    return cell_dict


def rna_overview(df):

    #Initialising variables
    states = {}

    #Looping over the three genes
    for gene in ["Abd-A", "Abd-B", "Ubx"]:
        cols = ["cellNumber", "embNumber", "segNumber"]
        cols.extend([col for col in df.columns if gene in col])

        #Getting the number of molecules per cell
        mini_df = df[cols]
        molecules = mini_df.iloc[:, 3:].sum(axis=1)

        #Convert the number of molecules to expression states
        states[gene] = list(molecules.where(molecules < 1, 1))

    #Combining the data for all three genes
    exp_states = [",".join([str(a), str(b), str(u)]) for a, b, u in zip(states["Abd-A"], states["Abd-B"], states["Ubx"])]

    return np.array(exp_states)


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialsing variables
    path = os.path.dirname(os.getcwd())
    cut_off = 52 * args.perc

    if args.verbose:
        if interpolate:
            print("\nInterpolating missing coordinates...")
            print("Calculating pairwise distances for cells that contain at least {} barcodes...".format(str(cut_off)))
        else:
            print("\nCalculating pairwise distances for cells that contain at least {} barcodes...".format(str(cut_off)))

    #Reading the input DNA data as a dataframe
    dna1_df = pd.read_csv(os.path.join(path, args.output, "raw/dnaData_exp1.csv"))
    dna2_df = pd.read_csv(os.path.join(path, args.output, "raw/dnaData_exp2.csv"))

    #Getting pairwise distances
    pairwise_dict1 = dna_overview(dna1_df, "A", cut_off, args.learn, args.interpol, args.verbose)
    pairwise_dict2 = dna_overview(dna2_df, "B", cut_off, args.learn, args.interpol, args.verbose)

    pairwise = {}
    pairwise.update(pairwise_dict1)
    pairwise.update(pairwise_dict2)

    #Getting the cell IDs
    cells1 = [int(cell.split("_")[1]) for cell in pairwise_dict1.keys()]
    cells2 = [int(cell.split("_")[1]) for cell in pairwise_dict2.keys()]

    #Reading the input RNA data as a dataframe
    rna1_df = pd.read_csv(os.path.join(path, args.output, "raw/rnaData_exp1.csv"))
    rna2_df = pd.read_csv(os.path.join(path, args.output, "raw/rnaData_exp2.csv"))

    if args.verbose:
        print("\nCalculating single cell expression states of Abd-A, Abd-B and Ubx...")

    #Getting the expression states of the genes Abd-A, Abd-B and Ubx
    exp_states1 = rna_overview(rna1_df)
    exp_states2 = rna_overview(rna2_df)

    #Filtering the expression states to only get those of cells with sufficient barcodes
    exp_states_filtered1 = exp_states1[cells1]
    exp_states_filtered2 = exp_states2[cells2]
    exp_states_combined = np.concatenate([exp_states_filtered1, exp_states_filtered2])

    if args.verbose:
        print("Cell expression states:")
        for key in ["0,0,0", "0,0,1", "0,1,0", "1,0,0", "0,1,1", "1,0,1", "1,1,0", "1,1,1"]:
            print("{}: {}".format(key, Counter(exp_states_combined)[key]))
        print("Total: {}".format(str(exp_states_combined.shape[0])))

    #Defining the explanatory and response data
    if args.learn == "machine":
        X = pd.DataFrame.from_dict(pairwise, orient="index", columns=[str(c1) + "_" + str(c2) for c1, c2 in combinations(list(range(1, 53)), 2)])
    else:
        X = list(pairwise.keys())
    y = exp_states_combined

    #Encoding the target labels
    le = LabelEncoder()
    le.fit(y)

    #Splitting the data into (temporary) train and test
    if args.verbose:
        print("\nSplitting the dataset and into train ({}%) and test ({}%)...".format(str(int((1-float(args.test))*100)), str(int(float(args.test)*100))))
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=args.test, stratify=y)

    #Splitting the temporary training dataset into K training and validation datasets
    if args.verbose:
        print("\nSplitting the temporary training dataset into {} folds...".format(args.kfold))

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True)

    #Looping over the k folds
    for idx, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        if args.verbose:
            print("TRAIN:", train_index, "VAL:", val_index)

        X_train, X_val = np.array(X_temp)[train_index], np.array(X_temp)[val_index]
        y_train, y_val = np.array(y_temp)[train_index], np.array(y_temp)[val_index]

        #Saving the train and validation datasets
        if args.learn == "machine":
            train_idx = [X_temp.index.values.tolist()[i] for i in train_index]
            val_idx = [X_temp.index.values.tolist()[i] for i in val_index]

            X_train = pd.DataFrame(X_train, index=train_idx, columns=X_temp.columns.values.tolist())
            X_train.to_csv(os.path.join(path, "data", "train", "X_train_{}_{}.txt".format(args.learn, idx)))
            np.savetxt(os.path.join(path, "data", "train", "y_train_{}_{}.txt".format(args.learn, idx)), le.transform(y_train), fmt='%s')

            X_val = pd.DataFrame(X_val, index=val_idx, columns=X_temp.columns.values.tolist())
            X_val.to_csv(os.path.join(path, "data", "val", "X_val_{}_{}.txt".format(args.learn, idx)))
            np.savetxt(os.path.join(path, "data", "val", "y_val_{}_{}.txt".format(args.learn, idx)), le.transform(y_val), fmt='%s')
        else:
            X_train = {key: pairwise[key] for key in X_train}
            np.save(os.path.join(path, "data", "train", "X_train_{}_{}.npy".format(args.learn, idx)), X_train)
            np.savetxt(os.path.join(path, "data", "train", "y_train_{}_{}.npy".format(args.learn, idx)), y_train, fmt='%s')

            X_val = {key: pairwise[key] for key in X_val}
            np.save(os.path.join(path, "data", "val", "X_val_{}_{}.npy".format(args.learn, idx)), X_val)
            np.savetxt(os.path.join(path, "data", "val", "y_val_{}_{}.npy".format(args.learn, idx)), y_val, fmt='%s')

    #Saving the test dataset
    if args.learn == "machine":
        X_test.to_csv(os.path.join(path, "data", "test", "X_test_{}.txt".format(args.learn)))
        np.savetxt(os.path.join(path, "data", "test", "y_test_{}.txt".format(args.learn)), le.transform(y_test), fmt='%s')
    else:
        X_test = {key: pairwise[key] for key in X_test}
        np.save(os.path.join(path, "data", "test", "X_test_{}.npy".format(args.learn)), X_test)
        np.savetxt(os.path.join(path, "data", "test", "y_test_{}.npy".format(args.learn)), y_test, fmt='%s')

    if args.verbose:
        print("\nSaved the training datasets to: {}".format(os.path.join(path, "data", "train")))
        print("Saved the validation datasets to: {}".format(os.path.join(path, "data", "val")))
        print("Saved the test dataset to: {}".format(os.path.join(path, "data", "test")))


if __name__ == '__main__':
    main()
