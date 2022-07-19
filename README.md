# BX-C_Chromatin_Structure

This project came about when I found an article by [Rajpurkar et al](https://www.nature.com/articles/s41467-021-23831-4#Sec2)<sup>1</sup>.
In the first part of this article the researchers train a model to predict RNA expression for 3 genes of the Bithorax complex (BX-C) in *D. melanogaster*, using
only 3D chromatin information. The data used seemed really challening and interesting to work with, and because I wondered why the researchers didn't use any
pretrained CNN models for transfer learning, I chose to reproduce the first part of the research. I tried to stick as close as possible to the analysis, but
for various reasons I made some slight modifications, which I will discuss in the sections below.

## Overview

The analysis can be run by providing [`pipeline.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/pipeline.py) a command and an input
[.yaml file](https://github.com/Marmzy/BX-C_Chromatin_Structure/tree/master/config). As data preprocessing wasn't discussed in depth in the article, I tried
to infer it through the images in the article and by looking at the source code, though it was difficult to understand at times. Combined with my guess that the
input data has changed, this means that I probably didn't replicate data preprocessing to a T.

Explanation of the .yaml file parameters:

```
data:
  cutoff:       Cut-off value for min amount of records to be present in 3D chromatin data
  dirname:      Name of directory that will contain all data
  download:     Boolean to download the 3D chromatin data
  interpolate:  Boolean to interpolate missing values in the 3D chromatin data
  visualise:    Boolean to create images from the 3D chromatin data
model:
  early_stop:   None or number after which if no improvement is seen, the model will stop prematurely
  name:         Name of the machine/deep learning model/architecture
  params:
    batch:      Deep learning only: minibatch size
    lr:         Deep learning only: learning rate
    decay:      Deep learning only: decay rate
    epochs:     Deep learning only: number of epochs
  target:       Target BX-C gene (Abd-A, Abd-B or Ubx)
  type:         Type learning algorithm (machine or deep)
pipeline:
  cv:           Number of cross-validation folds to split the data into
  split:
    test:       Amount of data the test data should contain
random_state:   Random state number
verbose:        Verbose output boolean
```

## Preprocessing

Providing the command "prep-data" to [`pipeline.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/pipeline.py) will preprocess the data through
[`data_prep.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/src/data_prep.py). The script `data_prep.py` will:

1. Download the 3D chromatin data
2. Create images from the coordinates
3. Split the data into train, val and test

**Changes from Rajpurkar et al.**:

- Increased the cutoff value from 0.5 to 0.75. Only data records that contain no more than 25% missing values are used in the analysis. Less data is interpolated
this way and analysis speed is shortened due to fewer data being used.
- Number of cross-validation folds is reduced from 10 to 5, to speed up the analysis. The goal of this project is to reproduce the analysis and not to try and
train a better model.
- Both interpolated and non-interpolated datasets were created. If I'm not mistaken, Rajpurkar et al. only used interpolated data


| Interpolated | Non-interpolated |
| --- | --- |
| ![interpolated](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/data/raw/images/interpolate/A_2.png) | ![missing](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/data/raw/images/missing/A_2.png) |

Example usage:
```bash
python3 pipeline.py prep-data config/customcnn1.yaml
```

## Training

Providing the command "train-model" to [`pipeline.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/pipeline.py) will train a model through
[`data_train.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/src/data_train.py). The script `data_train.py` will:

- Train a model on the training dataset for each cross-validation fold
- Evaluate the model at each epoch on the train and val datasets
- Save the model with the highest ROC-AUC score on the val dataset
- Stop the model early if no improvement is seen (only if early stopping is selected)

**Changes from Rajpurkar et al.**:

- Transfer learning ResNet50 and VGG16 models. Transfer learning was not implemented in the paper
- No optimization of hyperparameters. For each model, only models with the best hyperparameters (as seen in the Supplementary information) were trained

Example usage:
```bash
python3 pipeline.py train-model config/customdnn3.yaml
```

## Evaluation

Providing the command "train-model" to [`pipeline.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/pipeline.py) will train a model through
[`data_eval.py`](https://github.com/Marmzy/BX-C_Chromatin_Structure/blob/master/src/data_eval.py). The script `data_eval.py` will:

- Evaluate the best trained model for each cross-validation fold on the test dataset of that fold

**Changes from Rajpurkar et al.**:

- Calculation of ROC-AUC only. No other metrics are calculated, neither are any images output

Example usage:
```bash
python3 pipeline.py eval-model config/resnet50.yaml
```
