#!/usr/bin/env python

import typer

from src.utils.file_helper import read_yaml
from src.data_prep import main_prep
from src.data_train import main_train
from src.data_eval import main_eval


app = typer.Typer(add_completion=False)

@app.command()
def prep_data(
    config: str
) -> None:
    """Prepare data for training

    Args:
        config (str): Full path to configuration (yaml) file
    """

    conf_dict = read_yaml(config)
    main_prep(conf_dict)


@app.command()
def train_model(
    config: str
) -> None:
    """Train model on training dataset and tune hyperparameters
        on validation dataset

    Args:
        config (str): Full path to configuration (yaml) file
    """

    conf_dict = read_yaml(config)
    main_train(conf_dict)


@app.command()
def eval_model(
    config: str
) -> None:
    """Evaluate model on test dataset

    Args:
        config (str): Full path to configuration (yaml) file
    """

    conf_dict = read_yaml(config)
    main_eval(conf_dict)


if __name__ == "__main__":
    app()