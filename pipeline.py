#!/usr/bin/env python

import typer

from src.utils.file_helper import read_yaml
from src.data_prep import main_prep


app = typer.Typer(add_completion=False)

@app.command()
def prep_data(
    config: str
) -> None:

    conf_dict = read_yaml(config)
    main_prep(conf_dict)


@app.command()
def test():
    return "This is a placeholder"


if __name__ == "__main__":
    app()