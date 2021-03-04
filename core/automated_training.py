# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from argparse import ArgumentParser

from AutomatedTraining.AutomatedTraining import optimize, int_or_list

from core import datasets, architecture
from core.architecture import RCNN2D


parser = ArgumentParser()
parser.add_argument(
    "--params",
    type=str,
    default="Hyperparameters",
    help="Name of hyperparameters from `RCNN2D` to use.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="Dataset1Dsmall",
    help="Name of dataset from `DefinedDataset` to use.",
)
parser.add_argument(
    "--gpus",
    type=int_or_list,
    default=None,
    help=(
        "Either the quantity of GPUs or a list of GPU IDs to use in data "
        "creation, training and inference. Use a string representation "
        "for lists of GPU IDs, e.g. `'[0, 1]'` or `[0,1]`. By default, "
        "use all available GPUs."
    ),
)
parser.add_argument(
    "--debug",
    action='store_true',
    help="Generate a small dataset of 5 examples.",
)
parser.add_argument(
    "--eager",
    action='store_true',
    help="Run the Keras model eagerly, for debugging.",
)
args, config = parser.parse_known_args()
config = {
    name[2:]: eval(value) for name, value in zip(config[::2], config[1::2])
}

args.dataset = getattr(datasets, args.dataset)()
args.params = getattr(architecture, args.params)()

if args.debug:
    config["epochs"] = 1
    config["steps_per_epoch"] = 5

optimize(
    nn=RCNN2D,
    params=args.params,
    dataset=args.dataset,
    gpus=args.gpus,
    debug=args.debug,
    eager=args.eager,
    **config,
)
