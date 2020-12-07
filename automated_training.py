# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from argparse import ArgumentParser

from AutomatedTraining.AutomatedTraining import optimize

from deep_learning_velocity_estimation import datasets, architecture
from deep_learning_velocity_estimation.architecture import RCNN2D


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
config = {name[2:]: eval(value) for name, value
          in zip(config[::2], config[1::2])}

args.dataset = getattr(datasets, args.dataset)()
args.params = getattr(architecture, args.params)()

if args.debug:
    config["epochs"] = 1
    config["steps_per_epoch"] = 5

optimize(
    architecture=RCNN2D,
    params=args.params,
    dataset=args.dataset,
    ngpu=2,
    debug=args.debug,
    eager=args.eager,
    **config,
)
