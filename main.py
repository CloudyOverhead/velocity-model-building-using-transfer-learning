# -*- coding: utf-8 -*-
"""Launch dataset generation, training or testing."""

from argparse import ArgumentParser
from os import pardir
from os.path import join, split
from importlib.util import find_spec, spec_from_file_location, module_from_spec


# Import `Geoflow..main.main` function.
geoflow_init_path = find_spec("GeoFlow").origin
geoflow_path, _ = split(geoflow_init_path)
main_path = join(geoflow_path, pardir, "main.py")
main_spec = spec_from_file_location("main", main_path)
main_module = module_from_spec(main_spec)
main_spec.loader.exec_module(main_module)
main = main_module.main


if __name__ == "__main__":
    from GeoFlow import RCNN2D

    from deep_learning_velocity_estimation import datasets

    # Initialize argument parser.
    parser = ArgumentParser()

    # Add arguments to parse for training.
    parser.add_argument(
        "--architecture",
        type=str,
        default="RCNN2D",
        help="Name of the architecture from `RCNN2D` to use.",
    )
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
        help=(
            "Name of dataset from `deep_learning_velocity_estimation."
            "datasets` to use."
        ),
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory in which to store the checkpoints.",
    )
    parser.add_argument(
        "--training",
        type=int,
        default=0,
        help=(
            "0: create dataset only; 1: training only; 2: training+dataset; "
            "3: testing."
        ),
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=2,
        help="Quantity of GPUs for data creation.",
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        help="Validate data by plotting.",
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

    args = parser.parse_args()
    args.architecture = getattr(RCNN2D, args.architecture)
    args.dataset = getattr(datasets, args.dataset)()
    args.params = getattr(RCNN2D, args.params)()
    main(args)
