# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from argparse import ArgumentParser

from GeoFlow.AutomatedTraining.AutomatedTraining import optimize

from deep_learning_velocity_estimation import datasets, architecture
from deep_learning_velocity_estimation.architecture import RCNN2D


parser = ArgumentParser()
parser.add_argument("--params")
parser.add_argument("--dataset")
args = parser.parse_args()

args.params = getattr(architecture, args.params)()
args.dataset = getattr(datasets, args.dataset)()
optimize(
    architecture=RCNN2D,
    params=args.params,
    dataset=args.dataset,
    ngpu=2,
)
