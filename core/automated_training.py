# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from GeoFlow.AutomatedTraining.AutomatedTraining import optimize

from core.__main__ import parse_args
from core.architecture import RCNN2D


args = parse_args()

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(
    nn=RCNN2D,
    params=args.params,
    dataset=args.dataset,
    gpus=args.gpus,
    debug=args.debug,
    eager=args.eager,
    **args.params,
)
