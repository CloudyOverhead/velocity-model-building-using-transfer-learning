# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from GeoFlow.AutomatedTraining.AutomatedTraining import optimize

from core.__main__ import parse_args


args = parse_args()

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(args)
