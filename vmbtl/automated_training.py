# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from GeoFlow.AutomatedTraining.AutomatedTraining import optimize

from vmbtl.__main__ import parser
from vmbtl import architecture, datasets


args, config = parser.parse_known_args()
config = {
    name[2:]: eval(value) for name, value in zip(config[::2], config[1::2])
}
args.nn = getattr(architecture, args.nn)
args.params = getattr(architecture, args.params)
args.params = args.params(is_training=True)
args.dataset = getattr(datasets, args.dataset)(args.noise)

if args.debug:
    args.params["epochs"] = 1
    args.params["steps_per_epoch"] = 5

optimize(args, **config)
