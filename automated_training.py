# -*- coding: utf-8 -*-
"""Launch hyperoptimization and chain training stages."""

from GeoFlow.RCNN2D import Hyperparameters
from GeoFlow import Dataset2Dtest
from GeoFlow.AutomatedTraining.AutomatedTraining import optimize


optimize(
    params=Hyperparameters(),
    case=Dataset2Dtest(),
    epochs=(100, 100, 50),
    steps=20,
    lr=.0002,
    beta_1=.9,
    beta_2=.98,
    eps=1e-5,
    batchsize=4,
    loss_ref=(.5, .0, .0),
    loss_vrms=(.5, .7, .0),
    loss_vint=(.0, .3, 1.),
    loss_vdepth=(.0, .0, .0),
    nmodel=1,
    ngpu=2,
    noise=0,
    plot=0,
    no_weights=False,
    restore_from=None,
)
