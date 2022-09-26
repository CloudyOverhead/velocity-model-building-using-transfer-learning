# -*- coding: utf-8 -*-

from argparse import Namespace
from os import makedirs, listdir
from os.path import join, exists, split
from copy import deepcopy
from datetime import datetime

import numpy as np

from vmbtl.__main__ import main as global_main
from vmbtl.architecture import Hyperparameters1D, Hyperparameters2D

from vmbtl.postprocess.constants import IGNORE_NNS


def launch_both_inferences(args, nn, dataset, batch_size=None):
    params_1d = Hyperparameters1D(is_training=False)
    params_2d = Hyperparameters2D(is_training=False)
    for logdir, savedir, params in zip(
        [args.logdir_1d, args.logdir_2d],
        ["Pretraining", "EndResults"],
        [params_1d, params_2d],
    ):
        launch_inference(
            nn=nn,
            params=params,
            dataset=dataset,
            logdir=logdir,
            gpus=args.gpus,
            savedir=savedir,
        )


def launch_inference(
    nn, params, dataset, logdir, gpus, savedir, batch_size=None,
):
    print("Launching inference.")
    print("NN:", nn.__name__)
    print("Hyperparameters:", type(params).__name__)
    print("Weights:", logdir)
    print("Case:", savedir)

    if batch_size is not None:
        if isinstance(gpus, int) and gpus > batch_size:
            gpus = batch_size
        elif isinstance(gpus, list) and len(gpus) > batch_size:
            gpus = gpus[:batch_size]
        params = deepcopy(params)
        params.batch_size = batch_size

    logdirs = sorted(listdir(logdir))
    for i, current_logdir in enumerate(logdirs):
        if int(current_logdir) in IGNORE_NNS:
            continue
        print(f"Using NN {i+1} out of {len(logdirs)}.")
        print(f"Started at {datetime.now()}.")
        current_logdir = join(logdir, current_logdir)
        current_savedir = f"{savedir}_{i}"
        current_args = Namespace(
            nn=nn,
            params=params,
            dataset=dataset,
            logdir=current_logdir,
            generate=False,
            train=False,
            test=True,
            gpus=gpus,
            savedir=current_savedir,
            plot=False,
            debug=False,
            eager=False,
        )
        global_main(current_args)
    print(f"Finished at {datetime.now()}.")

    combine_predictions(dataset, logdir, savedir)


def combine_predictions(dataset, logdir, savedir):
    print("Averaging predictions.")
    logdirs = sorted(listdir(logdir))
    dataset._getfilelist()
    for filename in dataset.files["test"]:
        preds = {key: [] for key in dataset.generator.outputs}
        for i, current_logdir in enumerate(logdirs):
            if int(current_logdir) in IGNORE_NNS:
                continue
            current_load_dir = f"{savedir}_{i}"
            current_preds = dataset.generator.read_predictions(
                filename, current_load_dir,
            )
            for key, value in current_preds.items():
                preds[key].append(value)
        average = {
            key: np.mean(value, axis=0) for key, value in preds.items()
        }
        std = {
            key: np.std(value, axis=0) for key, value in preds.items()
        }
        directory, filename = split(filename)
        filedir = join(directory, savedir)
        if not exists(filedir):
            makedirs(filedir)
        dataset.generator.write_predictions(
            None, filedir, average, filename=filename,
        )
        if not exists(f"{filedir}_std"):
            makedirs(f"{filedir}_std")
        dataset.generator.write_predictions(
            None, f"{filedir}_std", std, filename=filename,
        )
