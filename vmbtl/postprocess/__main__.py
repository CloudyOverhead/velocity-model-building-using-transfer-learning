# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from os import makedirs
from os.path import join, exists

import numpy as np
from matplotlib import pyplot as plt
from GeoFlow.__main__ import int_or_list

from vmbtl.architecture import (
    RCNN2D, RCNN2DUnpackReal, Hyperparameters1D, Hyperparameters2D,
    Hyperparameters2DNoTL, Hyperparameters2DSteep,
)
from vmbtl.datasets import Article2D, USGS, Article2DSteep, Marmousi

from .constants import FIGURES_DIR
from .launch_inference import launch_inference, launch_both_inferences
from .utils import compare_preds
from .error import plot_error
from .examples import plot_example, plot_ensemble
from .losses import plot_losses
from .usgs import plot_real_data, plot_semblance, plot_ensemble_real
from .marmousi import (
    plot_examples_steep, plot_marmousi, plot_ensemble_marmousi,
)

plt.rcParams.update(
    {
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.titlepad': 4,
        'figure.figsize': [4.33, 2.5],
        'figure.dpi': 1000,
    }
)

# Patch `plt.savefig`.
_savefig = plt.savefig


def savefig(filename, plot=False):
    _savefig(
        join(FIGURES_DIR, filename + ".eps"),
        bbox_inches="tight",
        dpi=1000,
    )
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


plt.savefig = savefig


def main(args):
    dataset = Article2D()
    dataset._getfilelist()
    dataset_train = Article2D()
    dataset_train.testsize = dataset_train.trainsize
    dataset_train.datatest = dataset_train.datatrain
    dataset_train._getfilelist()
    dataset_real = USGS()
    dataset_real._getfilelist()
    dataset_marmousi = Marmousi()
    dataset_marmousi._getfilelist()
    dataset_steep = Article2DSteep()
    dataset_steep._getfilelist()

    if not exists(FIGURES_DIR):
        makedirs(FIGURES_DIR)

    if not args.no_inference:
        launch_inference(
            RCNN2D,
            Hyperparameters2D(is_training=False),
            dataset_train,
            args.logdir_2d,
            args.gpus,
            "Training",
        )
        launch_both_inferences(args, RCNN2D, dataset)
        for lr in ['8E-4', '8E-5']:
            launch_inference(
                RCNN2D,
                Hyperparameters2DNoTL(is_training=False),
                dataset,
                args.logdir_2d_no_tl + '_' + lr,
                args.gpus,
                "NoTransferLearning" + lr,
            )
        launch_both_inferences(
            args, RCNN2DUnpackReal, dataset_real, batch_size=2,
        )
        launch_inference(
            RCNN2D,
            Hyperparameters2DSteep(is_training=False),
            dataset_steep,
            args.logdir_2d + '_steep',
            args.gpus,
            "Steep",
        )
        launch_inference(
            RCNN2D,
            Hyperparameters2DSteep(is_training=False),
            dataset_marmousi,
            args.logdir_2d + '_steep',
            args.gpus,
            "Steep",
            batch_size=1,
        )

    compare_preds(dataset_train, savedir="Training")
    compare_preds(dataset, savedir="Pretraining")
    for lr in ['8E-4', '8E-5']:
        compare_preds(dataset, savedir="NoTransferLearning" + lr)
    similarities = compare_preds(dataset, savedir="EndResults")

    plot_error(dataset, plot=args.plot)

    for percentile in [10, 50, 90]:
        score = np.percentile(
            similarities, percentile, interpolation="nearest",
        )
        idx = np.argwhere(score == similarities)[0, 0]
        print(f"SSIM {percentile}th percentile: {score} for example {idx}.")
        plot_example(
            dataset=dataset,
            filename=dataset.files["test"][idx],
            figure_name=f"results_{percentile}th_percentile",
            plot=args.plot,
        )
        if percentile == 50:
            for output_name in ['vint', 'vdepth']:
                plot_ensemble(
                    dataset=dataset,
                    output_name=output_name,
                    filename=dataset.files["test"][idx],
                    plot=args.plot,
                )
    plot_losses(
        logdir_1d=args.logdir_1d,
        params_1d=Hyperparameters1D(is_training=True),
        logdir_2d=args.logdir_2d,
        params_2d=Hyperparameters2D(is_training=True),
        plot=args.plot,
    )
    plot_real_data(
        dataset=dataset_real,
        plot=args.plot,
    )
    plot_semblance(dataset_real, plot=args.plot)
    for output_name in ['vint', 'vdepth']:
        plot_ensemble_real(
            dataset=dataset_real,
            output_name=output_name,
            plot=args.plot,
        )
    plot_examples_steep(dataset=dataset_steep, plot=args.plot)
    plot_marmousi(dataset=dataset_marmousi, plot=args.plot)
    plot_ensemble_marmousi(dataset=dataset_marmousi, plot=args.plot)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir_1d",
        type=str,
        default='logs/weights_1d',
        help="Directory in which the checkpoints for the 1D case are stored.",
    )
    parser.add_argument(
        "--logdir_2d",
        type=str,
        default='logs/weights_2d',
        help="Directory in which the checkpoints for the 2D case are stored.",
    )
    parser.add_argument(
        "--logdir_2d_no_tl",
        type=str,
        default='logs/weights_2d_no_tl',
        help=(
            "Directory in which the checkpoints for the 2D case without "
            "transfer learning are stored."
        ),
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
        "--no-inference",
        action="store_true",
        help=(
            "Do not run inference if predictions have already been generated."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the produced figures."
    )

    args = parser.parse_args()
    main(args)
