# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vmbtl.postprocess.utils import load_events
from vmbtl.postprocess.constants import TABLEAU_COLORS


def plot_losses(logdir_1d, params_1d, logdir_2d, params_2d, plot=True):
    mean_1d, std_1d = load_events(logdir_1d)
    mean_2d, std_2d = load_events(logdir_2d)
    mean = pd.concat([mean_1d, mean_2d], ignore_index=True)
    std = pd.concat([std_1d, std_2d], ignore_index=True)
    qty_stages_1d = len(params_1d.loss_scales)
    qty_stages_2d = len(params_2d.loss_scales)
    epochs_1d = (params_1d.epochs,) * qty_stages_1d
    epochs_2d = (params_2d.epochs,) * qty_stages_2d
    epochs = epochs_1d + epochs_2d

    LABEL_NAMES = {
        'loss': "Total loss",
        'ref_loss': "Primaries",
        'vrms_loss': "$v_\\mathrm{RMS}(t, x)$",
        'vint_loss': "$v_\\mathrm{int}(t, x)$",
        'vdepth_loss': "$v_\\mathrm{int}(z, x)$",
    }
    mean.columns = [column.split('/')[-1] for column in mean.columns]
    std.columns = [column.split('/')[-1] for column in std.columns]
    for column in mean.columns:
        if column not in LABEL_NAMES.keys():
            del mean[column]
            del std[column]
    plt.figure(figsize=[3.33, 2.5])
    for i, column in enumerate(LABEL_NAMES.keys()):
        iters = (np.arange(len(mean[column]))+1) * params_1d.steps_per_epoch
        current_mean = mean[column].map(lambda x: 10**x)
        if column == 'loss':
            plt.plot(
                iters,
                current_mean,
                label=LABEL_NAMES[column],
                zorder=100,
                lw=2.5,
                color=TABLEAU_COLORS[i],
            )
        else:
            plt.plot(
                iters,
                current_mean,
                label=LABEL_NAMES[column],
                color=TABLEAU_COLORS[i],
            )
        upper = mean[column].add(std[column]).map(lambda x: 10**x)
        lower = mean[column].sub(std[column]).map(lambda x: 10**x)
        plt.fill_between(
            iters, lower, upper, color=TABLEAU_COLORS[i], lw=0, alpha=.2,
        )
    limits = np.cumsum((0,) + epochs)
    limits[0] = 1
    limits *= params_1d.steps_per_epoch
    colormaps = [plt.get_cmap('Blues'), plt.get_cmap('Oranges')]
    sample_colormaps = [
        np.linspace(.4, .8, n_stages)
        for n_stages in [qty_stages_1d, qty_stages_2d]
    ]
    colors = []
    for sample_colormap, colormap in zip(sample_colormaps, colormaps):
        for sample in sample_colormap:
            colors.append(colormap(sample))
    for [x_min, x_max], color in zip(zip(limits[:-1], limits[1:]), colors):
        plt.fill_betweenx(
            [1E-16, 1E16], x_min, x_max, color=color, alpha=.2,
        )
    plt.xlim([limits[0], limits[-1]])
    plt.semilogy()
    vmax, vmin = mean.values.max(), mean.values.min()
    diff = vmax - vmin
    plt.ylim([10**(vmin-.1*diff), 10**(vmax+.1*diff)])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(.5, 1),
        ncol=len(LABEL_NAMES),
        handlelength=.25,
        handletextpad=.5,
        columnspacing=1.0,
    )
    plt.minorticks_on()
    plt.grid(which='major', alpha=.6)
    plt.grid(which='minor', alpha=.15)

    plt.savefig("losses", plot=plot)
