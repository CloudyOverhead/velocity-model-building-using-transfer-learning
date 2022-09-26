# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from vmbtl.postprocess.utils import load_all


def plot_error(dataset, plot=True):
    savedir = "EndResults"
    _, all_labels, all_weights, all_preds = load_all(dataset, savedir)

    rmses = np.array([])
    thicknesses = np.array([])
    velocities = np.array([])
    depths = np.array([])
    vmin, vmax = dataset.model.properties['vp']
    dt = dataset.acquire.dt * dataset.acquire.resampling / 2
    for label, weight, pred in zip(
        all_labels["vint"], all_weights["vint"], all_preds["vint"],
    ):
        label = label * weight
        pred = pred * weight
        for slice_label, slice_pred in zip(label.T, pred.T):
            interfaces = np.nonzero(np.diff(slice_label))
            interfaces = interfaces[0] + 1
            current_depth = 0
            for start, end in zip([0, *interfaces[:-1]], interfaces):
                temp_label = slice_label[start:end]
                temp_pred = slice_pred[start:end]
                rmse = np.sqrt(np.mean((temp_label-temp_pred)**2))
                velocity = temp_label[0]*(vmax-vmin) + vmin
                rmses = np.append(rmses, rmse)
                thicknesses = np.append(thicknesses, end-start)
                velocities = np.append(velocities, velocity)
                depths = np.append(depths, current_depth)
                current_depth += (end-start) * velocity * dt
    rmses *= vmax - vmin
    thicknesses *= velocities * dt

    samples = [velocities, thicknesses, depths, rmses]
    bins = [np.linspace(a.min(), a.max(), 101) for a in samples]
    log_thicknesses = np.log10(thicknesses)
    bins[1] = np.logspace(log_thicknesses.min(), log_thicknesses.max(), 101)
    bins[-1] = np.logspace(np.log10(2), np.log10(1900), 101)
    hist, _ = np.histogramdd(samples, bins=bins, normed=True)

    fig, axs = plt.subplots(nrows=3, figsize=[3.33, 7.5], sharex=True)

    for i, (ax, y) in enumerate(zip(axs, bins[:-1])):
        x = bins[-1]
        x = np.repeat(x[None, :], len(x), axis=0)
        y = np.repeat(y[::-1, None], len(bins[-1]), axis=1)
        other_axis = tuple(axis for axis in range(4) if axis not in [i, 3])
        errors = np.sum(hist, axis=other_axis)
        average = np.ma.average(
            (x[:-1, 1:]+x[1:, 1:])/2,
            axis=-1,
            weights=errors,
        )
        errors = np.log10(errors)
        ax.pcolor(x, y, errors[::-1], cmap='Greys')
        if i in [0, 2]:
            y = (y[:-1, 0]+y[1:, 1]) / 2
            ax.plot(average[::-1], y, lw=1, ls='--', c='k')

    axs[-1].set_xlabel("RMSE (m/s)")
    axs[0].set_ylabel("$v_\\mathrm{int}(t, x)$ (m/s)")
    axs[1].set_ylabel("Thickness (m)")
    axs[2].set_ylabel("Depth (m)")

    axs[0].set_xscale('log')
    axs[1].set_yscale('log')

    for ax, letter in zip(axs.flatten(), range(ord('a'), ord('c')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig("error", plot=plot)
