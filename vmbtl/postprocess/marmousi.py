# -*- coding: utf-8 -*-

from os import listdir
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from GeoFlow.SeismicUtilities import sortcmp

from .utils import compare_preds
from .constants import IGNORE_IDX


def plot_examples_steep(dataset, plot=True):
    toinputs = []
    tooutputs = ['vint']

    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=[3.3, 6],
        constrained_layout=False,
    )

    meta = dataset.outputs['vint']

    dt = dataset.acquire.dt * dataset.acquire.resampling
    vmin, vmax = dataset.model.properties['vp']
    diff = vmax - vmin
    tdelay = dataset.acquire.tdelay
    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps = cmps[10:-10]
    cmps /= 1000

    similarities = compare_preds(dataset, "Steep")
    for percentile, row_axs in zip([90, 50, 10], axs):
        score = np.percentile(
            similarities, percentile, interpolation="higher",
        )
        idx = np.argwhere(score == similarities)[0, 0]
        print(f"SSIM {percentile}th percentile: {score} for example {idx}.")

        filename = dataset.files["test"][idx]

        _, label, weight, filename = dataset.get_example(
            filename=filename,
            phase='test',
            toinputs=toinputs,
            tooutputs=tooutputs,
        )
        label = label['vint']
        label = meta.postprocess(label)
        weight = weight['vint']

        pred = dataset.generator.read_predictions(filename, "Steep")
        pred = pred['vint']
        pred = meta.postprocess(pred)
        std = dataset.generator.read_predictions(filename, "Steep_std")
        std = std['vint'] * diff

        crop_top = np.nonzero(np.diff(label, axis=0).any(axis=1))[0][1] * .95
        crop_top = int(crop_top)
        label = label[crop_top:]
        weight = weight[crop_top:]
        pred = pred[crop_top:]
        std = std[crop_top:]

        start_time = crop_top*dt - tdelay
        time = np.arange(len(label))*dt + start_time

        for array, ax in zip([label], [row_axs]):
            im, = meta.plot(array, weights=weight, axs=[ax])
            im.set_extent(
                [cmps.min(), cmps.max(), time.max(), time.min()]
            )
            ax.set_title("")
            cbar = im.colorbar
            if cbar is not None:
                cbar.remove()

    for ax in axs[:-1].flatten():
        ax.set_xticklabels([])
    axs[-1].set_xlabel("$v_\\mathrm{int}(t, x)$ (km/s)")
    for ax in axs[:]:
        ax.set_ylabel("$t$ (s)")
    for ax in axs.flatten():
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    for ax in [axs[-1]]:
        ax.set_xlabel("$x$ (km)")

    ticks = np.arange(2000, 5000, 1000)
    cax = fig.add_subplot(3, 1, 1)
    left, bottom, width, height = cax.get_position().bounds
    bottom += 1.2 * height
    height /= 8
    width /= 2
    left += width / 2
    cax.set_position([left, bottom, width, height])
    cbar = plt.colorbar(axs[0].images[0], cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel("$v_\\mathrm{int}(t, x)$ (km/s)")
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks/1000)

    plt.savefig('examples_steep', plot=plot)


def plot_marmousi(dataset, plot=True):
    compare_preds(dataset, "Steep")

    filename = join(dataset.basepath, dataset.name, "test", "example_0")
    inputs, labels, _ = dataset.generator.read(filename)
    preds = dataset.generator.read_predictions(filename, "Steep")

    fig, axs = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=[4.33, 4],
        constrained_layout=False,
        gridspec_kw={"width_ratios": [95, 5], "hspace": .3, "wspace": .05},
    )
    for ax in axs[1:, 1]:
        ax.remove()
    cax = axs[0, 1]
    axs = axs[:, 0]
    for shared_axes in [cax.get_shared_x_axes(), cax.get_shared_y_axes()]:
        shared_axes.remove(cax)

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    # cmps = cmps[10:-10]

    resampling = dataset.acquire.resampling
    dt = dataset.acquire.dt * resampling
    tdelay = dataset.acquire.tdelay
    offsets = np.arange(
        dataset.acquire.gmin,
        dataset.acquire.gmax,
        dataset.acquire.dg,
        dtype=float,
    )
    offsets *= dataset.model.dh

    ref = preds['ref'] > .1
    crop_top = int(np.nonzero(ref.any(axis=1))[0][0] * .95)
    start_time = crop_top*dt - tdelay
    END_TIME = 3
    crop_bottom = int((END_TIME+tdelay) / dt)

    vint_meta = dataset.outputs['vint']

    props, *_ = dataset.model.generate_model()
    label_vint, _ = dataset.outputs['vint'].generate(None, props)
    label_vint, _ = dataset.outputs['vint'].preprocess(label_vint, None)
    label_vint = vint_meta.postprocess(label_vint)
    pred_vint = vint_meta.postprocess(preds['vint'])
    label_vint = label_vint[crop_top:crop_bottom]
    pred_vint = pred_vint[crop_top:crop_bottom]

    vint_meta.plot(
        label_vint, axs=[axs[0]], vmin=1500, vmax=4500, cmap='inferno',
    )
    vint_meta.plot(
        pred_vint, axs=[axs[1]], vmin=1500, vmax=4500, cmap='inferno',
    )

    extent = [cmps.min()/1000, cmps.max()/1000, END_TIME, start_time]

    for ax in axs:
        ax.images[0].set_extent(extent)
        ax.set_title("")
        if ax.images:
            cbar = ax.images[-1].colorbar
            if cbar is not None:
                cbar.remove()

    for ax in axs[:-1]:
        ax.set_xticklabels([])

    for ax in axs:
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()

    plt.xlabel("$x$ (km)")
    for ax in axs:
        ax.set_ylabel("$t$ (s)")

    cbar = plt.colorbar(axs[0].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity (km/s)")
    cbar.set_ticks(range(2000, 5000, 1000))
    cbar.set_ticklabels(range(2, 5, 1))

    for ax, letter in zip(axs, range(ord('a'), ord('b')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig("marmousi", plot=plot)


def plot_ensemble_marmousi(dataset, plot):
    output_name = 'vint'
    filename = join(dataset.datatest, 'example_0')

    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=[6.5, 3.33],
        constrained_layout=False,
        gridspec_kw={"width_ratios": [*(1 for _ in range(3)), .2]},
    )
    cax_std = axs[0, -1]
    cax = axs[1, -1]
    axs = axs[:, :-1]

    meta = dataset.outputs[output_name]

    _, labels, weights, _ = dataset.get_example(
        filename=filename, phase='test',
    )
    label = labels[output_name]
    weight = weights[output_name]

    ensemble = []
    savedirs = [
        dir for dir in listdir(dataset.datatest)
        if (
            "Steep_" in dir
            and "std" not in dir
            and int(dir.split('_')[-1]) not in IGNORE_IDX
        )
    ]
    for savedir in savedirs:
        preds = dataset.generator.read_predictions(filename, savedir)
        ensemble.append(preds[output_name])
    mean = dataset.generator.read_predictions(filename, "Steep")
    mean = mean[output_name]
    std = dataset.generator.read_predictions(filename, "Steep_std")
    std = std[output_name]

    similarities = np.array([])
    rmses = np.array([])
    for pred in ensemble:
        similarity = ssim(label*weight, pred*weight)
        similarities = np.append(similarities, similarity)
        rmse = np.sqrt(np.mean((label*weight-pred*weight)**2))
        rmses = np.append(rmses, rmse)
    vmin, vmax = dataset.model.properties['vp']
    rmses *= vmax - vmin

    ref = labels['ref']
    crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0] * .95)
    dh = dataset.model.dh
    dt = dataset.acquire.dt * dataset.acquire.resampling
    diff = vmax - vmin
    water_v = float(labels['vint'][0, 0])*diff + vmin
    tdelay = dataset.acquire.tdelay
    if output_name == 'vdepth':
        crop_top = int((crop_top-tdelay/dt)*dt/2*water_v/dh)
        mask = weights['vdepth']
        crop_bottom = int(np.nonzero((~mask.astype(bool)).all(axis=1))[0][0])
    else:
        crop_bottom = None
    for i, pred in enumerate(ensemble):
        ensemble[i] = pred[crop_top:crop_bottom]
    label = label[crop_top:crop_bottom]
    std = std[crop_top:crop_bottom]
    weight = weight[crop_top:crop_bottom]

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps = cmps[10:-10]
    cmps /= 1000
    if output_name == 'vdepth':
        dh = dataset.model.dh
        src_rec_depth = dataset.acquire.source_depth
        start = crop_top*dh + src_rec_depth
        start /= 1000
        end = (len(labels['vdepth'])-1)*dh + start
        end /= 1000
    else:
        tdelay = dataset.acquire.tdelay
        start = crop_top*dt - tdelay
        end = (len(label)-1)*dt + start

    far = np.argsort(similarities)
    print("Farthest SSIMs:", similarities[far[:3]])
    print("Farthest RMSEs:", rmses[far[:3]])
    closest = np.argmax(similarities)
    print("Closest SSIM:", similarities[closest])
    print("Closest RMSE:", rmses[closest])
    arrays = np.array(
        [
            [label, ensemble[closest], std],
            [ensemble[i] for i in far[:3]],
        ]
    )
    for i, (array, ax) in enumerate(
        zip(arrays.reshape([-1, *label.shape]), axs.flatten())
    ):
        if i != 2:
            array = meta.postprocess(array)
            cmap = 'inferno'
            vmin, vmax = 1500, 4500
        else:
            vmin, vmax = dataset.model.properties["vp"]
            array = array * (vmax-vmin)
            vmin, vmax = 0, 1000
            cmap = 'afmhot_r'
        meta.plot(
            array,
            weights=None,
            axs=[ax],
            ims=[None],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )

    for ax in axs.flatten():
        ax.images[0].set_extent([cmps.min(), cmps.max(), end, start])
    for ax in axs.flatten():
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()

    for ax in axs.flatten():
        ax.set_title("")
        if ax.images:
            cbar = ax.images[-1].colorbar
            if cbar is not None:
                cbar.remove()

    for ax in axs[:, 0]:
        if output_name != 'vdepth':
            ax.set_ylabel("$t$ (s)")
        else:
            ax.set_ylabel("$z$ (km)")
    for ax in axs[:, 1:].flatten():
        ax.set_yticklabels([])
    for ax in axs[-1, :]:
        ax.set_xlabel("$x$ (km)")
    for ax in axs[:-1, :].flatten():
        ax.set_xticklabels([])

    cbar = plt.colorbar(axs[0, -1].images[0], cax=cax_std)
    cbar.ax.set_ylabel("Standard\ndeviation\n(km/s)")
    cbar.set_ticks(np.arange(0, 1000, 300))
    cbar.set_ticklabels(np.around(np.arange(0, 1, .3), 1))

    cbar = plt.colorbar(axs[0, 0].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity\n(km/s)")
    cbar.set_ticks(range(2000, 5000, 1000))
    cbar.set_ticklabels(range(2, 5, 1))

    for ax, letter in zip(axs.flatten(), range(ord('a'), ord('g')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(f"ensemble_{output_name}_marmousi", plot=plot)
