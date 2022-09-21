# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
from copy import deepcopy

import segyio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import compare_ssim as ssim
from GeoFlow.SeismicUtilities import (
    sortcmp, semblance_gather, nmo_correction,
)

from .utils import data_preprocess, stack_2d
from .constants import IGNORE_IDX, TABLEAU_COLORS


def plot_real_data(dataset, plot=True):
    filename = join(dataset.basepath, dataset.name, "test", "example_1")
    inputs, _, _ = dataset.generator.read(filename)

    pretrained = dataset.generator.read_predictions(filename, "Pretraining")
    preds = dataset.generator.read_predictions(filename, "EndResults")
    nt = inputs['shotgather'].shape[0]
    TARGET_NT = 3071
    pad = TARGET_NT - nt
    for name, input in inputs.items():
        inputs[name] = np.pad(input, [[pad, 0], [0, 0]])
    for outputs in [pretrained, preds]:
        for name, output in outputs.items():
            outputs[name] = np.pad(
                output, [[pad, 0], [0, 0]], constant_values=output[100, 1000],
            )

    plot_real_models(dataset, pretrained, preds, plot=plot)
    plot_real_stacks(dataset, inputs, preds, plot=plot)


def plot_real_models(dataset, pretrained, preds, plot=True):
    fig, axs = plt.subplots(
        ncols=2,
        nrows=3,
        figsize=[4.33, 5],
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
    cmps = cmps[10:-10]

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
    END_TIME = 10
    crop_bottom = int((END_TIME+tdelay) / dt)

    dh = dataset.model.dh
    TOP_VINT = 1500
    start_depth = (start_time+tdelay) / 2 * TOP_VINT
    crop_top_d = int(start_depth / dh)
    END_DEPTH = 10000
    crop_bottom_d = int(END_DEPTH / dh)

    data_meta = deepcopy(dataset.inputs['shotgather'])
    data_meta.acquire.singleshot = True
    vint_meta = dataset.outputs['vint']

    pretrained_vint = vint_meta.postprocess(pretrained['vint'])
    pred_vint = vint_meta.postprocess(preds['vint'])
    pred_vdepth = vint_meta.postprocess(preds['vdepth'])
    pretrained_vint = gaussian_filter(pretrained_vint, [5, 15])
    pred_vint = gaussian_filter(pred_vint, [5, 15])
    pred_vdepth = gaussian_filter(pred_vdepth, [5, 15])
    pretrained_vint = pretrained_vint[crop_top:crop_bottom]
    pred_vint = pred_vint[crop_top:crop_bottom]
    pred_vdepth = pred_vdepth[crop_top_d:crop_bottom_d]

    vint_meta.plot(
        pretrained_vint, axs=[axs[0]], vmin=1400, vmax=3100, cmap='jet',
    )
    vint_meta.plot(
        pred_vint, axs=[axs[1]], vmin=1400, vmax=3100, cmap='jet',
    )
    vint_meta.plot(
        pred_vdepth, axs=[axs[2]], vmin=1400, vmax=3100, cmap='jet',
    )

    extent = [cmps.min()/1000, cmps.max()/1000, END_TIME, start_time]
    extent_d = [
        cmps.min()/1000, cmps.max()/1000, END_DEPTH/1000, start_depth/1000,
    ]

    for i, ax in enumerate(axs[:-1]):
        ax.images[0].set_extent(extent)
    axs[-1].images[0].set_extent(extent_d)

    for ax in axs:
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
    for ax in axs[:-1]:
        ax.set_ylabel("$t$ (s)")
    axs[-1].set_ylabel("$z$ (km)")

    cbar = plt.colorbar(axs[0].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity (km/s)")
    cbar.set_ticks(range(2000, 5000, 1000))
    cbar.set_ticklabels(range(2, 5, 1))

    for ax, letter in zip(axs, range(ord('a'), ord('e')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig("real_models", plot=plot)


def plot_real_stacks(dataset, inputs, preds, plot=True):
    fig, axs = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=[4.33, 3],
        constrained_layout=False,
        gridspec_kw={"hspace": .3},
    )

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps = cmps[10:-10]

    data_meta = deepcopy(dataset.inputs['shotgather'])
    data_meta.acquire.singleshot = True
    vint_meta = dataset.outputs['vint']

    resampling = dataset.acquire.resampling
    dt = dataset.acquire.dt * resampling
    tdelay = dataset.acquire.tdelay
    nt = dataset.acquire.NT
    times = np.arange(nt//resampling)*dt - tdelay
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
    END_TIME = 10
    crop_bottom = int((END_TIME+tdelay) / dt)

    shotgather = inputs['shotgather']
    if data_meta.skip_preprocess:
        # Trigger first preprocess skipping.
        data_meta.preprocess(None, None)
    shotgather = data_meta.preprocess(shotgather, None, use_agc=False)
    shotgather = shotgather[..., 0]

    stacked_filepath = join(dataset.basepath, dataset.name, "CSDS32_1.SGY")
    with segyio.open(stacked_filepath, "r", ignore_geometry=True) as segy:
        stacked_usgs = [segy.trace[trid] for trid in range(segy.tracecount)]
        stacked_usgs = np.array(stacked_usgs)
        stacked_usgs = stacked_usgs.T
    stacked_usgs = stacked_usgs[:, -2350:-160]
    stacked_usgs = stacked_usgs[:, ::-1]
    stacked_usgs = data_preprocess(stacked_usgs)

    print("Stacking.")
    pred_vrms = vint_meta.postprocess(preds['vrms'])
    shotgather *= times[:, None, None]**2
    pred_stacked = stack_2d(shotgather, times, offsets, pred_vrms)
    pred_stacked = data_preprocess(pred_stacked)

    pred_stacked = pred_stacked[crop_top:crop_bottom]
    stacked_usgs = stacked_usgs[crop_top:crop_bottom]

    data_meta.plot(pred_stacked, axs=[axs[0]], vmin=0, clip=4E-2)
    data_meta.plot(stacked_usgs, axs=[axs[1]], vmin=0, clip=1.5E-1)

    extent = [cmps.min()/1000, cmps.max()/1000, END_TIME, start_time]

    for i, ax in enumerate(axs):
        ax.images[0].set_extent(extent)

    for ax in axs:
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

    for ax, letter in zip(axs, range(ord('a'), ord('e')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig("real_stacks", plot=plot)


def plot_semblance(dataset, plot=True):
    filename = join(dataset.basepath, dataset.name, "test", "example_1")
    inputs, _, _ = dataset.generator.read(filename)
    data_meta = deepcopy(dataset.inputs['shotgather'])
    data_meta.acquire.singleshot = True
    shotgather = inputs['shotgather']
    try:
        data_meta.preprocess(None, None)
    except AttributeError:
        pass
    shotgather = data_meta.preprocess(shotgather, None)

    filename = dataset.files["test"][0]
    preds = dataset.generator.read_predictions(filename, "EndResults")
    preds_std = dataset.generator.read_predictions(filename, "EndResults_std")
    for key, value in preds.items():
        value = dataset.outputs[key].postprocess(value)
        preds[key] = value
    for key, value in preds_std.items():
        vmin, vmax = dataset.model.properties["vp"]
        preds_std[key] = value * (vmax-vmin)

    resampling = dataset.acquire.resampling
    dt = dataset.acquire.dt * resampling
    tdelay = dataset.acquire.tdelay
    nt = dataset.acquire.NT
    times = np.arange(nt//resampling)*dt - tdelay
    offsets = np.arange(
        dataset.acquire.gmin,
        dataset.acquire.gmax,
        dataset.acquire.dg,
        dtype=float,
    )
    offsets *= dataset.model.dh
    velocities = np.arange(1300, 3500, 50)

    extent_gather = [
        offsets.min() / 1000,
        offsets.max() / 1000,
        times.max(),
        times.min(),
    ]
    extent_semblance = [
        velocities.min() / 1000,
        velocities.max() / 1000,
        times.max(),
        times.min(),
    ]

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=[4.33, 8],
        sharex='col',
        sharey=True,
        gridspec_kw={'wspace': .05},
    )
    TITLES = {
        'vrms': "$v_\\mathrm{RMS}(t, x)$",
        'vint': "$v_\\mathrm{int}(t, x)$",
    }
    for i, cmp in enumerate([250, 1000, 1750]):
        temp_shotgather = shotgather[..., cmp, 0]
        temp_shotgather /= np.amax(temp_shotgather)
        vmax = 4E-1
        axs[i, 0].imshow(
            temp_shotgather,
            aspect='auto',
            cmap='Greys',
            extent=extent_gather,
            vmin=0,
            vmax=vmax,
        )

        semblance = semblance_gather(
            temp_shotgather, times, offsets, velocities,
        )
        axs[i, 1].imshow(
            semblance,
            aspect='auto',
            cmap='Greys',
            extent=extent_semblance,
            alpha=.8,
        )
        for color, pred_name in zip(TABLEAU_COLORS, ['vrms', 'vint']):
            pred = preds[pred_name][:, cmp] / 1000
            std = preds_std[pred_name][:, cmp] / 1000
            axs[i, 1].plot(
                pred, times, lw=.5, color=color, label=TITLES[pred_name],
            )
            axs[i, 1].fill_betweenx(
                times,
                pred-std,
                pred+std,
                color=color,
                lw=0,
                alpha=.4,
            )
        axs[i, 1].set_xlim([velocities.min() / 1000, velocities.max() / 1000])

        v = preds['vrms'][:, cmp]
        corrected = nmo_correction(temp_shotgather, times, offsets, v)
        axs[i, 2].imshow(
            corrected,
            aspect='auto',
            cmap='Greys',
            extent=extent_gather,
            vmin=0,
            vmax=vmax,
        )

    for ax in axs.flatten():
        plt.sca(ax)
        plt.minorticks_on()

    for ax in axs.flatten():
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()

    start_idx = np.nonzero(preds['vint'][:, 1750] > 2000)[0][0] - 50
    start_time = times[start_idx]
    END_TIME = 10
    for ax in axs.flatten():
        ax.set_ylim([start_time, END_TIME])

    for ax in axs[:, 0]:
        ax.invert_xaxis()
        ax.invert_yaxis()

    axs[-1, 0].set_xlabel("$h$ (km)")
    axs[-1, 1].set_xlabel("Velocity (km/s)")
    axs[-1, 2].set_xlabel("$h$ (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("$t$ (s)")

    for ax, letter in zip(axs.T.flatten(), range(ord('a'), ord('i')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    axs[0, 1].legend(
        loc='center right',
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.savefig("semblance", plot=plot)


def plot_ensemble_real(dataset, output_name, plot):
    filename = dataset.files["test"][0]
    fig, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=[3.33, 5],
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1, .1], "hspace": .3},
    )
    cax = axs[0, -1]
    cax_std = axs[1, -1]
    for ax in axs[2:, -1]:
        ax.remove()
    axs = axs[:, :-1]

    meta = dataset.outputs[output_name]

    ensemble = []
    savedirs = [
        dir for dir in listdir(dataset.datatest)
        if (
            "EndResults_" in dir
            and "std" not in dir
            and int(dir.split('_')[-1]) not in IGNORE_IDX
        )
    ]
    for savedir in savedirs:
        preds = dataset.generator.read_predictions(filename, savedir)
        ensemble.append(preds[output_name])
    mean = dataset.generator.read_predictions(filename, "EndResults")
    mean = mean[output_name]
    std = dataset.generator.read_predictions(filename, "EndResults_std")
    std = std[output_name]

    similarities = np.array([])
    for pred in ensemble:
        similarity = ssim(mean, pred)
        similarities = np.append(similarities, similarity)

    ref = preds['ref'] > .2
    crop_top = int(np.nonzero(ref.any(axis=1))[0][0] * .95)
    dt = dataset.acquire.dt * dataset.acquire.resampling
    tdelay = dataset.acquire.tdelay
    start = crop_top*dt - tdelay

    if output_name == 'vdepth':
        dh = dataset.model.dh
        TOP_VINT = 1500
        start = (start+tdelay) / 2 * TOP_VINT
        crop_top = int(start / dh)
        end = 10000
        crop_bottom = int(end / dh)
    else:
        end = 10
        crop_bottom = int((end+tdelay) / dt)
    for i, pred in enumerate(ensemble):
        ensemble[i] = pred[crop_top:crop_bottom]
    std = std[crop_top:crop_bottom]

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps = cmps[10:-10]
    cmps /= 1000
    if output_name == 'vdepth':
        dh = dataset.model.dh
        src_rec_depth = dataset.acquire.source_depth
        start = crop_top*dh + src_rec_depth
        start /= 1000
        end = (len(ensemble[0])-1)*dh + start
        end /= 1000
    else:
        tdelay = dataset.acquire.tdelay
        start = crop_top*dt - tdelay
        end = (len(ensemble[0])-1)*dt + start

    far = np.argsort(similarities)
    print("Farthest SSIMs:", similarities[far[:3]])
    closest = np.argmax(similarities)
    print("Closest SSIM:", similarities[closest])
    arrays = np.array(
        [ensemble[closest], std, *[ensemble[i] for i in far[:3]]]
    )
    for i, (array, ax) in enumerate(
        zip(arrays.reshape([-1, *ensemble[0].shape]), axs.flatten())
    ):
        if i != 1:
            array = meta.postprocess(array)
            cmap = 'jet'
            vmin, vmax = 1400, 3500
        else:
            vmin, vmax = dataset.model.properties["vp"]
            array = array * (vmax-vmin)
            vmin, vmax = 0, 1000
            cmap = 'afmhot_r'
        array = gaussian_filter(array, [5, 15])
        meta.plot(
            array,
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

    cbar = plt.colorbar(axs[0, 0].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity\n(km/s)")
    cbar.set_ticks(range(2000, 5000, 1000))
    cbar.set_ticklabels(range(2, 5, 1))

    cbar = plt.colorbar(axs[1, 0].images[0], cax=cax_std)
    cbar.ax.set_ylabel("Standard\ndeviation\n(km/s)")
    cbar.set_ticks(np.arange(0, 1000, 300))
    cbar.set_ticklabels(np.arange(0, 1, .3))

    for ax, letter in zip(axs.flatten(), range(ord('a'), ord('g')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(f"ensemble_{output_name}_real", plot=plot)
