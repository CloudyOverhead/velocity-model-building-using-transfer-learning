# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from os import makedirs, listdir
from os.path import join, exists, split
from copy import deepcopy
from datetime import datetime

import segyio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TABLEAU_COLORS
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from tensorflow.compat.v1.train import summary_iterator
from GeoFlow.__main__ import int_or_list
from GeoFlow.SeismicUtilities import sortcmp, stack, semblance_gather

from vmbtl.__main__ import main as global_main
from vmbtl.architecture import (
    RCNN2D, RCNN2DUnpackReal, Hyperparameters1D, Hyperparameters2D,
    Hyperparameters2DNoTL,
)
from vmbtl.datasets import Article2D, USGS

FIGURES_DIR = "figures"
TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint', 'vdepth']

plt.rcParams.update(
    {
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.titlepad': 4,
        'figure.figsize': [4.33, 2.5],
        'figure.dpi': 1200,
    }
)

TABLEAU_COLORS = [color[1] for color in TABLEAU_COLORS.items()]


def main(args):
    dataset = Article2D()
    dataset._getfilelist()
    dataset_train = Article2D()
    dataset_train.testsize = dataset_train.trainsize
    dataset_train.datatest = dataset_train.datatrain
    dataset_train._getfilelist()
    dataset_real = USGS()
    dataset_real._getfilelist()

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
        launch_inference(
            RCNN2D,
            Hyperparameters2DNoTL(is_training=False),
            dataset,
            args.logdir_2d_no_tl,
            args.gpus,
            "NoTransferLearning",
        )
        launch_both_inferences(args, RCNN2DUnpackReal, dataset_real)

    compare_preds(dataset_train, savedir="Training")
    compare_preds(dataset, savedir="Pretraining")
    compare_preds(dataset, savedir="NoTransferLearning")
    inputs, labels, weights, preds, similarities = compare_preds(
        dataset, savedir="EndResults",
    )

    for percentile in [10, 50, 90]:
        score = np.percentile(
            similarities, percentile, interpolation="nearest",
        )
        idx = np.argwhere(score == similarities)[0, 0]
        print(f"SSIM {percentile}th percentile:", score)
        plot_example(
            dataset=dataset,
            filename=dataset.files["test"][idx],
            figure_name=f"results_{percentile}th_percentile.pdf",
            plot=args.plot,
        )
        if percentile == 50:
            plot_no_transfer_learning(
                dataset=dataset,
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


def launch_both_inferences(args, nn, dataset):
    params_1d = Hyperparameters1D(is_training=False)
    params_1d.batch_size = 2
    params_2d = Hyperparameters2D(is_training=False)
    for logdir, savedir, params in zip(
        [args.logdir_1d, args.logdir_2d],
        ["Pretraining", "EndResults"],
        [params_1d, params_2d],
    ):
        launch_inference(nn, params, dataset, logdir, args.gpus, savedir)


def launch_inference(nn, params, dataset, logdir, gpus, savedir):
    print("Launching inference.")
    print("NN:", nn.__name__)
    print("Hyperparameters:", type(params).__name__)
    print("Weights:", logdir)
    print("Case:", savedir)

    logdirs = listdir(logdir)
    for i, current_logdir in enumerate(logdirs):
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
    logdirs = listdir(logdir)
    for filename in dataset.files["test"]:
        preds = {key: [] for key in dataset.generator.outputs}
        for i in range(len(logdirs)):
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


def compare_preds(dataset, savedir):
    print(f"Comparing predictions for directory {savedir}.")
    all_inputs = {}
    all_labels = {}
    all_weights = {}
    all_preds = {}
    for example in dataset.files["test"]:
        inputs, labels, weights, filename = dataset.get_example(
            example,
            phase='test',
            toinputs=RCNN2D.toinputs,
            tooutputs=RCNN2D.tooutputs,
        )
        preds = dataset.generator.read_predictions(filename, savedir)
        target_dicts = [all_inputs, all_labels, all_weights, all_preds]
        current_dicts = [inputs, labels, weights, preds]
        for target_dict, current_dict in zip(target_dicts, current_dicts):
            for key in current_dict.keys():
                current_array = np.expand_dims(current_dict[key], axis=0)
                if key in target_dict.keys():
                    target_dict[key] = np.append(
                        target_dict[key], current_array, axis=0,
                    )
                else:
                    target_dict[key] = current_array

    similarities = np.array([])
    rmses = np.array([])
    for labels, weights, preds in zip(
        all_labels["vint"], all_weights["vint"], all_preds["vint"],
    ):
        temp_labels = labels * weights
        temp_preds = preds * weights
        similarity = ssim(temp_labels, temp_preds)
        similarities = np.append(similarities, similarity)
        rmse = np.sqrt(np.mean((temp_labels-temp_preds)**2))
        rmses = np.append(rmses, rmse)
    vmin, vmax = dataset.model.properties['vp']
    rmses *= vmax - vmin

    print("Average SSIM:", np.mean(similarities))
    print("Standard deviation on SSIM:", np.std(similarities))
    print("Average RMSE:", np.mean(rmses))
    print("Standard deviation on RMSE:", np.std(rmses))

    return all_inputs, all_labels, all_weights, all_preds, similarities


def plot_example(dataset, filename, figure_name, plot=True):
    inputs, labels, weights, filename = dataset.get_example(
        filename=filename,
        phase='test',
        toinputs=TOINPUTS,
        tooutputs=TOOUTPUTS,
    )
    inputs_meta = {input: dataset.inputs[input] for input in TOINPUTS}
    outputs_meta = {output: dataset.outputs[output] for output in TOOUTPUTS}
    cols_meta = [inputs_meta, outputs_meta, outputs_meta, outputs_meta]

    pretrained = dataset.generator.read_predictions(filename, "Pretraining")
    pretrained = {name: pretrained[name] for name in TOOUTPUTS}
    preds = dataset.generator.read_predictions(filename, "EndResults")
    preds = {name: preds[name] for name in TOOUTPUTS}
    cols = [inputs, pretrained, preds, labels]

    ref = labels['ref']
    crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0] * .95)
    dh = dataset.model.dh
    dt = dataset.acquire.dt * dataset.acquire.resampling
    vmin, vmax = dataset.model.properties['vp']
    diff = vmax - vmin
    water_v = float(labels['vint'][0, 0])*diff + vmin
    tdelay = dataset.acquire.tdelay
    crop_top_depth = int((crop_top-tdelay/dt)*dt/2*water_v/dh)
    mask = weights['vdepth']
    crop_bottom_depth = int(np.nonzero((~mask.astype(bool)).all(axis=1))[0][0])
    for col in [*cols, weights]:
        for row_name, row in col.items():
            if row_name != 'vdepth':
                col[row_name] = row[crop_top:]
            else:
                col[row_name] = row[crop_top_depth:crop_bottom_depth]

    tdelay = dataset.acquire.tdelay
    start_time = crop_top*dt - tdelay
    time = np.arange(len(labels['ref']))*dt + start_time
    dh = dataset.model.dh
    src_rec_depth = dataset.acquire.source_depth
    start_depth = crop_top_depth*dh + src_rec_depth
    depth = np.arange(len(labels['vdepth']))*dh + start_depth

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps /= 1000
    depth /= 1000

    NROWS = 5
    QTY_IMS = 14
    NCOLS = 3

    fig = plt.figure(figsize=[6.5, 7.5], constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=NROWS,
        ncols=NCOLS*2+2,
        width_ratios=[.2, *(.5 for _ in range(NCOLS*2+1))],
    )
    axs = []
    ax = fig.add_subplot(gs[0, 2:4])
    axs.append(ax)
    ax = fig.add_subplot(gs[0, 4:6])
    axs.append(ax)
    for j in range(1, 7, 2):
        for i in range(NROWS-1):
            ax = fig.add_subplot(gs[i+1, j:j+2])
            axs.append(ax)
    ims = [None for _ in range(QTY_IMS)]

    n = 0
    for col, col_meta in zip(cols, cols_meta):
        for row_name in col:
            naxes = col_meta[row_name].naxes
            input_ims = ims[n:n+naxes]
            input_axs = axs[n:n+naxes]
            data = col[row_name]
            try:
                data = col_meta[row_name].postprocess(data)
            except AttributeError:
                pass
            if row_name != 'vdepth':
                mask = weights['vrms']
            else:
                mask = weights['vdepth']
            output_ims = col_meta[row_name].plot(
                data, weights=mask, axs=input_axs, ims=input_ims,
            )
            for im in output_ims:
                ims[n] = im
                n += 1

    offsets = np.arange(
        dataset.acquire.gmin,
        dataset.acquire.gmax,
        dataset.acquire.dg,
        dtype=float,
    )
    offsets *= dataset.model.dh
    axs[0].images[0].set_extent(
        [offsets.min()/1000, offsets.max()/1000, time.max(), time.min()]
    )
    axs[0].invert_xaxis()
    for ax in axs[1:]:
        ax.images[0].set_extent(
            [cmps.min(), cmps.max(), time.max(), time.min()]
        )
    for ax in axs[:1:-NROWS+1]:
        ax.images[0].set_extent(
            [cmps.min(), cmps.max(), depth.max(), depth.min()]
        )
    for ax in axs:
        ax.tick_params(which='minor', length=2)
        ax.minorticks_on()

    for ax in axs:
        ax.set_title("")
        if ax.images:
            cbar = ax.images[-1].colorbar
            if cbar is not None:
                cbar.remove()

    axs[0].set_title("First CMP\ngather")
    axs[0].set_ylabel("$t$ (s)")
    axs[0].set_xlabel("$h$ (km)")

    axs[1].set_title("Constant-offset\ngather")
    axs[1].set_yticklabels([])
    axs[1].set_xlabel("$x$ (km)")

    vmin -= .05 * diff
    vmax += .05 * diff
    TO_SLICE = ['vrms', 'vint', 'vdepth']
    START_AX_IDX = [3, 4, 5]
    LINE_LABELS = ["Pretraining", "End estimate", "Ground truth"]
    ZORDERS = [2, 3, 1]
    line_axs = []
    for i, (label_name, start_idx) in enumerate(zip(TO_SLICE, START_AX_IDX)):
        line_ax = fig.add_subplot(gs[i+2, 7])
        line_axs.append(line_ax)
        for ax, label, zorder in zip(
            axs[start_idx:start_idx+3*4:4], LINE_LABELS, ZORDERS,
        ):
            data = ax.images[0].get_array()
            center_data = data[:, data.shape[1] // 2] / 1000
            if label_name != 'vdepth':
                y_min, y_max = time.min(), time.max()
                line_ax.plot(center_data, time, zorder=zorder, label=label)
            else:
                y_min, y_max = depth.min(), depth.max()
                line_ax.plot(center_data, depth, zorder=zorder, label=label)
            height = y_max-y_min
            x = cmps[data.shape[1]//2]
            dcmp = cmps[1] - cmps[2]
            rect = Rectangle(
                xy=(x-.5*dcmp, y_min+.01*height),
                width=dcmp,
                height=height*.98,
                ls=(0, (5, 5)),
                lw=.5,
                ec='w',
                fc='none',
            )
            ax.add_patch(rect)
        line_ax.set_xlim(vmin/1000, vmax/1000)
        line_ax.set_ylim(y_max, y_min)
        line_ax.set_yticklabels([])
        line_ax.grid()
        if i == 0:
            line_ax.legend(
                loc='lower center',
                bbox_to_anchor=(.5, 1.125),
                fontsize=6,
                handlelength=.2,
            )
        if i == len(TO_SLICE) - 1:
            line_ax.set_xlabel("Velocity (km/s)")
        else:
            line_ax.set_xticklabels([])

    gs.update(wspace=.15, hspace=.2)
    for ax in axs[:2]:
        box = ax.get_position()
        box.y0 += .05
        box.y1 += .05
        ax.set_position(box)
    TITLES = {
        'ref': "Primaries",
        'vrms': "$v_\\mathrm{RMS}(t, x)$",
        'vint': "$v_\\mathrm{int}(t, x)$",
        'vdepth': "$v_\\mathrm{int}(z, x)$",
    }
    for i, label_name in enumerate(TOOUTPUTS):
        axs[2+i].annotate(
            TITLES[label_name],
            (-.35, .5),
            xycoords="axes fraction",
            va="center",
            ha="center",
            rotation=90,
        )
        if label_name != 'vdepth':
            axs[2+i].set_ylabel("$t$ (s)")
        else:
            axs[2+i].set_ylabel("$z$ (km)")
    for ax in axs[2:2+NROWS]:
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    for i in range(NROWS-2):
        for j in range(NCOLS):
            axs[2+i+(NROWS-1)*j].set_xticklabels([])
    for i in range(NROWS-1):
        for j in range(1, NCOLS):
            axs[2+i+(NROWS-1)*j].set_yticklabels([])
    for i in range(NCOLS):
        axs[2+(i+1)*(NROWS-1)-1].set_xlabel("$x$ (km)")
    axs[2].set_title("Pretraining")
    axs[2+NROWS-1].set_title("End estimate")
    axs[2+2*(NROWS-1)].set_title("Ground truth")

    position = gs[1, 7].get_position(fig)
    left, bottom, width, height = position.bounds
    unpad_y = .4 * height
    unpad_x = .4 * width
    cax = fig.add_axes(
        [left, bottom+unpad_y, width-2*unpad_x, height-unpad_y]
    )
    cbar = plt.colorbar(axs[3].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity\n(km/s)")
    cbar.set_ticks(range(2000, 5000, 1000))
    cbar.set_ticklabels(range(2, 5, 1))

    temp_axs = [*axs[:2], *np.array(axs[2:]).reshape([3, 4]).T.flatten()]
    temp_axs.insert(-6, line_axs[0])
    temp_axs.insert(-3, line_axs[1])
    temp_axs.append(line_axs[2])
    for ax, letter in zip(temp_axs, range(ord('a'), ord('q')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(join(FIGURES_DIR, figure_name), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


def load_events(logdir):
    data = []
    for i in listdir(logdir):
        current_logdir = join(logdir, i)
        events_path = [
            path for path in listdir(current_logdir) if "events" in path
        ]
        assert len(events_path) == 1
        events_path = join(current_logdir, events_path[0])
        current_data = pd.DataFrame([])
        events = summary_iterator(events_path)
        for event in events:
            if hasattr(event, 'step'):
                step = event.step
                for value in event.summary.value:
                    column = value.tag
                    value = value.simple_value
                    current_data.loc[step, column] = np.log10(value)
        data.append(current_data)
    data = pd.concat(data)
    by_index = data.groupby(data.index)
    return by_index.mean(), by_index.std()


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
    plt.savefig(join(FIGURES_DIR, "losses.pdf"), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


def plot_real_data(dataset, plot=True):
    filename = join(dataset.basepath, dataset.name, "test", "example_1")
    inputs, outputs, _ = dataset.generator.read(filename)

    pretrained = dataset.generator.read_predictions(filename, "Pretraining")
    pretrained = {name: pretrained[name] for name in TOOUTPUTS}
    preds = dataset.generator.read_predictions(filename, "EndResults")
    preds = {name: preds[name] for name in TOOUTPUTS}

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
        pretrained_vint, axs=[axs[0]], vmin=1400, vmax=3500, cmap='jet',
    )
    vint_meta.plot(
        pred_vint, axs=[axs[1]], vmin=1400, vmax=3500, cmap='jet',
    )
    vint_meta.plot(
        pred_vdepth, axs=[axs[2]], vmin=1400, vmax=3500, cmap='jet',
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

    plt.savefig(join(FIGURES_DIR, "real_models.pdf"), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


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
    shotgather = data_meta.preprocess(shotgather, None)
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
    pred_stacked = stack_2d(shotgather, times, offsets, pred_vrms)
    pred_stacked = data_preprocess(pred_stacked)

    pred_stacked = pred_stacked[crop_top:crop_bottom]
    stacked_usgs = stacked_usgs[crop_top:crop_bottom]

    data_meta.plot(pred_stacked, axs=[axs[0]], vmin=0, clip=2.5E-1)
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

    plt.savefig(join(FIGURES_DIR, "real_stacks.pdf"), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


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
    preds = {name: preds[name] for name in TOOUTPUTS}
    preds_std = dataset.generator.read_predictions(filename, "EndResults_std")
    preds_std = {name: preds_std[name] for name in TOOUTPUTS}
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
        ncols=2,
        figsize=[3.33, 8],
        sharex='col',
        sharey=True,
        gridspec_kw={'wspace': .01},
    )
    for i, cmp in enumerate([250, 1000, 1750]):
        temp_shotgather = shotgather[..., cmp, 0]
        # temp_shotgather *= 1 / np.sqrt(offsets[None, :])
        temp_shotgather *= times[:, None]**2
        temp_shotgather /= np.amax(temp_shotgather)
        vmax = 5E-2
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
            axs[i, 1].plot(pred, times, lw=.5, color=color)
            axs[i, 1].fill_betweenx(
                times,
                pred-std,
                pred+std,
                color=color,
                lw=0,
                alpha=.4,
            )
        axs[i, 1].set_xlim([velocities.min() / 1000, velocities.max() / 1000])

    for ax in axs.flatten():
        plt.sca(ax)
        plt.minorticks_on()
        # plt.grid(True, which='major', c='tab:red', alpha=.4)
        # plt.grid(True, which='minor', c='tab:red', alpha=.4)

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
    for ax in axs[:, 0]:
        ax.set_ylabel("$t$ (s)")

    for ax, letter in zip(axs.flatten(), range(ord('a'), ord('f')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(join(FIGURES_DIR, "semblance.pdf"), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


def data_preprocess(data):
    eps = np.finfo(np.float32).eps
    trace_rms = np.sqrt(np.sum(data**2, axis=0, keepdims=True))
    data /= trace_rms + eps
    data = np.expand_dims(data, axis=-1)
    return data


def stack_2d(cmps, times, offsets, velocities):
    stacked = []
    cmps = cmps.transpose([2, 0, 1])
    velocities = velocities.T
    for i, (cmp, velocities_1d) in enumerate(zip(cmps, velocities)):
        stacked.append(stack(cmp, times, offsets, velocities_1d))
    stacked = np.array(stacked).T
    return stacked


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
