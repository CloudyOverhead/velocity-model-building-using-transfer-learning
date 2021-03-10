# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace
from os import makedirs, listdir, pardir
from os.path import join, exists

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim
from tensorflow.compat.v1.train import summary_iterator
from GeoFlow.SeismicUtilities import sortcmp

from main import main as global_main, int_or_list
from core.architecture import RCNN2D, Hyperparameters1D, Hyperparameters2D
from deep_learning_velocity_estimation.datasets import Article2D

FIGURES_DIR = join(pardir, "figures")
TOINPUTS = ['shotgather']
TOOUTPUTS = ['ref', 'vrms', 'vint']

plt.rcParams.update(
    {'font.size': 8, 'figure.figsize': [4.33, 2.5], 'figure.dpi': 1200}
)


def main(args):
    dataset = Article2D()
    dataset._getfilelist()

    if not exists(FIGURES_DIR):
        makedirs(FIGURES_DIR)

    if not args.no_inference:
        launch_inference(args, dataset)

    inputs, labels, weights, preds, similarities = compare_preds(dataset)

    for percentile in [10, 50, 90]:
        score = np.percentile(
            similarities, percentile, interpolation="nearest",
        )
        idx = np.argwhere(score == similarities)[0, 0]
        print(f"SSIM {percentile}th percentile:", score)
        plot_example(
            args,
            dataset=dataset,
            filename=dataset.files["test"][idx],
            plot=args.plot,
        )
        if percentile == 50:
            plot_transfer_learning(
                plot_idx=idx,
                plot=args.plot,
            )
    plot_losses(
        logdir_1d=args.logdir_1d,
        params_1d=Hyperparameters1D(is_training=True),
        logdir_2d=args.logdir_2d,
        params_2d=Hyperparameters2D(is_training=True),
        plot=args.plot,
    )


def launch_inference(args, dataset):
    for logdir, savedir in zip(
        [args.logdir_1d, args.logdir_2d], ["Pretraining", "PostTraining"],
    ):
        args = Namespace(
            nn=RCNN2D,
            params=Hyperparameters2D(is_training=False),
            dataset=dataset,
            logdir=logdir,
            training=3,
            gpus=args.gpus,
            savedir=savedir,
            plot=False,
            debug=False,
            eager=False,
        )
        global_main(args)


def plot_example(args, dataset, filename, plot=True):
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
    preds = dataset.generator.read_predictions(filename, "PostTraining")
    preds = {name: preds[name] for name in TOOUTPUTS}
    cols = [inputs, pretrained, preds, labels]

    ref = labels['ref']
    crop_top = int(np.nonzero(ref.astype(bool).any(axis=1))[0][0] * .95)
    for col in [*cols, weights]:
        for row_name, row in col.items():
            col[row_name] = row[crop_top:]

    dt = dataset.acquire.dt * dataset.acquire.resampling
    water_dmin = dataset.model.water_dmin
    water_vmin = dataset.model.water_vmin
    tdelay = dataset.acquire.tdelay
    start_time = crop_top*dt + 2*water_dmin/water_vmin - tdelay
    time = np.arange(len(labels['ref']))*dt + start_time

    src_pos, rec_pos = dataset.acquire.set_rec_src()
    _, cmps = sortcmp(None, src_pos, rec_pos)
    cmps /= 1000

    NROWS = 4
    QTY_IMS = 11
    NCOLS = 3

    fig = plt.figure(figsize=[6.66, 6], constrained_layout=False)
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

    mask = weights['vrms']
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
            output_ims = col_meta[row_name].plot(
                data, weights=mask, axs=input_axs, ims=input_ims,
            )
            for im in output_ims:
                ims[n] = im
                n += 1

    axs[0].images[0].set_extent(
        [rec_pos[0].min()/1000, rec_pos[0].max()/1000, time.max(), time.min()]
    )
    for ax in axs[1:]:
        ax.images[0].set_extent(
            [cmps.min(), cmps.max(), time.max(), time.min()]
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
    axs[0].set_ylabel("Two-way\ntraveltime (s)")
    axs[0].set_xlabel("Receiver position (km)")

    axs[1].set_title("Nearest offset\ngather")
    axs[1].set_yticklabels([])
    axs[1].set_xlabel("CMP (km)")

    vmin, vmax = dataset.model.properties['vp']
    diff = vmax - vmin
    vmin -= .05 * diff
    vmax += .05 * diff
    TO_SLICE = ['vint', 'vrms']
    START_AX_IDX = [3, 4]
    LINE_LABELS = ["Pretraining", "Post-training", "Expected"]
    line_axs = []
    for i, (label_name, start_idx) in enumerate(zip(TO_SLICE, START_AX_IDX)):
        line_ax = fig.add_subplot(gs[i+2, 7])
        line_axs.append(line_ax)
        for ax, label in zip(axs[start_idx:start_idx+3*3:3], LINE_LABELS):
            data = ax.images[0].get_array()
            center_data = data[:, data.shape[1] // 2]
            y_min, y_max = time.min(), time.max()
            line_ax.plot(center_data, time, label=label)
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
        line_ax.set_xlim(vmin, vmax)
        line_ax.set_ylim(y_max, y_min)
        line_ax.set_yticklabels([])
        line_ax.grid()
        if i == 0:
            line_ax.legend(
                loc='lower right',
                bbox_to_anchor=(1.1, 1.),
                fontsize=4,
                handlelength=.25,
            )
        if i == len(TO_SLICE) - 1:
            line_ax.set_xlabel("Velocity (m/s)")
        else:
            line_ax.set_xticklabels([])

    gs.update(wspace=.15, hspace=.2)
    for ax in axs[:2]:
        box = ax.get_position()
        box.y0 += .08
        box.y1 += .08
        ax.set_position(box)
    TITLES = {
        'ref': "Primaries\nidentification",
        'vrms': "$v_\\mathrm{RMS}(t, x)$",
        'vint': "$v_\\mathrm{int}(t, x)$",
    }
    for i, label_name in enumerate(TOOUTPUTS):
        axs[2+i].annotate(
            TITLES[label_name],
            (-.65, .5),
            xycoords="axes fraction",
            va="center",
            ha="center",
            rotation=90,
        )
        axs[2+i].set_ylabel("Two-way\ntraveltime (s)")
    for ax in axs[2:2+NROWS]:
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    for i in range(NROWS-2):
        for j in range(NCOLS):
            axs[2+i+(NROWS-1)*j].set_xticklabels([])
    for i in range(NROWS-1):
        for j in range(1, NCOLS):
            axs[2+i+(NROWS-1)*j].set_yticklabels([])
    for i in range(NCOLS):
        axs[2+(i+1)*(NROWS-1)-1].set_xlabel("CMP (km)")
    axs[2].set_title("Pretraining")
    axs[2+NROWS-1].set_title("Post-training")
    axs[2+2*(NROWS-1)].set_title("Expected")

    position = gs[1, 7].get_position(fig)
    left, bottom, width, height = position.bounds
    unpad_y = .3 * height
    unpad_x = .4 * width
    cax = fig.add_axes(
        [left, bottom+unpad_y, width-2*unpad_x, height-unpad_y]
    )
    cbar = plt.colorbar(axs[3].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity\n(m/s)")
    cbar.set_ticks(range(2000, 5000, 1000))

    temp_axs = [*axs[:2], *np.array(axs[2:]).reshape([3, 3]).T.flatten()]
    temp_axs.insert(-3, line_axs[0])
    temp_axs.append(line_axs[1])
    for ax, letter in zip(temp_axs, range(ord('a'), ord('m')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(
        join(FIGURES_DIR, "results_synthetic.pdf"), bbox_inches="tight",
    )
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


def compare_preds(dataset):
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
        preds = dataset.generator.read_predictions(filename, "PostTraining")
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
    for labels, weights, preds in zip(
        all_labels["vint"], all_weights["vint"], all_preds["vint"],
    ):
        temp_labels = labels * weights
        temp_preds = preds * weights
        similarity = ssim(temp_labels, temp_preds)
        similarities = np.append(similarities, similarity)

    print("Average SSIM:", np.mean(similarities))
    print("Standard deviation on SSIM:", np.std(similarities))

    return all_inputs, all_labels, all_weights, all_preds, similarities


def plot_transfer_learning(plot_idx, plot=True):
    plt.savefig(
        join(FIGURES_DIR, "transfer_learning.pdf"), bbox_inches="tight",
    )
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


def load_events(logdir):
    events_path = [path for path in listdir(logdir) if "events" in path]
    assert len(events_path) == 1
    events_path = join(logdir, events_path[0])
    data = pd.DataFrame([])
    events = summary_iterator(events_path)
    for event in events:
        if hasattr(event, 'step'):
            step = event.step
            for value in event.summary.value:
                column = value.tag
                value = value.simple_value
                data.loc[step, column] = value
    return data


def plot_losses(logdir_1d, params_1d, logdir_2d, params_2d, plot=True):
    data_1d = load_events(logdir_1d)
    data_2d = load_events(logdir_2d)
    data = pd.concat([data_1d, data_2d], ignore_index=True)
    qty_stages_1d = len(params_1d.loss_scales)
    qty_stages_2d = len(params_2d.loss_scales)
    epochs_1d = (params_1d.epochs,) * qty_stages_1d
    epochs_2d = (params_2d.epochs,) * qty_stages_2d
    epochs = epochs_1d + epochs_2d

    LABEL_NAMES = {
        'loss': "Total loss",
        'ref_loss': "Primaries identification",
        'vrms_loss': "$v_\\mathrm{RMS}$",
        'vint_loss': "$v_\\mathrm{int}$",
    }
    data.columns = [column.split('/')[-1] for column in data.columns]
    for column in data.columns:
        if column not in LABEL_NAMES.keys():
            del data[column]
    for column in LABEL_NAMES.keys():
        if column == 'loss':
            plt.plot(
                data[column], label=LABEL_NAMES[column], zorder=100, lw=2.5,
            )
        else:
            plt.plot(data[column], label=LABEL_NAMES[column])
    limits = np.cumsum((0,) + epochs)
    limits[0] = 1
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
    vmax, vmin = data.values.max(), data.values.min()
    diff = np.log10(vmax) - np.log10(vmin)
    plt.ylim([10**(np.log10(vmin)-.1*diff), 10**(np.log10(vmax)+.1*diff)])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='lower center', bbox_to_anchor=(.5, 1), ncol=4)
    plt.minorticks_on()
    plt.grid(which='major', alpha=.6)
    plt.grid(which='minor', alpha=.15)
    plt.savefig(join(FIGURES_DIR, "losses.pdf"), bbox_inches="tight")
    if plot:
        plt.gcf().set_dpi(200)
        plt.show()
    else:
        plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir_1d",
        type=str,
        help="Directory in which the checkpoints for the 1D case are stored.",
    )
    parser.add_argument(
        "--logdir_2d",
        type=str,
        help="Directory in which the checkpoints for the 2D case are stored.",
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
        "--no_inference",
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
