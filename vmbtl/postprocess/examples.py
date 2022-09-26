# -*- coding: utf-8 -*-

from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import compare_ssim as ssim
from GeoFlow.SeismicUtilities import sortcmp, semblance_gather

from vmbtl.postprocess.constants import IGNORE_IDX


def plot_example(dataset, filename, figure_name, plot=True):
    inputs, labels, weights, filename = dataset.get_example(
        filename=filename, phase='test',
    )
    inputs_meta = dataset.inputs
    outputs_meta = dataset.outputs
    cols_meta = [inputs_meta, outputs_meta, outputs_meta, outputs_meta]

    pretrained = dataset.generator.read_predictions(filename, "Pretraining")
    pretrained_std = dataset.generator.read_predictions(
        filename, "Pretraining_std",
    )
    preds = dataset.generator.read_predictions(filename, "EndResults")
    preds_std = dataset.generator.read_predictions(
        filename, "EndResults_std",
    )
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
    for col in [*cols, weights, pretrained_std, preds_std]:
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
    cmps = cmps[10:-10]
    cmps /= 1000
    depth /= 1000

    NROWS = 5
    QTY_IMS = 14
    NCOLS = 3

    fig = plt.figure(figsize=[6.5, 7.5], constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=NROWS,
        ncols=NCOLS*2+3,
        width_ratios=[.2, *(.5 for _ in range(NCOLS*2+1)), .15],
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
            if row_name == 'vrms':
                vmax_ = 2500
            else:
                vmax_ = None
            output_ims = col_meta[row_name].plot(
                data,
                weights=mask,
                axs=input_axs,
                ims=input_ims,
                vmax=vmax_,
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
    STDS = [pretrained_std, preds_std, None]
    line_axs = []
    for i, (label_name, start_idx) in enumerate(zip(TO_SLICE, START_AX_IDX)):
        line_ax = fig.add_subplot(gs[i+2, 7])
        line_axs.append(line_ax)
        for ax, label, zorder, std in zip(
            axs[start_idx:start_idx+3*4:4], LINE_LABELS, ZORDERS, STDS,
        ):
            data = ax.images[0].get_array()
            center_data = data[:, data.shape[1] // 2] / 1000
            if std is not None:
                std = std[label_name]
                std = std * diff
                center_std = std[:, data.shape[1] // 2] / 1000
            if label_name != 'vdepth':
                y_min, y_max = time.min(), time.max()
                y_values = time
            else:
                y_min, y_max = depth.min(), depth.max()
                y_values = depth
            line_ax.plot(
                center_data, y_values, lw=.5, zorder=zorder, label=label,
            )
            if std is not None:
                line_ax.fill_betweenx(
                    y_values,
                    center_data-center_std,
                    center_data+center_std,
                    lw=0,
                    alpha=.4,
                )
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
        if label_name == 'vrms':
            gather = inputs['shotgather'][:, :, data.shape[1] // 2, 0]
            velocities = np.linspace(vmin, vmax, 100)
            line_ax.imshow(
                semblance_gather(gather, time, offsets, velocities),
                aspect='auto',
                cmap='Greys',
                extent=[vmin/1000, vmax/1000, y_max, y_min],
                alpha=.8,
            )
        line_ax.set_xlim(vmin/1000, vmax/1000)
        line_ax.set_ylim(y_max, y_min)
        line_ax.set_yticklabels([])
        line_ax.grid()
        if i == 0:
            legend = line_ax.legend(
                loc='lower center',
                bbox_to_anchor=(.6, 1.125),
                fontsize=6,
                handlelength=.2,
            )
            for line in legend.get_lines():
                line.set_linewidth(2)
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
    for i, label_name in enumerate(TITLES.keys()):
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

    ticks = [
        np.arange(1500, 2500, 500),
        np.arange(2000, 5000, 1000),
        np.arange(2000, 5000, 1000),
    ]
    for i, current_ticks in enumerate(ticks):
        cax = fig.add_subplot(gs[i+2, -1])
        cbar = plt.colorbar(axs[i+3].images[0], cax=cax)
        cbar.ax.set_ylabel("Velocity (km/s)")
        cbar.set_ticks(current_ticks)
        cbar.set_ticklabels(current_ticks/1000)

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

    plt.savefig(figure_name, plot=plot)


def plot_ensemble(dataset, output_name, filename, plot):
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
        filename=filename,
        phase='test',
    )
    label = labels[output_name]
    weight = weights[output_name]

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
            vmin, vmax = None, None
        else:
            vmin, vmax = dataset.model.properties["vp"]
            array = array * (vmax-vmin)
            vmin, vmax = 0, 1000
            cmap = 'afmhot_r'
        meta.plot(
            array,
            weights=weight,
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
    cbar.ax.yaxis.set_major_formatter(lambda x, _: str(round(x/1000, 1)))

    cbar = plt.colorbar(axs[0, 0].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity\n(km/s)")
    cbar.ax.yaxis.set_major_formatter(lambda x, _: str(round(x/1000)))

    for ax, letter in zip(axs.flatten(), range(ord('a'), ord('g')+1)):
        letter = f"({chr(letter)})"
        plt.sca(ax)
        x0, _ = plt.xlim()
        y1, y0 = plt.ylim()
        height = y1 - y0
        plt.text(x0, y0-.02*height, letter, va='bottom')

    plt.savefig(f"ensemble_{output_name}", plot=plot)
