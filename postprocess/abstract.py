# -*- coding: utf-8 -*-

import os
from argparse import Namespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from Cases_define import Case2Dtest_complexity
from vrmslearn.Tester import Tester
from vrmslearn.Sequence import Sequence


OUT_NAMES = ["ref", "vrms", "vint"]

case = Case2Dtest_complexity()
sizes = case.get_dimensions()
sequence = Sequence(is_training=False,
                    case=case,
                    batch_size=2,
                    input_size=sizes[0],
                    depth_size=sizes[-1][0],
                    out_names=OUT_NAMES)
nn = Namespace()
nn.out_names = OUT_NAMES

tester = Tester(nn=nn, sequence=sequence, case=case)
savepath = os.path.join(case.datatest, "pred")
is_2d = sizes[0][2] != 1

examples = [os.path.basename(f) for f in case.files["test"]
            if os.path.basename(f) in os.listdir(savepath)]

labels, preds = tester.get_preds(OUT_NAMES, savepath, examples=examples)
datas = labels['input']
datas = [np.reshape(el, [el.shape[0], -1]) for el in datas]

rmse = []
for label, pred, weight in zip(labels["vrms"], preds["vrms"],
                               labels["tweight"]):
    temp_label = label * weight
    temp_pred = pred * weight
    count = np.count_nonzero(weight)
    rmse.append(np.sqrt(np.sum((temp_label-temp_pred)**2) / count))
rmse = np.array(rmse)

vmin = case.model.vp_min
vmax = case.model.vp_max
rmse *= vmax - vmin

p_score = np.percentile(rmse, 90, interpolation="nearest")
p_idx = np.argwhere(p_score == rmse)[0, 0]


def plot(case, labels, preds, data):
    fig, axs = plt.subplots(nrows=3,
                            ncols=5,
                            figsize=(10, 6),
                            squeeze=False,
                            sharex="col",
                            sharey=True,
                            gridspec_kw={'width_ratios': [1, .2, 1, 1, .5]})
    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, 0])
    clip = 0.01
    vmax = np.max(datas) * clip
    vmin = -vmax

    src_pos, _ = case.acquire.set_rec_src()
    qty_shots = src_pos.shape[1]
    data = data.reshape([data.shape[0], -1, qty_shots])
    print(data.shape)
    labeld = {la: labels[la] for la in OUT_NAMES}
    predd = {la: preds[la] for la in OUT_NAMES}
    label, pred = case.label.postprocess(labeld, predd)
    weight = labels["tweight"].astype(bool)
    valid_rows = weight.any(axis=1)
    last_idx = np.nonzero(~valid_rows)[0].min()
    dt = case.acquire.dt
    resampling = case.acquire.resampling
    tdelay = case.acquire.tdelay
    first_idx = round(tdelay / (dt*resampling))

    axbig.imshow(data[..., qty_shots//2],
                 vmin=vmin,
                 vmax=vmax,
                 aspect='auto',
                 cmap=plt.get_cmap('Greys'))
    title = f"Center shot gather\n({qty_shots//2+1} out of {qty_shots})"
    axbig.set_title(title)
    rect = Rectangle((0, first_idx), data.shape[1]-1, last_idx-first_idx,
                     linewidth=2, linestyle=":", edgecolor='r',
                     facecolor='none')
    # Add the patch to the Axes
    axbig.add_patch(rect)
    axbig.set_ylabel("Two-way traveltime")
    axbig.set_xlabel("Receiver")
    for ii, labelname in enumerate(OUT_NAMES):
        pred[labelname] = pred[labelname].astype(float)
        pred[labelname] = pred[labelname].astype(float)
        pred[labelname][~weight] = np.nan
        label[labelname][~weight] = np.nan
        if labelname == "ref":
            vmin, vmax = -.2, 1
        else:
            vmin = case.model.vp_min
            vmax = case.model.vp_max
        y = np.arange(pred[labelname].shape[0])

        if labelname == "ref":
            cmap = "Greys"
        else:
            cmap = "inferno"
        axs[ii, 2].imshow(label[labelname],
                          vmin=vmin,
                          vmax=vmax,
                          animated=True,
                          cmap=cmap,
                          aspect='auto')
        axs[ii, 3].imshow(pred[labelname],
                          vmin=vmin,
                          vmax=vmax,
                          animated=True,
                          cmap=cmap,
                          aspect='auto')
        if labelname != "ref":
            center_label = label[labelname][:, qty_shots//2]
            center_pred = pred[labelname][:, qty_shots//2]
            axs[ii, 4].plot(center_label, y, label="Expected values")
            axs[ii, 4].plot(center_pred, y, label="Estimated values")
            axs[ii, 4].set_xlim(vmin, vmax)
            axs[ii, 4].invert_yaxis()
            axs[ii, 4].legend(loc="upper center", fontsize=6)

    plt.tight_layout()
    for ii, labelname in enumerate(["Primaries identification",
                                    "$\\mathbf{v_\\mathrm{RMS}}$",
                                    "$\\mathbf{v_\\mathrm{int}}$"]):
        axs[ii, 2].annotate(labelname, (-.325, .5), fontweight="bold",
                            xycoords="axes fraction",
                            va="center",
                            ha="center", rotation=90)
        axs[ii, 2].set_ylabel("Two-way traveltime")
    for ax in axs.flatten():
        ax.set_ylim(last_idx, first_idx)
    for ax in axs[:, 2]:
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    for ax in axs[:, 1]:
        ax.remove()
    for ax in axs[2, [2, 3]]:
        ax.set_xlabel("Shot")
    axs[0, 2].set_title("Expected values")
    axs[0, 3].set_title("Estimated values")
    axs[1, 4].set_title(f"Center slice\n({qty_shots//2+1} out of {qty_shots})")
    axs[2, 4].set_xlabel("Velocity [m/s]")
    left, bottom, width, height = axs[0, 4].get_position().bounds
    unpad_y = .3 * height
    unpad_x = .4 * width
    cax = fig.add_axes([left, bottom+unpad_y, width-2*unpad_x, height-unpad_y])
    cbar = plt.colorbar(axs[1, 2].images[0], cax=cax)
    cbar.ax.set_ylabel("Velocity [m/s]")
    cbar.set_ticks(range(2000, 5000, 1000))
    axs[0, 4].remove()
    plt.savefig("abstract_figure", bbox_inches="tight")
    plt.show()


current_labels = {out_name: labels[out_name][p_idx]
                  for out_name in labels.keys()}
current_preds = {out_name: preds[out_name][p_idx]
                 for out_name in preds.keys()}
current_data = datas[p_idx]
print(p_score, min(rmse), max(rmse))
plot(case, current_labels, current_preds, current_data)
