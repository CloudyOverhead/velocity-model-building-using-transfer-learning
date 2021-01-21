# -*- coding: utf-8 -*-
"""Define parameters for different Datasets"""

from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import Reftime, Vrms, Vint, Vdepth, ShotGather


class GeoDataset(GeoDataset):
    def plot_example(
        self, filename=None, phase='train', toinputs=None, tooutputs=None,
        plot_preds=False, apply_weights=True, nn_name=None, ims=None,
    ):
        fig, axs, ims = super().plot_example(
            filename, phase, toinputs, tooutputs, plot_preds, apply_weights,
            nn_name, ims,
        )

        cols_meta = [self.outputs]
        if not apply_weights:
            weights_meta = deepcopy(self.outputs)
            for output in weights_meta.values():
                output.meta_name = "Weights"
            cols_meta.append(weights_meta)
        if plot_preds:
            preds_meta = deepcopy(self.outputs)
            for output in preds_meta.values():
                output.meta_name = "Predictions"
            cols_meta.append(preds_meta)
        ncols = len(cols_meta)
        ims_per_row = [
            sum(col[name].naxes for name in col) for col in cols_meta
        ]
        qty_ims = sum(ims_per_row)
        nrows = np.lcm.reduce(ims_per_row)

        mask = np.isnan(ims[3].get_array())
        for im in [ims[1], ims[2], ims[2+nrows]]:
            data = im.get_array()
            data[mask] = np.nan
            im.set_array(data)

        if fig is not None:
            fig.set_size_inches(16, 10)
            for ax in axs:
                ax.set_title("")
                cbar = ax.images[-1].colorbar
                if cbar is not None:
                    cbar.remove()
            gs = fig.add_gridspec(
                ncols=ncols+3,
                nrows=nrows,
                width_ratios=[1, .2, *(1 for _ in range(ncols)), .5],
            )

            axs[0].set_position([0, 0, 0, 0])

            axs[1].set_position(gs[:, 0].get_position(fig))
            axs[1].set_subplotspec(gs[:, 0])
            axs[1].set_title("Zero offset gather")
            axs[1].set_ylabel("Two-way traveltime")
            axs[1].set_xlabel("Shot")

            # Transpose axes positions.
            for i in range(nrows):
                for j in range(ncols):
                    current_gs = gs[i, j+2]
                    ax_id = 2 + i + nrows*j
                    axs[ax_id].set_position(current_gs.get_position(fig))
                    axs[ax_id].set_subplotspec(current_gs)

            for i in range(ncols):
                for ax in axs[2+1+i*nrows:2+(i+1)*nrows]:
                    axs[2+i*nrows].get_shared_x_axes().join(axs[2+i*nrows], ax)

            vmin, vmax = self.model.properties["vp"]
            diff = vmax - vmin
            vmin -= .05 * diff
            vmax += .05 * diff
            for i, label_name in enumerate(self.outputs.keys()):
                label_ax = axs[2+i]
                if plot_preds:
                    pred_ax = axs[2+i+nrows]
                    label_ax.get_shared_y_axes().join(label_ax, pred_ax)
                if label_name != 'ref':
                    label = label_ax.images[0].get_array()
                    qty_shots = label.shape[-1]
                    center_label = label[:, qty_shots//2]
                    y_min, y_max = label_ax.get_ylim()
                    y = np.linspace(y_max, y_min, len(center_label))
                    ax = plt.subplot(gs[i, 2+ncols])
                    ax.plot(center_label, y, label="Expected values")
                    if plot_preds:
                        pred = pred_ax.images[0].get_array()
                        center_pred = pred[:, qty_shots//2]
                        ax.plot(center_pred, y, label="Estimated values")
                    ax.set_xlim(vmin, vmax)
                    ax.set_ylim(y_max, y_min)
                    ax.set_yticklabels([])
                    ax.invert_yaxis()
                    label_ax.get_shared_y_axes().join(label_ax, ax)
                    if "prev_ax" in locals().keys():
                        ax.get_shared_x_axes().join(ax, prev_ax)
                    if i == 1:
                        ax.legend(loc="upper center", fontsize=6)
                        ax.set_title(
                            f"Center slice\n({qty_shots//2+1} out of "
                            f"{qty_shots})"
                        )
                    if i == len(self.outputs) - 1:
                        ax.set_xlabel("Velocity [m/s]")
                    else:
                        ax.set_xticklabels([])
                    prev_ax = ax

            plt.tight_layout()
            TITLES = {
                'ref': "Primaries identification",
                'vrms': "$v_\\mathrm{RMS}(t, x)$",
                'vint': "$v_\\mathrm{int}(t, x)$",
                'vdepth': "$v_\\mathrm{int}(z, x)$",
            }
            for i, label_name in enumerate(self.outputs.keys()):
                axs[2+i].annotate(
                    TITLES[label_name],
                    (-.325, .5),
                    xycoords="axes fraction",
                    va="center",
                    ha="center",
                    rotation=90,
                )
                axs[2+i].set_ylabel("Two-way traveltime")
            for ax in axs[2:2+nrows]:
                ax.yaxis.set_tick_params(which='both', labelleft=True)
            for i in range(nrows-1):
                for j in range(ncols):
                    axs[2+i+nrows*j].set_xticklabels([])
            for i in range(nrows):
                for j in range(1, ncols):
                    axs[2+i+nrows*j].set_yticklabels([])
            for i in range(ncols):
                axs[2+(i+1)*nrows-1].set_xlabel("Shot")
            axs[2].set_title("Expected values")
            if plot_preds:
                axs[2+nrows].set_title("Estimated values")
            position = gs[0, 2+ncols].get_position(fig)
            left, bottom, width, height = position.bounds
            unpad_y = .3 * height
            unpad_x = .4 * width
            cax = fig.add_axes(
                [left, bottom+unpad_y, width-2*unpad_x, height-unpad_y]
            )
            cbar = plt.colorbar(axs[3].images[0], cax=cax)
            cbar.ax.set_ylabel("Velocity [m/s]")
            cbar.set_ticks(range(2000, 5000, 1000))
        return fig, axs, ims


class Dataset1DArticle(GeoDataset):
    name = "Dataset1DArticle"

    def set_dataset(self):
        self.trainsize = 10000
        self.validatesize = 0
        self.testsize = 100

        model = MarineModel()
        model.NX = 500
        model.NZ = 300
        model.dh = dh = 10
        model.water_dmin = 300
        model.water_dmax = 500
        model.layer_num_min = 48
        model.layer_dh_min = 20
        model.layer_dh_max = 50
        model.water_vmin = 1430
        model.water_vmax = 1560
        model.vp_min = 1300.0
        model.vp_max = 4000.0

        acquire = Acquisition(model=model)
        acquire.dt = 0.0008
        acquire.NT = 2560
        acquire.resampling = 10
        acquire.dg = 6
        acquire.gmin = acquire.Npad + 4*acquire.dg
        acquire.gmax = model.NX - acquire.gmin - acquire.Npad
        acquire.peak_freq = 26
        acquire.df = 5
        acquire.wavefuns = [0, 1]
        acquire.source_depth = (acquire.Npad+4) * dh
        acquire.receiver_depth = (acquire.Npad+4) * dh
        acquire.tdelay = 2.0 / (acquire.peak_freq-acquire.df)
        acquire.singleshot = True

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {Reftime.name: Reftime(model=model, acquire=acquire),
                   Vrms.name: Vrms(model=model, acquire=acquire),
                   Vint.name: Vint(model=model, acquire=acquire),
                   Vdepth.name: Vdepth(model=model, acquire=acquire)}
        for name in inputs:
            inputs[name].train_on_shots = True
            inputs[name].mute_dir = True
        for name in outputs:
            outputs[name].train_on_shots = True
            outputs[name].identify_direct = False

        return model, acquire, inputs, outputs

    def __init__(self, noise=0):
        if noise:
            self.name = self.name + "_noise"

        super().__init__()
        if noise:
            for name in self.inputs:
                self.inputs[name].random_static = True
                self.inputs[name].random_static_max = 1
                self.inputs[name].random_noise = True
                self.inputs[name].random_noise_max = 0.02


class Dataset1DArticleMoreRecs(Dataset1DArticle):
    name = "Dataset1DArticleMoreRecs"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        acquire.dg = 4
        return model, acquire, inputs, outputs


class Dataset2DArticle(Dataset1DArticle):
    name = "Dataset2DArticle"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        model.max_deform_freq = 0.06
        model.min_deform_freq = 0.0001
        model.amp_max = 8
        model.max_deform_nfreq = 40
        model.prob_deform_change = 0.7
        model.dip_max = 10
        model.ddip_max = 4

        acquire.ds = 16
        acquire.singleshot = False

        return model, acquire, inputs, outputs


class Dataset2DArticleMoreRecs(Dataset2DArticle):
    name = "Dataset2DArticleMoreRecs"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()
        acquire.dg = 4
        # for name in inputs:
        #     inputs[name].train_on_shots = False
        # for name in outputs:
        #     outputs[name].train_on_shots = False
        return model, acquire, inputs, outputs


class USGS(GeoDataset):
    name = "USGS"

    def set_case(self):
        model, acquire, inputs, outputs = super().__init__()
        self.trainsize = 0
        self.validatesize = 0
        self.testsize = 1
        return model, acquire, inputs, outputs


class Mercier(GeoDataset):
    name = "Mercier"

    def set_case(self):
        model, acquire, inputs, outputs = super().__init__()
        self.trainsize = 41
        self.validatesize = 0
        self.testsize = 10
        return model, acquire, inputs, outputs


if __name__ == "__main__":
    dataset = Dataset1DArticle()
    dataset.model.animated_dataset()

    dataset = Dataset2DArticle()
    dataset.model.animated_dataset()
