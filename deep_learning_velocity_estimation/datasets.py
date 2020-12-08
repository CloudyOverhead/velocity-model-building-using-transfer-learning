# -*- coding: utf-8 -*-
"""Define parameters for different Datasets"""

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import Reftime, Vrms, Vint, Vdepth, ShotGather


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
        acquire.dg = 12
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

        acquire.ds = 24
        acquire.singleshot = False

        return model, acquire, inputs, outputs


class Mercier(GeoDataset):
    name = "Mercier"

    def set_case(self):
        model, acquire, label = super().__init__()
        self.trainsize = 41
        self.validatesize = 0
        self.testsize = 10

        return model, acquire, label


if __name__ == "__main__":
    dataset = Dataset1DArticle()
    dataset.model.animated_dataset()

    dataset = Dataset2DArticle()
    dataset.model.animated_dataset()
