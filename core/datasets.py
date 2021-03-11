# -*- coding: utf-8 -*-
"""Define parameters for different datasets."""

from os.path import abspath

from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import Reftime, Vrms, Vint, Vdepth, ShotGather


class Dataset(GeoDataset):
    basepath = abspath("datasets")


class Article1D(Dataset):
    name = "Article1D"

    def set_dataset(self):
        self.trainsize = 10000
        self.validatesize = 0
        self.testsize = 100

        model = MarineModel()
        model.dh = 6.25
        model.NX = 692 * 2
        model.NZ = 752 * 2
        model.layer_num_min = 48
        model.layer_dh_min = 20
        model.layer_dh_max = 50
        model.water_vmin = 1430
        model.water_vmax = 1560
        model.water_dmin = .9 * model.water_vmin
        model.water_dmax = 3.1 * model.water_vmax
        model.vp_min = 1300.0
        model.vp_max = 4000.0

        acquire = Acquisition(model=model)
        acquire.dt = .0004
        acquire.NT = int(8 / acquire.dt)
        acquire.resampling = 10
        acquire.dg = 8
        acquire.ds = 8
        acquire.gmin = int(470 / model.dh)
        acquire.gmax = int((470+72*acquire.dg*model.dh) / model.dh)
        acquire.minoffset = 470
        acquire.peak_freq = 26
        acquire.df = 5
        acquire.wavefuns = [0, 1]
        acquire.source_depth = 6.4  # Approximate average value.
        acquire.receiver_depth = 13.7  # Approximate average value.
        acquire.tdelay = 3.0 / (acquire.peak_freq-acquire.df)
        acquire.singleshot = True
        acquire.configuration = 'inline'

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {Reftime.name: Reftime(model=model, acquire=acquire),
                   Vrms.name: Vrms(model=model, acquire=acquire),
                   Vint.name: Vint(model=model, acquire=acquire),
                   Vdepth.name: Vdepth(model=model, acquire=acquire)}

        for name in inputs:
            inputs[name].train_on_shots = True  # 1D shots are CMPs.
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


class Article2D(Article1D):
    name = "Article2D"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 1000
        self.validatesize = 0
        self.testsize = 100

        model.max_deform_freq = .06
        model.min_deform_freq = .0001
        model.amp_max = 8
        model.max_deform_nfreq = 40
        model.prob_deform_change = .7
        model.dip_max = 10
        model.ddip_max = 4

        acquire.singleshot = False
        for name in inputs:
            inputs[name].train_on_shots = False
        for name in outputs:
            outputs[name].train_on_shots = False
        return model, acquire, inputs, outputs


class USGS(Article2D):
    name = "USGS"

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 1
        self.validatesize = 0
        self.testsize = 1

        NS = 964
        model.NX = NS*acquire.ds + acquire.gmax + 2*acquire.Npad

        acquire.NT = 3071 * acquire.resampling

        for name in inputs:
            inputs[name].mute_dir = False

        return model, acquire, inputs, outputs
