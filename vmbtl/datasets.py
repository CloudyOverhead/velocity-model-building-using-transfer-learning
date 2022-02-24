# -*- coding: utf-8 -*-
"""Define parameters for different datasets."""

from os import makedirs
from os.path import abspath, exists, join
from urllib.request import urlretrieve
import gzip
import shutil

import numpy as np
from scipy.signal import convolve
from skimage.transform import rescale
import segyio
from ModelGenerator import (
    Sequence, Stratigraphy, Deformation, Property, Lithology,
)
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import Reftime, Vrms, Vint, Vdepth, ShotGather

from vmbtl.download_real_test_data import NS


class Dataset(GeoDataset):
    basepath = abspath("datasets")


class Article1D(Dataset):
    def set_dataset(self):
        self.trainsize = 20000
        self.validatesize = 0
        self.testsize = 10

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
        model.dzmax = 1000
        model.accept_decrease = .65

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
        acquire.source_depth = (acquire.Npad+4) * model.dh
        acquire.receiver_depth = (acquire.Npad+4) * model.dh
        acquire.tdelay = 3.0 / (acquire.peak_freq-acquire.df)
        acquire.singleshot = True
        acquire.configuration = 'inline'

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire),
            Vint.name: Vint(model=model, acquire=acquire),
            Vdepth.name: Vdepth(model=model, acquire=acquire),
        }

        for input in inputs.values():
            input.train_on_shots = True  # 1D shots are CMPs.
            input.mute_dir = True
        for output in outputs.values():
            output.train_on_shots = True
            output.identify_direct = False

        return model, acquire, inputs, outputs

    def __init__(self, noise=False):
        super().__init__()
        if noise:
            for input in self.inputs.values():
                input.random_static = True
                input.random_static_max = 1
                input.random_noise = True
                input.random_noise_max = 0.02
                input.random_time_scaling = True


class Article2D(Article1D):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 2000
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

        inputs = {
            ShotGatherCrop.name: ShotGatherCrop(model=model, acquire=acquire)
        }
        outputs = {
            ReftimeCrop.name: ReftimeCrop(model=model, acquire=acquire),
            VrmsCrop.name: VrmsCrop(model=model, acquire=acquire),
            VintCrop.name: VintCrop(model=model, acquire=acquire),
            VdepthCrop.name: VdepthCrop(model=model, acquire=acquire),
        }
        for input in inputs.values():
            input.train_on_shots = False
            input.mute_dir = True
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs


class USGS(Article2D):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 1
        self.validatesize = 0
        self.testsize = 1

        model.NX = NS*acquire.ds + acquire.gmax + 2*acquire.Npad
        model.NZ = 2000

        dt = acquire.dt * acquire.resampling
        real_tdelay = 0
        pad = int((acquire.tdelay-real_tdelay) / dt)
        acquire.NT = (3071+pad) * acquire.resampling

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire),
            Vint.name: Vint(model=model, acquire=acquire),
            Vdepth.name: Vdepth(model=model, acquire=acquire),
        }
        for input in inputs.values():
            input.mute_dir = False
            input.train_on_shots = False
            input.preprocess = decorate_preprocess(input)
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs


def decorate_preprocess(self):
    # Preprocessing is costly, but it is run once in order to initialize the
    # NN. We can skip the first preprocess, because we can infer the shapes
    # manually.
    self.skip_preprocess = True

    def preprocess_real_data(data, labels, use_agc=True):
        if not self.skip_preprocess:
            data = data.reshape([3071, -1, 72])
            NT = int(self.acquire.NT / self.acquire.resampling)
            pad = NT - data.shape[0]
            data = np.pad(data, [[pad, 0], [0, 0], [0, 0]])
            data = data.swapaxes(1, 2)

            END_CMP = 2100
            data = data[:, :, :END_CMP]

            eps = np.finfo(np.float32).eps
            if use_agc:
                agc_kernel = np.ones([21, 5, 1])
                agc_kernel /= agc_kernel.size
                pads = [[int(pad//2), int(pad//2)] for pad in agc_kernel.shape]
                gain = convolve(
                    np.pad(data, pads, mode='symmetric')**2,
                    agc_kernel,
                    'valid',
                )
                gain[gain < eps] = eps
                gain = 1 / np.sqrt(gain)
            vmax = np.amax(data, axis=0)
            first_arrival = np.argmax(data > .4*vmax[None], axis=0)
            dt = self.acquire.dt * self.acquire.resampling
            pad = int(1 / self.acquire.peak_freq / dt)
            mask = np.ones_like(data, dtype=bool)
            for (i, j), trace_arrival in np.ndenumerate(first_arrival):
                mask[:trace_arrival-pad, i, j] = False
            data[~mask] = 0
            if use_agc:
                data[mask] *= gain[mask]

            trace_rms = np.sqrt(np.sum(data**2, axis=0, keepdims=True))
            data /= trace_rms + eps
            panel_max = np.amax(data, axis=(0, 1), keepdims=True)
            data /= panel_max + eps

            data *= 1000
            data = np.expand_dims(data, axis=-1)
            return data
        else:
            self.skip_preprocess = False
            return data
    return preprocess_real_data


class Article2DSteep(Article2D):
    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 2000
        self.testsize = 100

        model.NX = int(7000 / 5)
        model.NZ = int(2800 / 5)
        model.dh = 1.25 * 5
        model.dip_max = 45
        model.ddip_max = 8
        model.water_dmin = 450
        model.water_dmax = 1200
        model.vp_max = 4500
        model.dzmax = 1200
        model.accept_decrease = .65
        model.max_deform_freq = .01
        model.min_deform_freq = .0005
        model.amp_max = 160
        model.max_deform_nfreq = 20
        model.prob_deform_change = .2
        model.deform_cumulative = True

        model.fault_dip_min = 35
        model.fault_dip_max = 89
        model.fault_displ_min = -800
        model.fault_displ_max = 0
        model.fault_x_lim = [int(.25*model.NX), int(.75*model.NX)]
        model.fault_y_lim = [int(.25*model.NZ), int(.75*model.NZ)]
        model.fault_nmax = 3
        model.fault_prob = [.4, .3, .2]
        model.generate_model = self.decorate_generate_model(model)

        acquire.dt = .0002
        acquire.resampling = 10
        acquire.NT = int(3 / acquire.dt)
        acquire.dg = 8
        acquire.ds = 8
        acquire.gmin = 0
        acquire.gmax = int(4000 / model.dh)
        acquire.minoffset = 0
        acquire.source_depth = (acquire.Npad+4) * model.dh
        acquire.receiver_depth = (acquire.Npad+4) * model.dh
        acquire.tdelay = 3.0 / (acquire.peak_freq-acquire.df)
        acquire.configuration = 'inline'

        return model, acquire, inputs, outputs

    def decorate_generate_model(self, model):
        old_generate_model = model.generate_model

        def generate_model(*args, seed=None, **kwargs):
            props, layerids, layers = old_generate_model(
                *args, seed=seed, **kwargs,
            )
            source_depth = self.acquire.source_depth
            dh = self.model.dh
            water_top = int(source_depth / dh * 2)
            props['vp'][:water_top] = 1500
            return props, layerids, layers
        return generate_model


class Marmousi(Article2DSteep):
    FILENAME = "vp_marmousi-ii.segy"
    URL = "http://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz"
    WATER_PAD = 0
    RESCALE_FACTOR = 1 / 5
    NX = 13600
    NZ = 2800

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 0
        self.validatesize = 0
        self.testsize = 1

        model.generate_model = self.load_model
        if acquire.gmax is not None:
            self.lateral_pad = int(acquire.Npad + acquire.gmax/2)
        else:
            self.lateral_pad = 0
        model.NX = int(self.NX*self.RESCALE_FACTOR+2*self.lateral_pad)
        model.NZ = int(self.NZ*self.RESCALE_FACTOR+self.WATER_PAD)

        acquire.NT = int(3 / acquire.dt)

        inputs = {ShotGather.name: ShotGather(model=model, acquire=acquire)}
        outputs = {
            Reftime.name: Reftime(model=model, acquire=acquire),
            Vrms.name: Vrms(model=model, acquire=acquire),
            Vint.name: Vint(model=model, acquire=acquire),
            Vdepth.name: Vdepth(model=model, acquire=acquire),
        }
        for input in inputs.values():
            input.mute_dir = True
            input.train_on_shots = False
        for output in outputs.values():
            input.train_on_shots = False
            output.identify_direct = False

        return model, acquire, inputs, outputs

    def load_model(self, *args, **kwargs):
        filepath = join(self.basepath, self.name, self.FILENAME)
        if not exists(filepath):
            makedirs(join(self.basepath, self.name))
            urlretrieve(self.URL, filename=filepath + '.gz')
            with gzip.open(filepath + '.gz', 'rb') as compressed_f:
                with open(filepath, 'wb') as decompressed_f:
                    shutil.copyfileobj(compressed_f, decompressed_f)
        vp = segyio.open(filepath, ignore_geometry=True)
        vp = np.concatenate([np.copy(t)[:, None] for t in vp.trace[:]], axis=1)
        vp *= 1000
        vp = vp[1:, 1:]
        vp = rescale(vp, self.RESCALE_FACTOR)
        vp = np.pad(vp, [[self.WATER_PAD, 0], [0, 0]], constant_values=1500)
        pad = self.lateral_pad
        vp = np.pad(vp, [[0, 0], [pad, pad]], mode='edge')
        props = {
            'vp': vp,
            'vs': np.zeros_like(vp),
            'rho': np.full_like(vp, 2000),
        }
        return props, None, None


class MarineModel(MarineModel):
    def generate_model(self, *args, seed=None, **kwargs):
        is_2d = self.dip_max > 0
        self.layer_num_min = 5
        if seed is None:
            seed = np.random.randint(0, 20000)
        if not is_2d:
            if seed < 5000:
                self.layer_dh_max = 500
            if seed < 10000:
                self.layer_dh_max = 200
            else:
                self.layer_dh_max = 50
        else:
            self.layer_num_min = 50
        return super().generate_model(*args, seed=seed, **kwargs)

    def build_stratigraphy(self):
        self.thick0min = int(self.water_dmin/self.dh)
        self.thick0max = int(self.water_dmax/self.dh)

        vp = Property(
            name="vp", vmin=self.water_vmin, vmax=self.water_vmax, dzmax=0,
        )
        vs = Property(name="vs", vmin=0, vmax=0)
        rho = Property(name="rho", vmin=2000, vmax=2000)
        water = Lithology(name='water', properties=[vp, vs, rho])
        vp = Property(
            name="vp",
            vmin=self.vp_min,
            vmax=self.vp_max,
            texture=self.max_texture,
            trend_min=self.vp_trend_min,
            trend_max=self.vp_trend_max,
            dzmax=self.dzmax,
            filter_decrease=self.accept_decrease > 0,
        )
        roc = Lithology(name='roc', properties=[vp, vs, rho])
        if self.amp_max > 0 and self.max_deform_nfreq > 0:
            deform = Deformation(
                max_deform_freq=self.max_deform_freq,
                min_deform_freq=self.min_deform_freq,
                amp_max=self.amp_max,
                max_deform_nfreq=self.max_deform_nfreq,
                prob_deform_change=self.prob_deform_change,
            )
        else:
            deform = None
        waterseq = Sequence(
            lithologies=[water],
            ordered=False,
            thick_min=self.thick0min,
            thick_max=self.thick0max,
            nmin=1,
        )
        rocseq = Sequence(
            lithologies=[roc],
            ordered=False,
            deform=deform,
            accept_decrease=self.accept_decrease,
        )
        strati = Stratigraphy(sequences=[waterseq, rocseq])
        properties = strati.properties()

        return strati, properties


class ShotGather(ShotGather):
    def plot(
        self, data, weights=None, axs=None, cmap='Greys', vmin=0, vmax=None,
        clip=.08, ims=None,
    ):
        return super().plot(data, weights, axs, cmap, vmin, vmax, clip, ims)


class ReftimeCrop(Reftime):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VrmsCrop(Vrms):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VintCrop(Vint):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class VdepthCrop(Vdepth):
    def preprocess(self, label, weight):
        label, weight = super().preprocess(label, weight)
        return label[:, 10:-10], weight[:, 10:-10]


class ShotGatherCrop(ShotGather):
    def preprocess(self, data, label):
        data = super().preprocess(data, label)
        return data[:, :, 10:-10]

    def plot(self, *args, **kwargs):
        with CropAcquisition(self.acquire):
            return super().plot(*args, **kwargs)


class CropAcquisition:
    def __init__(self, acquisition):
        self.acquisition = acquisition
        self._set_rec_src = acquisition.set_rec_src

    def __enter__(self):
        self.acquisition.set_rec_src = self.set_rec_src

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.acquisition.set_rec_src = self._set_rec_src

    def set_rec_src(self):
        src_pos, rec_pos = self._set_rec_src()
        drop_ids = [*range(0, 10), *range(78, 88)]
        src_pos = src_pos[:, ~np.isin(src_pos[3], drop_ids)]
        rec_pos = rec_pos[:, ~np.isin(rec_pos[3], drop_ids)]
        src_pos[3] -= 10
        rec_pos[3] -= 10
        return src_pos, rec_pos
