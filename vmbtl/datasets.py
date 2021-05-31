# -*- coding: utf-8 -*-
"""Define parameters for different datasets."""

from os.path import abspath, join

import numpy as np
from scipy.signal import convolve, hann
from GeoFlow.GeoDataset import GeoDataset
from GeoFlow.EarthModel import MarineModel
from GeoFlow.SeismicGenerator import Acquisition
from GeoFlow.GraphIO import Reftime, Vrms, Vint, Vdepth, ShotGather

from vmbtl.download_real_test_data import NS, NG


class Dataset(GeoDataset):
    basepath = abspath("datasets")


class Article1D(Dataset):
    def set_dataset(self):
        self.trainsize = 5000
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
        outputs = {Reftime.name: Reftime(model=model, acquire=acquire),
                   Vrms.name: Vrms(model=model, acquire=acquire),
                   Vint.name: Vint(model=model, acquire=acquire),
                   Vdepth.name: Vdepth(model=model, acquire=acquire)}

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

        self.trainsize = 500
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
        for input in inputs.values():
            input.train_on_shots = False
        for output in outputs.values():
            output.train_on_shots = False
        return model, acquire, inputs, outputs


class USGS(Article2D):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        filename = join(self.basepath, self.name, "test", "example_1")
        inputs, _, _ = self.generator.read(filename)
        d = inputs['shotgather']
        d = d.reshape([d.shape[0], -1, NG])
        d = d[:, 1000]
        self.acquire.source_generator = real_source_generator(self, d)

    def set_dataset(self):
        model, acquire, inputs, outputs = super().set_dataset()

        self.trainsize = 1
        self.validatesize = 0
        self.testsize = 1

        model.NX = NS*acquire.ds + acquire.gmax + 2*acquire.Npad
        model.NZ = 2000

        dt = acquire.dt * acquire.resampling
        real_tdelay = 3 / 8
        unpad = int((real_tdelay-acquire.tdelay) / dt)
        acquire.NT = (3071-unpad) * acquire.resampling

        for input in inputs.values():
            input.mute_dir = False
            input.preprocess = decorate_preprocess(input)

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
            data = data[-NT:]
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
                pad = int(1.5 * self.acquire.tdelay / dt)
                mask = np.ones_like(data, dtype=bool)
                for (i, j), trace_arrival in np.ndenumerate(first_arrival):
                    mask[:trace_arrival-pad, i, j] = False
                data[~mask] = 0
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


def real_source_generator(self, d, nt_wav=41):
    def zero_phase_real_wavelet():
        source = np.zeros(self.acquire.NT)
        real_tdelay = 3 / 8
        t0 = int(real_tdelay / self.acquire.dt)
        w = estimate_zero_phase_wavelet(d, nt_wav=nt_wav)
        r = self.acquire.resampling
        w = np.repeat(w, r)
        source[t0-nt_wav*r:t0+(nt_wav-1)*r] = w
        return source
    return lambda: zero_phase_real_wavelet


def estimate_zero_phase_wavelet(d, ntmax=-1, nt_wav=41):
    # Estimate wavelet spectrum.
    dwind = d[:ntmax, ...]
    dwind = dwind * np.reshape(hann(dwind.shape[0]), [-1, 1])
    nfft = int(2**np.ceil(np.log2(dwind.shape[0])))
    wav_est_fft = np.mean(
        np.abs(np.fft.fft(dwind, nfft, axis=0)),
        axis=tuple(ii for ii in range(1, len(dwind.shape))),
    )

    wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = wav_est * hann(wav_est.shape[0])
    return wav_est
