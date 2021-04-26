# -*- coding: utf-8 -*-
"""Build the neural network for predicting v_p in 2D and in depth."""

from os import mkdir
from os.path import join, isdir

import numpy as np
import tensorflow as tf
from GeoFlow.DefinedNN.RCNN2D import RCNN2D, Hyperparameters


class RCNN2DUnpackReal(RCNN2D):
    def __init__(
        self, input_shapes, params, dataset, checkpoint_dir, devices,
        run_eagerly,
    ):
        ng = (dataset.acquire.gmax-dataset.acquire.gmin) // dataset.acquire.dg
        ng = int(ng)
        nt = dataset.acquire.NT // dataset.acquire.resampling
        nt = int(nt)

        is_1d = "1D" in type(params).__name__
        if is_1d:
            self.receptive_field = 1
            self.cmps_per_iter = 61
        else:
            self.receptive_field = 31
            self.cmps_per_iter = 2*self.receptive_field - 1

        input_shapes = {'shotgather': (nt, ng, self.cmps_per_iter, 1)}
        params.batch_size = 1
        super().__init__(
            input_shapes, params, dataset, checkpoint_dir, devices,
            run_eagerly,
        )

    @property
    def dbatch(self):
        return self.cmps_per_iter - 2*(self.receptive_field//2)

    def launch_testing(self, tfdataset, savedir):
        if savedir is None:
            savedir = type(self).__name__
        savedir = join(self.dataset.datatest, savedir)
        if not isdir(savedir):
            mkdir(savedir)

        for data, _ in tfdataset:
            evaluated = {key: [] for key in self.tooutputs}
            shotgather = data['shotgather'][0]
            filename = data['filename'][0]
            qty_cmps = shotgather.shape[2]
            shotgather = self.split_data(shotgather)
            for i, slice in enumerate(shotgather):
                print(f"Processing slice {i+1} out of {len(shotgather)}.")
                evaluated_slice = self.predict(
                    {
                        'filename': tf.expand_dims(filename, axis=0),
                        'shotgather': tf.expand_dims(slice, axis=0),
                    },
                    batch_size=1,
                    max_queue_size=10,
                    use_multiprocessing=False,
                )
                for key, pred in evaluated_slice.items():
                    evaluated[key].append(pred[0])
            print("Joining slices.")
            evaluated = self.unsplit_predictions(evaluated, qty_cmps)
            for lbl, out in evaluated.items():
                evaluated[lbl] = out[..., 0]

            example = filename.numpy().decode("utf-8")
            exampleid = int(example.split("_")[-1])
            example_evaluated = {
                lbl: out for lbl, out in evaluated.items()
            }
            self.dataset.generator.write_predictions(
                exampleid, savedir, example_evaluated,
            )

    def split_data(self, data):
        rf = self.receptive_field
        cmps_per_iter = self.cmps_per_iter
        dbatch = self.dbatch

        qty_cmps = data.shape[2]
        start_idx = np.arange(0, qty_cmps-rf//2, dbatch)
        batch_idx = np.arange(cmps_per_iter)
        select_idx = (
            np.expand_dims(start_idx, 0) + np.expand_dims(batch_idx, 1)
        )
        qty_batches = len(start_idx)
        end_pad = dbatch*qty_batches + 2*(rf//2) - qty_cmps
        data = np.pad(data, [[0, 0], [0, 0], [0, end_pad], [0, 0]])
        data = np.take(data, select_idx, axis=2)
        data = np.transpose(data, [3, 0, 1, 2, 4])
        return data

    def unsplit_predictions(self, predictions, qty_cmps):
        rf = self.receptive_field
        dbatch = self.dbatch

        is_1d = "1D" in type(self.params).__name__
        for key, pred in predictions.items():
            if not is_1d:
                for i, slice in enumerate(pred):
                    if i == 0:
                        pred[i] = slice[:, :-(rf//2)]
                    elif i != len(pred) - 1:
                        pred[i] = slice[:, rf//2:-(rf//2)]
            unpad_end = dbatch*len(pred) + 2*(rf//2) - qty_cmps
            if unpad_end:
                pred[-1] = pred[-1][:, rf//2:-unpad_end]
            else:
                pred[-1] = pred[-1][:, rf//2:]
        for key, pred in predictions.items():
            predictions[key] = np.concatenate(pred, axis=1)
        return predictions


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.epochs = 20
        self.steps_per_epoch = 100
        self.batch_size = 24

        self.learning_rate = 8E-4

        if is_training:
            self.loss_scales = (
                {'ref': .8, 'vrms': .2, 'vint': .0, 'vdepth': .0},
                {'ref': .1, 'vrms': .7, 'vint': .2, 'vdepth': .0},
                {'ref': .1, 'vrms': .3, 'vint': .5, 'vdepth': .1},
            )
            self.seed = (0, 1, 2)


class Hyperparameters2D(Hyperparameters1D):
    def __init__(self, is_training=True):
        super().__init__(is_training=is_training)

        self.batch_size = 2

        self.learning_rate = 8E-5

        self.encoder_kernels = [
            [15, 1, 1],
            [1, 9, 9],
            [15, 1, 1],
            [1, 9, 9],
        ]
        self.rcnn_kernel = [15, 3, 3]

        if is_training:
            CHECKPOINT_1D = (
                "/home/CloudyOverhead/jacques/Drive/Drive/GitHub"
                "/velocity-model-building-using-transfer-learning/logs"
                "/test-on-usgs/4d92542_Reintroduce_deformations_in_dataset"
                "/0/model/lambda_2021-02-11_16-38-47"
                "/lambda_85f1b_00000_0_2021-02-11_16-38-47/checkpoint_150"
            )
            self.restore_from = (CHECKPOINT_1D, None, None)
            self.seed = (3, 4, 5)


class Hyperparameters2DNoTL(Hyperparameters2D):
    def __init__(self, is_training=True):
        super().__init__(is_training=is_training)

        self.epochs *= 2

        self.restore_from = None
