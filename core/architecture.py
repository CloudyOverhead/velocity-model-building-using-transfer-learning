# -*- coding: utf-8 -*-
"""Build the neural network for predicting v_p in 2D and in depth."""

from os import isdir, mkdir
from os.path import join

from DefinedNN.RCNN2D import RCNN2D, Hyperparameters


class RCNN2D(RCNN2D):
    pass


class RCNN2DUnpackReal(RCNN2D):
    def launch_testing(self, tfdataset, savedir):
        if savedir is None:
            savedir = type(self).__name__
        savedir = join(self.dataset.datatest, savedir)
        if not isdir(savedir):
            mkdir(savedir)
        if self.dataset.testsize % self.params.batch_size != 0:
            raise ValueError(
                "Your batch size must be a divisor of your dataset length."
            )

        for data, _ in tfdataset:
            evaluated = self.predict(
                data,
                batch_size=self.params.batch_size,
                max_queue_size=10,
                use_multiprocessing=False,
            )
            for lbl, out in evaluated.items():
                evaluated[lbl] = out[..., 0]

            for i, example in enumerate(data["filename"]):
                example = example.numpy().decode("utf-8")
                exampleid = int(example.split("_")[-1])
                example_evaluated = {
                    lbl: out[i] for lbl, out in evaluated.items()
                }
                self.dataset.generator.write_predictions(
                    exampleid, savedir, example_evaluated,
                )


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.epochs = 40
        self.steps_per_epoch = 100
        self.batch_size = 24

        self.learning_rate = 8E-4

        if is_training:
            self.loss_scales = (
                {'ref': .8, 'vrms': .2, 'vint': .0, 'vdepth': .0},
                {'ref': .1, 'vrms': .7, 'vint': .2, 'vdepth': .0},
                {'ref': .1, 'vrms': .1, 'vint': .8, 'vdepth': .0},
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
