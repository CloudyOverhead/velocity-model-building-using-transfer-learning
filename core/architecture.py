# -*- coding: utf-8 -*-
"""Build the neural network for predicting v_p in 2D and in depth."""

from DefinedNN.RCNN2D import RCNN2D, Hyperparameters


class RCNN2D(RCNN2D):
    pass


class Hyperparameters1D(Hyperparameters):
    def __init__(self, is_training=True):
        super().__init__()

        self.epochs = 50
        self.steps_per_epoch = 100
        self.batch_size = 24

        self.learning_rate = 8E-4

        if is_training:
            self.loss_scales = (
                {'ref': .8, 'vrms': .2, 'vint': .0, 'vdepth': .0},
                {'ref': .1, 'vrms': .7, 'vint': .2, 'vdepth': .0},
                {'ref': .1, 'vrms': .1, 'vint': .8, 'vdepth': .0},
            )


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
