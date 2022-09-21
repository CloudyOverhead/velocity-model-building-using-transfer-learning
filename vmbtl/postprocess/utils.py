# -*- coding: utf-8 -*-

from os.path import listdir, join, exists

import numpy as np
import pandas as pd
from skimage.measure import compare_ssim as ssim
from tensorflow.compat.v1.train import summary_iterator
from GeoFlow.SeismicUtilities import stack, vint2vrms

from .constants import IGNORE_NNS


def compare_preds(dataset, savedir):
    print(f"Comparing predictions for directory {savedir}.")

    nt = dataset.acquire.NT
    dt = dataset.acquire.dt
    resampling = dataset.acquire.resampling
    time = np.arange(nt // resampling) * dt * resampling
    time = time[:, None]

    similarities = np.array([])
    rmses = np.array([])
    rmses_rms = np.array([])

    for i, example in enumerate(dataset.files["test"]):
        if (i+1) % 20 == 0:
            print(f"Processing example {i+1} of {len(dataset.files['test'])}.")
        _, labels, weights, filename = dataset.get_example(
            example, phase='test',
        )
        preds = dataset.generator.read_predictions(filename, savedir)
        vint = labels['vint']
        weight = weights['vint']
        vint_pred = preds['vint']
        vrms_pred = preds['vrms']

        vint = vint * weight
        vint_pred = vint_pred * weight
        similarity = ssim(vint, vint_pred)
        similarities = np.append(similarities, similarity)
        rmse = np.sqrt(np.mean((vint-vint_pred)**2))
        rmses = np.append(rmses, rmse)

        vrms_pred = vrms_pred * weight
        vrms_converted = vint2vrms(vint_pred, time)
        rmse_rms = np.sqrt(np.mean((vrms_pred-vrms_converted)**2))
        rmses_rms = np.append(rmses_rms, rmse_rms)

    vmin, vmax = dataset.model.properties['vp']
    rmses *= vmax - vmin
    print("Average SSIM:", np.mean(similarities))
    print("Standard deviation on SSIM:", np.std(similarities))
    print("Average RMSE:", np.mean(rmses))
    print("Standard deviation on RMSE:", np.std(rmses))

    rmses_rms *= vmax - vmin
    print("Average RMSE of RMS conversion:", np.mean(rmses_rms))
    print("Standard deviation on RMSE of RMS conversion:", np.std(rmses_rms))

    return similarities


def load_all(dataset, savedir):
    all_inputs = {}
    all_labels = {}
    all_weights = {}
    all_preds = {}
    for example in dataset.files["test"]:
        inputs, labels, weights, filename = dataset.get_example(
            example, phase='test',
        )
        preds = dataset.generator.read_predictions(filename, savedir)
        target_dicts = [all_inputs, all_labels, all_weights, all_preds]
        current_dicts = [inputs, labels, weights, preds]
        for target_dict, current_dict in zip(target_dicts, current_dicts):
            for key in current_dict.keys():
                current_array = np.expand_dims(current_dict[key], axis=0)
                if key in target_dict.keys():
                    target_dict[key] = np.append(
                        target_dict[key], current_array, axis=0,
                    )
                else:
                    target_dict[key] = current_array
    return all_inputs, all_labels, all_weights, all_preds


def load_events(logdir):
    data = []
    for i in listdir(logdir):
        if int(i) in IGNORE_NNS:
            continue
        current_logdir = join(logdir, i)
        events_path = [
            path for path in listdir(current_logdir) if "events" in path
        ]
        if events_path:
            events_path = join(current_logdir, events_path[-1])
            current_data = pd.DataFrame([])
            events = summary_iterator(events_path)
            for event in events:
                if hasattr(event, 'step'):
                    step = event.step
                    for value in event.summary.value:
                        column = value.tag
                        value = value.simple_value
                        current_data.loc[step, column] = np.log10(value)
        else:
            events_path = join(logdir, 'progress.csv')
            assert exists(events_path)
            current_data = pd.read_csv(events_path)
            current_data = np.log10(current_data)
        data.append(current_data)
    data = pd.concat(data)
    by_index = data.groupby(data.index)
    return by_index.mean(), by_index.std()


def data_preprocess(data):
    eps = np.finfo(np.float32).eps
    trace_rms = np.sqrt(np.sum(data**2, axis=0, keepdims=True))
    data /= trace_rms + eps
    data = np.expand_dims(data, axis=-1)
    return data


def stack_2d(cmps, times, offsets, velocities):
    stacked = []
    cmps = cmps.transpose([2, 0, 1])
    velocities = velocities.T
    for i, (cmp, velocities_1d) in enumerate(zip(cmps, velocities)):
        stacked.append(stack(cmp, times, offsets, velocities_1d))
    stacked = np.array(stacked).T
    return stacked
