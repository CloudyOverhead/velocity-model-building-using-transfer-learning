# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
from multiprocessing import Process, Queue

from matplotlib import pyplot as plt
import numpy as np
from segyio import open as open_segy

from Cases_define import Mercier
from vrmslearn.DatasetGenerator import DatasetProcess, SampleGenerator


DATASET_DIR = "../../mercier_timestack_brute"
DESTINATION_DIR = "Datasets/Mercier"

NG = 48
XSHOTS_0 = 76  # 70.5 m of landstreamer + 5.5 m offset from source.
DS = 4.5  # Distance between shots (m).
DG = 1.5
DT = .25  # ms.
T_MAX = 1000  # ms.
OFFSETS = np.arange(5.5, 77.5, 1.5)
TIME = np.arange(0, 1000, DT)
NT = len(TIME)
NSHOTS = 9

X_SRC = np.arange(XSHOTS_0, XSHOTS_0+(NSHOTS)*DS, DS)
X_SRC = np.repeat(X_SRC, NG)

X_RCV = np.arange((NG-1)*DG, -1, -DG)
X_RCV = np.tile(X_RCV, NSHOTS)
X_RCV = X_RCV + X_SRC - XSHOTS_0


class MercierConverter(SampleGenerator):
    def generate(self, number):
        segy_path = f"brute_stack_line{number}.sgy"
        data = load_line(segy_path)
        labels, weights = None, None
        return data, labels, weights


def load_line(segy_path):
    with open_segy(segy_path, 'rb', ignore_geometry=True) as f:
        data = np.empty([NT, NG, NSHOTS])
        for shot in range(NSHOTS):
            for receiver in range(shot*NG, (shot+1)*NG):
                data[:, receiver, shot] = f.trace[receiver]
    return data


def plot_shot_from_segy(segy_path, shot_no, scale_factor=1e-3):
    data = load_line(segy_path)
    data = data[:, :, shot_no]
    fig, ax = plt.subplots(figsize=(10, 10))

    for o, t in zip(OFFSETS, data):
        x = o + t*scale_factor*(0.01*o)
        ax.plot(x, TIME, 'k-')
        ax.fill_betweenx(TIME, o, x, where=(x > o), color='k')

    ax.set_ylim(0, 1000)
    ax.set_xlim(10, 80)
    ax.invert_yaxis()
    ax.set_xlabel('Position relative du géophone (m)', fontsize=14)
    ax.set_ylabel('Temps (ms)', fontsize=14)


files = listdir(DATASET_DIR)
files.remove("Log_files")
numbers = [int(file[16:-4]) for file in files]
filepath = join(DATASET_DIR, files[0])
plot_shot_from_segy(filepath, 2, scale_factor=2)


sample_generator = MercierConverter(model=None, acquire=None, label=None)
seeds_train, seeds_test = ...
for phase, seeds in zip(["train", "test"], [seeds_train, seeds_test]):
    queue = Queue()  # We use a queue for compatibility purposes.
    for seed in seeds:
        queue.put(seed)
    savepath = join(DATASET_DIR, phase)
    dataset_process = DatasetProcess(savepath, sample_generator, seeds=queue)
    dataset_process.run()
