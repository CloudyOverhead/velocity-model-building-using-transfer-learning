from os import listdir
from os.path import join

from matplotlib import pyplot as plt
import numpy as np
from segyio import open as open_segy

DATASET_DIR = "E:\\mercier_timestack_brute"

NG = 48
XSHOTS_0 = 76  # 70.5 de landstreamer + 5.5 d'offset avec la source
DS = 4.5  # distance entre les shots
DG = 1.5
DT = .25  # ms
T_MAX = 1000  # ms
OFFSETS = np.arange(5.5, 77.5, 1.5)
TIME = np.arange(0, 1000, DT)
NSHOTS = 9

X_SRC = np.arange(XSHOTS_0, XSHOTS_0+(NSHOTS)*DS, DS)
X_SRC = np.repeat(X_SRC, NG)

X_RCV = np.arange((NG-1)*DG, -1, -DG)
X_RCV = np.tile(X_RCV, NSHOTS)
X_RCV = X_RCV + X_SRC - XSHOTS_0

print(X_SRC.shape)
print(X_RCV.min())
print(X_RCV.shape)


def plot_shot_from_segy(segy_in, shot_no, scale_factor=1e-3):
    seismic_data = []

    with open_segy(segy_in, 'rb', ignore_geometry=True) as f:
        print("Quantity of shots:", len(f.trace) // NG)
        for i in np.arange(shot_no*NG, (shot_no+1)*NG):
            seismic_data.append(f.trace[i])
        seismic_data = np.array(seismic_data)

    fig, ax = plt.subplots(figsize=(10, 10))

    times = seismic_data
    for o, t in zip(OFFSETS, times):
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
filepath = join(DATASET_DIR, files[0])
plot_shot_from_segy(filepath, 2, scale_factor=2)
