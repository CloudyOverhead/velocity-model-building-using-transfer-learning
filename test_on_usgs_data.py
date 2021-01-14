# -*- coding: utf-8 -*-
"""
This file is modified from `gfabieno/Deep_1D_velocity/realdata
/Process_realdata.py`](https://github.com/GeoCode-polymtl/Deep_1D_velocity/blob
/master/realdata/Process_realdata.py) which is licensed under the MIT license:

    MIT License

    Copyright (c) 2019 Gabriel Fabien-Ouellet

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

The processed real data is available publicly at
https://cmgds.marine.usgs.gov/fan_info.php?fan=1978-015-FA
"""

from os import mkdir
from os.path import join, isfile, isdir
from urllib.request import urlretrieve

import segyio
from segyio.TraceField import FieldRecord, TraceNumber
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


if __name__ == "__main__":
    """Download the data."""
    DATAPATH = "./USGS_line32"
    PREFIX = "http://cotuit.er.usgs.gov/files/1978-015-FA"
    files = {
        "32obslog.pdf": join(PREFIX, "NL/001/01/32-obslogs/32obslog.pdf"),
        "report.pdf": "https://pubs.usgs.gov/of/1995/0027/report.pdf",
        "CSDS32_1.SGY": join(PREFIX, "/SE/001/39/CSDS32_1.SGY"),
    }
    FILES_PREFIX = "SE/001/18"
    dfiles = {
        "U32A_01.SGY": join(PREFIX, FILES_PREFIX, "U32A_01.SGY"),
        "U32A_02.SGY": join(PREFIX, FILES_PREFIX, "U32A_02.SGY"),
        "U32A_03.SGY": join(PREFIX, FILES_PREFIX, "U32A_03.SGY"),
        "U32A_04.SGY": join(PREFIX, FILES_PREFIX, "U32A_04.SGY"),
        "U32A_05.SGY": join(PREFIX, FILES_PREFIX, "U32A_05.SGY"),
        "U32A_06.SGY": join(PREFIX, FILES_PREFIX, "U32A_06.SGY"),
        "U32A_07.SGY": join(PREFIX, FILES_PREFIX, "U32A_07.SGY"),
        "U32A_08.SGY": join(PREFIX, FILES_PREFIX, "U32A_08.SGY"),
        "U32A_09.SGY": join(PREFIX, FILES_PREFIX, "U32A_09.SGY"),
    }
    # "U32A_10.SGY": join(PREFIX, FILES_PREFIX, "U32A_10.SGY"),
    # "U32A_11.SGY": join(PREFIX, FILES_PREFIX, "U32A_11.SGY"),
    # "U32A_12.SGY": join(PREFIX, FILES_PREFIX, "U32A_12.SGY"),
    # "U32A_13.SGY": join(PREFIX, FILES_PREFIX, "U32A_13.SGY"),
    # "U32A_14.SGY": join(PREFIX, FILES_PREFIX, "U32A_14.SGY"),
    # "U32A_15.SGY": join(PREFIX, FILES_PREFIX, "U32A_15.SGY"),
    # "U32A_16.SGY": join(PREFIX, FILES_PREFIX, "U32A_16.SGY"),
    # "U32A_17.SGY": join(PREFIX, FILES_PREFIX, "U32A_17.SGY"),
    # "U32A_18.SGY": join(PREFIX, FILES_PREFIX, "U32A_18.SGY"),
    # "U32A_19.SGY": join(PREFIX, FILES_PREFIX, "U32A_19.SGY"),
    # "U32A_20.SGY": join(PREFIX, FILES_PREFIX, "U32A_20.SGY"),
    # "U32A_21.SGY": join(PREFIX, FILES_PREFIX, "U32A_21.SGY")}

    fkeys = sorted(list(dfiles.keys()))
    if not isdir(DATAPATH):
        mkdir(DATAPATH)

    for file in files:
        if not isfile(join(DATAPATH, file)):
            urlretrieve(files[file], join(DATAPATH, file))

    for file in dfiles:
        if not isfile(join(DATAPATH, file)):
            print(file)
            urlretrieve(dfiles[file], join(DATAPATH, file))

    """Read the segy into numpy."""
    data = []
    fid = []
    cid = []
    NT = 3071
    for file in fkeys:
        print(file)
        file = join(DATAPATH, file)
        with segyio.open(file, "r", ignore_geometry=True) as segy:
            file_data = []
            file_fid = []
            file_cid = []
            for trid in range(segy.tracecount):
                file_data.append(segy.trace[trid])
                file_fid.append(segy.header[trid][FieldRecord])
                file_cid.append(segy.header[trid][TraceNumber])
            file_data = file_data.T
            file_data = file_data[:NT]
            data.append(file_data)
            fid.append(file_fid)
            cid.append(file_cid)

    """Remove bad shots."""
    # Correct `fid`.
    if len(fid) > 16:
        fid[16] = [id if id < 700 else id + 200 for id in fid[16]]
    if len(fid) > 6:
        fid[6] = fid[6][:12180]
        cid[6] = cid[6][:12180]
        data[6] = data[6][:, :12180]
    if len(fid) > 7:
        fid[7] = fid[7][36:]
        cid[7] = cid[7][36:]
        data[7] = data[7][:, 36:]
    if len(fid) > 2:  # repeated shots between files 03 and 04
        fid[2] = fid[2][:8872]
        cid[2] = cid[2][:8872]
        data[2] = data[2][:, :8872]
    fid = np.concatenate(fid)
    cid = np.concatenate(cid)
    data = np.concatenate(data, axis=1)

    # recnoSpn = InterpText()
    # recnoSpn.read('recnoSpn.txt')

    # recnoDelrt = InterpText()
    # recnoDelrt.read('recnoDelrt.txt')

    prev_fldr = -9999
    fldr_bias = 0
    shot = np.full_like(cid, -1)
    delrt = np.full_like(cid, -1)

    NOT_SHOTS = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 211, 213, 225, 279,
        335, 387, 400, 493, 528, 553, 561, 571,
        668, 669, 698, 699, 700, 727, 728, 780, 816, 826, 1073, 1219,
        1253, 1254, 1300, 1301, 1418, 1419, 1527, 1741, 2089, 2170,
        2303, 2610, 2957, 2980, 3021, 3104, 3167, 3223, 3268, 3476,
        3707, 3784, 3831, 3934, 4051, 4472, 4671, 4757, 4797,
    ]

    for ii in range(fid.shape[0]):
        fldr = fid[ii]
        tracf = cid[ii]

        if fldr < prev_fldr:
            fldr_bias += 1000

        fldr += fldr_bias
        if fldr not in NOT_SHOTS:
            shot[ii] = 6102 - fldr

        # The time 0 of different files changes. We prepad with zeros so that
        # all shots begin at time 0.
        if fldr < 15:
            delrt[ii] = 4000
        elif fldr < 20:
            delrt[ii] = 5000
        elif fldr < 1043:
            delrt[ii] = 4000
        elif fldr < 1841:
            delrt[ii] = 3000
        elif fldr < 2199:
            delrt[ii] = 2000
        elif fldr < 2472:
            delrt[ii] = 1000
        else:
            delrt[ii] = 0

        prev_fldr = fldr

    valid = shot > 0
    shot = shot[valid]
    delrt = delrt[valid]
    data = data[:, valid]

    plt.plot(shot)
    plt.show()

    DT = 4  # Time step, in milliseconds.
    for ii in range(data.shape[1]):
        pad_width = delrt[ii] // DT
        padded_data = np.pad(
            data[:, ii],
            constant_values=0,
            pad_width=(pad_width, 0),
        )
        data[:, ii] = padded_data[:NT]

    # Open the hdf5 file in which to save the pre-processed data.
    savefile = h5.File("survey.hdf5", "w")
    savefile["data"] = data

    """Trace interpolation."""
    # From the observer log, we get the acquisition parameters:
    DS = 50  # Shot point spacing.
    DG1 = 100  # Geophone spacing for channels 1-24.
    DG2 = 50  # Geophone spacing for channels 25-48.
    VWATER = 1533
    NS = data.shape[1] // 48
    NG = 72
    dg = 50
    NEAROFF = 470  # Varies for several shots. We take the most common value.

    data_i = np.zeros([data.shape[0], NS*NG])
    t0off = 2 * np.sqrt((NEAROFF/2)**2+3000**2) / VWATER
    for ii in range(NS):
        data_i[:, NG*ii:NG*ii+23] = data[:, ii*48:ii*48+23]
        data_roll = data[:, ii*48+23:(ii+1)*48]
        n = data_roll.shape[1]
        for jj in range(n):
            toff = 2*np.sqrt(((NEAROFF+DG1*(n-jj))/2)**2+3000**2)/VWATER - t0off
            data_roll[:, jj] = np.roll(data_roll[:, jj], -toff//0.004)
        data_roll = ndimage.zoom(data_roll, [1, 2], order=1)
        n = data_roll.shape[1]
        for jj in range(n):
            toff = 2*np.sqrt(((NEAROFF+DG2*(n-jj))/2)**2+3000**2)/VWATER - t0off
            data_roll[:, jj] = np.roll(data_roll[:, jj], toff//0.004)
        data_i[:, NG*ii+23:NG*(ii+1)] = data_roll[:, :-1]

    savefile['data_i'] = data_i

    """Resort accorging to CMP."""
    NS = data_i.shape[1] // 72
    shots = np.arange(NEAROFF+NG*dg, NEAROFF+NG*dg+NS*DS, DS)
    recs = np.concatenate(
        [np.arange(0, NG*dg, dg)+n*DS for n in range(NS)],
        axis=0,
    )
    shots = np.repeat(shots, NG)
    cmps = ((shots+recs)/2/50).astype(int) * 50
    offsets = shots - recs

    ind = np.lexsort((offsets, cmps))
    cmps = cmps[ind]
    unique_cmps, counts = np.unique(cmps, return_counts=True)
    firstcmp = unique_cmps[np.argmax(counts == 72)]
    lastcmp = unique_cmps[-np.argmax(counts[::-1] == 72) - 1]
    ind1 = np.argmax(cmps == firstcmp)
    ind2 = np.argmax(cmps > lastcmp)
    ntraces = cmps[ind1:ind2].shape[0]
    data_cmp = np.zeros([data_i.shape[0], ntraces])

    n = 0
    for ii, jj in enumerate(ind):
        if ii >= ind1 and ii < ind2:
            data_cmp[:, n] = data_i[:, jj]
            n += 1

    savefile['data_cmp'] = data_cmp
    savefile.close()

    """Plots for quality control."""
    # Plot some CMP gather.
    CLIP = 0.05
    vmax = np.max(data_cmp[:, 0]) * CLIP
    vmin = -vmax
    plt.imshow(
        data_cmp[:, :200],
        interpolation='bilinear',
        cmap=plt.get_cmap('Greys'),
        vmin=vmin, vmax=vmax,
        aspect='auto',
    )
    plt.show()

    # Constant offset plot.
    plt.imshow(
        data_cmp[:, ::72],
        interpolation='bilinear',
        cmap=plt.get_cmap('Greys'),
        vmin=vmin, vmax=vmax,
        aspect='auto',
    )
    plt.show()
