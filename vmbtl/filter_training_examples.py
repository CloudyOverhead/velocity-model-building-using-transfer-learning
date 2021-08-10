from argparse import ArgumentParser
from os import listdir, remove
from os.path import join

import numpy as np
from h5py import File


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = args.dataset
    dataset_dir = join('datasets', dataset, 'train')

    for filename in listdir(dataset_dir):
        if 'example' in filename:
            filepath = join(dataset_dir, filename)
            f = File(filepath, 'r')
            for label in ['vrms', 'vint', 'vdepth']:
                label = np.array(f[label][:])
                if (np.diff(label, axis=0) < 1E-5).all():
                    print(f"Discarding example {filename}.")
                    remove(filepath)
                    break
