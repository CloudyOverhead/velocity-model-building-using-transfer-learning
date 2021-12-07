from argparse import ArgumentParser
from os import listdir, remove
from os.path import join

import numpy as np
from h5py import File
import tensorflow as tf

from GeoFlow.Losses import v_compound_loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = args.dataset
    dataset_dir = join('datasets', dataset, 'train')

    loss = v_compound_loss(normalize=True)
    for filename in sorted(listdir(dataset_dir)):
        if 'example' in filename:
            filepath = join(dataset_dir, filename)
            f = File(filepath, 'r')
            for name in ['vrms', 'vint', 'vdepth']:
                label = np.array(f[name][:])
                weight = np.array(f[name + '_w'][:])

                loss_value = loss(
                    tf.convert_to_tensor([[label, weight]]),
                    tf.convert_to_tensor(label[None, ..., None]),
                )
                loss_value = loss_value.numpy()[0]
                has_no_interface = (np.diff(label, axis=0) < 1E-5).all()
                if has_no_interface or np.isnan(loss_value) or loss_value > 1E9:
                    print(f"Discarding example {filename}.")
                    remove(filepath)
                    break
