# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


def main(args):
    dataset = args.dataset
    dataset.tooutputs = ['ref', 'vrms', 'vint']
    dataset._getfilelist()

    inputs, _, _, _ = dataset.get_example(toinputs=args.nn.toinputs)
    input_shapes = {name: input.shape for name, input in inputs.items()}
    nn = args.nn(
        dataset=dataset,
        input_shapes=input_shapes,
        params=args.params,
        checkpoint_dir=args.logdir,
        run_eagerly=False,
    )
    tfdataset = dataset.tfdataset(
        phase='test',
        tooutputs=nn.tooutputs,
        toinputs=nn.toinputs,
        batch_size=args.params.batch_size,
    )

    if args.launch_testing:
        nn.launch_testing(tfdataset)

    inputs, labels, weights, preds, p_idx = compare_preds(dataset, nn)
    dataset.plot_example(
        dataset.files["test"][p_idx],
        phase='test',
        toinputs=nn.toinputs,
        tooutputs=nn.tooutputs,
        plot_preds=True,
        nn_name=type(nn).__name__,
    )
    plt.savefig("results_2d_synthetic", bbox_inches="tight")
    plt.show()


def compare_preds(dataset, nn):
    all_inputs = {}
    all_labels = {}
    all_weights = {}
    all_preds = {}
    for example in dataset.files["test"]:
        inputs, labels, weights, filename = dataset.get_example(
            example,
            phase='test',
            toinputs=nn.toinputs,
            tooutputs=nn.tooutputs,
        )
        preds = dataset.generator.read_predictions(filename, type(nn).__name__)
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

    similarities = np.array([])
    for labels, weights, preds in zip(
                all_labels["vint"], all_weights["vint"], all_preds["vint"]
            ):
        temp_labels = labels * weights
        temp_preds = preds * weights
        similarity = ssim(temp_labels, temp_preds)
        similarities = np.append(similarities, similarity)

    p_score = np.percentile(similarities, 50, interpolation="nearest")
    p_idx = np.argwhere(p_score == similarities)[0, 0]
    print("Minimum SSIM:        ", min(similarities))
    print("SSIM 50th percentile:", p_score)
    print("Maximum SSIM:        ", max(similarities))
    return all_inputs, all_labels, all_weights, all_preds, p_idx


if __name__ == "__main__":
    from deep_learning_velocity_estimation import architecture
    from deep_learning_velocity_estimation import datasets

    # Initialize argument parser.
    parser = ArgumentParser()

    # Add arguments to parse for training.
    parser.add_argument(
        "--nn",
        type=str,
        default="RCNN2D",
        help="Name of the architecture from `architecture` to use.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="Hyperparameters",
        help="Name of hyperparameters from `architecture` to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset1Dsmall",
        help="Name of dataset from `datasets.datasets` to use.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory in which to store the checkpoints.",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=2,
        help="Quantity of GPUs for data creation.",
    )
    parser.add_argument(
        "--launch_testing",
        action="store_true",
        help="Run inference.",
    )

    args, unknown_args = parser.parse_known_args()
    args.nn = getattr(architecture, args.nn)
    args.dataset = getattr(datasets, args.dataset)()
    args.params = getattr(architecture, args.params)()
    for arg, value in zip(unknown_args[::2], unknown_args[1::2]):
        arg = arg.strip('-')
        if arg in args.params.__dict__.keys():
            setattr(args.params, arg, value)
        else:
            raise ValueError(
                f"Argument `{arg}` not recognized. Could not match it to an "
                f"existing hyerparameter."
            )
    main(args)
