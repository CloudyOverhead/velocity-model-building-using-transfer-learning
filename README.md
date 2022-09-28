[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7120739.svg)](https://doi.org/10.5281/zenodo.7120739)

# Hierarchical transfer learning for deep learning velocity model building

## Prerequisites and installation

Install the latest version of `git`. **Make sure to update `git` if it is already installed.**

Clone this repository through:
```git clone https://github.com/CloudyOverhead/velocity-model-building-using-transfer-learning.git```

This should install all required packages, including [SeisCL](https://github.com/gfabieno/SeisCL). However, please follow the additional steps for completing the installation of SeisCL in the package's `README.md`.


## Generating the seismic data and training the neural networks

This section may be skipped entirely if you download the files from [Dataverse]() and put them at the root of the project repository.

### Generating velocity models and the seismic data

Use the following commands to generate each dataset:

```python vmbtl --dataset Article1D --generate```

```python vmbtl --dataset Article2D --generate```

```python vmbtl --dataset USGS --generate```

```python vmbtl --dataset Article1DSteep --generate```

```python vmbtl --dataset Article2DSteep --generate```

```python vmbtl --dataset Marmousi --generate```

You may use the `--gpus` option to control the quantity of GPUs dedicated to generation and the `--plot` option to ensure the results are correct.

### Training the neural networks

Use the following commands to produce the train the neural networks. Make sure to create the `./logs` directory if necessary (`mkdir logs`). Use `export i=0` through `export i=15`, except `export i=2` and `export i=9` for each given command to generate the ensemble of 14 networks.

```python vmbtl/automated_training.py --dataset Article1D --params Hyperparameters1D --noise --seed "($(($i*3)), $(($i*3+1)), $(($i*3+2)))" --destdir logs/weights_1d/$i --gpus 2```

```python vmbtl/automated_training.py --dataset Article2D --params Hyperparameters2D --noise --seed "($(($i*3+100)), $(($i*3+101)), $(($i*3+102)))" --restore_from "('$(pwd)/logs/weights_1d/$i/checkpoint_000060', None, None)" --destdir logs/weights_2d/$i --gpus 2```

```python vmbtl/automated_training.py --dataset Article2D --params Hyperparameters2DNoTL --noise --seed "($(($i*3+200)), $(($i*3+201)), $(($i*3+202)))" --learning_rate 8E-4 --destdir logs/weights_2d_no_tl_8E-4/$i --gpus 2```

```python vmbtl/automated_training.py --dataset Article2D --params Hyperparameters2DNoTL --noise --seed "($(($i*3+300)), $(($i*3+301)), $(($i*3+302)))" --learning_rate 8E-5 --destdir logs/weights_2d_no_tl_8E-5/$i --gpus 2```

```python vmbtl/automated_training.py --dataset Article1DSteep --params Hyperparameters1DSteep --seed "($(($i*3+1000)), $(($i*3+1001)), $(($i*3+1002)))" --destdir logs/weights_1d_steep/$i --destdir logs/weights_1d_steep/$i --gpus 2```

```python vmbtl/automated_training.py --dataset Article2DSteep --params Hyperparameters2DSteep --seed "($(($i*3+1100)), $(($i*3+1101)), $(($i*3+1102)))" --restore_from "('$(pwd)/logs/weights_1d_steep/$i/checkpoint_000060', None, None)" --destdir logs/weights_2d_steep/$i --gpus 2```


## Reproducing the figures

To generate the figures, use the following command, which will produce the predictions on the test examples as well. If you have downloaded the files from [Dataverse](), use the `--no-inference` option.

```python vmbtl/postprocess```

The figures will be available under the `./figures` directory.
