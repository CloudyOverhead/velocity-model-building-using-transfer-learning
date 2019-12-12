"""
Classes and functions that combines a SeismicGenerator and a ModelGenerator
to produce a dataset on multiple GPUs. Used by the Case class (Case.py).
"""
import numpy as np
import os
import h5py as h5
from vrmslearn.ModelGenerator import ModelGenerator
from vrmslearn.SeismicGenerator import SeismicGenerator
from vrmslearn.ModelParameters import ModelParameters
from multiprocessing import Process, Queue, Event, Value


class SampleGenerator:
    """
    Class to create one example: 1- generate models 2-simulate the data
    """
    def __init__(self, pars: ModelParameters, gpu: int = 0):
        """
        Create the ModelGenerator and SeismicGenerator objects according to pars.
        @params:
        pars (ModelParameters): Parameters for data and model creation
        gpu  (int): The GPU id to use for data computations

        """

        self.model_gen = ModelGenerator(pars)
        self.data_gen = SeismicGenerator(pars, gpu=gpu,
                                         workdir="workdir%d" % gpu)
        self.files_list = {}

    def generate(self, seed):
        """
        Generate one example
        @params:
        seed (int): Seed of the model to generate

        """
        vp, vs, rho = self.model_gen.generate_model(seed=seed)
        data = self.data_gen.compute_data(vp, vs, rho)
        vp, valid = self.model_gen.generate_labels(vp, vs, rho)
        return data, [vp, valid]

    def write(self, exampleid , savedir, data, labels, filename=None):
        """
        This method writes one example in the hdf5 format

        @params:
        exampleid (int):        The example id number
        savedir (str)   :       A string containing the directory in which to
                                save the example
        data (numpy.ndarray)  : Contains the modelled seismic data
        labels (list)  :       List of numpy array containing the labels
        filename (str):      If provided, save the example in filename.

        @returns:
        """

        if filename is None:
            filename = os.path.join(savedir, "example_%d" % exampleid)
        else:
            filename = os.path.join(savedir, filename)

        file = h5.File(filename, "w")
        file["data"] = data
        for ii, label in enumerate(labels):
            file["label%d" % ii] = label
        file.close()


class DatasetProcess(Process):
    """
    This class creates a new process to generate seismic data.
    """

    def __init__(self,
                 savepath: str,
                 sample_generator: SampleGenerator,
                 seeds: Queue):
        """
        Initialize the DatasetGenerator

        @params:
        savepath (str)   :     Path in which to create the dataset
        sample_generator (SampleGenerator): A SampleGenerator object to create
                                            examples
        seeds (Queue):   A Queue containing the seeds of models to create
        """
        super().__init__()

        self.savepath = savepath
        self.sample_generator = sample_generator
        self.seeds = seeds
        if not os.path.isdir(savepath):
            os.mkdir(savepath)

    def run(self):
        """
        Start the process to generate data
        """

        while not self.seeds.empty():
            try:
                seed = self.seeds.get(timeout=1)
            except Queue.Full:
                break
            filename = "example_%d" % seed
            if not os.path.isfile(os.path.join(self.savepath, filename)):
                data, labels = self.sample_generator.generate(seed)
                self.sample_generator.write(seed, self.savepath, data, labels,
                                            filename=filename)

def generate_dataset(pars: ModelParameters,
                     savepath: str,
                     nexamples: int,
                     seed0: int = None,
                     ngpu: int = 3):
    """
    This function creates a dataset on multiple GPUs.

    @params:
    pars (ModelParameter): A ModelParameter object containg the parameters for
                            creating examples.
    savepath (str)   :     Path in which to create the dataset
    nexamples (int):       Number of examples to generate
    seed0 (int):           First seed of the first example in the dataset. Seeds
                           are incremented by 1 for subsequents examples.
    ngpu (int):            Number of available gpus for data creation

    """

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    exampleids = Queue()
    for el in np.arange(seed0, seed0 +nexamples):
        exampleids.put(el)

    generators = []
    for jj in range(ngpu):
        thisgen = DatasetProcess(savepath,
                                 SampleGenerator(pars, gpu=jj),
                                 exampleids)
        thisgen.start()
        generators.append(thisgen)
    for gen in generators:
        gen.join()

def aggregate(examples):
    """
    This method aggregates a batch of examples created by a SampleGenerator
    object. To be used with Inputqueue.

    @params:
    examples (list):       A list of numpy arrays that contain a list with
                        all elements of example.

    @returns:
    batch (list): A list of numpy arrays that contains all examples
                           for each element of a batch.

    """
    nel = len(examples[0])
    batch = [[] for _ in range(nel)]
    for ii in range(nel):
        batch[ii] = np.stack([el[ii] for el in examples])
    #data = np.stack([el[0] for el in batch])
    batch[0] = np.expand_dims(batch[0], axis=-1)

    return batch