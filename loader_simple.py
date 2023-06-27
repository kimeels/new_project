import os
import h5py
import yaml
import numpy as np
import tensorflow as tf
from util import print_level, gen_header, parse_param_keys
from keras.utils import Sequence

class Data:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.n_params = 1   # simple
        self.n_slices = {}  # filled in later
        
    def load_config(self):
        """Loads the YAML config file stored at config_path and stores
        in self.config
        
        """
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.batch_size = int(self.config['batch_size'])
            self.epochs = int(self.config['epochs'])
            self.verbose = bool(self.config['verbose'])
            self.out_dir = self.config['out_dir']
            self.inp_dir = self.config['inp_dir']
            self.weight_dir = self.config['weight_dir']
            self.load_norms = bool(self.config['load_norms'])
            try:
                self.generator = bool(self.config['generator'])
            except KeyError:
                self.generator = False
            
            assert(self.config['mode'] in ['test', 'train', 'both'])

            # Set variables which dictate how the network runs
            self.train = True
            self.test = True
            self.datasets = ['train', 'valid', 'test']
            if self.config['mode'] == 'train':
                self.test = False
                self.datasets = ['train', 'valid']
            if self.config['mode'] == 'test':
                self.train = False
                self.datasets = ['test']

            # Make the weights directory
            os.makedirs(self.out_dir, exist_ok=True)
            # Make the output directory
            os.makedirs(self.out_dir, exist_ok=True)
                
            print_level(f'read config file {self.config_path:s}',
                        1,
                        self.verbose)
            print_level(f'running in {self.config["mode"]} mode',
                        2,
                        self.verbose)
            print_level(f'using a batch size of {self.batch_size:d}',
                        2,
                        self.verbose)
            print_level(f'using {self.epochs:d} epochs',
                        2,
                        self.verbose)

    def get_shuffle_idxs(self, ds):
        """Generates a set of random indexes to shuffle the training
        and validation data, but only does this once. Also generates
        indices for shuffling the test data, but these won't be used.

        """
        shuffle_attr = 'shuffle_idxs_' + ds
        if hasattr(self, shuffle_attr):
            shuffle_idxs = getattr(self, shuffle_attr)
        else:
            n_choice = self.n_slices[ds]
            shuffle_idxs = np.random.choice(n_choice,
                                            n_choice,
                                            replace=False)
            setattr(self, shuffle_attr, shuffle_idxs)

        return shuffle_idxs

    def load_data(self, ds, shuffle=False, ret_z=False):
        """Load all the data for a given dataset, returns input and
        output numpy arrays

        Parameters
        ----------
        ds : str
            Type of dataset to load up (e.g. 'train')

        shuffle : bool
            Whether to randomly shuffle the input and
            output data

        ret_z : bool
            Whether to return the redshifts of the
            slices

        """

        print_level(f'loading {ds:s} data',
                    1,
                    self.verbose)
        
        hf = self.get_data_path(ds)

        with h5py.File(hf, 'r') as hid:
            x = np.array(hid['x'])
            y = np.array(hid['y'])

        self.n_slices[ds] = y.shape[0]
        # massage arrs into this size for legacy reasons
        x = x.reshape((*x.shape, 1))
        y = y.reshape((*y.shape, 1))
        
        if shuffle:
            shuffle_idxs = self.get_shuffle_idxs(ds)
            x = x[shuffle_idxs, :, :, :]
            y = y[shuffle_idxs, :]

        return x, y

    
    def get_data_path(self, ds):
        return os.path.join(self.inp_dir[ds], ds+'.h5')

    
    def get_n_data(self, ds):
        hf = self.get_data_path(ds)
        with h5py.File(hf, 'r') as hid:
            n_data = hid['y'].size

        print(f'-- file {hf:s} contains {n_data:d} data points')

        return n_data

    def get_dim(self, ds):
        hf = self.get_data_path(ds)
        with h5py.File(hf, 'r') as hid:
            dim = hid['x'][0, :, :].shape

        print(f'-- slices in file {hf:s} have dimension', dim)

        return dim


class DataGenerator(Sequence):
    """Based on
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly"""

    def __init__(self, D, ds, shuffle=True):
        """A DataGenerator to provided batched data on-demand to a network

        Parameters
        ----------
        D : Data
            Instance of the Data class, which contains information
            about the data and network parameters
        ds : str
            Which dataset to load out of 'train', 'valid' and 'test'
        shuffle : bool
            Whether to shuffle the data between epochs, default is
            True

        Examples
        --------
        d = Data(config_path)
        x_train = DataGenerator(d, 'train')
        y_train = None
        x_valid = DataGenerator(d, 'valid')
        y_valid = None

        train_network(model, x_train=x_train, ...)

        """
        self.D = D
        self.ds = ds
        self.n_data = D.get_n_data(ds)
        self.dim = D.get_dim(ds)
        self.shuffle = shuffle
        self.gen_idxs()
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch

        """
        return int(self.n_data // self.D.batch_size)

    def __getitem__(self, i):
        i0 = i * self.D.batch_size
        i1 = (i + 1) * self.D.batch_size
        idxs_batch = self.idxs[i0:i1]

        x, y = self.load_data(idxs_batch)

        return x, y

    def gen_idxs(self):
        """Generates a 1D array containing the indices of all the
        slices, which we can then sample from to access the slice we
        want. """
        # We split up the input datasets by group, not by slice (so a
        # whole lightcone is shown to the network at a time)
        self.idxs = np.arange(self.n_data,
                              dtype=int)


    def load_data(self, idxs_batch):
        # d = '{0:05d}/'
        x = np.zeros((self.D.batch_size,
                      *self.dim, 1),
                     dtype=np.float32)
        y = np.zeros((self.D.batch_size,
                      self.D.n_params),
                     dtype=np.float32)
        
        hf = self.D.get_data_path(self.ds)
        with h5py.File(hf, 'r') as hid:
            for i, idx in enumerate(idxs_batch):
                x[i, :, :, 0] = np.array(hid['x'][idx, :, :], dtype=np.float32)
                y[i, :] = np.float32(hid['y'][idx])
                
        return x, y


    def on_epoch_end(self):
        """Shuffle the slices at the end of each epoch

        """
        if self.shuffle:
            np.random.shuffle(self.idxs)
