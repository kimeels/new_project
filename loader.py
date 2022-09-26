import h5py
import yaml
import numpy as np
import tensorflow as tf
from util import print_level

class Data:
    def __init__(self, config_path):
        self.datasets = ['train', 'valid', 'test']
        self.config_path = config_path
        self.load_config()
        self.get_data_info()
        self.compute_param_norms()


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
            self.param_keys = self.config['param_keys']

            assert(self.config['mode'] in ['test', 'train', 'both'])

            # Set variables which dictate how the network runs
            self.train = True
            self.test = True
            if self.config['mode'] == 'train':
                self.test = False
            if self.config['mode'] == 'test':
                self.train = False
            
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
            

            
    def get_data_info(self):
        """Reads the properties of the HDF5 data file
        
        """
        self.n_params = len(self.param_keys)

        with h5py.File(self.config['data_path'], 'r') as hf:
            groups = list(hf.keys())  # all groups
            g0 = groups[0]
            self.n_groups = len(groups)
            self.n_slices = {ds: hf[g0+'/'+ds].shape[0]
                             for ds in self.datasets }  # per group
            self.dim = (hf[g0+'/'+self.datasets[0]].shape[1],
                        hf[g0+'/'+self.datasets[0]].shape[2],
                        1)  # dimensions of a single input image 

        print_level(f'read HDF5 file {self.config["data_path"]:s}',
                    1,
                    self.verbose)
        print_level(f'using {self.n_params:d} params',
                    2,
                    self.verbose)
        print_level(f'have {self.n_groups:d} groups, where group each has:',
                    2,
                    self.verbose)
        for ds in self.datasets:
            print_level(f'{self.n_slices[ds]:d} {ds} slices',
                        3,
                        self.verbose)
        print_level(f'and each slice has dimension {self.dim}',
                    3,
                    self.verbose)
        
            
    def compute_param_norms(self):
        """Computes the mean and standard deviation of each of the
        parameters, for use in normalisation

        """
        params = np.zeros((self.n_groups, self.n_params),
                          dtype=np.float32)

        with h5py.File(self.config['data_path'], 'r') as hf:
            for i, group in enumerate(hf.keys()):
                for j, param_key in enumerate(self.param_keys):
                    params[i, j] = hf[group].attrs[param_key]

        self.norms = {'mu': np.mean(params, axis=1),
                      'sig': np.std(params, axis=1)}


    def get_shuffle_idxs(self):
        """Generates a set of random indexes to shuffle the training
        and validation data, but only does this once. Also generates
        indices for shuffling the test data, but these won't be used.

        """
        shuffle_attr = 'shuffle_idxs'
        if hasattr(self, shuffle_attr):
            shuffle_idxs = getattr(self, shuffle_attr)
        else:
            shuffle_idxs = {ds:
                            np.random.choice(self.n_groups * self.n_slices[ds],
                                             self.n_groups * self.n_slices[ds],
                                             replace=False)
                            for ds in self.datasets}
            setattr(self, shuffle_attr, shuffle_idxs)

        return shuffle_idxs


    def load_data(self, ds, shuffle=False):
        """Load all the data for a given dataset, returns input and
        output numpy arrays

        Parameters
        ----------
        ds : str
            Type of dataset to load up (e.g. 'train')

        shuffle : bool
            Whether to randomly shuffle the input and
            output data

        """

        print_level(f'loading {ds:s} data',
                    1,
                    self.verbose)
        
        ns = self.n_slices[ds]   # number of slices per group
        nt = ns * self.n_groups  # total number of slices
        x = np.zeros((nt, *self.dim), dtype=np.float32)      # input
        y = np.zeros((nt, self.n_params), dtype=np.float32)  # output

        with h5py.File(self.config['data_path'], 'r') as hf:
            groups = list(hf.keys())

            for i, group in enumerate(groups):
                # Positions in the main output array
                i0 = i * ns
                i1 = (i + 1) * ns

                # Temporaray work arrays
                _x = np.array(hf[group+'/'+ds],
                              dtype=np.float32).reshape((ns, *self.dim))
                _y = np.array([hf[group].attrs[param_key]
                               for param_key in self.param_keys],
                              dtype=np.float32).reshape((1, self.n_params))
                # These groups share output params
                _y = np.tile(_y, ns).reshape((ns, self.n_params))

                # Store
                x[i0:i1, :, :, :] = _x
                y[i0:i1, :] = _y

        # Normalise parameters
        for i in range(self.n_params):
            y[:, i] = (y[:, i] - self.norms['mu'][i]) / self.norms['sig'][i]

        if shuffle:
            shuffle_idxs = self.get_shuffle_idxs()[ds]
            x = x[shuffle_idxs, :, :, :]
            y = y[shuffle_idxs, :]
            
        return x, y        
