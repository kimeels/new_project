# new_project

## Simulations

The preprocessing script has been designed to work with the coeval 21
cm brightness temperature cubes produced by
[simfast21](https://github.com/mariogrs/Simfast21), though any
simulation should work if it is put in the proper format (see
pre/preprocessing.py). We use a slightly modified version of
simfast21, available [here](https://github.com/lconaboy/Simfast21)
that allows peculiar velocities to be turned off and explicitly
enforces a flat comsology by setting Omega_lambda = 1 - Omega_matter.

Individual simulations are expected to be stored in zero-padded
directories, e.g.

    00000/
    00001/
    00002/
    ...
    ?????/

with each directory containing (at least) the deltaTb/ and
Output_text_files/ directories produced by simfast21 and the
simfast21.ini file.

## Preprocessing

The `Data` class in loader.py expects an HDF5 file that has been
produced by pre/preprocessing.py. Parameters for the preprocessing are
set in a YAML file and passed on the command line as

    $ python preprocessing.py config.yaml

and an example YAML file can be found at pre/config.yaml.example --
copy and modify it to your needs.

If using pre/preprocessing.py, the steps performed are:

1. Load up the text file listing the redshifts and average neutral
fraction of each output cube

2. Flatten the neutral fraction distribution by binning the average
neutral fraction into `nbin` bins, and randomly sampling up to `nbox`
boxes from each bin -- this is done independently for every redshift

3. Load up each simulation box and sample `nslc_dim` slices from each
dimension in `slc_idxs`, where the location of the starting slice is
randomly decided and the sampled slices are spaced `nx // nslc_dim`
slices apart

4. The simulation box then has the average brightness temperature
subtracted and is converted to units of mK

5. Slices are randomly split into train/validation/test according to
the fractions specified by `tr_frac`, `va_frac` and `te_frac`,
respectively

6. Slices are written to the HDF5 file `deltaTb_slc/deltaTb_slc.h5`,
stored in sequential groups, where each group is a simulation box at a
given redshift, containing a `train`, `valid` and `test` dataset.

Simulation parameters (`omega_matter`, `sigma8`, `hubble` and `fesc`)
and other information (`z` and `xHI`) are stored as group attributes.

## Running

As with preprocessing, the actual training and testing of the network
is controlled by a YAML file (see network.yaml.example). Parameters
than be specified in this file are: the path to the preprocessed data
(`data_path`); the output directory (`out_dir`); training parameters
(`batch_size` and `epochs`); a list of simulation parameters to train
on (`param_keys`); the mode of operation (`mode`, can be `train`,
`test` or `both`) and a switch for verbosity (`verbose`, can be `True`
or `False`).

Network architecture is dictated in training_functions.py and the size
of the final layer is detected automatically by the length of the
`param_keys` list. Once the parameters have been set in the
network.yaml file, running is as easy as

    $ python main.py network.yaml

Data loading is handled by the `Data` class in loader.py, which also
handles the normalisation of the output parameters, done as

    y_i = (y_i - mu_i) - sig_i

where `mu_i` and `sig_i` are the mean and standard deviation of the
output parameter y_i. The network predictions are scaled back to their
original range before being output.
