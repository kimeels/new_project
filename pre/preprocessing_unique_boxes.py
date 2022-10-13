import os
import re
import sys
import h5py
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
# import 

def bin_x(x, nbin):
    """Bin the neutral fraction x into nbin bins between 0 and 1

    Parameters
    ----------
    x : arr
        Neutral fraction values
    nbin : int
        Number of bins

    Examples
    --------
    h, bin_edges, bin_indices = bin_x(x, nbin)

    """
    
    h, bedg, bidx = binned_statistic(x=x, 
                                     values=x,           # values is ignored
                                     statistic='count', 
                                     bins=nbin,
                                     range=[0.0, 1.0]) 

    return h, bedg, bidx


def get_params(data_path):
    """Parses the simfast21.ini file in a simulation directory to get
    the value of omega_matter, hubble, sigma8 and fesc for that
    simulation

    Parameters
    ----------
    data_path : str
        Path to simulation directory (e.g. 00001)

    """
    ini_path = os.path.join(data_path,
                            'simfast21.ini')
    fields = ['omega_matter', 'hubble', 'sigma8', 'fesc']
    params = {}
    np = len(fields)
    ip = 0

    with open(ini_path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            elif ip == np:
                break
            if fields[ip] in line:
                v = line[line.find('=')+1:-1].strip('\n')
                params[fields[ip]] = float(v)
                ip += 1

    return params


def get_zidx(z, zs):
    """Returns the index of an array of redshifts zs that is closest to z

    Parameters
    ----------
    z : float
        Target redshift
    zs : arr
        Array of redshifts

    """
    return np.argmin(np.abs(zs - z))


def get_t_avg(data_dirs):
    """Loads up the value of <T_21> for all redshifts for all
    simulations, return a dictionary of arrays where the dictionary
    key is the simulation and the array has a column of redshifts and
    a column of <T_21> (in K)

    Parameters
    ----------
    data_dirs : arr
        All of the simulation directories to load up <T_21> for

    Examples
    --------
    FIXME: Add docs.

    """
    # Load up all the average temperatures in one go
    t_avg = {data_dir: np.loadtxt(os.path.join(data_dir,
                                              f'Output_text_files/t21_av_N200_L150.0.dat')) for data_dir in data_dirs}

    return t_avg


def get_x_avg(data_dirs):
    """Loads up the value of <x_HI> for all redshifts for all
    simulations, return a dictionary of arrays where the dictionary
    key is the simulation and the array has a column of redshifts and
    a column of <x_HI>

    Parameters
    ----------
    data_dirs : arr
        All of the simulation directories to load up <x_HI> for

    Examples
    --------
    FIXME: Add docs.

    """
    
    # Load up all the average temperatures in one go
    x_avg = {data_dir: np.loadtxt(os.path.join(data_dir,
                                              f'Output_text_files/x_HI_N200_L150.0.dat')) for data_dir in data_dirs}

    return x_avg


def get_data_dirs(data_dirs, z, zs, x_avg, nbox, nbin):
    """Decides which data_dirs to keep at redshift z, by making a cut on xHI (0.01 < xHI < 0.99) and randomly sampling nbox boxes from nbin bins of xHI

    Parameters
    ----------
    data_dirs : arr
        The simulation directories (e.g. 00001)
    z : float
        Current redshift
    zs : arr
        Array of all redshifts
    x_avg : dict
        Dictionary of neutral fractions loaded by get_x_avg
    nbox : int
        Number of boxes to sample from histogram of xHI
    nbin : int
        Number of bins to use in histogram of xHI

    """

    # Now load up the data
    zidx = get_zidx(z, zs)
    ns = data_dirs.shape[0]
    x = np.zeros(ns)
    keep_dir = np.ones(ns, dtype=bool)
    x_hi = 0
    x_lo = 0
    for i, data_dir in enumerate(data_dirs):
        # Do preprocessing on xHI
        ix = x_avg[data_dir]  # extract xHI for this z
        x[i] = ix[zidx]

        # Make cuts on x
        if x[i] > 0.99:
            keep_dir[i] = False
            x_hi += 1
        elif x[i] < 0.01:
            keep_dir[i] = False
            x_lo += 1

    x = x[keep_dir]
    data_dirs = data_dirs[keep_dir]

    print('---- xHI < 0.01', x_lo)
    print('---- xHI > 0.99', x_hi)

    # Bin x
    h, bedg, bidx = bin_x(x, nbin)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.stairs(h, edges=bedg, fill=False, color='k')
    ax.set_xlabel(r'${\sf x_{HI}}$')
    ax.set_ylabel('number of boxes')
    ax.text(0.95, 0.95, f'z = {z:.0f}', transform=ax.transAxes, ha='right',
            va='top', fontsize='small', backgroundcolor='w')
    ax.set_ylim([0, 100])
    ax.set_xlim([0., 1.])
    fig.savefig(f'xhi_dist_z{z:.0f}.pdf', bbox_inches='tight', pad_inches=0.02)

    # Normalise x
    ns = x.shape[0]
    keep_dir = np.zeros(ns, dtype=bool)

    for i in range(1, nbin+1):
        # Pick out which data_dirs fall into this x bin
        cur_dir = (bidx == i)
        ndir = cur_dir.sum()

        # This is activated only when we have more boxes in this bin
        # (ndir) than the maximum allowed (nbox)
        while ndir > nbox:
            idx = np.random.randint(low=0, high=ndir)
            arg = np.argwhere(cur_dir)[idx]
            cur_dir[arg] = False
            ndir = cur_dir.sum()  # could just take one off each time, but
            # better to sum I think

        keep_dir[cur_dir] = True

    # Extract the x values to keep
    x = x[keep_dir]
    data_dirs = data_dirs[keep_dir]

    h, bedg, bidx = bin_x(x, nbin)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.stairs(h, edges=bedg, fill=False, color='k')
    ax.set_xlabel(r'${\sf x_{HI}}$')
    ax.set_ylabel('number of boxes')
    ax.text(0.95, 0.95, f'z = {z:.0f}', transform=ax.transAxes, ha='right',
            va='top', fontsize='small', backgroundcolor='w')
    ax.set_ylim([0, 1.2 * nbox])
    ax.set_xlim([0., 1.])
    fig.savefig(f'xhi_dist_bal_z{z:.0f}.pdf',
                bbox_inches='tight',
                pad_inches=0.02)

    return data_dirs


def load_box(data_dir, nx):
    """Loads a simfast21 output cube, assumes data are floats

    Parameters
    ----------
    data_dir : str
        Simulation directory
    nx : int
        Number of simulation elements per dimension

    Examples
    --------
    FIXME: Add docs.

    """
    data_path = os.path.join(data_dir,
                             f'deltaTb/deltaTb_z{z:.3f}_N{nx:d}_L150.0.dat')
    data = np.fromfile(data_path, dtype=np.float32, count=-1)
    assert(data.shape[0] == nn), f'Read {data.shape[0]:d} but expected {nn:d} elements'
    data = data.reshape((nx, nx, nx), order='C')

    return data


def get_slc_idxs(nx, nslc_dim):
    """Generates indices for taking slices of data starting from a random slice within the box

    Parameters
    ----------
    nx : int
        Number of resolution elements per dimension
    nslc_dim : int
        Number of slices to take per dimension

    Examples
    --------
    FIXME: Add docs.

    """
    step = int(nx // nslc_dim)
    idxs = np.arange(nslc_dim) * step
    idx0 = np.random.randint(nx)  # choose random start
    idxs += idx0
    idxs = np.mod(idxs, nx)  # wrap around

    return idxs
    

# Read in config file
if len(sys.argv) < 2:
    config_fn = 'config.yaml'
else:
    config_fn = sys.argv[1]

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

nx = int(config['nx']) # 200  # number of cells per dim (N_smooth)
nn = nx ** 3
slc_idxs = config['slc_idxs']
nslc_dim = int(config['nslc_dim'])
nslc = nslc_dim * len(slc_idxs)

# Train/validation/test fractions
trf = float(config['tr_frac'])
vaf = float(config['va_frac'])
tef = float(config['te_frac'])

assert(abs(1.0 - trf - vaf - tef) < 1e-6), '-- data splits do not sum to one'

box_step = 10  # out of every box_step boxes, we split the data
ntr = int(trf * box_step)
nva = int(vaf * box_step)
nte = box_step - ntr - nva

# Keys for splitting the boxes
tr_keys = ['train' for i in range(ntr)]
va_keys = ['valid' for i in range(ntr, ntr + nva)]
te_keys = ['test' for i in range(ntr + nva, ntr + nva + nte)]
ds_keys = [*te_keys, *tr_keys, *va_keys]

nbox = int(config['nbox'])
nbin = int(config['nbin'])

print(f'-- using {ntr:d} train, {nva:d} validation and {nte:d} test boxes every {box_step:d} boxes')
print(f'-- sampling a maximum of {nbox:d} boxes from {nbin:d} bins in neutral fraction')

# Output file
hf_dir = config['out_path']
os.makedirs(hf_dir, exist_ok=True)
# Dump the yaml
with open(os.path.join(hf_dir, config_fn), 'w') as f:
    yaml.dump(config, f)

pattern = re.compile(r'(\d{5})')
K_to_mK = 1e3  # covert from K to mK

# Directories of simfast21 outputs labelled 00000/ 00001/ ...
data_path = config['data_path']
data_dirs = np.array(glob.glob(os.path.join(data_path, '?????')))

# Load up the redshift list
zs = np.loadtxt(os.path.join(data_dirs[0], 
                             'Output_text_files/zsim.txt'))
nz = zs.shape[0]

# Load up all the neutral fractions
x_avg = get_x_avg(data_dirs)

# Extract the directories which satsify the xHI constraints
data_dirs_z = [None for x in range(nz)]
all_z = [None for x in range(nz)]  # Store the redshifts
for i, z in enumerate(zs):
    print(f'-- working on z = {z:.0f}')
    data_dirs_z[i] = get_data_dirs(data_dirs=data_dirs,
                                   z=z,
                                   zs=zs,
                                   x_avg=x_avg,
                                   nbox=nbox,
                                   nbin=nbin)
    all_z[i] = np.ones(data_dirs_z[i].shape[0]) * z
    print(f'---- {data_dirs_z[i].shape[0]:d} boxes kept')

# Concatenate and shuffle all the arrays so we can extract the right
# proportions of arrays for all the data
data_dirs_z = np.concatenate(data_dirs_z)
all_z = np.concatenate(all_z)
shuffle_idxs = np.random.choice(data_dirs_z.shape[0],
                                data_dirs_z.shape[0],
                                replace=False)
data_dirs_z = data_dirs_z[shuffle_idxs]
all_z = all_z[shuffle_idxs]

# Now we want to load the average temperature and neutral fraction for
# all the remaining groups
t_avg = get_t_avg(data_dirs)

gidx = 0  # counter for storing the data in groups
n_remaining = data_dirs_z.shape[0]
j = 0
while n_remaining > box_step:
    # Extract box_step boxes to be shared amongst the train/val/test
    j0 = box_step * j
    j1 = box_step * (j + 1)
    _data_dirs_z = data_dirs_z[j0:j1]
    _all_z = all_z[j0:j1]
    for i, (z, data_dir) in enumerate(zip(_all_z, _data_dirs_z)):
        # Get the average T21 and xHI for this sim and z -- the
        # z-ordering should be the same as zs but just in case I've
        # put this assert in
        iz = get_zidx(z, zs)
        it_avg = t_avg[data_dir][iz, 1]
        ix_avg = x_avg[data_dir][iz]  # weirdly xHI is just one column
        assert(abs(t_avg[data_dir][iz, 0] - z) < 1e-6), '<T21> z different to z from zs'
    
        slcs = np.zeros((nslc, nx, nx), dtype=np.float32)
        data = load_box(data_dir, nx)
        
        # Slice the data
        for islc, slc_idx in enumerate(slc_idxs):
            idxs = get_slc_idxs(nx=nx, nslc_dim=nslc_dim)
            if slc_idx == 0:
                slc_data = data[idxs, :, :]
            elif slc_idx == 1:
                slc_data = data[:, idxs, :]
            elif slc_idx == 2:
                slc_data = data[:, :, idxs]

            # Move the sliced index to the front of the array so it
            # has shape (nslc_dim, nx, nx)
            slc_data = np.moveaxis(slc_data, slc_idx, 0)
            slcs[islc*nslc_dim : (islc+1)*nslc_dim, :, :]  = slc_data
            
        # Subtract the mean from the data and convert to mK
        slcs = (slcs - it_avg) * K_to_mK
        
        # Parameters for this simulation box
        params = get_params(data_dir)
        params['z'] = z
        params['xHI'] = ix_avg

        with h5py.File(os.path.join(hf_dir, 'deltaTb_slc.h5'), 'a') as hf:
            # Just use sequential index as group
            g = hf.create_group(f'{gidx:05d}')
            d = g.create_dataset(ds_keys[i], data=slcs)
            for k, v in params.items():
                g.attrs[k] = v

        # Output bookkeeping
        gidx += 1

    # Input bookkeeping
    n_remaining -= box_step
    j += 1
