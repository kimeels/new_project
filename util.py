def print_level(s, level, verbose=True):
    """Print a message at different levels of indentation

    Parameters
    ----------
    s : str
        Message to print
    level : int
        Level of indentation
    verbose : bool
        Whether to print
    """
    
    if verbose:
        pre = '--' * level
        print(pre + ' ' + s)
    else:
        return


def gen_header(d):
    """Makes a header suitable for use with savetxt (add \n for plain
    write) with the space-delimited parameter names

    Parameters
    ----------
    d : Data
        Data instance

    """
    header = d.param_keys[0]
    for i in range(1, d.n_params):
        header += ' ' + d.param_keys[i]

    return header


def parse_param_keys(fn='inp_pred.txt'):
    """Reads a prediction or normalisation parameter file and returns
    the param_keys stored in the header

    Parameters
    ----------
    fn : str
        Path to file

    """
    with open(fn, 'r') as f:
        l = f.readline()
        assert(l[0] == '#'), 'Header line missing'

    l = l.strip('\n').split(' ')
    param_keys =  [x for x in l if x[1:]]

    return param_keys

