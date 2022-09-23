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
