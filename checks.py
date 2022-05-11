
############################################################################
####                            Libraries                               ####
############################################################################

import os

from pathlib import Path

############################################################################
####                        Routines & definitions                      ####
############################################################################


def check_out(*args):
    '''
        Check whether output dir exist
    '''
    for arg in args:
        if not os.path.isdir(arg):
            os.mkdir(arg)


def check_pathlib_Path(path):
    '''
        Check if the provided path is a pathlib.Path object

        Parameters
        ----------
        path            : `string` or `pathlib.Path`
            Path to the images

        Returns
        -------

    '''
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise RuntimeError('The provided path ({}) is neither a String nor'
                        ' a pathlib.Path object.'.format(arg))
