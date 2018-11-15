"""
utility to read chain files
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.io import ascii

colnames = {0:'col1',
            1:'col2',
            2:'col3',
            3:'col4',
            4:'col5',
            5:'col6',
            6:'col7',
            7:'col8',
            8:'col9',
            9:'col10',
            10:'col11'}


def open_chains(fname, nwalkers):
    """
    return array of chains
    """

    data = ascii.read(fname)
    nsteps = len(data)//nwalkers
    return data[:nsteps*nwalkers]


def return_parameter_chains(paramater, fname, nwalkers):
    """
    """
    data = open_chains(fname, nwalkers)
    nsteps = len(data)//nwalkers
    p = np.array(data[paramater])
    p = p.reshape((nsteps,nwalkers)).T
    return p


def return_final_step(fname, nwalkers):
    """
    """

    data = open_chains(fname, nwalkers)
    nparams = len(data.colnames) - 1

    j = -1
    pos = np.zeros((nwalkers, nparams))
    for i in range(nparams):
        p = return_parameter_chains(colnames[i+1], fname, nwalkers)
        for j in range(nwalkers):
            pos[j,i] = p[j,-1]

    return pos


