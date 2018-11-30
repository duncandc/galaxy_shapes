"""
"""
from __future__ import print_function, division
import numpy as np
import emcee
from astropy.table import Table
import matplotlib.pyplot as plt
import sys


def main():

    if len(sys.argv)>1:
        sample = sys.argv[1]
    else:
        print("The first positional argument must be the galaxy sample, e.g. 'sample_1'.")
        sys.exit()

    fpath = './figures/' + sample + '/'

    # load sample and run information
    _temp = __import__(sample+'_fitting_params')
    params = _temp.params

    nwalkers = params['nwalkers']
    mag_lim = params['mag_lim']

    reader = emcee.backends.HDFBackend("chains/"+sample+"_chain.hdf5")
    data = reader.get_chain()

    nsteps = len(data)
    print("{0} steps with {1} walkers.".format(nsteps, nwalkers))

   
if __name__ == '__main__':
    main()