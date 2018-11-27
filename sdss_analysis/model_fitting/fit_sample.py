"""
run mcmc chain for a galaxy sample
"""

import numpy as np
from astropy.table import Table
from astropy.io import ascii
import time
import emcee
from prob import lnprob
import sys
from chains.chain_utils import return_final_step
from multiprocessing import Pool, cpu_count
from contextlib import closing
import os
os.environ["OMP_NUM_THREADS"] = "1"

def main():

    if len(sys.argv)>1:
        sample = sys.argv[1]
    else:
        print("The first positional argument must be the galaxy sample, e.g. 'sample_1'.")
        sys.exit()

    chain_dir = './chains/'

    # load parameters for sample
    _temp = __import__(sample+'_fitting_params')
    params = _temp.params

    # retreive parameters for the galaxy sample
    mag_lim = params['mag_lim']
    ndim = params['ndim']
    nwalkers = params['nwalkers']
    nthreads = params['nthreads']
    nsteps = params['nsteps']

    # initialize walkers
    pos0 = [params['theta0'] + params['dtheta']*np.random.randn(params['ndim']) for i in range(params['nwalkers'])]
    
    # check multiprocessing arguments
    ncpu = cpu_count()
    print("Using {0} CPU cores out of a possible {1}.".format(nthreads, ncpu))

    # load sdss measurements
    t = Table.read(params['comparison_fname'], format='ascii')
    y = t['frequency']
    yerr = t['err']

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = chain_dir + sample + '_chain.hdf5'
    backend = emcee.backends.HDFBackend(filename)

    if params['continue_chain'] == False:
        backend.reset(nwalkers, ndim)
    else:
        print("Initial number of steps: {0}".format(backend.iteration))
        # retrieve final position of chains
        samples = backend.get_chain()
        pos0 = samples.T[:,:,-1].T

    with closing(Pool(processes=nthreads)) as pool:
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(y, yerr, mag_lim),  pool=pool)

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(pos0, iterations=nsteps, progress=True):
            continue

    print("Final number of steps: {0}".format(backend.iteration))


if __name__ == '__main__':
    main()


