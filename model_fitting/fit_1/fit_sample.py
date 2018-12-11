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
from multiprocessing import Pool, cpu_count
from contextlib import closing
import os
os.environ["OMP_NUM_THREADS"] = "1"

def main():

    nchunk = 25 # numnber of steps to take before reinitializing pool

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
    mag_lim = params['mag_lim'][0]
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
    
    # run first batch of steps
    print('starting initial pool...')
    pool = Pool(processes=nthreads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(y, yerr, mag_lim),  pool=pool)
    if nchunk > nsteps:
        nsteps0 = nsteps
    else:
        nsteps0 = nchunk
    sampler.run_mcmc(pos0, nsteps0, progress=True)
    print('closing pool...')
    pool.close()

    # loop through the remaining steps
    for i in range(1,nsteps//nchunk):
        print('starting new pool...')
        pool = Pool(processes=nthreads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(y, yerr, mag_lim),  pool=pool)
        sampler.run_mcmc(None, nchunk, progress=True)
        print('closing pool...')
        pool.close()

    # take ramaining steps
    if nchunk > nsteps:
        nremainder = 0.0
    else:
        nremainder = nsteps%nchunk
    
    if nremainder > 0:
        print('starting new pool...')
        pool = Pool(processes=nthreads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=(y, yerr, mag_lim),  pool=pool)
        sampler.run_mcmc(None, nremainder, progress=True)
        print('closing pool...')
        pool.close()
    
    print("Final number of steps: {0}".format(backend.iteration))


if __name__ == '__main__':
    main()


