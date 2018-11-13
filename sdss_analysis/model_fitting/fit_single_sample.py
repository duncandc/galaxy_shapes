"""
script to run MCMC fitting routine for shape model
"""

from __future__ import print_function, division, absolute_import 
import numpy as np
from astropy.table import Table
import time
import emcee
from prob import lnprob

def main():
    
    galaxy_sample = '1'
    t = Table.read('../data/sample_'+galaxy_sample+'_shapes.dat', format='ascii')

    # set initial parameters
    theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04]
    
    # set MCMC parameters
    ndim, nwalkers = 8, 50
    nthreads = 2
    nsteps = 10

    # set magnitude limit for galaxy sample
    if   galaxy_sample == '1':
    	mag_lim = -17
    elif galaxy_sample == '2':
    	mag_lim = -18
    elif galaxy_sample == '3':
    	mag_lim = -19
    elif galaxy_sample == '4':
    	mag_lim = -20
    elif galaxy_sample == '5':
    	mag_lim = -21
    elif galaxy_sample == '6':
    	mag_lim = -22

    pos = [theta0 + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t['frequency'], t['err'], mag_lim), threads=nthreads)

    sampler.run_mcmc(pos, nsteps)
    print(sampler.chain)


if __name__ == '__main__':
    main()
