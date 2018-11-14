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
    
    galaxy_sample = '4'
    t = Table.read('../data/sample_'+galaxy_sample+'_shapes.dat', format='ascii')

    f = open("./chains/sample_"+galaxy_sample+"_chain.dat", "w")
    f.close()

    # set initial parameters
    theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.1]
    
    # set MCMC parameters
    ndim = len(theta0)
    nwalkers = 50
    nthreads = 2
    nsteps = 100

    # set magnitude limit for galaxy sample
    if   galaxy_sample == '1':
    	mag_lim = -17
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.8686]
    elif galaxy_sample == '2':
    	mag_lim = -18
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.8788]
    elif galaxy_sample == '3':
    	mag_lim = -19
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.6713]
    elif galaxy_sample == '4':
    	mag_lim = -20
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.4712]
    elif galaxy_sample == '5':
    	mag_lim = -21
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.3106]
    elif galaxy_sample == '6':
    	mag_lim = -22
        theta0 = [0.1071, 0.8, 0.0078, 0.0056, 0.16667, 0.4, 0.01068, 0.04, 0.2284]

    pos0 = [theta0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t['frequency'], t['err'], mag_lim), threads=nthreads)

    #sampler.run_mcmc(pos0, nsteps)
    
    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        position = result[0]
        f = open("./chains/sample_"+galaxy_sample+"_chain.dat", "a")
        for k in range(position.shape[0]):
            s = position[k]
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,s))))
        f.close()


if __name__ == '__main__':
    main()
