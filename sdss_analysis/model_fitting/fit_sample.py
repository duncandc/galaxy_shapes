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

    # if not continuing a chain from a previous run
    # create a new file to output chain
    # otherwise load chain
    if params['continue_chain'] == False:
        f = open("./chains/"+sample+"_chain.dat", "w")
        f.close()

        # initialize walkers
        pos0 = [params['theta0'] + params['dtheta']*np.random.randn(params['ndim']) for i in range(params['nwalkers'])]
    else:
        print('continuing chain.')
        # set intial position to the last complete step
        pos0 = return_final_step(chain_dir + sample + '_chain.dat', nwalkers)

    # load sdss measurements
    t = Table.read(params['comparison_fname'], format='ascii')
    y = t['frequency']
    yerr = t['err']

    # intialize sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, yerr, mag_lim), threads=nthreads)

    # save the progress after each step
    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        position = result[0]
        f = open(chain_dir + sample + '_chain.dat', 'a')
        for k in range(position.shape[0]):
            s = position[k]
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,s))))
        f.close()




if __name__ == '__main__':
    main()


