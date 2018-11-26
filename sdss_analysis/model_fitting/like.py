"""
define liklihood function
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.table import Table
from make_mock import make_galaxy_sample

__all__ = ['lnlike']

def lnlike(theta, y, yerr, mag_lim):
    """
    """

    # set model parameters
    d = {'disk_shape_mu_1_centrals': theta[0],
         'disk_shape_mu_2_centrals': theta[1],
         'disk_shape_sigma_1_centrals': theta[2],
         'disk_shape_sigma_2_centrals': theta[3],
         'elliptical_shape_mu_1_centrals': theta[4],
         'elliptical_shape_mu_2_centrals': theta[5],
         'elliptical_shape_sigma_1_centrals': theta[6],
         'elliptical_shape_sigma_2_centrals': theta[7],
         'f_disk':theta[8]}
    
    # simulate mock galaxy sample
    mock = make_galaxy_sample(mag_lim=mag_lim, size=10**5, **d)
    
    # measure shape distribution
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    mag_key = 'obs_Mag_r'
    mask = (mock[mag_key]<mag_lim) & (mock[mag_key]>(mag_lim-1.0))
    
    x = mock['galaxy_projected_b_to_a']
    counts = np.histogram(x[mask], bins=bins)[0]
    counts = 1.0*counts/np.sum(mask)/np.diff(bins)
    
    # estimate model error
    N = np.histogram(x[mask], bins=bins)[0]

    with np.errstate(divide='ignore',invalid='ignore'):
        model_err = np.where(N!=0, 1.0/np.sqrt(N)/np.sum(mask)/np.diff(bins), 0.0)
        
    # combine model and measurement error
    total_err_squared = model_err**2 + yerr**2

    mask = (total_err_squared==0.0)
    total_err_squared[mask] = 10**(-8)

    return -0.5*np.sum((counts - y)**2/total_err_squared)


