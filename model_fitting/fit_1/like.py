"""
define liklihood function
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.table import Table
from make_mock import make_galaxy_sample
from stat_utils import histogram as custom_histogram
import matplotlib.pyplot as plt

__all__ = ['lnlike']

# defualt comparison data
# load sdss measurements
t_1 = Table.read('../../sdss_measurements/data/sample_1_all_DEV_shapes.dat', format='ascii')
t_2 = Table.read('../../sdss_measurements/data/sample_2_all_DEV_shapes.dat', format='ascii')
t_3 = Table.read('../../sdss_measurements/data/sample_3_all_DEV_shapes.dat', format='ascii')
t_4 = Table.read('../../sdss_measurements/data/sample_4_all_DEV_shapes.dat', format='ascii')
t_5 = Table.read('../../sdss_measurements/data/sample_5_all_DEV_shapes.dat', format='ascii')
t_6 = Table.read('../../sdss_measurements/data/sample_6_all_DEV_shapes.dat', format='ascii')


def shape_dist(b_to_a, weights, mask=None):
    """
    measure the projected shape distribution in bins
    """
    if mask is None:
        mask = np.array([True]*len(b_to_a))

    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    counts, err = custom_histogram(b_to_a[mask], bins=bins, weights=weights[mask])
    n = np.sum(weights[mask])
    counts = 1.0*counts/n/np.diff(bins)
    err = 1.0*err/n/np.diff(bins)
    return counts, err


def lnlike(theta=None, y=None, yerr=None, mag_lim=-17, show_plots=False):
    """
    """

    # set model parameters
    if theta is None:
        gal_type='_centrals'
        d = {}
    else:
        gal_type = '_centrals' # hack to use halotools structure
        d = {'disk_shape_mu_1'+gal_type: theta[0],
             'disk_shape_mu_2'+gal_type: theta[1],
             'disk_shape_sigma_1'+gal_type: theta[2],
             'disk_shape_sigma_2'+gal_type: theta[3],
             'elliptical_shape_mu_1'+gal_type: theta[4],
             'elliptical_shape_mu_2'+gal_type: theta[5],
             'elliptical_shape_sigma_1'+gal_type: theta[6],
             'elliptical_shape_sigma_2'+gal_type: theta[7],
             'f_disk': theta[8]}

    # set default comparison data
    if y is None:
        y = np.array(t_1['frequency'])
    if yerr is None:
        yerr = np.array(t_1['err'])
    
    # simulate mock galaxy sample
    mock = make_galaxy_sample(mag_lim=mag_lim, size=10**5, **d)

    # weight by luminosity function
    w = mock['weight']
    
    mag_key = 'obs_Mag_r'
    mask = (mock[mag_key]<mag_lim) & (mock[mag_key]>(mag_lim-1.0))
    
    # measure shape distributions
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    counts, err = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask)

    if show_plots:
        plt.figure()
        plt.fill_between(bin_centers, counts-err, counts+err)
        plt.errorbar(bin_centers, y, yerr, fmt='o', ms=4)
        plt.ylim([0,3])
        plt.xlim([0,1])
        plt.show()

    # manually delete mock table after we are done with it
    # this might help with memory issues when running emcee
    mock = None

    # combine model and measurement errors
    err_squared = err**2 + yerr**2

    # calculate penalties
    # model counts or data must be greater than zero to have defined errors
    # otherwise, penalty is zero (they match)
    with np.errstate(invalid='ignore'):
        w = -0.5*np.sum(np.where((counts>0) | (y>0), (counts - y)**2/err_squared, 0.0))

    if show_plots:
        print(w)

    return w

if __name__ == '__main__':
    lnlike(show_plots=True)

