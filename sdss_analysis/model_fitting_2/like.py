"""
define liklihood function
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.table import Table
from make_mock import make_galaxy_sample
from astro_utils.schechter_functions import MagSchechter
import matplotlib.pyplot as plt
from stat_utils import histogram as custom_histogram
from lss_observations.luminosity_functions import Blanton_2003_phi

__all__ = ['lnlike']

# defualt comparison data
# load sdss measurements
default_y = np.zeros((6,19))
default_yerr = np.zeros((6,19))
t = Table.read('../data/sample_1_shapes.dat', format='ascii')
default_y[0,:] = t['frequency']
default_yerr[0,:] = t['err']
t = Table.read('../data/sample_2_shapes.dat', format='ascii')
default_y[1,:] = t['frequency']
default_yerr[1,:] = t['err']
t = Table.read('../data/sample_3_shapes.dat', format='ascii')
default_y[2,:] = t['frequency']
default_yerr[2,:] = t['err']
t = Table.read('../data/sample_4_shapes.dat', format='ascii')
default_y[3,:] = t['frequency']
default_yerr[3,:] = t['err']
t = Table.read('../data/sample_5_shapes.dat', format='ascii')
default_y[4,:] = t['frequency']
default_yerr[4,:] = t['err']
t = Table.read('../data/sample_6_shapes.dat', format='ascii')
default_y[5,:] = t['frequency']
default_yerr[5,:] = t['err']

def shape_dist(b_to_a, weights, mask=None):
    """
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

def luminosity_function(m, weights, lum_func):
    """
    """
    bins = np.arange(-23,-17,0.1)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    counts, err = custom_histogram(m, bins=bins, weights=weights)
    n = np.sum(counts)
    counts = counts/n/np.diff(bins)
    err = err/n/np.diff(bins)
    
    return counts, err

def lnlike(theta=None, y1=None, y1err=None):
    """
    log-liklihood
    """

    show_plots = True

    # set model parameters
    if theta is None:
        d = {}  # use default model components' paramaters
    else:
        d = {'disk_shape_mu_1_centrals': theta[0],
             'disk_shape_mu_2_centrals': theta[1],
             'disk_shape_sigma_1_centrals': theta[2],
             'disk_shape_sigma_2_centrals': theta[3],
             'elliptical_shape_mu_1_centrals': theta[4],
             'elliptical_shape_mu_2_centrals': theta[5],
             'elliptical_shape_sigma_1_centrals': theta[6],
             'elliptical_shape_sigma_2_centrals': theta[7],
             'morphology_m0':theta[8],
             'morphology_sigma':theta[9]}

    if y1 is None:
        y1 = default_y
    if y1err is None:
        y1err = default_yerr

    # simulate mock galaxy sample
    mock = make_galaxy_sample(mag_lim=-17, size=10**5, **d)

    # initialize luminosity function
    lum_func = MagSchechter(1.49 * 10**(-2), -20.44, -1.05)
    w = lum_func(mock['Mag_r'])
    
    # define samples
    mag_key = 'obs_Mag_r'
    mag_lims = np.array([-17.0, -18.0, -19.0, -20.0, -21.0, -22.0])
    mask_1 = (mock[mag_key]<mag_lims[0]) & (mock[mag_key]>(mag_lims[0]-1.0))
    mask_2 = (mock[mag_key]<mag_lims[1]) & (mock[mag_key]>(mag_lims[1]-1.0))
    mask_3 = (mock[mag_key]<mag_lims[2]) & (mock[mag_key]>(mag_lims[2]-1.0))
    mask_4 = (mock[mag_key]<mag_lims[3]) & (mock[mag_key]>(mag_lims[3]-1.0))
    mask_5 = (mock[mag_key]<mag_lims[4]) & (mock[mag_key]>(mag_lims[4]-1.0))
    mask_6 = (mock[mag_key]<mag_lims[5]) & (mock[mag_key]>(mag_lims[5]-1.0))
    
    # measure shape distributions
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    counts_1, err_1 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_1)
    counts_2, err_2 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_2)
    counts_3, err_3 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_3)
    counts_4, err_4 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_4)
    counts_5, err_5 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_5)
    counts_6, err_6 = shape_dist(mock['galaxy_projected_b_to_a'], w, mask=mask_6)

    if show_plots:
        plt.figure()
        plt.fill_between(bin_centers, counts_1-err_1, counts_1+err_1)
        plt.fill_between(bin_centers, counts_2-err_2, counts_2+err_1)
        plt.fill_between(bin_centers, counts_3-err_3, counts_3+err_1)
        plt.fill_between(bin_centers, counts_4-err_4, counts_4+err_1)
        plt.fill_between(bin_centers, counts_5-err_5, counts_5+err_1)
        plt.fill_between(bin_centers, counts_6-err_6, counts_6+err_1)
        plt.errorbar(bin_centers, y1[0], y1err[0], fmt='o', ms=4)
        plt.errorbar(bin_centers, y1[1], y1err[1], fmt='o', ms=4)
        plt.errorbar(bin_centers, y1[2], y1err[2], fmt='o', ms=4)
        plt.errorbar(bin_centers, y1[3], y1err[3], fmt='o', ms=4)
        plt.errorbar(bin_centers, y1[4], y1err[4], fmt='o', ms=4)
        plt.errorbar(bin_centers, y1[5], y1err[5], fmt='o', ms=4)
        plt.ylim([0,3])
        plt.xlim([0,1])
        plt.show()
        
    # combine model and measurement errors
    err_1_squared = err_1**2 + y1err[0]**2
    err_2_squared = err_2**2 + y1err[1]**2
    err_3_squared = err_3**2 + y1err[2]**2
    err_4_squared = err_4**2 + y1err[3]**2
    err_5_squared = err_5**2 + y1err[4]**2
    err_6_squared = err_6**2 + y1err[5]**2

    # calculate penalties
    # model counts or data must be greater than zero to have defined errors
    # otherwise, penalty is zero (they match)
    with np.errstate(invalid='ignore'):
        w_1 = -0.5*np.sum(np.where((counts_1>0) | (y1[0]>0), (counts_1 - y1[0])**2/err_1_squared, 0.0))
        w_2 = -0.5*np.sum(np.where((counts_2>0) | (y1[1]>0), (counts_2 - y1[1])**2/err_2_squared, 0.0))
        w_3 = -0.5*np.sum(np.where((counts_3>0) | (y1[2]>0), (counts_3 - y1[2])**2/err_3_squared, 0.0))
        w_4 = -0.5*np.sum(np.where((counts_4>0) | (y1[3]>0), (counts_4 - y1[3])**2/err_4_squared, 0.0))
        w_5 = -0.5*np.sum(np.where((counts_5>0) | (y1[4]>0), (counts_5 - y1[4])**2/err_5_squared, 0.0))
        w_6 = -0.5*np.sum(np.where((counts_6>0) | (y1[5]>0), (counts_6 - y1[5])**2/err_6_squared, 0.0))
     
    # calculate model luminosity fuction
    counts, err = luminosity_function(mock['obs_Mag_r'], w, lum_func)
    
    # calculate penalties
    bins = np.arange(-23,-17,0.1)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    nbins = len(bins)-1
    lum_func = Blanton_2003_phi(band='r')
    norm = lum_func.number_density(-17.0,-30.0)
    y = lum_func(bin_centers)/norm
    y_err = y*0.1 # add 10% error to each data point
    total_err_squared = np.sqrt(y_err**2 + err**2)
    w = -0.5*np.sum((counts - y)**2/err**2)/nbins*19

    if show_plots:
        plt.figure()
        l0 = plt.fill_between(bin_centers, counts-err, counts+err)
        #plt.errorbar(bin_centers, lum_func(bin_centers), y_err, fmt='o', ms=4)
        l1, = plt.plot(bin_centers, lum_func(bin_centers)/norm, '--')
        x  = lum_func.data['absolute_magnitude']
        y1 = lum_func.data['phi']-lum_func.data['sigma_phi']
        y2 = lum_func.data['phi']+lum_func.data['sigma_phi']
        l2 = plt.fill_between(x, y1/norm, y2/norm, alpha=0.5)
        plt.yscale('log')
        plt.ylim([10**-6,1])
        plt.legend([l0, l1, l2],['model','Blanton + (2003) fit', 'Blanton + (2003)'])
        plt.show()

    # delete mock
    mock = None
    print(w , w_1 , w_2, w_3, w_4, w_5, w_6)
    return w + w_1 + w_2 + w_3 + w_4 + w_5 + w_6


if __name__ == '__main__':
    lnlike()
