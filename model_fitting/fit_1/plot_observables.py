"""
utility to plot paramater chains
"""

import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np
from like import shape_dist
from make_mock import make_galaxy_sample
from astro_utils.schechter_functions import MagSchechter
from lss_observations.luminosity_functions import Blanton_2003_phi
from astropy.table import Table
import sys

def plot_shape_distributions(theta, sample='sample_1'):
    """
    """
    
    gal_type = '_centrals' # hack to use halotools structure
    d = {'disk_shape_mu_1'+gal_type: theta[0],
         'disk_shape_mu_2'+gal_type: theta[1],
         'disk_shape_sigma_1'+gal_type: theta[2],
         'disk_shape_sigma_2'+gal_type: theta[3],
         'elliptical_shape_mu_1'+gal_type: theta[4],
         'elliptical_shape_mu_2'+gal_type: theta[5],
         'elliptical_shape_sigma_1'+gal_type: theta[6],
         'elliptical_shape_sigma_2'+gal_type: theta[7],
         'f_disk':theta[8]}

    if sample   == 'sample_1':
        mag_bins = np.array([-17,-18])
    elif sample == 'sample_2':
        mag_bins = np.array([-18,-19])
    elif sample == 'sample_3':
        mag_bins = np.array([-19,-20])
    elif sample == 'sample_4':
        mag_bins = np.array([-20,-21])
    elif sample == 'sample_5':
        mag_bins = np.array([-21,-22])
    elif sample == 'sample_6':
        mag_bins = np.array([-22,-23])


    # simulate mock galaxy sample
    mag_lim = mag_bins[0]
    mock = make_galaxy_sample(mag_lim=mag_lim, size=10**7, **d)

    # weight by luminosity function
    w = mock['weight']

    fig, ax = plt.subplots(1, 1, figsize=(3.3,3.3))
        
    # load sdss measurements
    t = Table.read('../../sdss_measurements/data/'+sample+'_all_DEV_shapes.dat', format='ascii')
        
    # plot sdss distribution
    ax.errorbar(t['q'], t['frequency'], t['err'], fmt='o', ms=4, color='black')
    
    # measure shape distribution
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    mag_lim = mag_bins[0]
    mag_key = 'obs_Mag_r'
    mask = (mock[mag_key]<mag_lim) & (mock[mag_key]>(mag_lim-1.0))
    
    disks = mock['disk'] == True
    ellipticals = mock['elliptical'] == True

    f_disk = np.sum(w[disks & mask])/np.sum(w[mask])
    
    x = mock['galaxy_projected_b_to_a']

    counts, err = shape_dist(x, w, mask=mask)
    ax.plot(bin_centers, counts, color='black')

    counts_1, err_1 = shape_dist(x, w, mask=[mask & disks])
    counts_1 = 1.0*f_disk * counts_1
    ax.plot(bin_centers, counts_1, color='blue')

    counts_2, err_2 = shape_dist(x, w, mask=[mask & ellipticals])
    counts_2 = 1.0 * (1.0-f_disk) * counts_2
    ax.plot(bin_centers, counts_2, color='red')
        
    mag_key = 'Mag_r'
    mask = (mock[mag_key]<mag_lim) & (mock[mag_key]>(mag_lim-1.0)) & disks
    
    counts_3, err_3 = shape_dist(x, w, mask=mask)
    counts_3 = 1.0* f_disk * counts_3
    ax.plot(bin_centers, counts_3, '--', color='blue')

    ax.set_xlim([0,1])

    return fig


def main():
    """
    """

    fpath = './figures/'

    if len(sys.argv)>1:
        sample = sys.argv[1]
    else:
        print("The first positional argument must be the galaxy sample, e.g. 'sample_1'.")
        sys.exit()

    # load sample and run information
    _temp = __import__(sample+'_fitting_params')
    params = _temp.params

    nwalkers = params['nwalkers']
    mag_lim = params['mag_lim']

    reader = emcee.backends.HDFBackend("chains/"+sample+"_chain.hdf5")
    data = reader.get_chain()

    ndim = params['ndim']
    theta = [np.percentile(data.T[i][:,-1],[16, 50, 84])[1] for i in range(ndim)]

    fig1 = plot_shape_distributions(theta, sample)

    plt.show()


if __name__ == '__main__':
    main()