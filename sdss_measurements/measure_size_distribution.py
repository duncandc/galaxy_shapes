"""
script to measure galaxy shape distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from scipy import stats
from astropy.table import Table, Column
import time
from astropy.io import ascii
from bootstrap_frequency import bootstrap_frequency
import matplotlib.pyplot as plt


def main():

    # load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t_1 = Table.read(fpath+fname, path='data')

    # load meert catalog
    from astropy.table import Table
    fpath = '../data/UPENN/'
    fname = 'meert_vagc.hdf5'
    t_2 = Table.read(fpath+fname, path='data')

    # find matches in meert catalog
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    c = SkyCoord(ra=t_1['RA']*u.degree, dec=t_1['DEC']*u.degree)  
    catalog = SkyCoord(ra=t_2['ra']*u.degree, dec=t_2['dec']*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)  

    # add matched sizes into catalog
    t_1['r_tot'] = t_2['r_tot'][idx]

    disks = t_1['FRACPSF'][:,2] < 0.8
    ellipticals = t_1['FRACPSF'][:,2] >= 0.8

    mag_bins = np.arange(-24,-17,0.1)
    mag_bin_centers = (mag_bins[:-1]+mag_bins[1:])/2.0
    
    mask = (~np.isnan(t_1['ABSMAG_r0.1'])) & (t_1['r_tot']>0) & (t_1['r_tot']<100) & disks
    result_disks = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  t_1['r_tot'][mask], 'mean', bins=mag_bins)[0]
    err_disks = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  np.log10(t_1['r_tot'][mask]), 'std', bins=mag_bins)[0]

    mask = (~np.isnan(t_1['ABSMAG_r0.1'])) & (t_1['r_tot']>0) & (t_1['r_tot']<100) & ellipticals
    result_ellipticals = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  t_1['r_tot'][mask], 'mean', bins=mag_bins)[0]
    err_ellipticals = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  np.log10(t_1['r_tot'][mask]), 'std', bins=mag_bins)[0]

    plt.figure()
    plt.plot(t_1['ABSMAG_r0.1'],  t_1['r_tot'], '.', alpha=0.5, ms=1)
    plt.plot(mag_bin_centers, result_disks, 'o', color='blue')
    plt.plot(mag_bin_centers, result_disks-10**err_disks, '--', color='blue')
    plt.plot(mag_bin_centers, result_disks+10**err_disks, '--', color='blue')
    plt.plot(mag_bin_centers, result_ellipticals, 'o', color='orange')
    plt.plot(mag_bin_centers, result_ellipticals-10**err_ellipticals, '--', color='orange')
    plt.plot(mag_bin_centers, result_ellipticals+10**err_ellipticals, '--', color='orange')
    plt.xlim([-17,-23])
    plt.ylim([0.1,100])
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()