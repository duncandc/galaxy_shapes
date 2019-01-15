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

    # copnvert from angle to physical size
    from default_cosmology import cosmo
    kpc_per_arcsec = cosmo.angular_diameter_distance(t_1['Z'])*(10**3)*(2.0*np.pi/360.0)/(60*60)
    r = np.array(t_1['r_tot']) * kpc_per_arcsec.value

    disks = t_1['FRACPSF'][:,2] < 0.8
    ellipticals = t_1['FRACPSF'][:,2] >= 0.8

    mag_bins = np.arange(-24,-17,0.1)
    mag_bin_centers = (mag_bins[:-1]+mag_bins[1:])/2.0
    
    mask = (~np.isnan(t_1['ABSMAG_r0.1'])) & (r>0) & (r<100) & disks
    result_disks = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  r[mask], 'mean', bins=mag_bins)[0]
    err_disks = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  np.log10(r[mask]), 'std', bins=mag_bins)[0]

    mask = (~np.isnan(t_1['ABSMAG_r0.1'])) & (r>0) & (r<100) & ellipticals
    result_ellipticals = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  r[mask], 'mean', bins=mag_bins)[0]
    err_ellipticals = stats.binned_statistic(t_1['ABSMAG_r0.1'][mask],  np.log10(r[mask]), 'std', bins=mag_bins)[0]

    mu = effective_surface_brightness(t_1['ABSMAG_r0.1'], r)
    

    plt.figure()
    #plt.plot(t_1['ABSMAG_r0.1'],  t_1['r_tot'], '.', alpha=0.5, ms=1)
    plt.scatter(t_1['ABSMAG_r0.1'],  r, marker='o', s=1, c=mu, vmax=30,vmin=20)
    plt.plot(mag_bin_centers, result_disks, 'o', color='blue')
    plt.plot(mag_bin_centers, result_disks-10**err_disks, '--', color='blue')
    plt.plot(mag_bin_centers, result_disks+10**err_disks, '--', color='blue')
    plt.plot(mag_bin_centers, result_ellipticals, 'o', color='orange')
    plt.plot(mag_bin_centers, result_ellipticals-10**err_ellipticals, '--', color='orange')
    plt.plot(mag_bin_centers, result_ellipticals+10**err_ellipticals, '--', color='orange')
    plt.xlim([-17,-23])
    plt.ylim([0.1,100])
    plt.yscale('log')
    plt.colorbar()
    plt.show()

def effective_surface_brightness(m, r):
    return np.array(m + 2.5*np.log10(2.0*np.pi*r**2) + 36.57)


if __name__ == '__main__':
    main()