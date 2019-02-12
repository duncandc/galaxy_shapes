"""
script to measure galaxy color distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time
from astropy.io import ascii
from bootstrap_frequency import bootstrap_frequency
import sys
import matplotlib.pyplot as plt

# import astropy cosmology object to use for calculations
from default_cosmology import cosmo
from sdss_utils import maximum_redshift, maximum_volume


def main():

    Nboot = 100

    # weight galaxies by vmax
    use_vmax_weights = True

    # key in galaxy catalog that stores axis ratio
    sersic_index_key = 'SERSIC_N_r'

    # load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t = Table.read(fpath+fname, path='data')

    # make completeness cut
    from estimate_completeness import z_lim
    zz = maximum_redshift(t['ABSMAG_r0.1'], cosmo=cosmo)
    comp_mask = (t['Z'] <= zz)

    mag_bins = np.arange(-24, -16.99, 0.25)[::-1]
    mag_bin_centers = (mag_bins[:-1]+mag_bins[1:])/2.0
    N_mag_bins = len(mag_bins)-1

    # apply vmax weighting
    if use_vmax_weights:
        vmax = maximum_volume(t['ABSMAG_r0.1'], cosmo=cosmo)
    else:
        vmax = np.ones(len(t))

    # measure color distribution in bins
    bins = np.arange(0,1.5001,0.05)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    color = t['ABSMAG_g0.1'] - t['ABSMAG_r0.1']

    # save measurements
    fpath = './data/'

    plt.figure()
    for i in range(0, N_mag_bins):
        mask = (t['ABSMAG_r0.1']<=mag_bins[i]) & (t['ABSMAG_r0.1']>mag_bins[i+1])
        x = color[mask]
        w = 1.0/(t['FGOTMAIN'][mask]*vmax[mask])

        y, y_err = bootstrap_frequency(x, bins, weights=w, Nboot=Nboot)

        plt.errorbar(bin_centers, y, y_err, fmt='-o', ms=4)

        fname = 'sample_'+str(int(i))+'_g_minus_r.dat'
        ascii.write([bin_centers, y, y_err], fpath+fname,
                names=['g_minus_r', 'frequency', 'err'], overwrite=True)

    plt.xlim([0,1.5])
    plt.show(block=True)


if __name__ == '__main__':
    main()
