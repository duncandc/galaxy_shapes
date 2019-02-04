"""
script to measure galaxy sersic index distributions for samples fo SDSS galaxies
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

    # define luminosity bins
    mask_0 = (t['ABSMAG_r0.1'] <= -17) & comp_mask
    mask_1 = (t['ABSMAG_r0.1'] > -18.00) & (t['ABSMAG_r0.1'] <= -17.00) & comp_mask
    mask_2 = (t['ABSMAG_r0.1'] > -19.00) & (t['ABSMAG_r0.1'] <= -18.00) & comp_mask
    mask_3 = (t['ABSMAG_r0.1'] > -20.00) & (t['ABSMAG_r0.1'] <= -19.00) & comp_mask
    mask_4 = (t['ABSMAG_r0.1'] > -20.25) & (t['ABSMAG_r0.1'] <= -20.00) & comp_mask
    mask_5 = (t['ABSMAG_r0.1'] > -20.50) & (t['ABSMAG_r0.1'] <= -20.25) & comp_mask
    mask_6 = (t['ABSMAG_r0.1'] > -20.75) & (t['ABSMAG_r0.1'] <= -20.50) & comp_mask
    mask_7 = (t['ABSMAG_r0.1'] > -21.00) & (t['ABSMAG_r0.1'] <= -20.75) & comp_mask
    mask_8 = (t['ABSMAG_r0.1'] > -21.25) & (t['ABSMAG_r0.1'] <= -21.00) & comp_mask
    mask_9 = (t['ABSMAG_r0.1'] > -21.50) & (t['ABSMAG_r0.1'] <= -21.25) & comp_mask
    mask_10 = (t['ABSMAG_r0.1'] > -21.75) & (t['ABSMAG_r0.1'] <= -21.50) & comp_mask
    mask_11 = (t['ABSMAG_r0.1'] > -22.00) & (t['ABSMAG_r0.1'] <= -21.75) & comp_mask
    mask_12 = (t['ABSMAG_r0.1'] > -23.00) & (t['ABSMAG_r0.1'] <= -22.00) & comp_mask

    N_0 = np.sum(mask_0)
    N_1 = np.sum(mask_1)
    N_2 = np.sum(mask_2)
    N_3 = np.sum(mask_3)
    N_4 = np.sum(mask_4)
    N_5 = np.sum(mask_5)
    N_6 = np.sum(mask_6)
    N_7 = np.sum(mask_7)
    N_8 = np.sum(mask_8)
    N_9 = np.sum(mask_9)
    N_10 = np.sum(mask_10)
    N_11 = np.sum(mask_11)
    N_12 = np.sum(mask_12)
    print('number of galaxies in samples 1-9:')
    print(N_0, N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10, N_11, N_12)
    print('total number of galaxies:')
    print(np.sum(comp_mask))

    # apply vmax weighting
    if use_vmax_weights:
        vmax = maximum_volume(t['ABSMAG_r0.1'], cosmo=cosmo)
    else:
        vmax = np.ones(len(t))

    # measure axis ratio distribution in bins
    bins = np.logspace(-1,1,100)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    # full sample
    x_0 = t[sersic_index_key][mask_0]
    x_1 = t[sersic_index_key][mask_1]
    x_2 = t[sersic_index_key][mask_2]
    x_3 = t[sersic_index_key][mask_3]
    x_4 = t[sersic_index_key][mask_4]
    x_5 = t[sersic_index_key][mask_5]
    x_6 = t[sersic_index_key][mask_6]
    x_7 = t[sersic_index_key][mask_7]
    x_8 = t[sersic_index_key][mask_8]
    x_9 = t[sersic_index_key][mask_9]
    x_10 = t[sersic_index_key][mask_10]
    x_11 = t[sersic_index_key][mask_11]
    x_12 = t[sersic_index_key][mask_12]

    # weights
    w_0 = 1.0/(t['FGOTMAIN'][mask_0]*vmax[mask_0])
    w_1 = 1.0/(t['FGOTMAIN'][mask_1]*vmax[mask_1])
    w_2 = 1.0/(t['FGOTMAIN'][mask_2]*vmax[mask_2])
    w_3 = 1.0/(t['FGOTMAIN'][mask_3]*vmax[mask_3])
    w_4 = 1.0/(t['FGOTMAIN'][mask_4]*vmax[mask_4])
    w_5 = 1.0/(t['FGOTMAIN'][mask_5]*vmax[mask_5])
    w_6 = 1.0/(t['FGOTMAIN'][mask_6]*vmax[mask_6])
    w_7 = 1.0/(t['FGOTMAIN'][mask_7]*vmax[mask_7])
    w_8 = 1.0/(t['FGOTMAIN'][mask_8]*vmax[mask_8])
    w_9 = 1.0/(t['FGOTMAIN'][mask_9]*vmax[mask_9])
    w_10 = 1.0/(t['FGOTMAIN'][mask_10]*vmax[mask_10])
    w_11 = 1.0/(t['FGOTMAIN'][mask_11]*vmax[mask_11])
    w_12 = 1.0/(t['FGOTMAIN'][mask_12]*vmax[mask_12])

    y_0, y_err_0 = bootstrap_frequency(x_0, bins, weights=w_0, Nboot=Nboot)
    y_1, y_err_1 = bootstrap_frequency(x_1, bins, weights=w_1, Nboot=Nboot)
    y_2, y_err_2 = bootstrap_frequency(x_2, bins, weights=w_2, Nboot=Nboot)
    y_3, y_err_3 = bootstrap_frequency(x_3, bins, weights=w_3, Nboot=Nboot)
    y_4, y_err_4 = bootstrap_frequency(x_4, bins, weights=w_4, Nboot=Nboot)
    y_5, y_err_5 = bootstrap_frequency(x_5, bins, weights=w_5, Nboot=Nboot)
    y_6, y_err_6 = bootstrap_frequency(x_6, bins, weights=w_6, Nboot=Nboot)
    y_7, y_err_7 = bootstrap_frequency(x_7, bins, weights=w_7, Nboot=Nboot)
    y_8, y_err_8 = bootstrap_frequency(x_8, bins, weights=w_8, Nboot=Nboot)
    y_9, y_err_9 = bootstrap_frequency(x_9, bins, weights=w_9, Nboot=Nboot)
    y_10, y_err_10 = bootstrap_frequency(x_10, bins, weights=w_10, Nboot=Nboot)
    y_11, y_err_11 = bootstrap_frequency(x_11, bins, weights=w_11, Nboot=Nboot)
    y_12, y_err_12 = bootstrap_frequency(x_12, bins, weights=w_12, Nboot=Nboot)

    plt.figure()
    plt.plot(bin_centers, y_0, '-')
    plt.errorbar(bin_centers, y_1, yerr=y_err_1, fmt='o')
    plt.errorbar(bin_centers, y_2, yerr=y_err_2, fmt='o')
    plt.errorbar(bin_centers, y_3, yerr=y_err_3, fmt='o')
    plt.errorbar(bin_centers, y_4, yerr=y_err_4, fmt='o')
    plt.errorbar(bin_centers, y_5, yerr=y_err_5, fmt='o')
    plt.errorbar(bin_centers, y_6, yerr=y_err_6, fmt='o')
    plt.errorbar(bin_centers, y_7, yerr=y_err_7, fmt='o')
    plt.errorbar(bin_centers, y_8, yerr=y_err_8, fmt='o')
    plt.errorbar(bin_centers, y_9, yerr=y_err_9, fmt='o')
    plt.errorbar(bin_centers, y_10, yerr=y_err_10, fmt='o')
    plt.errorbar(bin_centers, y_11, yerr=y_err_11, fmt='o')
    plt.errorbar(bin_centers, y_12, yerr=y_err_12, fmt='o')
    plt.xscale('log')
    plt.xlim([0.35,10])
    plt.show()


    # save measurements
    fpath = './data/'

    fname = 'sample_1_all_sersic_n.dat'
    ascii.write([bin_centers, y_1, y_err_1], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_2_all_sersic_n.dat'
    ascii.write([bin_centers, y_2, y_err_2], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_3_all_sersic_n.dat'
    ascii.write([bin_centers, y_3, y_err_3], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_4_all_sersic_n.dat'
    ascii.write([bin_centers, y_4, y_err_4], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_5_all_sersic_n.dat'
    ascii.write([bin_centers, y_5, y_err_5], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_6_all_sersic_n.dat'
    ascii.write([bin_centers, y_6, y_err_6], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_7_all_sersic_n.dat'
    ascii.write([bin_centers, y_7, y_err_7], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_8_all_sersic_n.dat'
    ascii.write([bin_centers, y_8, y_err_8], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_9_all_sersic_n.dat'
    ascii.write([bin_centers, y_9, y_err_9], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_10_all_sersic_n.dat'
    ascii.write([bin_centers, y_10, y_err_10], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_11_all_sersic_n.dat'
    ascii.write([bin_centers, y_11, y_err_11], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)

    fname = 'sample_12_all_sersic_n.dat'
    ascii.write([bin_centers, y_12, y_err_12], fpath+fname,
                names=['n', 'frequency', 'err'], overwrite=True)





if __name__ == '__main__':
    main()
