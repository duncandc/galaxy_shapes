"""
script to measure galaxy shape distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time
from astropy.io import ascii
from bootstrap_frequency import bootstrap_frequency
import sys

# import astropy cosmology object to use for calculations
from default_cosmology import cosmo
from sdss_utils import maximum_redshift, maximum_volume


def main():

    # get arguments
    if len(sys.argv) == 2:
        shape_type = sys.argv[1]
        Nboot = sys.argv[2]
    else:
        shape_type = 'DEV'  # AB_EXP AB_DEV AB_ISO
        Nboot = 1000
    
    # weight galaxies by vmax
    use_vmax_weights = False

    # key in galaxy catalog that stores axis ratio
    shape_key = 'AB_' + shape_type

    # load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t = Table.read(fpath+fname, path='data')

    # define disk and elliptical samples
    disks = t['FRACPSF'][:,2] < 0.8
    ellipticals = t['FRACPSF'][:,2] >= 0.8

    # make completeness cut
    from estimate_completeness import z_lim
    zz = maximum_redshift(t['ABSMAG_r0.1'], cosmo=cosmo)
    comp_mask = (t['Z'] <= zz)

    # define luminosity bins
    mask_1 = (t['ABSMAG_r0.1'] > -18) & (t['ABSMAG_r0.1'] <= -17) & comp_mask
    mask_2 = (t['ABSMAG_r0.1'] > -19) & (t['ABSMAG_r0.1'] <= -18) & comp_mask
    mask_3 = (t['ABSMAG_r0.1'] > -20) & (t['ABSMAG_r0.1'] <= -19) & comp_mask
    mask_4 = (t['ABSMAG_r0.1'] > -21) & (t['ABSMAG_r0.1'] <= -20) & comp_mask
    mask_5 = (t['ABSMAG_r0.1'] > -22) & (t['ABSMAG_r0.1'] <= -21) & comp_mask
    mask_6 = (t['ABSMAG_r0.1'] > -23) & (t['ABSMAG_r0.1'] <= -22) & comp_mask

    N_1 = np.sum(mask_1)
    N_2 = np.sum(mask_2) 
    N_3 = np.sum(mask_3) 
    N_4 = np.sum(mask_4) 
    N_5 = np.sum(mask_5)
    N_6 = np.sum(mask_6)
    print('number of galaxies in samples 1-6:')
    print(N_1, N_2, N_3, N_4, N_5, N_6)
    print('total number of galaxies:')
    print(np.sum(comp_mask))

    # apply vmax weighting
    if use_vmax_weights:
        vmax = maximum_volume(t['ABSMAG_r0.1'], cosmo=cosmo)
    else:
        vmax = np.ones(len(t))

    # measure axis ratio distribution in bins
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    # full sample
    x_1 = t[shape_key][mask_1]
    x_2 = t[shape_key][mask_2]
    x_3 = t[shape_key][mask_3]
    x_4 = t[shape_key][mask_4]
    x_5 = t[shape_key][mask_5]
    x_6 = t[shape_key][mask_6]

    w_1 = 1.0/(t['FGOTMAIN'][mask_1]*vmax[mask_1])
    w_2 = 1.0/(t['FGOTMAIN'][mask_2]*vmax[mask_2])
    w_3 = 1.0/(t['FGOTMAIN'][mask_3]*vmax[mask_3])
    w_4 = 1.0/(t['FGOTMAIN'][mask_4]*vmax[mask_4])
    w_5 = 1.0/(t['FGOTMAIN'][mask_5]*vmax[mask_5])
    w_6 = 1.0/(t['FGOTMAIN'][mask_6]*vmax[mask_6])

    y_1, y_err_1 = bootstrap_frequency(x_1, bins, weights=w_1, Nboot=Nboot)
    y_2, y_err_2 = bootstrap_frequency(x_2, bins, weights=w_2, Nboot=Nboot)
    y_3, y_err_3 = bootstrap_frequency(x_3, bins, weights=w_3, Nboot=Nboot)
    y_4, y_err_4 = bootstrap_frequency(x_4, bins, weights=w_4, Nboot=Nboot)
    y_5, y_err_5 = bootstrap_frequency(x_5, bins, weights=w_5, Nboot=Nboot)
    y_6, y_err_6 = bootstrap_frequency(x_6, bins, weights=w_6, Nboot=Nboot)

    # disks
    x_1 = t[shape_key][mask_1 & disks]
    x_2 = t[shape_key][mask_2 & disks]
    x_3 = t[shape_key][mask_3 & disks]
    x_4 = t[shape_key][mask_4 & disks]
    x_5 = t[shape_key][mask_5 & disks]
    x_6 = t[shape_key][mask_6 & disks]

    w_1 = 1.0/(t['FGOTMAIN'][mask_1 & disks]*vmax[mask_1 & disks])
    w_2 = 1.0/(t['FGOTMAIN'][mask_2 & disks]*vmax[mask_2 & disks])
    w_3 = 1.0/(t['FGOTMAIN'][mask_3 & disks]*vmax[mask_3 & disks])
    w_4 = 1.0/(t['FGOTMAIN'][mask_4 & disks]*vmax[mask_4 & disks])
    w_5 = 1.0/(t['FGOTMAIN'][mask_5 & disks]*vmax[mask_5 & disks])
    w_6 = 1.0/(t['FGOTMAIN'][mask_6 & disks]*vmax[mask_6 & disks])

    y_1a, y_err_1a = bootstrap_frequency(x_1, bins, weights=w_1, Nboot=Nboot)
    y_2a, y_err_2a = bootstrap_frequency(x_2, bins, weights=w_2, Nboot=Nboot)
    y_3a, y_err_3a = bootstrap_frequency(x_3, bins, weights=w_3, Nboot=Nboot)
    y_4a, y_err_4a = bootstrap_frequency(x_4, bins, weights=w_4, Nboot=Nboot)
    y_5a, y_err_5a = bootstrap_frequency(x_5, bins, weights=w_5, Nboot=Nboot)
    y_6a, y_err_6a = bootstrap_frequency(x_6, bins, weights=w_6, Nboot=Nboot)

    # ellipticals
    x_1 = t[shape_key][mask_1 & ellipticals]
    x_2 = t[shape_key][mask_2 & ellipticals]
    x_3 = t[shape_key][mask_3 & ellipticals]
    x_4 = t[shape_key][mask_4 & ellipticals]
    x_5 = t[shape_key][mask_5 & ellipticals]
    x_6 = t[shape_key][mask_6 & ellipticals]

    w_1 = 1.0/(t['FGOTMAIN'][mask_1 & ellipticals]*vmax[mask_1 & ellipticals])
    w_2 = 1.0/(t['FGOTMAIN'][mask_2 & ellipticals]*vmax[mask_2 & ellipticals])
    w_3 = 1.0/(t['FGOTMAIN'][mask_3 & ellipticals]*vmax[mask_3 & ellipticals])
    w_4 = 1.0/(t['FGOTMAIN'][mask_4 & ellipticals]*vmax[mask_4 & ellipticals])
    w_5 = 1.0/(t['FGOTMAIN'][mask_5 & ellipticals]*vmax[mask_5 & ellipticals])
    w_6 = 1.0/(t['FGOTMAIN'][mask_6 & ellipticals]*vmax[mask_6 & ellipticals])

    y_1b, y_err_1b = bootstrap_frequency(x_1, bins, weights=w_1, Nboot=Nboot)
    y_2b, y_err_2b = bootstrap_frequency(x_2, bins, weights=w_2, Nboot=Nboot)
    y_3b, y_err_3b = bootstrap_frequency(x_3, bins, weights=w_3, Nboot=Nboot)
    y_4b, y_err_4b = bootstrap_frequency(x_4, bins, weights=w_4, Nboot=Nboot)
    y_5b, y_err_5b = bootstrap_frequency(x_5, bins, weights=w_5, Nboot=Nboot)
    y_6b, y_err_6b = bootstrap_frequency(x_6, bins, weights=w_6, Nboot=Nboot)

    # save measurements
    fpath = './data/'
    
    fname = 'sample_1_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_1, y_err_1], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_1_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_1a, y_err_1a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_1_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_1b, y_err_1b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_2_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_2, y_err_2], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_2_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_2a, y_err_2a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_2_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_2b, y_err_2b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_3_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_3, y_err_3], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_3_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_3a, y_err_3a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_3_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_3b, y_err_3b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_4_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_4, y_err_4], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_4_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_4a, y_err_4a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_4_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_4b, y_err_4b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_5_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_5, y_err_5], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_5_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_5a, y_err_5a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_5_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_5b, y_err_5b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_6_all_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_6, y_err_6], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_6_disks_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_6a, y_err_6a], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    fname = 'sample_6_ellipticals_'+shape_type+'_shapes.dat'
    ascii.write([bin_centers, y_6b, y_err_6b], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)


if __name__ == '__main__':
    main()