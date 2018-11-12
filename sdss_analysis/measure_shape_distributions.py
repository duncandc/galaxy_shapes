"""
script to measure galaxy shape distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time
from astropy.io import ascii
from bootstrap_frequency import bootstrap_frequency

def main():

	# load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t = Table.read(fpath+fname, path='data')

    disks = t['FRACPSF'][:,2] < 0.8
    ellipticals = t['FRACPSF'][:,2] >= 0.8

    mask_1 = (t['ABSMAG_r0.1'] > -18) & (t['ABSMAG_r0.1'] <= -17)
    mask_2 = (t['ABSMAG_r0.1'] > -19) & (t['ABSMAG_r0.1'] <= -18)
    mask_3 = (t['ABSMAG_r0.1'] > -20) & (t['ABSMAG_r0.1'] <= -19)
    mask_4 = (t['ABSMAG_r0.1'] > -21) & (t['ABSMAG_r0.1'] <= -20)
    mask_5 = (t['ABSMAG_r0.1'] > -22) & (t['ABSMAG_r0.1'] <= -21)
    mask_6 = (t['ABSMAG_r0.1'] > -23) & (t['ABSMAG_r0.1'] <= -22)

    N_1 = np.sum(mask_1)
    N_2 = np.sum(mask_2) 
    N_3 = np.sum(mask_3) 
    N_4 = np.sum(mask_4) 
    N_5 = np.sum(mask_5)
    N_6 = np.sum(mask_5)
    print('number of ghalaxies in samples 1-6:')
    print(N_1, N_2, N_3, N_4, N_5, N_6)

    # calculate vmax for each galaxy
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
    vmax = cosmo.comoving_volume(t['Z']).value
    vmax = vmax/np.mean(vmax)

    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2.0

    shape_key = 'AB_ISO'  # AB_EXP AB_DEV AB_ISO

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

    y_1, y_err_1 = bootstrap_frequency(x_1, bins, weights=w_1, Nboot=100)
    y_2, y_err_2 = bootstrap_frequency(x_2, bins, weights=w_2, Nboot=100)
    y_3, y_err_3 = bootstrap_frequency(x_3, bins, weights=w_3, Nboot=100)
    y_4, y_err_4 = bootstrap_frequency(x_4, bins, weights=w_4, Nboot=100)
    y_5, y_err_5 = bootstrap_frequency(x_5, bins, weights=w_5, Nboot=100)
    y_6, y_err_6 = bootstrap_frequency(x_6, bins, weights=w_6, Nboot=100)

    fpath = './data/'
    
    fname = 'sample_1_shapes.dat'
    ascii.write([bin_centers, y_1, y_err_1], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_2_shapes.dat'
    ascii.write([bin_centers, y_2, y_err_2], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_3_shapes.dat'
    ascii.write([bin_centers, y_3, y_err_3], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_4_shapes.dat'
    ascii.write([bin_centers, y_4, y_err_4], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_5_shapes.dat'
    ascii.write([bin_centers, y_5, y_err_5], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)
    
    fname = 'sample_6_shapes.dat'
    ascii.write([bin_centers, y_6, y_err_6], fpath+fname,
                names=['q', 'frequency', 'err'], overwrite=True)


if __name__ == '__main__':
    main()