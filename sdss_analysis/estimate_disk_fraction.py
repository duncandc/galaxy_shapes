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
    print('number of galaxies in samples 1-6:')
    print(N_1, N_2, N_3, N_4, N_5, N_6)

    # calculate vmax for each galaxy
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
    vmax = cosmo.comoving_volume(t['Z']).value
    vmax = vmax/np.mean(vmax)

    w = 1.0/vmax

    mask = mask_1 & disks
    f_disk_1 = 1.0*np.sum(w[mask])/np.sum(w[mask_1])

    mask = mask_2 & disks
    f_disk_2 = 1.0*np.sum(w[mask])/np.sum(w[mask_2])

    mask = mask_3 & disks
    f_disk_3 = 1.0*np.sum(w[mask])/np.sum(w[mask_3])

    mask = mask_4 & disks
    f_disk_4 = 1.0*np.sum(w[mask])/np.sum(w[mask_4])

    mask = mask_5 & disks
    f_disk_5 = 1.0*np.sum(w[mask])/np.sum(w[mask_5])

    mask = mask_6 & disks
    f_disk_6 = 1.0*np.sum(w[mask])/np.sum(w[mask_6])

    print('disk fraction in samples 1-6:')
    print(f_disk_1,f_disk_2,f_disk_3,f_disk_4,f_disk_5,f_disk_6)



if __name__ == '__main__':
    main()