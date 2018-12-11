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

    use_vmax_weights = True

    # load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t = Table.read(fpath+fname, path='data')

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)

    disks = t['FRACPSF'][:,2] < 0.8
    ellipticals = t['FRACPSF'][:,2] >= 0.8

    # make completeness cut
    from estimate_completeness import z_lim
    zz = z_lim(t['ABSMAG_r0.1'], cosmo)
    comp_mask = (t['Z'] <= zz)

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
    N_6 = np.sum(mask_5)
    print('number of galaxies in samples 1-6:')
    print(N_1, N_2, N_3, N_4, N_5, N_6)

    # calculate vmax for each galaxy
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
    
    if use_vmax_weights:
        from estimate_completeness import vmax as vmax_func
        vmax = vmax_func(t['ABSMAG_r0.1'], cosmo)
    else:
        vmax = np.ones(len(t))

    w = 1.0/(t['FGOTMAIN']*vmax)

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

    f_disk = np.array([f_disk_1, f_disk_2, f_disk_3, f_disk_4, f_disk_5, f_disk_6])
    bin_centers = [-17.5,-18.5,-19.5,-20.5,-21.5,-22.5]

    #fpath = './data/'
    #fname = 'disk_fraction.dat'
    #ascii.write([bin_centers, f_disk], fpath+fname,
    #            names=['mag', 'f_disk'], overwrite=True)

    # bootstrap error estimate
    Nboot = 1000
    N = len (w)
    inds = np.arange(0,N)

    f = t['FRACPSF'][:,2]
    m =t['ABSMAG_r0.1']

    f_disk_1 = np.zeros(Nboot)
    f_disk_2 = np.zeros(Nboot)
    f_disk_3 = np.zeros(Nboot)
    f_disk_4 = np.zeros(Nboot)
    f_disk_5 = np.zeros(Nboot)
    f_disk_6 = np.zeros(Nboot)
    for i in range(Nboot):
        idx = np.random.choice(inds, size=N)
        ww = w[idx]
        ff = f[idx]
        mm = m[idx]
        
        disks = (ff < 0.8)

        mask_1 = (mm > -18) & (mm <= -17)
        mask_2 = (mm > -19) & (mm <= -18)
        mask_3 = (mm > -20) & (mm <= -19)
        mask_4 = (mm > -21) & (mm <= -20)
        mask_5 = (mm > -22) & (mm <= -21)
        mask_6 = (mm > -23) & (mm <= -22)

        f_disk_1[i] = 1.0*np.sum(ww[mask_1 & disks])/np.sum(ww[mask_1])
        f_disk_2[i] = 1.0*np.sum(ww[mask_2 & disks])/np.sum(ww[mask_2])
        f_disk_3[i] = 1.0*np.sum(ww[mask_3 & disks])/np.sum(ww[mask_3])
        f_disk_4[i] = 1.0*np.sum(ww[mask_4 & disks])/np.sum(ww[mask_4])
        f_disk_5[i] = 1.0*np.sum(ww[mask_5 & disks])/np.sum(ww[mask_5])
        f_disk_6[i] = 1.0*np.sum(ww[mask_6 & disks])/np.sum(ww[mask_6])

    y = [np.mean(f_disk_1),np.mean(f_disk_2),np.mean(f_disk_3),np.mean(f_disk_4),np.mean(f_disk_5),np.mean(f_disk_6)]
    err = [np.std(f_disk_1),np.std(f_disk_2),np.std(f_disk_3),np.std(f_disk_4),np.std(f_disk_5),np.std(f_disk_6)]

    fpath = './data/'
    fname = 'disk_fraction.dat'
    ascii.write([bin_centers, y, err], fpath+fname,
                names=['mag', 'f_disk', 'err'], overwrite=True)



if __name__ == '__main__':
    main()