"""
script to measure galaxy shape distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time
from astropy.io import ascii
from bootstrap_frequency import bootstrap_frequency
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astro_utils.magnitudes import absolute_magnitude_lim, color_k_correct
from astropy.cosmology import FlatLambdaCDM


def z_lim(M, cosmo, mlim = 17.77, zmin=0.01, zmax=1.0):
    """
    calculate the maximumum redshift a galaxy with a r-band,
    k-corrected to z=0.1, absolute magnitude
    will be observed at in sdss

    Parameters
    ----------
    M : array_like

    cosmo : astropy.cosmology object

    mlim : 

    zmin : 

    zmax :

    Returns
    -------
    z_lim : numpy.array
    """

    # sample redshift range
    z = np.linspace(zmin, zmax, 1000)
    Mr_lim = absolute_magnitude_lim(z, mlim, cosmo=cosmo)

    # make an adjustment to account for k-correction
    # see Yang et al. (2009) eq. 3, 4, and 5
    Mr_lim_01 = Mr_lim + 2.5*np.log10((z+0.9)/1.1) - 1.62*(z-0.1) - 0.1

    # invert and interpolate relation
    f = interp1d(Mr_lim_01, z, kind='linear', fill_value='extrapolate')

    # return redshift as a function of absolute magnitude
    return f(M)


def vmax(M, cosmo, mlim = 17.77, zmin=0.01, zmax=1.0, omega=4.0*np.pi):
    """
    calculate the maximumum volume a galaxy with a r-band,
    k-corrected to z=0.1, absolute magnitude
    will be observed within in sdss

    Parameters
    ----------
    M : array_like

    cosmo : astropy.cosmology object

    mlim : float

    zmin : float

    zmax : float

    omega : float
        solid angle

    Returns
    -------
    vmax : numpy.array
    """
    
    f = omega/(4.0*np.pi)
    z = z_lim(M, cosmo, mlim, zmin, zmax)
    return f*cosmo.comoving_volume(z).value


def main():

    # load sample selection
    from astropy.table import Table
    fpath = '../data/SDSS_Main/'
    fname = 'sdss_vagc.hdf5'
    t = Table.read(fpath+fname, path='data')

    print(t.dtype.names)
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)

    z = np.linspace(0,0.5,100)
    Mr_lim = absolute_magnitude_lim(z, 17.77, cosmo=cosmo)

    # see Yang et al. (2009) eq. 3, 4, and 5
    Mr_lim_01 = Mr_lim + 2.5*np.log10((z+0.9)/1.1) - 1.62*(z-0.1) - 0.1

    color = np.array(t['ABSMAG_g0.1'] - t['ABSMAG_r0.1'])

    plt.figure()
    plt.scatter(t['ABSMAG_r0.1'], color,
                s=1, c=color, vmin=0.2, vmax=1.2, cmap='jet')
    plt.xlim([-17,-24])
    plt.ylim([0,1.5])
    plt.xlabel('z')
    plt.ylabel('g-r')
    plt.colorbar(label='g-r')
    plt.show()

    plt.figure()
    plt.scatter(t['Z'], t['ABSMAG_r0.1'],
                s=1, c=color, vmin=0.2, vmax=1.2, cmap='jet')
    plt.plot(z, Mr_lim, '-', color='red')
    plt.plot(z, Mr_lim_01, '--', color='red')
    plt.xlim([0,0.4])
    plt.ylim([-17,-24])
    plt.xlabel('z')
    plt.ylabel(r'$M_{r0.1}$')
    plt.show()

if __name__ == '__main__':
    main()