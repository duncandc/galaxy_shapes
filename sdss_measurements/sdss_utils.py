"""
script to measure galaxy shape distributions for samples fo SDSS galaxies
"""

from __future__ import print_function, division
import numpy as np
from scipy.interpolate import interp1d
from astro_utils.magnitudes import absolute_magnitude_lim, color_k_correct

# import default astropy cosmology object to use for calculations
from default_cosmology import cosmo as default_cosmo


__all__=('maximum_redshift', 'maximum_volume')


def maximum_redshift(M, app_mag_lim=17.77, cosmo=None, dz=1e-5, zmax=1.0):
    """
    calculate the maximumum redshift a galaxy with a r-band,
    k-corrected to z=0.1, absolute magnitude
    will be observed at in sdss

    Parameters
    ----------
    M : array_like
        r-band absolute magnitude k-corrected to z=0.1

    app_mag_lim : float
        apparent magnitude limit 

    cosmo : astropy.cosmology object

    dz : float
        redshift spacing when building interpolation function

    zmax : float
        maximum redshift to use for interpolation

    Returns
    -------
    z_lim : numpy.array
    """

    if cosmo is None:
        cosmo = default_cosmo

    # sample redshift range
    zmin = dz
    z = np.arange(zmin, zmax, dz)
    Mr_lim = absolute_magnitude_lim(z, app_mag_lim, cosmo=cosmo)

    # make an adjustment to account for k-correction
    # see Yang et al. (2009) eq. 3, 4, and 5
    Mr_lim_01 = Mr_lim + 2.5*np.log10((z+0.9)/1.1) - 1.62*(z-0.1) - 0.1

    # invert and interpolate relation
    f = interp1d(Mr_lim_01, z, kind='linear', fill_value='extrapolate')

    # return redshift as a function of absolute magnitude
    return f(M)


def maximum_volume(M, cosmo=None, app_mag_lim=17.77, dz=1e-5, zmax=1.0, omega=4.0*np.pi):
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

    if cosmo is None:
        cosmo = default_cosmo
    
    f = omega/(4.0*np.pi)
    z = maximum_redshift(M, cosmo=cosmo, app_mag_lim=app_mag_lim, dz=dz, zmax=zmax)
    return f*cosmo.comoving_volume(z).value
