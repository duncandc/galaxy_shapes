"""
utilities for processing lrg sample
"""

from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
import numpy as np

__all__=['interpolated_distmod']
__author__ = ['Duncan Campbell']


def interpolated_distmod(z, cosmo):
    """
	Paramaters
	----------
	z : array_like
	    array of redshifts

	cosmo : astropy.cosmnology object
	    astropy cosmology object indicating cosmology to use
    """

    z = np.atleast_1d(z)

    z_sample = np.linspace(0.000001, 1.0, 1000)
    d = cosmo.distmod(z_sample)
    f = interpolate.InterpolatedUnivariateSpline(z_sample, d, k=1, ext=3, check_finite=True)

    return f(z)