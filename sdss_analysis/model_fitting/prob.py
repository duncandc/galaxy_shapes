"""
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from like import lnlike
from priors import lnprior

__all__ = ['lnprob']

def lnprob(theta, y, yerr, mag_lim):
    """
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, yerr, mag_lim)
