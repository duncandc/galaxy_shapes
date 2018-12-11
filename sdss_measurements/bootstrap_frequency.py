"""
function used to caclulate the the distribvution of a quantity in bins where
errors on the relative frequency are calculated using bootstrap resampling.
"""

from __future__ import print_function, division
import numpy as np

__all__=['bootstrap_frequency', ]

def bootstrap_frequency(x, bins, weights=None, Nboot=100):
    """
    Measure of the relative frequency of `x` in bins where
    errors are calculated using bootstrap sampling.

    Parameters
    -----------
    x : array_like
       array of values

    bins : array_like
        array of bins

    weights : array_like, optional
       array of floats to use for weights counts

    Nboot : int
       number of bootstrap samples to use

    Returns
    -------
    f, err : list
       list of number arrays containing the relative frequency and error
    """

    x = np.atleast_1d(x)
    N = len(x)
    
    if weights is None:
        weights = np.ones(N)
    
    inds = np.arange(0,N)
    
    f = np.zeros((Nboot, len(bins)-1))
    for i in range(0,Nboot):
        idx = np.random.choice(inds, size=N, replace=True)
        xx = x[idx]
        ww = weights[idx]
        counts = np.histogram(xx, weights=ww, bins=bins)[0]
        f[i,:] = 1.0*counts/np.sum(ww)/np.diff(bins)
    
    return np.mean(f, axis=0), np.std(f, axis=0)

