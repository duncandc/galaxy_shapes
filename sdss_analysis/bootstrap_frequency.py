"""
"""

from __future__ import print_function, division
import numpy as np

__all__=['bootstrap_frequency']

def bootstrap_frequency(x, bins, weights=None, Nboot=100):
    """
    Measure of the relative frequency of x in bins.
    Errors are calculated using bootstrap sampling.
    """
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