r"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import rv_continuous
from scipy.special import erf


__all__ = ('')
__author__ = ('Duncan Campbell')


class LogitNormal(rv_continuous):
    r"""
    logit-normal distribution
    """

    def _argcheck(self, mu, sigma):
        r"""
        check arguments
        """

        mu = np.aarray(mu)
        sigma = np.aarray(sigma)
        
        self.a = 0.0  # lower bound
        self.b = 1.0  # upper bound
        
        return (k == k) & (sigma == sigma)


    def _pdf(self, x, mu, sigma):
        r"""
        probability distribution function
        
        Parameters
        ----------
       
        Notes
        -----
        """
        
        sigma = np.atleast_1d(sigma).astype(np.float64)
        mu = np.atleast_1d(mu).astype(np.float64)

        norm = 1.0/(sigma * np.sqrt(2.0*np.pi))
        return norm * 1.0/(x*(1-x)) * np.exp(-1.0*(logit(x)-mu)**2/(2.0*sigma**2))

    def _cdf(self, x, mu, sigma):
        r"""
        cumulative probability distribution function
        
        Parameters
        ----------
       
        Notes
        -----
        """
        
        sigma = np.atleast_1d(sigma).astype(np.float64)
        mu = np.atleast_1d(mu).astype(np.float64)

        norm = 1.0/2.0
        return norm * (1.0 + erf((logit(x)-mu)/(np.sqrt(2.0*sigma**2))))

        

    