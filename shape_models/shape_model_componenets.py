r"""
halotools model components
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm


__all__ = ('EllipticalShapes',)
__author__ = ('Duncan Campbell',)


class EllipticalShapes(object):
    r"""
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        Notes
        -----
        """

        self.gal_type = gal_type
        self._mock_generation_calling_sequence = (['assign_b_to_a', 'assign_c_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_b_to_a'), 'f4'),
             (str('galaxy_c_to_a'), 'f4'),
             (str('galaxy_c_to_b'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)


    def set_params(self, **kwargs):
        """
        """

        param_dict = ({'shape_gamma_'+self.gal_type: 0.57,
                       'shape_sigma_gamma_'+self.gal_type: 0.21,
                       'shape_mu_'+self.gal_type: -2.2,
                       'shape_sigma_'+self.gal_type: 1.4})

        self.param_dict = param_dict


    def assign_b_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)


        mu = self.param_dict['shape_mu_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_'+self.gal_type]

        log_epsilon = norm.rvs(loc=mu, scale=sigma, size=N)

        epsilon = 10.0**log_epsilon

        b_to_a = 1.0 - epsilon

        mask = (table['gal_type'] == self.gal_type)
        table['galaxy_b_to_a'][mask] = b_to_a[mask]


    def assign_c_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)


        mu = self.param_dict['shape_gamma_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_gamma_'+self.gal_type]

        x = norm.rvs(loc=mu, scale=sigma, size=N)

        c_to_b = 1.0 - x
        b_to_a = np.array(table['galaxy_b_to_a'])*1.0
        c_to_a = c_to_b*b_to_a

        mask = (table['gal_type'] == self.gal_type)
        table['galaxy_c_to_a'][mask] = c_to_a[mask]
        table['galaxy_c_to_b'][mask] = c_to_b[mask]

