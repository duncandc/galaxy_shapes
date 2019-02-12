r"""
halotools style model components used to model galaxy color
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm

__all__ = ('GalaxyColors', 'DoubleGaussGalaxyColor')
__author__ = ('Duncan Campbell',)


class GalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of two gaussians
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        # morphological type
        self.early_type_key = 'elliptical'
        self.late_type_key = 'disk'

        self._mock_generation_calling_sequence = (['assign_color'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('galaxy_g_minus_r', 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self):
        """
        """

        # set default params
        self.params = {'color_m1': -0.10301,
                       'color_b1': -1.37177,
                       'color_m2': -0.03347,
                       'color_b2': 0.246085,
                       'color_s1': 0.4**2,
                       'color_s2': 0.2**2
                       }

    def lt_mean_color(self, mag):
        """
        mean of the blue distribution
        """
        m = self.params['color_m1']
        b = self.params['color_b1']
        return m*mag + b

    def et_mean_color(self, mag):
        """
        mean of the red distribution
        """
        m = self.params['color_m2']
        b = self.params['color_b2']
        return m*mag + b

    def lt_scatter_color(self, mag):
        """
        width of blue distribution
        """
        return self.params['color_s1'] + 0.0*mag

    def et_scatter_color(self, mag):
        """
        width of red distribution
        """
        return self.params['color_s2'] + 0.0*mag

    def lt_pdf_color(self, mag):
        """
        """
        mean_colors = self.lt_mean_color(mag)
        scatters = self.lt_scatter_color(mag)
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def lt_rvs_color(self, mag):
        """
        """
        mean_colors = self.lt_mean_color(mag)
        scatters = self.lt_scatter_color(mag)
        return norm.rvs(loc=mean_colors, scale=scatters)

    def et_pdf_color(self, mag):
        """
        """
        mean_colors = self.et_mean_color(mag)
        scatters = self.et_scatter_color(mag)
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def et_rvs_color(self, mag):
        """
        """
        mean_colors = self.et_mean_color(mag)
        scatters = self.et_scatter_color(mag)
        return norm.rvs(loc=mean_colors, scale=scatters)

    def assign_color(self, **kwargs):
        """
        """

        table = kwargs['table']
        mag = table['Mag_r']

        # assign late type colors
        mask = table[self.late_type_key]
        table['galaxy_g_minus_r'][mask] = self.lt_rvs_color(mag[mask])

        # assign early type colors
        mask = table[self.early_type_key]
        table['galaxy_g_minus_r'][mask] = self.et_rvs_color(mag[mask])

        return table


class DoubleGaussGalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of two gaussians
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        # morphological type
        self.early_type_key = 'elliptical'
        self.late_type_key = 'disk'

        self._mock_generation_calling_sequence = (['assign_color'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('galaxy_g_minus_r', 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self):
        """
        """

        # set default params
        self.param_dict = {'color_mu_1': 0.35,
                       'color_sigma_1': 0.1,
                       'color_mu_2': 0.85,
                       'color_sigma_2': 0.05
                       }

    def lt_pdf_color(self, mag):
        """
        """
        mean_colors = self.param_dict['color_mu_1']
        scatters = self.param_dict['color_sigma_1']
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def lt_rvs_color(self, size):
        """
        """
        mean_colors = self.param_dict['color_mu_1']
        scatters = self.param_dict['color_sigma_1']
        return norm.rvs(size=size, loc=mean_colors, scale=scatters)

    def et_pdf_color(self, mag):
        """
        """
        mean_colors = self.param_dict['color_mu_2']
        scatters = self.param_dict['color_sigma_2']
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def et_rvs_color(self, size):
        """
        """
        mean_colors = self.param_dict['color_mu_2']
        scatters = self.param_dict['color_sigma_2']
        return norm.rvs(size=size, loc=mean_colors, scale=scatters)

    def assign_color(self, **kwargs):
        """
        """

        table = kwargs['table']
        N=len(table)

        table['galaxy_g_minus_r'] = -99.00

        # assign late type colors
        mask = table[self.late_type_key]
        table['galaxy_g_minus_r'][mask] = self.lt_rvs_color(N)[mask]

        # assign early type colors
        mask = table[self.early_type_key]
        table['galaxy_g_minus_r'][mask] = self.et_rvs_color(N)[mask]

        return table


def blue_fraction(mag, m0=-21.5, s=1.5):
    """
    model for the blue fraction of galaxies

    Parameters
    ----------
    mag : array_like
        array of absolute magtnitudes

    Returns
    -------
    f_blue : numpy.array
        fraction of blue galaxies at tge given magnitude
    """

    loc = m0
    scale = s

    x = (mag-loc)
    return (1.0)/(1.0+np.exp(-1.0*x/scale))




