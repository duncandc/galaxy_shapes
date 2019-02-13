r"""
halotools style model components used to model galaxy color
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm

__all__ = ('GalaxyColors',
           'DoubleGaussGalaxyColor',
           'BinaryGalaxyColor'
           )
__author__ = ('Duncan Campbell',)


class GalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of two gaussians
    where the mean and scale of each gaussian varies linearly with absoluet magnitude
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        # morphological type
        self.early_type_key = ['quiescent',True]
        self.late_type_key = ['quiescent',False]

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
        self.param_dict = {'color_m1': -0.08042807,
                           'color_b1': -1.08652303,
                           'color_m2': -0.02695986,
                           'color_b2': 0.38075935,
                           'color_m3': 0.0,
                           'color_b3': 0.10226203,
                           'color_m4': 0.00359192,
                           'color_b4': 0.12373306,
                           }

    def lt_mean_color(self, mag):
        """
        mean of the blue distribution
        """
        m = self.param_dict['color_m1']
        b = self.param_dict['color_b1']
        return m*mag + b

    def et_mean_color(self, mag):
        """
        mean of the red distribution
        """
        m = self.param_dict['color_m2']
        b = self.param_dict['color_b2']
        return m*mag + b

    def lt_scatter_color(self, mag):
        """
        width of blue distribution
        """
        m = self.param_dict['color_m3']
        b = self.param_dict['color_b3']
        return m*mag + b

    def et_scatter_color(self, mag):
        """
        width of red distribution
        """
        m = self.param_dict['color_m4']
        b = self.param_dict['color_b4']
        return m*mag + b

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
        mask = (table[self.late_type_key[0]]==self.late_type_key[1])
        table['galaxy_g_minus_r'][mask] = self.lt_rvs_color(mag[mask])

        # assign early type colors
        mask = (table[self.early_type_key[0]]==self.early_type_key[1])
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
        self.early_type_key = ['quiescent',True]
        self.late_type_key = ['quiescent',False]

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
        mask = (table[self.late_type_key[0]]==self.late_type_key[1])
        table['galaxy_g_minus_r'][mask] = self.lt_rvs_color(N)[mask]

        # assign early type colors
        mask = (table[self.early_type_key[0]]==self.early_type_key[1])
        table['galaxy_g_minus_r'][mask] = self.et_rvs_color(N)[mask]

        return table


class BinaryGalaxyColor(object):
    """
    class to model galaxies as either 'red' or 'blue'
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.band = 'r'
        self.prim_gal_prop = 'Mag_'+self.band

        self._mock_generation_calling_sequence = (['assign_red_blue'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('red', 'bool'),('blue', 'bool')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)


    def set_params(self):
        """
        """
        self.param_dict = {'f_blue_m0': -20.36269151,
                           'f_blue_sigma': 1.60570487,
                           }

    def blue_fraction(self, mag):
        """
        model for the blue fraction of galaxies

        Parameters
        ----------
        mag : array_like
            array of absolute magtnitudes

        Returns
        -------
        f_blue : numpy.array
            fraction of blue galaxies at the given magnitude
        """

        loc = self.param_dict['f_blue_m0']
        scale = self.param_dict['f_blue_sigma']

        x = (mag-loc)
        f=(1.0)/(1.0+np.exp(-1.0*x/scale))
        return f

    def assign_red_blue(self, **kwargs):
    	"""
    	"""

    	table = kwargs['table']
        N=len(table)

        m = table[self.prim_gal_prop]

        ran_num = np.random.random(N)
        f = self.blue_fraction(m)

        blue_mask = (ran_num <=f)
        type_mask = table[self.late_type_key] & blue_mask

        table['blue'][blue_mask & type_mask] = True
        table['blue'][~blue_mask & type_mask] = False
        table['red'][~blue_mask & type_mask] = True
        table['red'][blue_mask & type_mask] = False

        return table





