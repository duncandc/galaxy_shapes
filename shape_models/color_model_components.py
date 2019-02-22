r"""
halotools style model components used to model galaxy color
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm, exponnorm

__all__ = ('BimodalGalaxyColor',
           'TrimodalGalaxyColor',
           'DoubleGaussGalaxyColor',
           'TripleGaussGalaxyColor',
           'BinaryGalaxyColor'
           'TrinaryGalaxyColor'
           )
__author__ = ('Duncan Campbell',)


class BimodalGalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of two gaussians
    where the mean and scale of each gaussian varies linearly with absolute magnitude
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        # morphological type
        self.early_type_key = ['quiescent',True]
        self.late_type_key  = ['quiescent',False]

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
        self.param_dict = {}

        # late type parameters
        self.param_dict['lt_color_mean_slope'] = -0.08042807
        self.param_dict['lt_color_mean_intercept'] = -1.08652303
        self.param_dict['lt_color_scatter_slope'] = 0.0
        self.param_dict['lt_color_scatter_intercept'] = 0.10226203

        # early type parameters
        self.param_dict['et_color_mean_slope'] = -0.02695986
        self.param_dict['et_color_mean_intercept'] = 0.38075935
        self.param_dict['et_color_scatter_slope'] = 0.00359192
        self.param_dict['et_color_scatter_intercept'] = 0.12373306

    def lt_mean_color(self, mag):
        """
        mean of the blue distribution
        """
        m = self.param_dict['lt_color_mean_slope']
        b = self.param_dict['lt_color_mean_intercept']
        return m*mag + b

    def et_mean_color(self, mag):
        """
        mean of the red distribution
        """
        m = self.param_dict['et_color_mean_slope']
        b = self.param_dict['et_color_mean_intercept']
        return m*mag + b

    def lt_scatter_color(self, mag):
        """
        width of blue distribution
        """
        m = self.param_dict['lt_color_scatter_slope']
        b = self.param_dict['lt_color_scatter_intercept']
        return m*mag + b

    def et_scatter_color(self, mag):
        """
        width of red distribution
        """
        m = self.param_dict['et_color_scatter_slope']
        b = self.param_dict['et_color_scatter_intercept']
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


class TrimodalGalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of three gaussians
    where the mean and scale of each gaussian varies linearly with absolute magnitude
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        self.late_type_key = ['quiescent', False]
        self.early_type_key = ['quiescent', True]
        self.green_valley_key = ['green_valley', True]

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
        self.param_dict = {}

        self.param_dict['lt_color_mean_slope'] = -0.080
        self.param_dict['lt_color_mean_intercept'] = -1.08
        self.param_dict['lt_color_scatter_slope'] = 0.00
        self.param_dict['lt_color_scatter_intercept'] = 0.100

        self.param_dict['et_color_mean_slope'] = -0.027
        self.param_dict['et_color_mean_intercept'] = 0.38
        self.param_dict['et_color_scatter_slope'] = 0.003
        self.param_dict['et_color_scatter_intercept'] = 0.120

        self.param_dict['gv_color_mean_slope'] = -0.027
        self.param_dict['gv_color_mean_intercept'] = 0.28
        self.param_dict['gv_color_scatter_slope'] = 0.0
        self.param_dict['gv_color_scatter_intercept'] = 0.075

    def lt_mean_color(self, mag):
        """
        mean of the late type (blue) distribution
        """
        m = self.param_dict['lt_color_mean_slope']
        b = self.param_dict['lt_color_mean_intercept']
        return m*mag + b

    def et_mean_color(self, mag):
        """
        mean of the early type (red) distribution
        """
        m = self.param_dict['et_color_mean_slope']
        b = self.param_dict['et_color_mean_intercept']
        return m*mag + b

    def gv_mean_color(self, mag):
        """
        mean of the green valley (green) distribution
        """
        m = self.param_dict['gv_color_mean_slope']
        b = self.param_dict['gv_color_mean_intercept']
        return m*mag + b

    def lt_scatter_color(self, mag):
        """
        width of late type (blue) distribution
        """
        m = self.param_dict['lt_color_scatter_slope']
        b = self.param_dict['lt_color_scatter_intercept']
        return m*mag + b

    def et_scatter_color(self, mag):
        """
        width of early type (red) distribution
        """
        m = self.param_dict['et_color_scatter_slope']
        b = self.param_dict['et_color_scatter_intercept']
        return m*mag + b

    def gv_scatter_color(self, mag):
        """
        width of green valley (green) distribution
        """
        m = self.param_dict['gv_color_scatter_slope']
        b = self.param_dict['gv_color_scatter_intercept']
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

    def gv_pdf_color(self, mag):
        """
        """
        mean_colors = self.gv_mean_color(mag)
        scatters = self.gv_scatter_color(mag)
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def gv_rvs_color(self, mag):
        """
        """
        mean_colors = self.gv_mean_color(mag)
        scatters = self.gv_scatter_color(mag)
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

        # assign green valley colors
        mask = (table[self.green_valley_key[0]]==self.green_valley_key[1])
        table['galaxy_g_minus_r'][mask] = self.gv_rvs_color(mag[mask])

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


class TripleGaussGalaxyColor(object):
    """
    class to model the intrinsic g-r color of galaxies as a sum of three gaussians
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        # morphological type
        self.early_type_key = ['quiescent',True]
        self.late_type_key = ['quiescent',False]
        self.green_valley_key = ['green_valley',True]

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
                           'color_sigma_1': 0.10,
                           'color_mu_2': 0.85,
                           'color_sigma_2': 0.05,
                           'color_mu_3': 0.75,
                           'color_sigma_3': 0.07
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

    def gv_pdf_color(self, mag):
        """
        """
        mean_colors = self.param_dict['color_mu_3']
        scatters = self.param_dict['color_sigma_3']
        return norm.pdf(mag, loc=mean_colors, scale=scatters)

    def gv_rvs_color(self, size):
        """
        """
        mean_colors = self.param_dict['color_mu_3']
        scatters = self.param_dict['color_sigma_3']
        return norm.rvs(size=size, loc=mean_colors, scale=scatters)

    def lt_pdf_extincted_color(self, mag):
        """
        approximate pdf of the extincted color distribution
        """
        mean_colors = self.param_dict['color_mu_1']
        scatters = self.param_dict['color_sigma_1']
        ks = 1.0/(scatters*7.1765)
        return exponnorm.pdf(mag, ks, loc=mean_colors, scale=scatters)

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

        # assign green valley colors
        mask = (table[self.green_valley_key[0]]==self.green_valley_key[1])
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
                           'f_blue_sigma': 0.60570487,
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


class TrinaryGalaxyColor(object):
    """
    class to model galaxies as either 'red' or 'blue' or 'green'
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.band = 'r'
        self.prim_gal_prop = 'Mag_'+self.band

        self._mock_generation_calling_sequence = (['assign_red_blue_green'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('red', 'bool'),
             ('blue', 'bool'),
             ('green', 'bool')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)


    def set_params(self):
        """
        """
        self.param_dict = {'f_blue_m0': -20.36269151,
                           'f_blue_k': 0.60570487,
                           'f_blue_min': 0.0,
                           'f_blue_max': 1.0,
                           'f_green_m0': -22.2166054,
                           'f_green_k': 6.97800719,
                           'f_green_min' : 0.0,
                           'f_green_max' : 0.12
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
        k = self.param_dict['f_blue_k']
        f_min = self.param_dict['f_blue_min']
        f_max = self.param_dict['f_blue_max']

        f = _sigmoid(mag, x0=loc, k=k, ymin=f_min, ymax=f_max)
        return f

    def green_fraction(self, mag):
        """
        model for the green fraction of galaxies

        Parameters
        ----------
        mag : array_like
            array of absolute magtnitudes

        Returns
        -------
        f_green : numpy.array
            fraction of blue galaxies at the given magnitude
        """

        loc = self.param_dict['f_green_m0']
        k = self.param_dict['f_green_k']
        f_min = self.param_dict['f_green_min']
        f_max = self.param_dict['f_green_max']

        f = _sigmoid(mag, x0=loc, k=k, ymin=f_min, ymax=f_max)
        return f

    def assign_red_blue(self, **kwargs):
        """
        """

        table = kwargs['table']
        N=len(table)

        m = table[self.prim_gal_prop]

        ran_num = np.random.random(N)
        f1 = self.blue_fraction(m)
        f2 = self.green_fraction(m)

        f_blue = (1.0-f2)*f1
        f_red = (1.0-f2)*(1-f1)
        f_green = f2

        blue_mask = (ran_num <=f_blue)
        green_mask = (ran_num >f_blue) & (ran_num <= (f_blue+f_green))
        red_mask = (ran_num > (f_blue+f_green))

        type_mask = table[self.late_type_key] & blue_mask

        table['blue'][blue_mask] = True
        table['blue'][~blue_mask] = False
        
        table['red'][red_mask] = True
        table['red'][~red_mask] = False

        table['green'][green_mask] = True
        table['green'][~green_mask] = False

        return table


def _sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    sigmoid function
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))

