r"""
halotools style model components used to model galaxy color
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn

__all__ = ('GalaxyColors')
__author__ = ('Duncan Campbell',)


class GalaxyColor(object):
    """
    class to model the g-r color of galaxies as a sum of 2 gaussians
    """

    def __init__(self, gal_type='centrals',  **kwargs):
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
        self.params = {'m1': 0.267,
                       'b1': -1.459,
                       'm2': 0.399,
                       'b2': -2.306,
                       's1': 0.2,
                       's2': 0.15
                       }

    def lt_mean_color(self, mag):
        """
        mean of the blue distribution
        """
        m = self.params['m1']
        b = self.params['b1']
        return m*mag + b

    def et_mean_color(self, mag):
        """
        mean of the red distribution
        """
        m = self.params['m2']
        b = self.params['b2']
        return m*mag + b

    def lt_scatter_color(self, mag):
        """
        width of blue distribution
        """
        return self.params['s1'] + 0.0*mag

    def et_scatter_color(self, mag):
        """
        width of red distribution
        """
        return self.params['s2'] + 0.0*mag

    def lt_pdf_color(self, mag):
        """
        """
        mean_colors = self.lt_mean_color(mag)
        scatter = self.lt_scatter_color(mag)
        return norm.pdf(m, loc=mean_color, scale=scatter)

    def et_pdf_color(self, mag):
        """
        """
        mean_colors = self.et_mean_color(mag)
        scatter = self.et_scatter_color(mag)
        return norm.pdf(m, loc=mean_color, scale=scatter)

    def assign_color(self):
        """
        """

        table = kwargs['table']
        m = table['Mag_r']

        # assign late type colors
        mask = table[self.late_type_key]
        mean_colors = self.lt_mean_color(m)
        scatter = self.lt_scatter_color(m)
        table['galaxy_g_minus_r'][mask] = norm.rvs(size=np.sum(mask),
                                                   loc=mean_color, scale=scatter)

        # assign early type colors
        mask = table[self.early_type_key]
        mean_colors = self.et_mean_color(m)
        scatter = self.et_scatter_color(m)
        table['galaxy_g_minus_r'][mask] = norm.rvs(size=np.sum(mask),
                                                   loc=mean_color, scale=scatter)

        return table




