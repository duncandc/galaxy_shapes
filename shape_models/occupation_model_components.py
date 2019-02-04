r"""
halotools style model components
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from astro_utils.magnitudes import luminosity_to_absolute_magnitude 


__all__ = ('MagnitudesSDSS')
__author__ = ('Duncan Campbell')


class MagnitudesSDSS(object):
    """
    """

    def __init__(self, gal_type, band='r', **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type
        self.band = band

        self._mock_generation_calling_sequence = (['assign_magnitude'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('Mag_'+str(self.band)), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

    def assign_magnitude(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        L = table['luminosity']

        M = luminosity_to_absolute_magnitude(L, band=self.band, system='SDSS_Blanton_2003_z0.1')
    
        mask = (table['gal_type'] == self.gal_type)
        table['Mag_'+str(self.band)][mask] = M[mask]

        return table
