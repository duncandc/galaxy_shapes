r"""
halotools style model components used to model galaxy morphology
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn

__all__ = ('Morphology',)


class Morphology(object):
    """
    clasify galaxies to be either a `disk` or `elliptical`.
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_morphology'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('disk'), 'bool'),
             (str('elliptical'), 'bool')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

    def assign_morphology(self, **kwargs):
        r"""
        """

        table = kwargs['table']

        mask = (table['quiescent'] == True)
        table['elliptical'] = False
        table['elliptical'][mask] = True

        mask = (table['quiescent'] == False)
        table['disk'] = False
        table['disk'][mask] = True


