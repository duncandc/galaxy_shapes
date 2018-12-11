r"""
halotools style model components used to model galaxy morphology
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn

__all__ = ('Morphology_1','Morphology_2')


class Morphology_1(object):
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

        return table


class Morphology_2(object):
    """
    clasify galaxies to be either a `disk` or `elliptical`.
    """

    def __init__(self, gal_type='centrals', **kwargs):
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
        self.set_params(**kwargs)


    def set_params(self, **kwargs):
        """
        """
        param_dict = {'morphology_m0': -21.5,
                      'morphology_sigma': 1.5 }
        self.param_dict = param_dict


    def disk_fraction(self, m, **kwargs):
        """
        """
        
        loc = self.param_dict['morphology_m0']
        scale = self.param_dict['morphology_sigma']

        x = (m-loc)
        return (1.0)/(1.0+np.exp(-1.0*x/scale))


    def assign_morphology(self, **kwargs):
        r"""
        """

        if 'table' in kwargs:
            table = kwargs['table']
            N = len(table)
            m = table['Mag_r']
        else:
            m = kwargs['Mag_r']
            N = len(m)

        ran_num = np.random.random(N)
        f_disk = self.disk_fraction(m)
         
        mask_2 = (ran_num < f_disk)
        
        if 'table' in kwargs:
            mask_1 = (table['gal_type'] == self.gal_type)
            table['elliptical'] = True
            table['elliptical'][mask_2 & mask_1] = False
            table['disk'] = False
            table['disk'][mask_2 & mask_1] = True
            return table
        else:
            return mask_2


