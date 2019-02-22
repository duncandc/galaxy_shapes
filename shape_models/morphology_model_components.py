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
    clasify galaxies to be either morphology type `disk` or `elliptical`
    based on their classification as 'quiescent' or 'star-forming'.

    see fig. 2 in Masters et al. (2009) for red disk fraction. 
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type
        self.prim_galprop = 'Mag_r'

        self._mock_generation_calling_sequence = (['assign_morphology'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('disk'), 'bool'),
             (str('elliptical'), 'bool')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params()

    def set_params(self, **kwargs):
        """
        """
        param_dict = {'morphology_m0': -22.0,
                      'morphology_k': -1.5,
                      'morphology_min': 0.0,
                      'morphology_max': 0.2,}
        self.param_dict = param_dict

    def quiescent_disk_fraction(self, m):
        """
        """
        m0 = self.param_dict['morphology_m0']
        k = self.param_dict['morphology_k']
        fmin = self.param_dict['morphology_min']
        fmax = self.param_dict['morphology_max']
        
        f_disk = _sigmoid(m, x0=m0, k=k, ymax=fmax, ymin=fmin)
        return f_disk

    def assign_morphology(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)
        
        m = table[self.prim_galprop]

        # intialize table
        table['elliptical'] = False
        table['disk'] = False


        # quiescent galaxies
        quiescent_mask = (table['quiescent'] == True)

        # quiscent disks
        f_disk = self.quiescent_disk_fraction(m)
        ran_num = np.random.random(N)
        quiscent_disk = (quiescent_mask) & (f_disk <= ran_num)
        table['elliptical'][quiscent_disk] = False
        table['disk'][quiscent_disk] = True
        
        # quiscent ellipticals
        quiescent_elliptical = (quiescent_mask) & (f_disk > ran_num)
        table['elliptical'][quiescent_elliptical] = True
        table['disk'][quiescent_elliptical] = False
        
        # star-forming galaxies
        sf_mask = (table['quiescent'] == False)

        # star-forming disks
        table['elliptical'][sf_mask] = False
        table['disk'][sf_mask] = True

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


def _sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    sigmoid function
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))
