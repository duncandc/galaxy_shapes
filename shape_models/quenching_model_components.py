r"""
halotools style model components used to model galaxy quenching
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm

__all__ = ('QuenchingCens',
           'QuenchingSats',
           )

__author__=('Duncan Campbell', 'Andrew Hearin')

class QuenchingCens(object):
    """
    class to model the quenching of central galaxies
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.prim_haloprop = 'halo_mvir'

        self._mock_generation_calling_sequence = (['assign_quiessence'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('quiescent', 'bool'),
             ('star_forming', 'bool')])

        self.list_of_haloprops_needed = [self.prim_haloprop]

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self):
        """
        """
        self.param_dict = {'quenching_'+gal_type+'_m0':10**12.25,
                           'quenching_'+gal_type+'_k':1.5,
                           'quenching_'+gal_type+'_max':1.0,
                           'quenching_'+gal_type+'_min':0.0}

    def quiscent_fraction(self, mhalo):
        """
        """
        m0 = self.param_dict['quenching_'+gal_type+'_m0']
        k = self.param_dict['quenching_'+gal_type+'_k']
        fmax = self.param_dict['quenching_'+gal_type+'_max']
        fmin = self.param_dict['quenching_'+gal_type+'_min']

        f_quiscent = _sigmoid(np.log10(mhalo), x0=np.log10(m0), k=k,
                              ymax=f_max, ymin=f_min)
        return f_quiscent

    def assign_quiessence(self, **kwargs):
        """
        """

        table = kwargs['table']
        N = len(table)

        mhalo = table[self.prim_haloprop]

        fq = quiscent_fraction(mhalo)

        ran_num = np.random.random(N)

        table['quiescent']=False
        table['star_forming']=False

        mask = (ran_num<fq)
        table['quiescent'][mask] = True
        table['star_forming'][~mask] = True

        return table


class QuenchingSats(object):
    """
    class to model the quenching of satellite galaxies
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.prim_haloprop = 'halo_mvir'

        self._mock_generation_calling_sequence = (['assign_quiessence'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [('quiescent', 'bool'),
             ('star_forming', 'bool')])

        self.list_of_haloprops_needed = [self.prim_haloprop]

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self):
        """
        """
        self.param_dict = {'quenching_'+gal_type+'_m0':10**12.25,
                           'quenching_'+gal_type+'_k':1.5,
                           'quenching_'+gal_type+'_max':1.0,
                           'quenching_'+gal_type+'_min':0.0}

    def quiscent_fraction_baseline(self, mhalo):
        """
        """
        m0 = self.param_dict['quenching_'+gal_type+'_m0']
        k = self.param_dict['quenching_'+gal_type+'_k']
        fmax = self.param_dict['quenching_'+gal_type+'_max']
        fmin = self.param_dict['quenching_'+gal_type+'_min']

        f_baseline = _sigmoid(np.log10(mhalo), x0=np.log10(m0), k=k,
                              ymax=f_max, ymin=f_min)
        return f_baseline

    def quiscent_fraction_boost(self, mhalo):
        """
        """
        m0 = self.param_dict['quenching_'+gal_type+'_m0']
        k = self.param_dict['quenching_'+gal_type+'_k']
        fmax = self.param_dict['quenching_'+gal_type+'_max']
        fmin = self.param_dict['quenching_'+gal_type+'_min']

        f_boost = _sigmoid(np.log10(mhalo), x0=np.log10(m0), k=k,
                              ymax=f_max, ymin=f_min)
        return f_boost

    def quiscent_fraction_boost(self, mhalo):
        """
        """
        f_base = quiscent_fraction_baseline(mhalo)
        f_boost = quiscent_fraction_boost(mhalo)
        return f_base + f_boost

    def assign_quiessence(self, **kwargs):
        """
        """

        table = kwargs['table']
        N = len(table)

        mhalo = table[self.prim_haloprop]

        fq = quiscent_fraction(mhalo)

        ran_num = np.random.random(N)

        table['quiescent']=False
        table['star_forming']=False

        mask = (ran_num<fq)
        table['quiescent'][mask] = True
        table['star_forming'][~mask] = True

        return table


def _sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    sigmoid function
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))

