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
    class to model the quenching of central galaxies based on host halo mass
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.prim_haloprop = 'halo_mvir'
        self.prim_galprop = 'Mag_r'

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
        self.param_dict = {'quenching_'+self.gal_type+'_m0':10**12.25,
                           'quenching_'+self.gal_type+'_k':2.0,
                           'quenching_'+self.gal_type+'_max':1.0,
                           'quenching_'+self.gal_type+'_min':0.0}

    def quiescent_fraction(self, mhalo):
        """
        """
        m0 = self.param_dict['quenching_'+self.gal_type+'_m0']
        k = self.param_dict['quenching_'+self.gal_type+'_k']
        fmax = self.param_dict['quenching_'+self.gal_type+'_max']
        fmin = self.param_dict['quenching_'+self.gal_type+'_min']

        f_quiescent = _sigmoid(np.log10(mhalo), x0=np.log10(m0), k=k,
                              ymax=fmax, ymin=fmin)
        return f_quiescent

    def assign_quiessence(self, **kwargs):
        """
        """

        table = kwargs['table']
        N = len(table)

        mhalo = table[self.prim_haloprop]

        fq = self.quiescent_fraction(mhalo)

        ran_num = np.random.random(N)

        # initialize table
        table['quiescent']=False
        table['star_forming']=False

        mask = (ran_num<fq)  & (table['gal_type']==self.gal_type)
        table['quiescent'][mask] = True
        table['star_forming'][~mask] = True

        return table


class QuenchingSats(object):
    """
    class to model the quenching of satellite galaxies based on stellar mass and host halo mass
    """

    def __init__(self, gal_type='satellites', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.prim_haloprop = 'halo_mvir'
        self.prim_galprop = 'Mag_r'

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
        self.param_dict = {'quenching_'+self.gal_type+'_base_m0':-20.8,
                           'quenching_'+self.gal_type+'_base_k':-1.5,
                           'quenching_'+self.gal_type+'_base_max':1.0,
                           'quenching_'+self.gal_type+'_base_min':0.0,
                           'quenching_'+self.gal_type+'_boost_m0':10**12.5,
                           'quenching_'+self.gal_type+'_boost_k':1.5,
                           'quenching_'+self.gal_type+'_boost_max':0.8,
                           'quenching_'+self.gal_type+'_boost_min':0.0,
                           }

    def quiescent_fraction_baseline(self, mag):
        """
        baseline quiescent fraction as a function of magnitude
        """
        m0 = self.param_dict['quenching_'+self.gal_type+'_base_m0']
        k = self.param_dict['quenching_'+self.gal_type+'_base_k']
        fmax = self.param_dict['quenching_'+self.gal_type+'_base_max']
        fmin = self.param_dict['quenching_'+self.gal_type+'_base_min']

        f_baseline = _sigmoid(mag, x0=m0, k=k,
                              ymax=fmax, ymin=fmin)
        return f_baseline

    def quiescent_fraction_boost(self, mhalo):
        """
        quiescent fraction boost as a function of host halo mass
        """
        m0 = self.param_dict['quenching_'+self.gal_type+'_boost_m0']
        k = self.param_dict['quenching_'+self.gal_type+'_boost_k']
        fmax = self.param_dict['quenching_'+self.gal_type+'_boost_max']
        fmin = self.param_dict['quenching_'+self.gal_type+'_boost_min']

        f_boost = _sigmoid(np.log10(mhalo), x0=np.log10(m0), k=k,
                           ymax=fmax, ymin=fmin)
        return f_boost

    def quiescent_fraction(self, mag, mhalo):
        """
        """
        f_base = self.quiescent_fraction_baseline(mag)
        f_boost = self.quiescent_fraction_boost(mhalo)
        return f_base + f_boost*(1.0-f_base)

    def assign_quiessence(self, **kwargs):
        """
        """

        table = kwargs['table']
        N = len(table)

        mhalo = table[self.prim_haloprop]
        mag = table[self.prim_galprop]

        fq = self.quiescent_fraction(mag, mhalo)

        ran_num = np.random.random(N)

        # initialize table
        table['quiescent']=False
        table['star_forming']=False

        mask = (ran_num<fq) & (table['gal_type']==self.gal_type)
        table['quiescent'][mask] = True
        table['star_forming'][~mask] = True

        return table


def _sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    sigmoid function
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))

