r"""
halotools style model components used to model galaxy sizes
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm, lognorm, truncnorm
from scipy.stats import beta as beta_dist
from stat_utils import TruncLogNorm
from halotools.utils import normalized_vectors, elementwise_dot
from rotations.vector_utilities import angles_between_list_of_vectors


__all__ = ('Shen03EllipticalGalaxySizes', 'Shen03DiskGalaxySizes',)
__author__ = ('Duncan Campbell',)


class Shen03EllipticalGalaxySizes(object):
    r"""
    model for elliptical galaxy shizes
    """

    def __init__(self, gal_type='centrals', **kwargs):
    	r"""
        Parameters
        ----------

        Notes
        -----
        """

        self.gal_type = gal_type
        self.band = 'r'
        self.primgal_prop_key = 'Mag_'+ self.band

        self._mock_generation_calling_sequence = (['assign_elliptical_size'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_R50'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        set default parameters

        values are taken from table 1 in in Shen et al. (2003)
        """

        # set default parameters
        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            # set dafault sample
            sample = 'fig4'

        if sample == 'fig4':
            param_dict = ({'elliptical_size_a':0.6,
        	               'elliptical_size_b':-4.63,
        	               'elliptical_size_sigma1':0.48,
        	               'elliptical_size_sigma2':0.25,
        	               'elliptical_size_m0':-20.52
        	               })
        elif sample == 'fig5':
        	param_dict = ({'elliptical_size_a':0.65,
        	               'elliptical_size_b':-5.06,
        	               'elliptical_size_sigma1':0.45,
        	               'elliptical_size_sigma2':0.27,
        	               'elliptical_size_m0':-20.91
        	               })
        elif sample == 'fig10':
        	param_dict = ({'elliptical_size_a':0.65,
        	               'elliptical_size_b':-5.22,
        	               'elliptical_size_sigma1':0.45,
        	               'elliptical_size_sigma2':030,
        	               'elliptical_size_m0':-21.57
        	               })
        elif sample == 'fig11':
        	param_dict = ({'elliptical_size_a':0.56,
        	               'elliptical_size_b':2.88*10^-6,
        	               'elliptical_size_sigma1':0.47,
        	               'elliptical_size_sigma2':0.34,
        	               'elliptical_size_m0':3.98*10**(10)
        	               })
        self.param_dict = param_dict

    def median_size_model(self, m):
    	"""
        median in the size lunminosity relation

    	Notes
        -----
    	see eq. 14 in Shen et al. (2003)
    	"""
    	a = self.param_dict['elliptical_size_a']
    	b = self.param_dict['elliptical_size_b']
    	return 10.0**(-0.4*a*m + b)

    def scatter_size_model(self, m):
    	"""
        dispersion in the size lunminosity relation

    	Notes
        -----
    	see eq. 16 in Shen et al. (2003)
    	"""
        s1 = self.param_dict['elliptical_size_sigma1']
    	s2 = self.param_dict['elliptical_size_sigma2']
    	m0 = self.param_dict['elliptical_size_m0']
    	return s2 + (s1 - s2)/(1.0+10.0**(-0.8*(m-m0)))

    def conditional_size_pdf(self, r, m):
    	"""
    	conditional probability density function

        Notes
        -----
    	see eq. 12 in Shen et al. (2003)
    	"""
    	rbar = self.median_size_model(m)
    	scatter = self.scatter_size_model(m)
    	return 1.0/(r*np.sqrt(2.0*np.pi)*scatter)*np.exp(-1.0*np.log(r/rbar)**2/(2.0*scatter**2))


    def assign_elliptical_size(self, **kwargs):
    	"""
    	"""

    	table = kwargs['table']
        N = len(table)
        m = table[self.primgal_prop_key]

        ln_r = np.log(self.median_size_model(m))
        ln_r = ln_r + np.random.normal(scale=self.scatter_size_model(m))
        r = np.exp(ln_r)

        table['galaxy_R50'] = r
        return table


class Shen03DiskGalaxySizes(object):
    r"""
    model for disk galaxy shizes
    """

    def __init__(self, gal_type='centrals', **kwargs):
    	r"""
        Parameters
        ----------

        Notes
        -----
        """

        self.gal_type = gal_type
        self.band = 'r'
        self.primgal_prop_key = 'Mag_'+ self.band

        self._mock_generation_calling_sequence = (['assign_elliptical_size'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_R50'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        set default parameters

        values are taken from table 1 in in Shen et al. (2003)
        """

        # set default parameters
        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            # set dafault sample
            sample = 'fig4'

        if sample == 'fig4':
            param_dict = ({'elliptical_size_alpha':0.21,
        	               'elliptical_size_beta':0.53,
        	               'elliptical_size_gamma':-1.31,
        	               'elliptical_size_sigma1':0.48,
        	               'elliptical_size_sigma2':0.25,
        	               'elliptical_size_m0':-20.52
        	               })
        elif sample == 'fig5':
        	param_dict = ({'elliptical_size_aalpha':0.26,
        	               'elliptical_size_beta':0.51,
        	               'elliptical_size_gamma':-1.71,
        	               'elliptical_size_sigma1':0.45,
        	               'elliptical_size_sigma2':0.27,
        	               'elliptical_size_m0':-20.91
        	               })
        elif sample == 'fig10':
        	param_dict = ({'elliptical_size_alpha':0.23,
        	               'elliptical_size_beta':0.53,
        	               'elliptical_size_gamma':-1.53,
        	               'elliptical_size_sigma1':0.45,
        	               'elliptical_size_sigma2':030,
        	               'elliptical_size_m0':-21.57
        	               })
        elif sample == 'fig11':
        	param_dict = ({'elliptical_size_alpha':0.14,
        	               'elliptical_size_beta':0.39,
        	               'elliptical_size_gamma':0.10,
        	               'elliptical_size_sigma1':0.47,
        	               'elliptical_size_sigma2':0.34,
        	               'elliptical_size_m0':3.98*10**(10)
        	               })
        self.param_dict = param_dict

    def median_size_model(self, m):
    	"""
        median in the size lunminosity relation

    	Notes
        -----
    	see eq. 15 in Shen et al. (2003)
    	"""
    	alpha = self.param_dict['elliptical_size_alpha']
    	beta = self.param_dict['elliptical_size_beta']
    	gamma = self.param_dict['elliptical_size_gamma']
        m0 = self.param_dict['elliptical_size_m0']
    	return 10.0**(-0.4*alpha*m + (beta-alpha)*np.log10(1.0+10.0**(-0.4*(m-m0)))+gamma)

    def scatter_size_model(self, m):
    	"""
        dispersion in the size lunminosity relation

    	Notes
        -----
    	see eq. 16 in Shen et al. (2003)
    	"""
        s1 = self.param_dict['elliptical_size_sigma1']
    	s2 = self.param_dict['elliptical_size_sigma2']
    	m0 = self.param_dict['elliptical_size_m0']
    	return s2 + (s1 - s2)/(1.0+10.0**(-0.8*(m-m0)))

    def conditional_size_pdf(self, r, m):
    	"""
    	conditional probability density function

        Notes
        -----
    	see eq. 12 in Shen et al. (2003)
    	"""
    	rbar = self.median_size_model(m)
    	scatter = self.scatter_size_model(m)
    	return 1.0/(r*np.sqrt(2.0*np.pi)*scatter)*np.exp(-1.0*np.log(r/rbar)**2/(2.0*scatter**2))


    def assign_elliptical_size(self, **kwargs):
    	"""
    	"""

    	table = kwargs['table']
        N = len(table)
        m = table[self.primgal_prop_key]

        ln_r = np.log(self.median_size_model(m))
        ln_r = ln_r + np.random.normal(scale=self.scatter_size_model(m))
        r = np.exp(ln_r)

        table['galaxy_R50'] = r
        return table

