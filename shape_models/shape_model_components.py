r"""
halotools style model components used to model galaxy shapes
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


__all__ = ('EllipticalGalaxyShapes', 'DiskGalaxyShapes',
           'PS08Shapes', 'ProjectedShapes')
__author__ = ('Duncan Campbell',)


class EllipticalGalaxyShapes(object):
    r"""
    3D ellipsoidal model for elliptical galaxy shapes
    """

    def __init__(self, gal_type='centrals', morphology_key='elliptical', **kwargs):
        r"""
        Parameters
        ----------
        morphology_key : string
            key word into the galaxy_table

        Notes
        -----
        This class models the minor axis ratio, :math:`c/a`, indirectly.
        Instead, :math:`c/b` is modelled.

        In this class, we define :math:`\gamma^{\prime}=1-c/b`, in order to
        distinguish it from :math:`\gamma = c/a`.
        """

        self.gal_type = gal_type
        self.morphology_key = morphology_key

        self._mock_generation_calling_sequence = (['assign_elliptical_b_to_a',
                                                   'assign_elliptical_c_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_b_to_a'), 'f4'),
             (str('galaxy_c_to_a'), 'f4'),
             (str('galaxy_c_to_b'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Notes
        -----
        The beta distributions in this model are parameterized by a mean and variance.
        """
        param_dict = ({'elliptical_shape_mu_1_'+self.gal_type:  0.1,
                       'elliptical_shape_mu_2_'+self.gal_type:  0.35,
                       'elliptical_shape_sigma_1_'+self.gal_type: 0.1,
                       'elliptical_shape_sigma_2_'+self.gal_type: 0.1})

        self.param_dict = param_dict

    def _epsilon_dist(self):
        """
        """
        mu = self.param_dict['elliptical_shape_mu_1_'+self.gal_type]
        sigma = self.param_dict['elliptical_shape_sigma_1_'+self.gal_type]

        alpha, beta = _beta_params(mu, sigma**2)

        d = beta_dist(alpha, beta)
        return d

    def _gamma_prime_dist(self):
        """
        gamma_prime = 1 - C/B
        """
        mu = self.param_dict['elliptical_shape_mu_2_'+self.gal_type]
        sigma = self.param_dict['elliptical_shape_sigma_2_'+self.gal_type]

        alpha, beta = _beta_params(mu, sigma**2)

        d = beta_dist(alpha, beta)
        return d

    def epsilon_pdf(self, x):
        """
        epsilon = 1 - B/A
        """

        dist = self._epsilon_dist()
        p =  dist.pdf(x)
        return p

    def gamma_prime_pdf(self, x):
        """
        gamma_prime = 1-C/B
        """

        dist = self._gamma_prime_dist()
        p =  dist.pdf(x)
        return p

    def assign_elliptical_b_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)

        dist = self._epsilon_dist()
        epsilon = dist.rvs(size=N)

        b_to_a = 1.0 - epsilon

        mask_1 = (table['gal_type'] == self.gal_type)
        mask_2 = (table[self.morphology_key] == True)
        mask = (mask_1 & mask_2)

        if 'galaxy_b_to_a' not in table.colnames:
            table['galaxy_b_to_a'] = 0.0

        table['galaxy_b_to_a'][mask] = b_to_a[mask]
        return table


    def assign_elliptical_c_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)

        dist = self._gamma_prime_dist()
        x = dist.rvs(size=N)

        # gamma_prime = 1-c/b
        # c/b = 1-gamma_prime
        c_to_b = 1.0 - x
        b_to_a = np.array(table['galaxy_b_to_a'])*1.0
        c_to_a = c_to_b*b_to_a

        mask_1 = (table['gal_type'] == self.gal_type)
        mask_2 = (table[self.morphology_key] == True)
        mask = (mask_1 & mask_2)

        if 'galaxy_c_to_a' not in table.colnames:
            table['galaxy_c_to_a'] = 0.0
        if 'galaxy_c_to_b' not in table.colnames:
            table['galaxy_c_to_b'] = 0.0

        table['galaxy_c_to_a'][mask] = c_to_a[mask]
        table['galaxy_c_to_b'][mask] = c_to_b[mask]
        return table


class DiskGalaxyShapes(object):
    r"""
    3D ellipsoidal model for disk galaxy shapes
    """

    def __init__(self, gal_type='centrals', morphology_key='disk', **kwargs):
        r"""
        Parameters
        ----------

        Notes
        -----
        """

        self.gal_type = gal_type
        self.morphology_key = morphology_key

        self._mock_generation_calling_sequence = (['assign_disk_b_to_a',
                                                   'assign_disk_c_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_b_to_a'), 'f4'),
             (str('galaxy_c_to_a'), 'f4'),
             (str('galaxy_c_to_b'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        """
        param_dict = ({'disk_shape_mu_1_'+self.gal_type: 0.05,
                       'disk_shape_mu_2_'+self.gal_type: 0.80,
                       'disk_shape_sigma_1_'+self.gal_type: 0.05,
                       'disk_shape_sigma_2_'+self.gal_type: 0.05})

        self.param_dict = param_dict

    def _epsilon_dist(self):
        """
        """
        mu = self.param_dict['disk_shape_mu_1_'+self.gal_type]
        sigma = self.param_dict['disk_shape_sigma_1_'+self.gal_type]

        alpha, beta = _beta_params(mu, sigma**2)

        d = beta_dist(alpha, beta)
        return d

    def _gamma_prime_dist(self):
        """
        gamma_prime = 1 - C/B
        """
        mu = self.param_dict['disk_shape_mu_2_'+self.gal_type]
        sigma = self.param_dict['disk_shape_sigma_2_'+self.gal_type]

        alpha, beta = _beta_params(mu, sigma**2)

        d = beta_dist(alpha, beta)
        return d

    def epsilon_pdf(self, x):
        """
        epsilon = 1 - B/A
        """

        dist = self._epsilon_dist()
        p =  dist.pdf(x)
        return p

    def gamma_prime_pdf(self, x):
        """
        gamma_prime = 1-C/B

        note: this is different from gamma = C/A
        """

        dist = self._gamma_prime_dist()
        p =  dist.pdf(x)
        return p

    def assign_disk_b_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)

        dist = self._epsilon_dist()
        epsilon = dist.rvs(size=N)

        b_to_a = 1.0 - epsilon

        mask_1 = (table['gal_type'] == self.gal_type)
        mask_2 = (table[self.morphology_key] == True)
        mask = (mask_1 & mask_2)

        if 'galaxy_b_to_a' not in table.colnames:
            table['galaxy_b_to_a'] = 0.0

        table['galaxy_b_to_a'][mask] = b_to_a[mask]
        return table


    def assign_disk_c_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)

        dist = self._gamma_prime_dist()
        x = dist.rvs(size=N)

        # gamma_prime = 1-c/b
        # c/b = 1-gamma_prime
        c_to_b = 1.0 - x
        b_to_a = np.array(table['galaxy_b_to_a'])*1.0
        c_to_a = c_to_b*b_to_a

        mask_1 = (table['gal_type'] == self.gal_type)
        mask_2 = (table[self.morphology_key] == True)
        mask = (mask_1 & mask_2)

        if 'galaxy_c_to_a' not in table.colnames:
            table['galaxy_c_to_a'] = 0.0
        if 'galaxy_c_to_b' not in table.colnames:
            table['galaxy_c_to_b'] = 0.0

        table['galaxy_c_to_a'][mask] = c_to_a[mask]
        table['galaxy_c_to_b'][mask] = c_to_b[mask]
        return table




class PS08Shapes(object):
    r"""
    Padilla & Strauss (2008) galaxy shape model, arxiv:0802.0877
    for assigning galaxies' axes ratios

    The distribution of galaxies' intermediate axis, :math:`b/a`, is modelled as a
    a clipped log-normal distribution in :math:`\epsilon` where:

    .. math::
    \epsilon = 1 - b/a

    The distribution of galaxies' minor axis, :math:`c/a`, is modelled as a
    a clipped normal distribution in :math:`\tidle{\gamma}` where:

    .. math::
        \tidle{\gamma} = 1 - c/a

    Notes
    -----
    In PS08, there is a typo (private communication).
    In the text, it says that the minor axis is modelled as:

    .. math::
        `\gamma^{\prime}` = 1 - c/b

    This is not correct.  Instead, it is modelled as :math:`\tidle{\gamma}`.

    In this class, we modell :math:`\gamma = c/a`.
    To do this, we take 1 and siubtract the :math:`\gamma` parameter
    from PS08.
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""

        Parameters
        ----------
        gal_type : string

        galaxy_type : string, optional
            string indicating 'spirial' or 'elliptical' when choosing predefined
            parameter sets from PS08.

        selection : string, optional
            string indicating the sample, e.g. 'all', when choosing predefined
            parameter sets from PS08.

        shape_E0 : float
            edge-on extinction in magnitudes.

        shape_mu : float
            log mean of distribution in :math:`\epsilon`.

        shape_sigma : float
            standard deviation of distribution in :math:`\epsilon`.

        shape_gamma : float
            mean of distribution in :math:`\gamma`.

        shape_sigma_gamma : float
            standard deviation of distribution in :math:`\gamma`.
        """

        self.gal_type = gal_type
        self._mock_generation_calling_sequence = (['assign_b_to_a',
                                                   'assign_c_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_b_to_a'), 'f4'),
             (str('galaxy_c_to_a'), 'f4'),
             (str('galaxy_c_to_b'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        """

        if 'morphology' in kwargs.keys():
            morphology = kwargs['morphology']
        else:
            morphology = 'elliptical'

        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            sample = 'all'

        if morphology == 'elliptical':
            # parameters from table 2 in PS08
            if sample=='luminosity_sample_1':
                param_dict = ({'shape_mu_' +          self.gal_type: -2.85,
                               'shape_sigma_' +       self.gal_type: 1.15,
                               'shape_gamma_' +       self.gal_type: 1.0-0.41,
                               'shape_sigma_gamma_' + self.gal_type: 0.17})
            elif sample=='luminosity_sample_2':
                param_dict = ({'shape_mu_' +          self.gal_type: -3.05,
                               'shape_sigma_' +       self.gal_type: 1.0,
                               'shape_gamma_' +       self.gal_type: 1.0-0.36,
                               'shape_sigma_gamma_' + self.gal_type: 0.21})
            elif sample=='luminosity_sample_3':
                param_dict = ({'shape_mu_' +          self.gal_type: -2.75,
                               'shape_sigma_' +       self.gal_type: 2.60,
                               'shape_gamma_' +       self.gal_type: 1.0-0.56,
                               'shape_sigma_gamma_' + self.gal_type: 0.25})
            elif sample=='luminosity_sample_4':
                param_dict = ({'shape_mu_' +          self.gal_type: -3.85,
                               'shape_sigma_' +       self.gal_type: 2.35,
                               'shape_gamma_' +       self.gal_type: 1.0-0.76,
                               'shape_sigma_gamma_' + self.gal_type: 0.17})
            elif sample=='all':
                param_dict = ({'shape_mu_' +          self.gal_type: -2.2,
                               'shape_sigma_' +       self.gal_type: 1.4,
                               'shape_gamma_' +       self.gal_type: 1.0-0.57,
                               'shape_sigma_gamma_' + self.gal_type: 0.21})
            # parameters from table 6 in PS08
            elif sample=='size_sample_1a':
                param_dict = ({'shape_mu_' +          self.gal_type: -3.18,
                               'shape_sigma_' +       self.gal_type: 0.75,
                               'shape_gamma_' +       self.gal_type: 1.0-0.445,
                               'shape_sigma_gamma_' + self.gal_type: 0.17})
            elif sample=='size_sample_1b':
                param_dict = ({'shape_mu_' +          self.gal_type: -2.03,
                               'shape_sigma_' +       self.gal_type: 1.6,
                               'shape_gamma_' +       self.gal_type: 1.0-0.325,
                               'shape_sigma_gamma_' + self.gal_type: 0.17})
            elif sample=='size_sample_1c':
                param_dict = ({'shape_mu_' +          self.gal_type: -1.18,
                               'shape_sigma_' +       self.gal_type: 1.6,
                               'shape_gamma_' +       self.gal_type: 1.0-0.19,
                               'shape_sigma_gamma_' + self.gal_type: 0.05})
            # parameters from table 7 in PS08
            elif sample=='size_sample_2a':
                param_dict = ({'shape_mu_' +          self.gal_type: -2.52,
                               'shape_sigma_' +       self.gal_type: 2.7,
                               'shape_gamma_' +       self.gal_type: 1.0-0.795,
                               'shape_sigma_gamma_' + self.gal_type: 0.22})
            elif sample=='size_sample_2b':
                param_dict = ({'shape_mu_' +          self.gal_type: -1.12,
                               'shape_sigma_' +       self.gal_type: 2.6,
                               'shape_gamma_' +       self.gal_type: 1.0-0.545,
                               'shape_sigma_gamma_' + self.gal_type: 0.13})
            elif sample=='size_sample_2c':
                param_dict = ({'shape_mu_' +          self.gal_type: -3.37,
                               'shape_sigma_' +       self.gal_type: 0.85,
                               'shape_gamma_' +       self.gal_type: 1.0-0.695,
                               'shape_sigma_gamma_' + self.gal_type: 0.17})
            else:
                msg = ('PS08 parameter set not recognized.')
                raise ValueError(msg)
        elif morphology == 'spiral':
            # parameters from table 4 in PS08
            if sample=='luminosity_sample_1':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.20,
                               'shape_mu_' +          self.gal_type: -2.13,
                               'shape_sigma_' +       self.gal_type: 0.73,
                               'shape_gamma_' +       self.gal_type: 1.0-0.79,
                               'shape_sigma_gamma_' + self.gal_type: 0.048})
            elif sample=='luminosity_sample_2':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.72,
                               'shape_mu_' +          self.gal_type: -2.41,
                               'shape_sigma_' +       self.gal_type: 0.76,
                               'shape_gamma_' +       self.gal_type: 1.0-0.79,
                               'shape_sigma_gamma_' + self.gal_type: 0.051})
            elif sample=='luminosity_sample_3':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.8,
                               'shape_mu_' +          self.gal_type: -2.17,
                               'shape_sigma_' +       self.gal_type: 0.70,
                               'shape_gamma_' +       self.gal_type: 1.0-0.74,
                               'shape_sigma_gamma_' + self.gal_type: 0.06})
            elif sample=='luminosity_sample_4':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.72,
                               'shape_mu_' +          self.gal_type: -2.17,
                               'shape_sigma_' +       self.gal_type: 0.79,
                               'shape_gamma_' +       self.gal_type: 1.0-0.62,
                               'shape_sigma_gamma_' + self.gal_type: 0.11})
            elif sample=='all':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.44,
                               'shape_mu_' +          self.gal_type: -2.33,
                               'shape_sigma_' +       self.gal_type: 0.79,
                               'shape_gamma_' +       self.gal_type: 1.0-0.79,
                               'shape_sigma_gamma_' + self.gal_type: 0.050})
            # parameters from table 5 in PS08
            elif sample=='color_sample_1':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.4,
                               'shape_mu_' +          self.gal_type: -2.13,
                               'shape_sigma_' +       self.gal_type: 0.7,
                               'shape_gamma_' +       self.gal_type: 1.0-0.80,
                               'shape_sigma_gamma_' + self.gal_type: 0.054})
            elif sample=='color_sample_2':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.6,
                               'shape_mu_' +          self.gal_type: -2.77,
                               'shape_sigma_' +       self.gal_type: 0.61,
                               'shape_gamma_' +       self.gal_type: 1.0-0.80,
                               'shape_sigma_gamma_' + self.gal_type: 0.054})
            elif sample=='color_sample_3':
                param_dict = ({'shape_E0_' +          self.gal_type:0.7,
                               'shape_mu_' +          self.gal_type: -2.45,
                               'shape_sigma_' +       self.gal_type: 0.91,
                               'shape_gamma_' +       self.gal_type: 1.0-0.80,
                               'shape_sigma_gamma_' + self.gal_type: 0.052})
            elif sample=='color_sample_4':
                param_dict = ({'shape_E0_' +          self.gal_type: 1.0,
                               'shape_mu_' +          self.gal_type: -2.41,
                               'shape_sigma_' +       self.gal_type: 0.73,
                               'shape_gamma_' +       self.gal_type: 1.0-0.79,
                               'shape_sigma_gamma_' + self.gal_type: 0.050})
            # parameters from table 6 in PS08
            elif sample=='size_sample_1a':
                param_dict = ({'shape_E0_' +          self.gal_type:0.6,
                               'shape_mu_' +          self.gal_type: -2.29,
                               'shape_sigma_' +       self.gal_type: 0.76,
                               'shape_gamma_' +       self.gal_type: 1.0-0.71,
                               'shape_sigma_gamma_' + self.gal_type: 0.05})
            elif sample=='size_sample_1b':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.16,
                               'shape_mu_' +          self.gal_type: -1.73,
                               'shape_sigma_' +       self.gal_type: 0.64,
                               'shape_gamma_' +       self.gal_type: 1.0-0.83,
                               'shape_sigma_gamma_' + self.gal_type: 0.02})
            elif sample=='size_sample_1c':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.16,
                               'shape_mu_' +          self.gal_type: -0.45,
                               'shape_sigma_' +       self.gal_type: 1.54,
                               'shape_gamma_' +       self.gal_type: 1.0-0.89,
                               'shape_sigma_gamma_' + self.gal_type: 0.01})
            # parameters from table 7 in PS08
            elif sample=='size_sample_2a':
                param_dict = ({'shape_E0_' +          self.gal_type: 1.9,
                               'shape_mu_' +          self.gal_type: -3.17,
                               'shape_sigma_' +       self.gal_type: 0.91,
                               'shape_gamma_' +       self.gal_type: 1.0-0.31,
                               'shape_sigma_gamma_' + self.gal_type: 0.04})
            elif sample=='size_sample_2b':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.72,
                               'shape_mu_' +          self.gal_type: -2.49,
                               'shape_sigma_' +       self.gal_type: 0.58,
                               'shape_gamma_' +       self.gal_type: 1.0-0.48,
                               'shape_sigma_gamma_' + self.gal_type: 0.14})
            elif sample=='size_sample_2c':
                param_dict = ({'shape_E0_' +          self.gal_type: 0.59,
                               'shape_mu_' +          self.gal_type: -2.13,
                               'shape_sigma_' +       self.gal_type: 0.79,
                               'shape_gamma_' +       self.gal_type: 1.0-0.66,
                               'shape_sigma_gamma_' + self.gal_type: 0.09})
            else:
                msg = ('PS08 parameter set not recognized.')
                raise ValueError(msg)
        else:
            msg = ("PS08 galaxy morphology must be 'elliptical' or 'spiral'.")
            raise ValueError(msg)

        # set parameters if passed, and override the
        # parameters set using the PS08 tabulated values.
        param_names = ['shape_E0','shape_mu', 'shape_sigma',
                       'shape_gamma', 'shape_sigma_gamma']
        for name in param_names:
            if name in kwargs.keys():
                param_dict[name + '_' + self.gal_type] = kwargs[name]

        self.param_dict = param_dict

    def _epsilon_dist(self):
        """
        """
        mu = self.param_dict['shape_mu_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_'+self.gal_type]

        scale = np.exp(mu)
        loc = 0.0

        myclip_a = 0
        myclip_b = 1.0
        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        trunc_lognorm = TruncLogNorm()
        d = trunc_lognorm(a=a, b=b, s=sigma, loc=loc, scale=scale)

        return d

    def _gamma_dist(self):
        """
        gamma = C/A
        """
        mu = self.param_dict['shape_gamma_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_gamma_'+self.gal_type]

        myclip_a = 0
        myclip_b = 1.0
        a, b = (myclip_a - mu) / sigma, (myclip_b - mu) / sigma
        return truncnorm(loc=mu, scale=sigma, a=a, b=b)

    def epsilon_pdf(self, x):
        """
        epsilon = 1 - B/A
        """

        dist = self._epsilon_dist()
        p =  dist.pdf(x)
        return p

    def gamma_pdf(self, x):
        """
        gamma = C/A
        """

        dist = self._gamma_dist()
        p =  dist.pdf(x)
        return p

    def assign_b_to_a(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
        else:
            N = kwargs['size']

        dist = self._epsilon_dist()
        #log_epsilon = dist.rvs(size=N)
        epsilon = dist.rvs(size=N)
        #epsilon = np.exp(log_epsilon)

        b_to_a = 1.0 - epsilon

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['galaxy_b_to_a'][mask] = b_to_a[mask]
            return table
        else:
            return b_to_a


    def assign_c_to_a(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
            b_to_a = np.array(table['galaxy_b_to_a'])*1.0
        else:
            b_to_a = kwargs['b_to_a']
            N = len(b_to_a)

        dist = self._gamma_dist()
        x = dist.rvs(size=N)

        #c_to_a = x
        c_to_a = 1.0 - x
        #c_to_b = x
        #c_to_b = 1.0-x
        #c_to_a = c_to_b*b_to_a

        # force consistency, a >= b >= c
        #mask = (c_to_a > b_to_a)
        #temp = 1.0*b_to_a
        #b_to_a[mask] = c_to_a[mask]
        #c_to_a[mask] = temp[mask]

        c_to_b = c_to_a / b_to_a

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['galaxy_c_to_a'][mask] = c_to_a[mask]
            table['galaxy_b_to_a'][mask] = b_to_a[mask]
            table['galaxy_c_to_b'][mask] = c_to_b[mask]
            return table
        else:
            return c_to_a, b_to_a, c_to_b


class ProjectedShapes(object):
    r"""
    model for projected galaxy shapes
    """

    def __init__(self, gal_type='centrals', los_dimension='z', **kwargs):
        r"""
        Parameters
        ----------
        gal_type : string

        los_dimension : string
            string indicating line-of-sight dimension: 'x', 'y', or 'z'.

        Notes
        -----
        """

        self.gal_type = gal_type
        self.los_dimension = los_dimension
        self._mock_generation_calling_sequence = (['assign_projected_b_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_projected_b_to_a'), 'f4'),
             (str('galaxy_theta'), 'f4'),
             (str('galaxy_phi'), 'f4')])

        self.list_of_haloprops_needed = []
        self._methods_to_inherit = ([])

        self.set_los_vector()

    def set_los_vector(self):
        """
        """

        if self.los_dimension == 'x':
            u_los = np.array([1.0, 0.0, 0.0])
        elif self.los_dimension == 'y':
            u_los = np.array([0.0, 1.0, 0.0])
        elif self.los_dimension == 'z':
            u_los = np.array([0.0, 0.0, 1.0])
        else:
            msg = ('los dimension not recognized')
            raise ValueError(msg)

        self.u_los = u_los

    def assign_projected_b_to_a(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
            # lookup galaxies' orientations
            minor_axis = normalized_vectors(np.vstack((table['galaxy_axisC_x'],
                                                       table['galaxy_axisC_y'],
                                                       table['galaxy_axisC_z'])).T)
            inter_axis = normalized_vectors(np.vstack((table['galaxy_axisB_x'],
                                                       table['galaxy_axisB_y'],
                                                       table['galaxy_axisB_z'])).T)
            major_axis = normalized_vectors(np.vstack((table['galaxy_axisA_x'],
                                                       table['galaxy_axisA_y'],
                                                       table['galaxy_axisA_z'])).T)
            # lookup axis ratios
            b_to_a = table['galaxy_b_to_a']
            c_to_a = table['galaxy_c_to_a']
        else:
            minor_axis = kwargs['minor_axis']
            inter_axis = kwargs['inter_axis']
            major_axis = kwargs['major_axis']
            b_to_a = kwargs['b_to_a']
            c_to_a = kwargs['c_to_a']
            N = len(b_to_a)

        # calculate inclination angle
        theta = angles_between_list_of_vectors(minor_axis, self.u_los)

        # calculate major-axis orientation angle
        u_n = normalized_vectors(np.cross(minor_axis, self.u_los))
        phi = angles_between_list_of_vectors(major_axis, u_n, vn=minor_axis) + np.pi

        # calulate projected axis ratio
        proj_b_to_a = self.projected_b_to_a(b_to_a, c_to_a, theta, phi)

        if 'table' in kwargs.keys():

            if 'galaxy_projected_b_to_a' not in table.colnames:
                table['galaxy_projected_b_to_a'] = 0.0
            if 'galaxy_theta' not in table.colnames:
                table['galaxy_theta'] = 0.0
            if 'galaxy_phi' not in table.colnames:
                table['galaxy_phi'] = 0.0

            # assign projected axis ratio
            mask = (table['gal_type'] == self.gal_type)
            table['galaxy_projected_b_to_a'][mask] = proj_b_to_a[mask]
            # assign galaxy orientation angles
            table['galaxy_theta'][mask] = theta[mask]
            table['galaxy_phi'][mask]   = phi[mask]
            return table
        else:
            return proj_b_to_a, theta

    def projected_b_to_a(self, b_to_a, c_to_a, theta, phi):
        r"""
        Calulate the projected minor-to-major semi-axis lengths ratios
        for the 2D projectyion of an 3D ellipsodial distributions.

        Parameters
        ----------
        b_to_a : array_like
            array of intermediate axis ratios, b/a

        c_to_a : array_like
            array of minor axis ratios, c/a

        theta : array_like
            orientation angle, where cos(theta) is bounded between :math:`[0,1]`

        phi : array_like
            orientation angle, where phi is bounded between :math:`[0,2\pi]`

        Returns
        -------
        proj_b_to_a : numpy.array
            array of projected minor-to-major axis ratios

        Notes
        -----
        For combinations of axis ratios and orientations that result in a sufficiently high
        projected ellipticity, numerical errors cause the projected axis ratio to be 
        approximated as 0.0.  
        """

        b_to_a = np.atleast_1d(b_to_a)
        c_to_a = np.atleast_1d(c_to_a)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
    
        # gamma
        g = c_to_a
        # ellipticity
        e = 1.0 - b_to_a

        V = (1 - e*(2 - e)*np.sin(phi)**2)*np.cos(theta)**2 + g**2*np.sin(theta)**2
        W = 4*e**2*(2-e)**2*np.cos(theta)**2*np.sin(phi)**2*np.cos(phi)**2
        Z = 1-e*(2-e)*np.cos(phi)**2
    
        # for very high ellipticity, e~1
        # numerical errors become an issue
        with np.errstate(invalid='ignore'):
            projected_b_to_a = np.sqrt((V+Z-np.sqrt((V-Z)**2+W))/(V+Z+np.sqrt((V-Z)**2+W)))
        
        # set b_to_a to 0.0 in this case
        return np.where(np.isnan(projected_b_to_a), 0.0, projected_b_to_a)


def _inv_beta_params(alpha, beta):
    """
    Return the mean and variane of a beta distribution given alpha and beta paramaters.

    Parameters
    ----------
    alpha : float
        beta distribution shape parameter 

    beta : float
        beta distribution shape parameter 
    """
    nu = alpha + beta
    mu = alpha/nu
    var = mu*(1.0-mu)/(nu + 1.0)
    return mu, var


def _beta_params(mu, var):
    """
    Return the alpha and beta parameters of a beta distribution given mean and variance.

    Parameters
    ----------
    mu : float
        beta distribution mean

    var : float
        beta distribution variance
    """
    nu = mu*(1.0-mu)/var - 1.0
    alpha = mu*nu
    beta = (1-mu)*nu
    return alpha, beta

