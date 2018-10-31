r"""
halotools model components
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.stats import norm, lognorm, truncnorm
from halotools.utils import normalized_vectors, elementwise_dot
from rotations.vector_utilities import angles_between_list_of_vectors


__all__ = ('PS08Shapes',)
__author__ = ('Duncan Campbell',)


class GalaxyShapes(object):
    r"""
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        Notes
        -----
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

        param_dict = ({'shape_alpha_1_'+self.gal_type: -2.85,
                       'shape_alpha_2_'+self.gal_type: 1.15,
                       'shape_beta_1_'+self.gal_type: 0.41,
                       'shape_beta_2_'+self.gal_type: 0.17})





class PS08Shapes(object):
    r"""
    Padilla & Strauss (2008) galaxy shape model, arxiv:0802.0877
    for assigning galaxies' axes ratios
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        Notes
        -----
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

        if 'galaxy_type' in kwargs.keys():
            galaxy_type = kwargs['galaxy_type']
        else:
            galaxy_type = 'elliptical'

        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            sample = 'all'

        if galaxy_type == 'elliptical':
            # parameters from table 2 in PS08
            if sample=='luminosity_sample_1':
                param_dict = ({'shape_mu_'+self.gal_type: -2.85,
                               'shape_sigma_'+self.gal_type: 1.15,
                               'shape_gamma_'+self.gal_type: 0.41,
                               'shape_sigma_gamma_'+self.gal_type: 0.17})
            elif sample=='luminosity_sample_2':
                param_dict = ({'shape_mu_'+self.gal_type: -3.05,
                               'shape_sigma_'+self.gal_type: 1.0,
                               'shape_gamma_'+self.gal_type: 0.36,
                               'shape_sigma_gamma_'+self.gal_type: 0.21})
            elif sample=='luminosity_sample_3':
                param_dict = ({'shape_mu_'+self.gal_type: -2.75,
                               'shape_sigma_'+self.gal_type: 2.60,
                               'shape_gamma_'+self.gal_type: 0.56,
                               'shape_sigma_gamma_'+self.gal_type: 0.25})
            elif sample=='luminosity_sample_4':
                param_dict = ({'shape_mu_'+self.gal_type: -3.85,
                               'shape_sigma_'+self.gal_type: 2.35,
                               'shape_gamma_'+self.gal_type: 0.76,
                               'shape_sigma_gamma_'+self.gal_type: 0.17})
            elif sample=='all':
                param_dict = ({'shape_mu_'+self.gal_type: -2.2,
                               'shape_sigma_'+self.gal_type: 1.4,
                               'shape_gamma_'+self.gal_type: 0.57,
                               'shape_sigma_gamma_'+self.gal_type: 0.21})
            # parameters from table 6 in PS08
            elif sample=='size_sample_1a':
                param_dict = ({'shape_mu_'+self.gal_type: -3.18,
                               'shape_sigma_'+self.gal_type: 0.75,
                               'shape_gamma_'+self.gal_type: 0.445,
                               'shape_sigma_gamma_'+self.gal_type: 0.17})
            elif sample=='size_sample_1b':
                param_dict = ({'shape_mu_'+self.gal_type: -2.03,
                               'shape_sigma_'+self.gal_type: 1.6,
                               'shape_gamma_'+self.gal_type: 0.325,
                               'shape_sigma_gamma_'+self.gal_type: 0.17})
            elif sample=='size_sample_1c':
                param_dict = ({'shape_mu_'+self.gal_type: -1.18,
                               'shape_sigma_'+self.gal_type: 1.6,
                               'shape_gamma_'+self.gal_type: 0.19,
                               'shape_sigma_gamma_'+self.gal_type: 0.05})
            # parameters from table 7 in PS08
            elif sample=='size_sample_2a':
                param_dict = ({'shape_mu_'+self.gal_type: -2.52,
                               'shape_sigma_'+self.gal_type: 2.7,
                               'shape_gamma_'+self.gal_type: 0.795,
                               'shape_sigma_gamma_'+self.gal_type: 0.22})
            elif sample=='size_sample_2b':
                param_dict = ({'shape_mu_'+self.gal_type: -1.12,
                               'shape_sigma_'+self.gal_type: 2.6,
                               'shape_gamma_'+self.gal_type: 0.545,
                               'shape_sigma_gamma_'+self.gal_type: 0.13})
            elif sample=='size_sample_2c':
                param_dict = ({'shape_mu_'+self.gal_type: -3.37,
                               'shape_sigma_'+self.gal_type: 0.85,
                               'shape_gamma_'+self.gal_type: 0.695,
                               'shape_sigma_gamma_'+self.gal_type: 0.17})
            else:
                msg = ('PS08 parameter set not recognized.')
                raise ValueError(msg)
        elif galaxy_type == 'spiral':
            # parameters from table 4 in PS08
            if sample=='luminosity_sample_1':
                param_dict = ({'shape_E0_'+self.gal_type:0.20,
                               'shape_mu_'+self.gal_type: -2.13,
                               'shape_sigma_'+self.gal_type: 0.73,
                               'shape_gamma_'+self.gal_type: 0.79,
                               'shape_sigma_gamma_'+self.gal_type: 0.048})
            elif sample=='luminosity_sample_2':
                param_dict = ({'shape_E0_'+self.gal_type:0.72,
                               'shape_mu_'+self.gal_type: -2.41,
                               'shape_sigma_'+self.gal_type: 0.76,
                               'shape_gamma_'+self.gal_type: 0.79,
                               'shape_sigma_gamma_'+self.gal_type: 0.051})
            elif sample=='luminosity_sample_3':
                param_dict = ({'shape_E0_'+self.gal_type:0.8,
                               'shape_mu_'+self.gal_type: -2.17,
                               'shape_sigma_'+self.gal_type: 0.70,
                               'shape_gamma_'+self.gal_type: 0.74,
                               'shape_sigma_gamma_'+self.gal_type: 0.06})
            elif sample=='luminosity_sample_4':
                param_dict = ({'shape_E0_'+self.gal_type:0.72,
                               'shape_mu_'+self.gal_type: -2.17,
                               'shape_sigma_'+self.gal_type: 0.79,
                               'shape_gamma_'+self.gal_type: 0.62,
                               'shape_sigma_gamma_'+self.gal_type: 0.11})
            elif sample=='all':
                param_dict = ({'shape_E0_'+self.gal_type:0.44,
                               'shape_mu_'+self.gal_type: -2.33,
                               'shape_sigma_'+self.gal_type: 0.79,
                               'shape_gamma_'+self.gal_type: 0.79,
                               'shape_sigma_gamma_'+self.gal_type: 0.050})
            # parameters from table 5 in PS08
            elif sample=='color_sample_1':
                param_dict = ({'shape_E0_'+self.gal_type:0.4,
                               'shape_mu_'+self.gal_type: -2.13,
                               'shape_sigma_'+self.gal_type: 0.7,
                               'shape_gamma_'+self.gal_type: 0.80,
                               'shape_sigma_gamma_'+self.gal_type: 0.054})
            elif sample=='color_sample_2':
                param_dict = ({'shape_E0_'+self.gal_type:0.6,
                               'shape_mu_'+self.gal_type: -2.77,
                               'shape_sigma_'+self.gal_type: 0.61,
                               'shape_gamma_'+self.gal_type: 0.80,
                               'shape_sigma_gamma_'+self.gal_type: 0.054})
            elif sample=='color_sample_3':
                param_dict = ({'shape_E0_'+self.gal_type:0.7,
                               'shape_mu_'+self.gal_type: -2.45,
                               'shape_sigma_'+self.gal_type: 0.91,
                               'shape_gamma_'+self.gal_type: 0.80,
                               'shape_sigma_gamma_'+self.gal_type: 0.052})
            elif sample=='color_sample_4':
                param_dict = ({'shape_E0_'+self.gal_type:1.0,
                               'shape_mu_'+self.gal_type: -2.41,
                               'shape_sigma_'+self.gal_type: 0.73,
                               'shape_gamma_'+self.gal_type: 0.79,
                               'shape_sigma_gamma_'+self.gal_type: 0.050})
            # parameters from table 6 in PS08
            elif sample=='size_sample_1a':
                param_dict = ({'shape_E0_'+self.gal_type:0.6,
                               'shape_mu_'+self.gal_type: -2.29,
                               'shape_sigma_'+self.gal_type: 0.76,
                               'shape_gamma_'+self.gal_type: 0.71,
                               'shape_sigma_gamma_'+self.gal_type: 0.05})
            elif sample=='size_sample_1b':
                param_dict = ({'shape_E0_'+self.gal_type:0.16,
                               'shape_mu_'+self.gal_type: -1.73,
                               'shape_sigma_'+self.gal_type: 0.64,
                               'shape_gamma_'+self.gal_type: 0.83,
                               'shape_sigma_gamma_'+self.gal_type: 0.02})
            elif sample=='size_sample_1c':
                param_dict = ({'shape_E0_'+self.gal_type:0.16,
                               'shape_mu_'+self.gal_type: -0.45,
                               'shape_sigma_'+self.gal_type: 1.54,
                               'shape_gamma_'+self.gal_type: 0.89,
                               'shape_sigma_gamma_'+self.gal_type: 0.01})
            # parameters from table 7 in PS08
            elif sample=='size_sample_2a':
                param_dict = ({'shape_E0_'+self.gal_type:1.9,
                               'shape_mu_'+self.gal_type: -3.17,
                               'shape_sigma_'+self.gal_type: 0.91,
                               'shape_gamma_'+self.gal_type: 0.31,
                               'shape_sigma_gamma_'+self.gal_type: 0.04})
            elif sample=='size_sample_2b':
                param_dict = ({'shape_E0_'+self.gal_type:0.72,
                               'shape_mu_'+self.gal_type: -2.49,
                               'shape_sigma_'+self.gal_type: 0.58,
                               'shape_gamma_'+self.gal_type: 0.48,
                               'shape_sigma_gamma_'+self.gal_type: 0.14})
            elif sample=='size_sample_2c':
                param_dict = ({'shape_E0_'+self.gal_type:0.59,
                               'shape_mu_'+self.gal_type: -2.13,
                               'shape_sigma_'+self.gal_type: 0.79,
                               'shape_gamma_'+self.gal_type: 0.66,
                               'shape_sigma_gamma_'+self.gal_type: 0.09})
            else:
                msg = ('PS08 parameter set not recognized.')
                raise ValueError(msg)
        else:
            msg = ("PS08 gaaxy type must be 'elliptical' or 'spiral'.")
            raise ValueError(msg)

        self.param_dict = param_dict


    def epsilon_pdf(self, x):
        """
        epsilon = 1 - B/A
        """

        mu = self.param_dict['shape_mu_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_'+self.gal_type]

        p = lognorm.pdf(x, scale=np.exp(mu), s=sigma, loc=0.0)
        return p


    def gamma_prime_pdf(self, x):
        """
        gamma_prime = 1-C/B

        note: this is different from gamma = C/A
        """

        mu = self.param_dict['shape_gamma_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_gamma_'+self.gal_type]

        myclip_a = 0
        myclip_b = 1.0
        a, b = (myclip_a - mu) / sigma, (myclip_b - mu) / sigma
        p = truncnorm.pdf(x, loc=mu, scale=sigma, a=a, b=b)
        return p


    def gamma_pdf(self, x):
        """
        gamma_prime = c/a
        """


        p = (self.gamma_prime_pdf(1.0-x))*(self.epsilon_pdf(1.0-x))
        return p


    def assign_b_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)


        mu = self.param_dict['shape_mu_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_'+self.gal_type]

        myclip_a = -1.0*np.inf
        myclip_b = 0.0
        a, b = (myclip_a - mu) / sigma, (myclip_b - mu) / sigma

        log_epsilon = truncnorm.rvs(loc=mu, scale=sigma, size=N, a=a, b=b)
        epsilon = np.exp(log_epsilon)

        b_to_a = 1.0 - epsilon

        mask = (table['gal_type'] == self.gal_type)
        table['galaxy_b_to_a'][mask] = b_to_a[mask]


    def assign_c_to_a(self, **kwargs):
        r"""
        """

        table = kwargs['table']
        N = len(table)


        mu = self.param_dict['shape_gamma_'+self.gal_type]
        sigma = self.param_dict['shape_sigma_gamma_'+self.gal_type]

        myclip_a = 0
        myclip_b = 1.0
        a, b = (myclip_a - mu) / sigma, (myclip_b - mu) / sigma
        x = truncnorm.rvs(loc=mu, scale=sigma, size=N, a=a, b=b)

        #gamma_prime = 1-c/b
        #c/b = 1-gamma
        c_to_b = 1.0 - x
        b_to_a = np.array(table['galaxy_b_to_a'])*1.0
        #c/a = c/b*b/a
        c_to_a = c_to_b*b_to_a

        mask = (table['gal_type'] == self.gal_type)
        table['galaxy_c_to_a'][mask] = c_to_a[mask]
        table['galaxy_c_to_b'][mask] = c_to_b[mask]


class ProjectedShape(object):
    r"""
    model for projected galaxy shapes
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        Notes
        -----
        """

        self.gal_type = gal_type
        self.los_dimension = 'z'
        self._mock_generation_calling_sequence = (['assign_projected_b_to_a'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_projected_b_to_a'), 'f4'),
             (str('galaxy_theta'), 'f4'),
             (str('galaxy_phi'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])
        #self.set_params(**kwargs)


    def assign_projected_b_to_a(self, **kwargs):
        r"""
        """
        table = kwargs['table']

        N = len(table)

        if self.los_dimension == 'x':
            u_los = np.array([1.0, 0.0, 0.0])
        elif self.los_dimension == 'y':
            u_los = np.array([0.0, 1.0, 0.0])
        elif self.los_dimension == 'z':
            u_los = np.array([0.0, 0.0, 1.0])
        else:
            msg = ('los dimension not recognized')
            raise ValueError(msg)

        minor_axis = normalized_vectors(np.vstack((table['galaxy_axisC_x'],
                                                   table['galaxy_axisC_y'],
                                                   table['galaxy_axisC_z'])).T)
        inter_axis = normalized_vectors(np.vstack((table['galaxy_axisB_x'],
                                                   table['galaxy_axisB_y'],
                                                   table['galaxy_axisB_z'])).T)
        major_axis = normalized_vectors(np.vstack((table['galaxy_axisA_x'],
                                                   table['galaxy_axisA_y'],
                                                   table['galaxy_axisA_z'])).T)

        # inclination angle
        theta = angles_between_list_of_vectors(minor_axis, u_los)

        # major-axis orientation angle
        u_n = normalized_vectors(np.cross(minor_axis, u_los))
        phi = angles_between_list_of_vectors(major_axis, u_n, vn=minor_axis) + np.pi

        b_to_a = table['galaxy_b_to_a']
        c_to_a = table['galaxy_c_to_a']

        proj_b_to_a = self.projected_b_to_a(b_to_a, c_to_a, theta, phi)

        mask = (table['gal_type'] == self.gal_type)
        table['galaxy_projected_b_to_a'][mask] = proj_b_to_a[mask]

        table['galaxy_theta'][mask] = theta[mask]

        table['galaxy_phi'][mask] = phi[mask]


    def projected_b_to_a(self, b_to_a, c_to_a, theta, phi):
        r"""
        Calulate the projected minor to major semi-axis lengths ratios
        for the 2D projectyion of an 3D ellipsodial distribution.

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

        Notes
        -----
        """

        g = c_to_a  # gamma
        e = 1.0 - b_to_a  # ellipticity

        V = (1 - e*(2 - e)*np.sin(phi)**2)*np.cos(theta)**2 + g**2*np.sin(theta)**2

        W = 4*e**2*(2-e)**2*np.cos(theta)**2*np.sin(phi)**2*np.cos(phi)**2

        Z = 1-e*(2-e)*np.cos(phi)**2

        projected_b_to_a = np.sqrt((V+Z-np.sqrt((V-Z)**2+W))/(V+Z+np.sqrt((V-Z)**2+W)))

        return projected_b_to_a
