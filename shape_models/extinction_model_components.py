r"""
halotools style model components used to model dust extinction
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn

__all__ = ('PS08DustExtinction', 'Shao07DustExtinction')
__author__ = ('Duncan Campbell',)


class PS08DustExtinction(object):
    """
    class to model the inclincation dependent affect of dust extinction on observed magnitude
    """

    def __init__(self, gal_type, band='r', **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type
        self.band = band # optical band for extinction model

        self._mock_generation_calling_sequence = (['assign_morphology'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_deltaMag_'+str(self.band)), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Notes
        -----
        The beta distributions in this model are parameterized by a mean and variance.
        """
        
        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            sample = 'all'

        if sample=='luminosity_sample_1':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.20})
        elif sample=='luminosity_sample_2':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.72})
        elif sample=='luminosity_sample_3':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.8})
        elif sample=='luminosity_sample_4':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.72})
        elif sample=='all':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.44})
        # parameters from table 5 in PS08
        elif sample=='color_sample_1':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.4})
        elif sample=='color_sample_2':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.6})
        elif sample=='color_sample_3':
            param_dict = ({'E0_' + 'r_' + self.gal_type:0.7})
        elif sample=='color_sample_4':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 1.0})
        # parameters from table 6 in PS08
        elif sample=='size_sample_1a':
            param_dict = ({'E0_' + 'r_' + self.gal_type:0.6})
        elif sample=='size_sample_1b':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.16})
        elif sample=='size_sample_1c':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.16})
        # parameters from table 7 in PS08
        elif sample=='size_sample_2a':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 1.9})
        elif sample=='size_sample_2b':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.72})
        elif sample=='size_sample_2c':
            param_dict = ({'E0_' + 'r_' + self.gal_type: 0.59})

        self.param_dict = param_dict


    def extinction_model(self, theta, y):
        """
        see equation 1 in PS08

        Paramaters
        ----------
        theta : array_like
            inclination angle in radians

        y : array_like
            y = c/b
        """
        
        E0 = self.param_dict['E0_'+self.band+'_'+self.gal_type]
        result = np.zeros(len(y)) + E0
        
        mask = (np.cos(theta)>y)
        e = (1.0 + y - np.cos(theta))*E0
        result[mask] = e[mask]

        return result


    def assign_deltaMag_r(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
            b_to_a = table['galaxy_b_to_a']
            c_to_a = table['galaxy_c_to_a']
            theta = table['galaxy_theta']
        else:
            mag = kwargs['mag']
            theta = kwargs['theta']
            b_to_a = kwargs['b_to_a']
            c_to_a = kwargs['c_to_a']

        #y = c/b
        y = c_to_a/b_to_a

        mask = (table['spiral']==True)
        result = np.zeros(N)

        result[mask] = self.extinction_model(mag, theta, y)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['galaxy_deltaMag_r'][mask] = result[mask]
        else:
            return result


class Shao07DustExtinction(object):
    """
    class to model inclincation dependent dust extinction
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_extinction'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('deltaMag_u'), 'f4'),
             (str('deltaMag_g'), 'f4'),
             (str('deltaMag_r'), 'f4'),
             (str('deltaMag_i'), 'f4'),
             (str('deltaMag_z'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        parameters from table 5 in Shao + (2007)
        """
        
        param_dict = ({'gamma_u_' + self.gal_type: 1.59,
                       'gamma_g_' + self.gal_type: 1.24,
                       'gamma_r_' + self.gal_type: 0.92,
                       'gamma_i_' + self.gal_type: 0.80,
                       'gamma_z_' + self.gal_type: 0.65
                       })
        self.param_dict = param_dict


    def extinction_model(self, theta):
        """
        see equation 10 in Shao + (2008)

        Paramaters
        ----------
        theta : array_like
            inclination angle in radians
        """
        
        theta = np.atleast_1d(theta)

        gamma_u = self.param_dict['gamma_u_' + self.gal_type]
        gamma_g = self.param_dict['gamma_g_' + self.gal_type]
        gamma_r = self.param_dict['gamma_r_' + self.gal_type]
        gamma_i = self.param_dict['gamma_i_' + self.gal_type]
        gamma_z = self.param_dict['gamma_z_' + self.gal_type]
        
        result = np.zeros((N,5))
        result[:,0] = gamma_u * np.log10(np.cos(theta))
        result[:,1] = gamma_g * np.log10(np.cos(theta))
        result[:,2] = gamma_r * np.log10(np.cos(theta))
        result[:,3] = gamma_i * np.log10(np.cos(theta))
        result[:,4] = gamma_z * np.log10(np.cos(theta))

        return result


    def assign_extinction(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            theta = table['galaxy_theta']
            N = len(table)
        else:
            theta = kwargs['theta']
            N = len(theta)

        result = self.extinction_model(theta)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['deltaMag_u'][mask] = result[mask,0]
            table['deltaMag_g'][mask] = result[mask,1]
            table['deltaMag_r'][mask] = result[mask,2]
            table['deltaMag_i'][mask] = result[mask,3]
            table['deltaMag_z'][mask] = result[mask,4]
        else:
            return result
