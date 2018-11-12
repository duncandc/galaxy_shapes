r"""
halotools style model components used to model dust extinction
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn

__all__ = ('PS08DustExtinction', 'Shao07DustExtinction', 'Unterborn08DustExtinction')
__author__ = ('Duncan Campbell',)


class PS08DustExtinction(object):
    """
    class to model the inclincation dependent affect of dust extinction on observed magnitude
    """

    def __init__(self, gal_type='centrals', band='r', morphology='disk', **kwargs):
        r"""
        Parameters
        ----------
        band : string
            optical band for extinction model

        morphology : string
            morphology of galaxies to apply extinction model to
 
        Notes
        -----
        When this model is used to populate a simulation, the galaxy table
        must contain a boolean column with name `morphology` indicating 
        whether the galaxy has that morphology.
        """

        self.gal_type = gal_type

        # optical band for extinction model
        self.band = band
        # morphological type
        self.morphology = morphology

        self._mock_generation_calling_sequence = (['assign_extinction'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('deltaMag_'+str(self.band)), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)


    def set_params(self, **kwargs):
        """
        see tables 4, 5, 6, and 7 in PS08
        """
        
        # values of parameter 'E0'
        table_4 = [0.20, 0.72, 0.80, 0.72, 0.44]
        table_5 = [0.40, 0.60, 0.70, 1.00]
        table_6 = [0.60, 0.16, 0.16]
        table_7 = [1.90, 0.72, 0.59]

        if 'sample' in kwargs.keys():
            sample = kwargs['sample']
        else:
            # set dafult sample
            sample = 'all'
        
        try:
            s1, s2, s3 = sample.split('_')
        except ValueError:
            s1 = sample

        if sample == 'all':
            E0 = table_4[4]
        elif s1 == 'luminosity':
            if   s3 == '1':
                E0 = table_4[0]
            elif s3 == '2':
                E0 = table_4[1]
            elif s3 == '3':
                E0 = table_4[2]
            elif s3 == '4':
                E0 = table_4[3]
            else:
                msg = ('`sample` not recognized.')
                raise ValueError(msg)
        elif s1 == 'color':
            if   s3 == '1':
                E0 = table_5[0]
            elif s3 == '2':
                E0 = table_5[1]
            elif s3 == '3':
                E0 = table_5[2]
            elif s3 == '4':
                E0 = table_5[3]
            else:
                msg = ('`sample` not recognized.')
                raise ValueError(msg)
        elif s1 == 'size':
            if   s3 == '1a':
                E0 = table_6[0]
            elif s3 == '1b':
                E0 = table_6[1]
            elif s3 == '1c':
                E0 = table_6[2]
            elif s3 == '2a':
                E0 = table_7[0]
            elif s3 == '2b':
                E0 = table_7[1]
            elif s3 == '2c':
                E0 = table_7[2]
            else:
                msg = ('`sample` not recognized.')
                raise ValueError(msg)
        else:
            msg = ('`sample` not recognized.')
            raise ValueError(msg)

        self.param_dict = {'E0_'+self.band+'_'+self.gal_type : E0}


    def extinction_model(self, theta, y):
        """
        see equation 1 in PS08

        Paramaters
        ----------
        theta : array_like
            inclination angle in radians

        y : array_like
            y = c/b

        Notes
        -----
        It is possible PS08 meant for y=c/a.
        """
        
        E0 = self.param_dict['E0_'+self.band+'_'+self.gal_type]
        result = np.zeros(len(y)) + E0
        
        mask = (np.cos(theta) > y)
        e = (1.0 + y - np.cos(theta))*E0
        result[mask] = e[mask]

        return result


    def assign_extinction(self, **kwargs):
        r"""
        Parameters
        ----------
        b_to_a : array_like

        c_to_a : array_like

        theta : array_like

        Returns
        -------
        delta_Mag : numpy.array
            array of extinctions in magnitudes
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            b_to_a = table['galaxy_b_to_a']
            c_to_a = table['galaxy_c_to_a']
            theta = table['galaxy_theta']
            N = len(table)
        else:
            theta = kwargs['theta']
            b_to_a = kwargs['b_to_a']
            c_to_a = kwargs['c_to_a']
            N = len(theta)

        # y = c/b
        y = c_to_a/b_to_a

        result = self.extinction_model(theta, y)

        if 'table' in kwargs.keys():

            if 'deltaMag_r' not in table.colnames:
                table['deltaMag_r'] = 0.0

            mask_1 = (table[self.morphology]==True)
            mask_2 = (table['gal_type'] == self.gal_type)
            mask = (mask_1 & mask_2)
            table['deltaMag_r'][mask] = result[mask]
        else:
            return result


class Shao07DustExtinction(object):
    """
    class to model inclincation dependent dust extinction
    """

    def __init__(self, gal_type='centrals', morphology='disk', **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type
        self.morphology = morphology

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
        
        N = len(theta)
        result = np.zeros((N,5))
        result[:,0] = -1.0*gamma_u * np.log10(np.fabs(np.cos(theta)))
        result[:,1] = -1.0*gamma_g * np.log10(np.fabs(np.cos(theta)))
        result[:,2] = -1.0*gamma_r * np.log10(np.fabs(np.cos(theta)))
        result[:,3] = -1.0*gamma_i * np.log10(np.fabs(np.cos(theta)))
        result[:,4] = -1.0*gamma_z * np.log10(np.fabs(np.cos(theta)))

        return result


    def assign_extinction(self, **kwargs):
        r"""
        Parameters
        ----------
        theta : array_like

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

            if 'deltaMag_u' not in table.colnames:
                table['deltaMag_u'] = 0.0
            if 'deltaMag_g' not in table.colnames:
                table['deltaMag_g'] = 0.0
            if 'deltaMag_r' not in table.colnames:
                table['deltaMag_r'] = 0.0
            if 'deltaMag_i' not in table.colnames:
                table['deltaMag_i'] = 0.0
            if 'deltaMag_z' not in table.colnames:
                table['deltaMag_z'] = 0.0

            mask_1 = (table[self.morphology]==True)
            mask_2 = (table['gal_type'] == self.gal_type)
            mask = (mask_1 & mask_2)
            table['deltaMag_u'][mask] = result[mask,0]
            table['deltaMag_g'][mask] = result[mask,1]
            table['deltaMag_r'][mask] = result[mask,2]
            table['deltaMag_i'][mask] = result[mask,3]
            table['deltaMag_z'][mask] = result[mask,4]

            return table
        else:
            return result


class Unterborn08DustExtinction(object):
    """
    class to model inclincation dependent dust extinction
    from Unterborn & Ryden (2008)
    """

    def __init__(self, gal_type='centrals', morphology='disk', **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type
        self.morphology = morphology

        self._mock_generation_calling_sequence = (['assign_extinction'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('deltaMag_r'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        """
        
        param_dict = ({
                       'beta_r_' + self.gal_type: 0.92
                       })
        self.param_dict = param_dict


    def extinction_model(self, q):
        """
        see equation 7 in Unterborn and Ryden + (2008)

        Paramaters
        ----------
        q : array_like
            array of observed projected axis ratio b/a
        """
        
        q = np.atleast_1d(q)

        beta_r = self.param_dict['beta_r_' + self.gal_type]
        
        N = len(q)
        result = np.zeros((N,1))
        result[:,0] = beta_r * np.log10(q)**2.0

        return result


    def assign_extinction(self, **kwargs):
        r"""
        Parameters
        ----------
        q : array_like
            array of observed projected axis ratio b/a
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            q = table['projected_b_to_a']
            N = len(table)
        else:
            q = kwargs['q']
            N = len(theta)

        result = self.extinction_model(q)

        if 'table' in kwargs.keys():
            
            if 'deltaMag_r' not in table.colnames:
                table['deltaMag_r'] = 0.0

            mask_1 = (table[self.morphology]==True)
            mask_2 = (table['gal_type'] == self.gal_type)
            mask = (mask_1 & mask_2)
            table['deltaMag_r'][mask] = result[mask,0]
        else:
            return result
