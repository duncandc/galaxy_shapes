r"""
halotools style model components used to model galaxies with Sersic profiles
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from scipy.special import kv, gamma

__all__ = ('SersicSize', 'SersicSurfaceBrightness', 'SersicIndex')
__author__ = ('Duncan Campbell',)


class SersicIndex(object):
    """
    assign a Sersic index to galaxies
    """

    def __init__(self, gal_type, **kwargs):
        r"""
        Parameters
        ----------

        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_sersic_index'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_sersic_index'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

    def assign_sersic_index(self, **kwargs):
        r"""
        """

        table = kwargs['table']

        # intialize key
        table['galaxy_sersic_index'] = 0.0

        # assign values for elliptical galaxies
        mask = (table['elliptical'] == True)
        table['galaxy_sersic_index'][mask] = 4.0

        # assign values for disk galaxies
        mask = (table['disk'] == True)
        table['galaxy_sersic_index'][mask] = 1.0

        return table


class SersicSize(object):
    r"""
    model for sersic effective radius
    """

    def __init__(self, gal_type='centrals', **kwargs):
        r"""
        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_projected_radius'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_projected_re'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])

    def r_half_mass(self, n):
        """
        fitting function for projected effecive radius

        Parameters
        ----------
        n : array_like
            sersic index

        Returns
        -------
        proj_to_3d_ratio : numpy.array
            effective projected radius divided by 3d radius

        Notes
        -----
        see eq. 21. Lima Neto + (1999)
        """
        nu = 1.0/n
        a = 0.0023
        b = 0.0293
        c = 1.356

        # ratio of r_3d/r_projected
        f = (c- b*nu + a*nu**2)

        return 1.0/f

    def assign_projected_radius(self, **kwargs):
        r"""
        """

        table = kwargs['table']

        # 3D half mass radius
        r = table['galaxy_r_half']
        # Sersic index
        n = table['galaxy_sersic_index']

        # projeted half mass radius
        R = self.r_half_mass(n)*r

        mask = (table['gal_type']==self.gal_type)

        table['galaxy_projected_re'][mask] = R[mask]
        return table


class SersicSurfaceBrightness(object):
    r"""
    model for sersic sruface brightness
    """

    def __init__(self, gal_type='centrals', band='r', **kwargs):
        r"""
        """

        self.gal_type = gal_type
        self.band = band

        self._mock_generation_calling_sequence = (['assign_central_brightness',
                                                   'assign_effective_brightness',
                                                   'assign_mean_brightness'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_central_brightness'), 'f4'),
             (str('galaxy_effective_brightness'), 'f4'),
             (str('galaxy_mean_brightness'), 'f4')])

        self.list_of_haloprops_needed = []

        self._methods_to_inherit = ([])


    def assign_central_brightness(self, **kwargs):
        """
        absolute central surface brightness
        """

        table = kwargs['table']

        # Sersic index
        n = table['galaxy_sersic_index']
        # effective projected radius in kpc/h
        Re = table['galaxy_projected_re']
        # projected ellipticity
        e = 1.0-table['galaxy_projected_b_to_a']
        # galaxy magnitude
        M = table['Mag_'+self.band]

        # luminosity
        Msun = 0.0
        L = 10.0**((M-Msun)/(-2.5))

        # central intensity
        b = bn(n)
        Re = Re*1000.0  # convert to pc
        I0 = L/(Re**2*(1.0-e)*(2.0*np.pi*n)/b**(2.0*n)*gamma(2.0*n))

        # central surface brightness
        m0 = -2.5*np.log10(I0) + 21.572

        mask = (table['gal_type']==self.gal_type)
        table['galaxy_central_brightness'][mask] = m0[mask]
        return table

    def assign_effective_brightness(self, **kwargs):
        """
        absolute effective surface brightness

        notes
        -----
        see eq. 7 in Graham & Driver (2005)
        """

        table = kwargs['table']

        # Sersic index
        n = table['galaxy_sersic_index']
        # effective projected radius
        Re = table['galaxy_projected_re']
        # central surface brightness
        m0 = table['galaxy_central_brightness']

        b = bn(n)
        me = m0 + 2.5*b/np.log(10)

        mask = (table['gal_type']==self.gal_type)
        table['galaxy_effective_brightness'][mask] = me[mask]
        return table

    def assign_mean_brightness(self, **kwargs):
        """
        absolute mean effective surface brightness

        notes
        -----
        see eq. 9 in Graham & Driver (2005)
        """

        table = kwargs['table']

        # Sersic index
        n = table['galaxy_sersic_index']
        # effective projected radius
        Re = table['galaxy_projected_re']
        # central surface brightness
        me = table['galaxy_effective_brightness']

        b = bn(n)
        f_n = (n*np.exp(b))/(b**(2.0*n)) * gamma(2.0*n)
        mean_me = me - 2.5*np.log10(f_n)

        mask = (table['gal_type']==self.gal_type)
        table['galaxy_mean_brightness'][mask] = mean_me[mask]
        return table


def bn(n):
    """
    fitting function for sersic constant

    Notes
    -----
    for n>0.36
    asymptotic expansion from Ciotti & Bertin (1999) (see eq. 5)

    for n<0.36
    polynomial expansion from MacArthur, Courteau, & Holtzman (2003)
    """

    n = np.atleast_1d(n)

    # for n>0.36
    n1 = 405.0
    n2 = 25515.0
    n3 = 1148175.0
    n4 = 30690717750.0
    result1 =  2.0*n - 1.0/3.0 + 4.0/(n1*n) + 46.0/(n2*n**2) + 131/(n3*n**3) - 2194697.0/(n4*n**4)

    # for n<0.36
    a0 = 0.01945
    a1 = -0.8902
    a2 = 10.95
    a3 = -19.67
    a4 = 13.43
    result2 = a0 + a1*n + a2*n**2 + a3*n**3 + a4*n**4

    return np.where(n>0.36, result1, result2)

