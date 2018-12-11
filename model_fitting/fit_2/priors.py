"""
define priors
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from galaxy_shapes.shape_models.shape_model_components import _beta_params

def lnprior(theta):
    """
    """

    # disk parameters
    disk_mu_1 = theta[0]
    disk_mu_2 = theta[1]
    disk_sigma_1 = theta[2]
    disk_sigma_2 = theta[3]
    disk_var_1 = theta[2]**2
    disk_var_2 = theta[3]**2

    # elliptical parameters
    elliptical_mu_1 = theta[4]
    elliptical_mu_2 = theta[5]
    elliptical_sigma_1 = theta[6]
    elliptical_sigma_2 = theta[7]
    elliptical_var_1 = theta[6]**2
    elliptical_var_2 = theta[7]**2

    # fraction of disk galaxies
    morphology_m0 =  theta[8]
    morphology_sigma = theta[9]

    # dust extinction
    gamma_r = theta[10]

    # luminosity function
    lum_M0 =theta[11]
    lum_alpha = theta[12]

    # keep track of prior
    result = 0.0

    # disk fraction model
    if (morphology_m0 > -23) * (morphology_m0 < -18):
        pass
    else:
        result += -np.inf
        print('hit prior 1')
    if (morphology_sigma > 0):
        pass
    else:
        result += -np.inf
        print('hit prior 2')

    # dust extinction model
    if (gamma_r > 0) & (gamma_r < 10):
        pass
    else:
        result += -np.inf
        print('hit prior 3')

    # luminosity function
    if (lum_M0 > -23) & (lum_M0 < -18):
        pass
    else:
        result += -np.inf
        print('hit prior 4')
    if (lum_alpha < -1.5) | (lum_alpha > -0.5):
        result += -np.inf
        print('hit prior 5')
    else:
        pass

    # mean axis ratio must be in the range [0,1]
    if (disk_mu_1>0.0) & (disk_mu_1<1.0):
        pass
    else:
        result += -np.inf
        print('hit prior 6')

    # mean axis ratio must be in the range [0,1]
    if (disk_mu_2>0.0) & (disk_mu_2<1.0):
        pass
    else:
        result += -np.inf
        print('hit prior 7')

    # variance must be > 0
    if (disk_sigma_1>0.0):
        pass
    else:
        result += -np.inf
        print('hit prior 8')

    # variance must be > 0
    if (disk_sigma_2>0.0):
        pass
    else:
        result += -np.inf
        print('hit prior 9')

    # mean axis ratio must be in the range [0,1]
    if (elliptical_mu_1>0.0) & (elliptical_mu_1<1.0):
        pass
    else:
        result += -np.inf
        print('hit prior 10')

    # mean axis ratio must be in the range [0,1]
    if (elliptical_mu_2>0.0) & (elliptical_mu_2<1.0):
        pass
    else:
        result += -np.inf
        print('hit prior 11')

    # variance must be > 0
    if (elliptical_sigma_1>0.0):
        pass
    else:
        result += -np.inf
        print('hit prior 12')

    # variance must be > 0
    if (elliptical_sigma_2>0.0):
        pass
    else:
        result += -np.inf
        print('hit prior 13')

    # requirement for paramaterization
    if disk_var_1 >= disk_mu_1*(1.0 - disk_mu_1):
        result += -np.inf
        print('hit prior 14')

    if disk_var_2 >= disk_mu_2*(1.0 - disk_mu_2):
        result += -np.inf
        print('hit prior 15')

    if elliptical_var_1 >= elliptical_mu_1*(1.0 - elliptical_mu_1):
        result += -np.inf
        print('hit prior 16')

    if elliptical_var_2 >= elliptical_mu_2*(1.0 - elliptical_mu_2):
        result += -np.inf
        print('hit prior 17')

    # shape disributions become 'U' shaped
    alpha, beta = _beta_params(disk_mu_1, disk_var_1)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
        print('hit prior 18')
    alpha, beta = _beta_params(disk_mu_2, disk_var_2)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
        print('hit prior 19')
    alpha, beta = _beta_params(elliptical_mu_1, elliptical_var_1)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
        print('hit prior 20')
    alpha, beta = _beta_params(elliptical_mu_2, elliptical_var_2)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
        print('hit prior 21')

    return result

