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
    disk_var_1 = theta[2]
    disk_var_2 = theta[3]

    # elliptical parameters
    elliptical_mu_1 = theta[4]
    elliptical_mu_2 = theta[5]
    elliptical_var_1 = theta[6]
    elliptical_var_2 = theta[7]

    # fraction of disk galaxies
    f_disk = theta[8]

    # keep track of prior
    result = 0.0

    # disk fraction must be in the range [0,1]
    if (f_disk>=0.0) & (f_disk<=1.0):
        pass
    else:
        result += -np.inf

    # mean axis ratio must be in the range [0,1]
    if (disk_mu_1>0.0) & (disk_mu_1<1.0):
        pass
    else:
        result += -np.inf

    # mean axis ratio must be in the range [0,1]
    if (disk_mu_2>0.0) & (disk_mu_2<1.0):
        pass
    else:
        result += -np.inf

    # variance must be > 0
    if (disk_var_1>0.0):
        pass
    else:
        result += -np.inf

    # variance must be > 0
    if (disk_var_2>0.0):
        pass
    else:
        result += -np.inf

    # mean axis ratio must be in the range [0,1]
    if (elliptical_mu_1>0.0) & (elliptical_mu_1<1.0):
        pass
    else:
        result += -np.inf

    # mean axis ratio must be in the range [0,1]
    if (elliptical_mu_2>0.0) & (elliptical_mu_2<1.0):
        pass
    else:
        result += -np.inf

    # variance must be > 0
    if (elliptical_var_1>0.0):
        pass
    else:
        result += -np.inf

    # variance must be > 0
    if (elliptical_var_2>0.0):
        pass
    else:
        result += -np.inf

    # requirement for paramaterization
    if disk_var_1 >= disk_mu_1*(1.0 - disk_mu_1):
        result += -np.inf

    if disk_var_2 >= disk_mu_2*(1.0 - disk_mu_2):
        result += -np.inf

    if elliptical_var_1 >= elliptical_mu_1*(1.0 - elliptical_mu_1):
        result += -np.inf

    if elliptical_var_2 >= elliptical_mu_2*(1.0 - elliptical_mu_2):
        result += -np.inf

    # shape disributions become 'U' shaped
    alpha, beta = _beta_params(disk_mu_1, disk_var_1)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
    alpha, beta = _beta_params(disk_mu_2, disk_var_2)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
    alpha, beta = _beta_params(elliptical_mu_1, elliptical_var_1)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf
    alpha, beta = _beta_params(elliptical_mu_2, elliptical_var_2)
    if (alpha <1.0) and (beta<1.0):
        result += -np.inf

    return result

