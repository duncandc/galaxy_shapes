"""
define priors
"""

from __future__ import print_function, division, absolute_import
import numpy as np

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
    
    result = 0.0
    
    if (disk_mu_1>0.0) & (disk_mu_1<1.0):
        pass
    else:
        result += -np.inf

    if (disk_mu_2>0.0) & (disk_mu_2<1.0):
        pass
    else:
        result += -np.inf
    
    if (disk_var_1>0.0):
        pass
    else:
        result += -np.inf
    
    if (disk_var_2>0.0):
        pass
    else:
        result += -np.inf
        
    if (elliptical_mu_1>0.0) & (elliptical_mu_1<1.0):
        pass
    else:
        result += -np.inf
    
    if (elliptical_mu_2>0.0) & (elliptical_mu_2<1.0):
        pass
    else:
        result += -np.inf
    
    if (elliptical_var_1>0.0):
        pass
    else:
        result += -np.inf
    
    if (elliptical_var_2>0.0):
        pass
    else:
        result += -np.inf
    
    return result

