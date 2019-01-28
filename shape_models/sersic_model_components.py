import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.special import kv, gamma


n  = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0]
nu = [-0.50000,0.00000,0.43675,0.47773,0.49231,0.49316,0.49280,0.50325,0.51140,0.52169,0.55823,0.58086,0.60463,0.61483,0.66995]
p  = [1.00000,0.00000,0.61007,0.77491,0.84071,0.87689,0.89914,0.91365,0.9244,0.93279,0.94451,0.95289,0.95904,0.96385,0.96731]
h1 = [0.00000,0.00000,-0.07257,-0.04963,-0.03313,-0.02282,-0.01648,-0.01248,-0.00970,-0.00773,-0.00522,-0.00369,-0.00272,-0.00206,-0.00164]
h2 = [0.00000,0.00000,-0.20048,-0.15556,-0.12070,-0.09611,-0.07919,-0.06747,-0.05829,-0.05106,-0.04060,-0.03311,-0.02768,-0.02353,-0.02053]
h3 = [0.00000,0.00000,0.01647,0.08284,0.14390,0.19680,0.24168,0.27969,0.31280,0.34181,0.39002,0.42942,0.46208,0.48997,0.51325]

t = Table([n, nu, p, h1, h2, h3], names=('n', 'nu', 'p', 'h1', 'h2', 'h3'))

from scipy.interpolate import interp1d

interp_h1 = interp1d(t['n'], t['h1'], kind='linear')
interp_h2 = interp1d(t['n'], t['h2'], kind='linear')
interp_h3 = interp1d(t['n'], t['h3'], kind='linear')
interp_nu = interp1d(t['n'], t['nu'], kind='linear')
interp_p  = interp1d(t['n'], t['p'],  kind='linear')


def total_luminosity(n, re, b_to_a, c_to_a, theta, phi, I0=1.0):
    """
    """
    e = 1.0 - projected_b_to_a(b_to_a, c_to_a, theta, phi)
    return I0 * re**2 * (1-e)*(2.0*np.pi*n)/(bn(n)**(2.0*n))*gamma(2.0*n)

def projected_b_to_a(b_to_a, c_to_a, theta, phi):
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

    I used the naming convention for variables employed in Ryden 2004.  I have included
    comments to connect back to Binney (1985) and Stark (1977).
    """

    b_to_a = np.atleast_1d(b_to_a)
    c_to_a = np.atleast_1d(c_to_a)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # gamma
    g = c_to_a
    # ellipticity
    e = 1.0 - b_to_a

    # called 'A'
    V = (1 - e*(2 - e)*np.sin(phi)**2)*np.cos(theta)**2 + g**2*np.sin(theta)**2
    # sometimes called 'B'
    W = 4*e**2*(2-e)**2*np.cos(theta)**2*np.sin(phi)**2*np.cos(phi)**2
    # sometimes called 'C'
    Z = 1-e*(2-e)*np.cos(phi)**2
    
    # for very high ellipticity, e~1
    # numerical errors become an issue
    with np.errstate(invalid='ignore'):
        projected_b_to_a = np.sqrt((V+Z-np.sqrt((V-Z)**2+W))/(V+Z+np.sqrt((V-Z)**2+W)))
        
    # set b_to_a to 0.0 in this case
    return np.where(np.isnan(projected_b_to_a), 0.0, projected_b_to_a)
    

def density_profile(zeta, n, re, theta, phi, b_to_a, c_to_a, I0=1.0):
    """
    """
    h = zeta/re
    b = bn(n)
    nu = interp_nu(n)
    p = interp_p(n)

    sqrt_f = f(theta, phi, b_to_a, c_to_a)**(0.5)
    
    return (sqrt_f * I0 * bn(n) * 2.0**((n-1)/(2.0*n)))/(re * n * np.pi) *\
           (h**(p*(1.0/n - 1.0))*kv(nu, bn(n)*h**(1.0/n)))/(1.0 - C(h, n))	


def C(h, n):
	"""
	"""
	h1 = interp_h1(n)
	h2 = interp_h2(n)
	h3 = interp_h3(n)
	return h1*np.log10(h)**2 + h2*np.log10(h) + h3


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


def f(theta, phi, b_to_a, c_to_a):
	"""
	see eq. 6 in binney 1985
	"""
	phi = phi
	return np.sin(theta)**2*np.cos(phi)**2 + np.sin(theta)**2*np.sin(phi)**2/b_to_a**2 + np.cos(theta)**2/c_to_a**2


def main():
    
    plt.figure()
    plt.plot(n, nu, '-o')
    plt.plot(n, p, '-o')
    plt.plot(n, h1, '-o')
    plt.plot(n, h2, '-o')
    plt.plot(n, h2, '-o')
    plt.show(block=False)

    n_sample = np.linspace(0.001,10,10000)

    mask = n_sample<0.36
    plt.figure()
    plt.plot(n_sample, bn(n_sample))
    plt.plot(n_sample[mask], np.fabs(bn(n_sample)[mask]), color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.show(block=False)

    zeta_sample = np.logspace(-3,3,1000)
    rho1 = density_profile(zeta_sample, n=0.5, re=1, theta=0, phi=0, b_to_a=1, c_to_a=1, I0=1.0)/\
                           total_luminosity(n=0.5, re=1, b_to_a=1, c_to_a=1, theta=0, phi=0)
    rho2 = density_profile(zeta_sample, n=1.0, re=1, theta=0, phi=0, b_to_a=1, c_to_a=1, I0=1.0)/\
                           total_luminosity(n=1, re=1, b_to_a=1, c_to_a=1, theta=0, phi=0)
    rho3 = density_profile(zeta_sample, n=2, re=1, theta=0, phi=0, b_to_a=1, c_to_a=1, I0=1.0)/\
                           total_luminosity(n=2, re=1, b_to_a=1, c_to_a=1, theta=0, phi=0)
    rho4 = density_profile(zeta_sample, n=4, re=1, theta=0, phi=0, b_to_a=1, c_to_a=1, I0=1.0)/\
                           total_luminosity(n=4, re=1, b_to_a=1, c_to_a=1, theta=0, phi=0)
    rho5 = density_profile(zeta_sample, n=10, re=1, theta=0, phi=0, b_to_a=1, c_to_a=1, I0=1.0)/\
                           total_luminosity(n=10, re=1, b_to_a=1, c_to_a=1, theta=0, phi=0)

    plt.figure()
    plt.plot(zeta_sample, rho1)
    plt.plot(zeta_sample, rho2)
    plt.plot(zeta_sample, rho3)
    plt.plot(zeta_sample, rho4)
    plt.plot(zeta_sample, rho5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([10**(-15), 10**5])
    plt.show(block=False)

    zeta_sample = np.logspace(-3,3,1000)
    rho1 = density_profile(zeta_sample, n=4.0, re=1, theta=0, phi=0, b_to_a=1.0, c_to_a=0.8, I0=1.0)/\
                           total_luminosity(n=4.0, re=1, b_to_a=1.0, c_to_a=0.8, theta=0.0, phi=0)

    rho2 = density_profile(zeta_sample, n=4.0, re=1, theta=0.2, phi=0, b_to_a=0.9, c_to_a=0.7, I0=1.0)/\
                           total_luminosity(n=4.0, re=1, b_to_a=0.9, c_to_a=0.7, theta=0.2, phi=0.)

    rho3 = density_profile(zeta_sample, n=4.0, re=1, theta=0.4, phi=0, b_to_a=0.8, c_to_a=0.6, I0=1.0)/\
                           total_luminosity(n=4.0, re=1, b_to_a=0.8, c_to_a=0.6, theta=0.4, phi=0)

    rho4 = density_profile(zeta_sample, n=4.0, re=1, theta=0.8, phi=0, b_to_a=0.7, c_to_a=0.5, I0=1.0)/\
                           total_luminosity(n=4.0, re=1, b_to_a=0.7, c_to_a=0.5, theta=0.8, phi=0)

    rho5 = density_profile(zeta_sample, n=4.0, re=1, theta=1.0, phi=0, b_to_a=0.6, c_to_a=0.4, I0=1.0)/\
                           total_luminosity(n=4.0, re=1, b_to_a=0.6, c_to_a=0.4, theta=1.0, phi=0)

    plt.figure()
    plt.plot(zeta_sample, rho1)
    plt.plot(zeta_sample, rho2)
    plt.plot(zeta_sample, rho3)
    plt.plot(zeta_sample, rho4)
    plt.plot(zeta_sample, rho5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([10**(-15), 10**5])
    plt.show()



if __name__ == '__main__':
    main()