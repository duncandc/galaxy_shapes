params = {
    'continue_chain' : True,
    'mag_lim'  : -17.0,
    'ndim'     : 13,
    'nwalkers' : 130,
    'nthreads' : 6,
    'nsteps'   : 100,
    'theta0'   : [0.0500,  # disk mu 1
                  0.8000,  # disk mu 2
                  0.0500,  # disk sigma 1
                  0.0500,  # disk sigma 2
                  0.1000,  # elliptical mu 1
                  0.3500,  # elliptical mu 2
                  0.1080,  # elliptical sigma 1
                  0.1000,  # elliptical sigma 2
                  -21.50,  # morphology m0
                  1.5000,  # morphology alpha
                  0.9200,  # gamma_r
                  -20.44,  # luminosity function M0
                  -1.000],   # luminosity function alpha
    'dtheta'   : [10**-4]*13
}
