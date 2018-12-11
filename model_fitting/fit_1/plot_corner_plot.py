"""
utility to plot paramater chains
"""

import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np
import sys

def plot_corner_plot(samples, theta0=None, ranges=None):
    """
    """
    
    ndim = len(samples.T)

    samples = samples.T[:,:,-1].T

    fig = corner.corner(samples, range=ranges)

    # Extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))

    if theta0 is not None:
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                x,y = theta0[xi], theta0[yi]
                ax.axvline(x, color="black")
                ax.axhline(y, color="black")
                ax.plot(x,y,"black")

    # show not allowed regions
    xx = np.linspace(0,1,100) # mu
    yy = np.sqrt((xx*(1.0-xx))) # sigma

    ax = axes[2, 0]
    ax.plot(xx,yy,color='black')
    ax.fill_between(xx, yy, yy*0.0 + 1.0, hatch="X", facecolor='none', linewidth=0.0)

    ax = axes[3, 1]
    ax.plot(xx,yy,color='black')
    ax.fill_between(xx, yy, yy*0.0 + 1.0, hatch="X", facecolor='none', linewidth=0.0)

    ax = axes[6, 4]
    ax.plot(xx,yy,color='black')
    ax.fill_between(xx, yy, yy*0.0 + 1.0, hatch="X", facecolor='none', linewidth=0.0)

    ax = axes[7, 5]
    ax.plot(xx,yy,color='black')
    ax.fill_between(xx, yy, yy*0.0 + 1.0, hatch="X", facecolor='none', linewidth=0.0)

    return fig



def main():
    """
    """

    fpath = './figures/'

    if len(sys.argv)>1:
        sample = sys.argv[1]
    else:
        print("The first positional argument must be the galaxy sample, e.g. 'sample_1'.")
        sys.exit()

    # load sample and run information
    _temp = __import__(sample + '_fitting_params')
    params = _temp.params

    nwalkers = params['nwalkers']
    mag_lim = params['mag_lim']

    reader = emcee.backends.HDFBackend("chains/"+sample+"_chain.hdf5")
    data = reader.get_chain()
    
    ranges = [(0,0.5),(0.5,1),(0,.3),(0,.3),(0,0.5),(0,0.5),(0,.3),(0,.3),(0,1)]

    fig = plot_corner_plot(data, params['theta0'], ranges)

    fname = sample+'_corner_plot.pdf'
    fig.savefig(fpath + fname, dpi=250)

    plt.show()


if __name__ == '__main__':
    main()