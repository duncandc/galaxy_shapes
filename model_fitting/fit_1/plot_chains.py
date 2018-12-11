"""
utility to plot paramater chains
"""

import matplotlib.pyplot as plt
import emcee
import sys

def plot_chains(samples):
    """
    """
    
    chains = samples.T
    n_chains = len(chains)
    n_walkers = len(chains[0])

    fig, axes = plt.subplots(n_chains, 1, figsize=(7.0,n_chains))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(0,n_chains):
        for j in range(0,n_walkers):
            axes[i].plot(chains[i][j])
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

    nsteps = len(data)
    print("{0} steps with {1} walkers.".format(nsteps, nwalkers))

    fig = plot_chains(data)

    fname = 'chains.pdf'
    fig.savefig(fpath + fname, dpi=250)

    plt.show()



if __name__ == '__main__':
    main()


