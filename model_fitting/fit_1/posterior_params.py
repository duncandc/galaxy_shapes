"""
save posterior parameter estimates
"""

import numpy as np
from astropy.table import Table
import emcee

def main():
    """
    """
    nwalkers = 100
    samples = ['sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5', 'sample_6']

    chain_dir = './chains/'
    f = open(chain_dir + 'posteriors.dat', 'w')

    def format(value):
        return "%.3f" % value

    for sample in samples:
        reader = emcee.backends.HDFBackend("chains/"+sample+"_chain.hdf5")
        data = reader.get_chain()

        nsteps = len(data)
        print(sample, "{0} steps with {1} walkers.".format(nsteps, nwalkers))

        p1 = data.T[0]
        p2 = data.T[1]
        p3 = data.T[2]
        p4 = data.T[3]
        p5 = data.T[4]
        p6 = data.T[5]
        p7 = data.T[6]
        p8 = data.T[7]
        p9 = data.T[8]

        y1_16, y1_50, y1_84 = np.percentile(p1[:,-1],[16, 50, 84])
        y2_16, y2_50, y2_84 = np.percentile(p2[:,-1],[16, 50, 84])
        y3_16, y3_50, y3_84 = np.percentile(p3[:,-1],[16, 50, 84])
        y4_16, y4_50, y4_84 = np.percentile(p4[:,-1],[16, 50, 84])
        y5_16, y5_50, y5_84 = np.percentile(p5[:,-1],[16, 50, 84])
        y6_16, y6_50, y6_84 = np.percentile(p6[:,-1],[16, 50, 84])
        y7_16, y7_50, y7_84 = np.percentile(p7[:,-1],[16, 50, 84])
        y8_16, y8_50, y8_84 = np.percentile(p8[:,-1],[16, 50, 84])
        y9_16, y9_50, y9_84 = np.percentile(p9[:,-1],[16, 50, 84])

        r = [y1_50, y1_16, y1_84,
              y2_50, y2_16, y2_84,
              y3_50, y3_16, y3_84,
              y4_50, y4_16, y4_84,
              y5_50, y5_16, y5_84,
              y6_50, y6_16, y6_84,
              y7_50, y7_16, y7_84,
              y8_50, y8_16, y8_84,
              y9_50, y9_16, y9_84]

        formatted = [format(v) for v in r] + ['\n']
        line = ' '.join(formatted)
        f.write(line)

    f.close()

if __name__ == '__main__':
    main()
