"""
save posterior parameter estimates
"""

import numpy as np
from astropy.table import Table
from chains.chain_utils import return_parameter_chains

def main():
    """
    """
    nwalkers = 50
    samples = ['sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5', 'sample_6']

    chain_dir = './chains/'
    f = open(chain_dir + 'posteriors.dat', 'w')

    def format(value):
        return "%.3f" % value

    for sample in samples:
        p1 = return_parameter_chains('col2', "chains/"+sample+"_chain.dat", nwalkers)
        p2 = return_parameter_chains('col3', "chains/"+sample+"_chain.dat", nwalkers)
        p3 = return_parameter_chains('col4', "chains/"+sample+"_chain.dat", nwalkers)
        p4 = return_parameter_chains('col5', "chains/"+sample+"_chain.dat", nwalkers)
        p5 = return_parameter_chains('col6', "chains/"+sample+"_chain.dat", nwalkers)
        p6 = return_parameter_chains('col7', "chains/"+sample+"_chain.dat", nwalkers)
        p7 = return_parameter_chains('col8', "chains/"+sample+"_chain.dat", nwalkers)
        p8 = return_parameter_chains('col9', "chains/"+sample+"_chain.dat", nwalkers)
        p9 = return_parameter_chains('col10', "chains/"+sample+"_chain.dat", nwalkers)

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
