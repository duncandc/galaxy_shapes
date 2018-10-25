"""
script to build indices into nyu vagc to select LRG sample
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astro_utils.magnitudes import color_k_correct
from astropy.cosmology import FlatLambdaCDM
from utils import interpolated_distmod


def main():

    # directory of nyu vagc files
    fpath = './'

	# load tables
    fname = 'object_sdss_tiling.fits'
    t_1 = Table.read(fpath + fname)

    fname = 'lss_index.dr72.fits'
    t_2 = Table.read(fpath + fname)

    fname = 'object_sdss_imaging.fits'
    t_3 = Table.read(fpath + fname)

    # load table from Eyal Kazin
    # http://cosmo.nyu.edu/~eak306/SDSS-LRG.html
    fname = 'DR7-Full.ascii'
    names=['ra', 'dec', 'z', 'M_g', 'sector_completeness', 'n(z)*1e4', 'radial_weight',
       'fiber_coll_weight (>=1)', 'fogtmain', 'ilss', 'icomb', 'sector']
    t_4 = Table.read(fpath + fname, format='ascii', names=names)

    # LRG selection mask
    # see table 27 in Stoughton + (2002)
    mask_a = (t_1['PRIMTARGET'] & 32) > 0
    # account for stripe 82 selection
    mask_b = (t_1['PRIMTARGET'] & 2**31) == 0
    mask_1 = (mask_a & mask_b)
    # redshift selection
    # sdss redshift
    mask_a = (t_2['ZTYPE'] == 1)
    # redshift range
    zmin, zmax = (0.16, 0.47)
    mask_b = (t_2['Z'] >=zmin) & (t_2['Z'] <=zmax)
    # not in star mask
    mask_c = (t_2['ICOMB'] != -1)
    mask_2 = (mask_a & mask_b) & mask_c

    # use sectors from Eyal Kazin's catalog
    sectors_to_keep = np.unique(t_4['sector'])
    mask_3 = np.in1d(np.array(t_2['SECTOR']), sectors_to_keep)

    # magnitude cut
    cosmo = FlatLambdaCDM(H0=70, Om0=0.25)
    mr=22.5-2.5*np.log10(t_3['PETROFLUX'][:,2])-t_3['EXTINCTION'][:,2] 
    z = t_2['Z']
    delta_g, u_minus_g, g_minus_r, r_minus_i = color_k_correct(z, galaxy_type = 'non-star-forming')
    z_calibration = 0.2
    Mg = mr - interpolated_distmod(z, cosmo) - delta_g + g_minus_r - z_calibration
    Mg = Mg - 5.0*np.log10(cosmo.h)
    mask_4 = (Mg >= -23.2) & (Mg <= -21.2)

    # combne masks
    mask = (mask_1 & mask_2) & (mask_3 & mask_4)

    # build array of indices
    N = len(t_1)
    inds = np.arange(0,N).astype('int')
    inds = inds[mask]

    # save indices
    t = Table()
    t['index'] = inds
    fname = 'lrg_sample_indices.dat'

    t.write(fpath + fname, format='ascii')
    


if __name__ == '__main__':
    main()