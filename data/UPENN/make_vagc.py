"""
script to make a value added galaxy catalogfor an SDSS main sample
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
from astropy.io import fits


def main():

    fpath = './'

    hdul_1 = fits.open(fpath + 'UPenn_PhotDec_CAST.fits')
    data_1 = hdul_1[1].data

    keys_1 = ['ra', 'dec', 'z', 'run', 'rerun', 'camCol', 'field', 'obj', 'objid', 'specobjid']  # columns to copy into vagc
    dts_1 = ['f4','f4','f4','i8','i8','i8','i8','i8', 'i8', 'i8']  # dtyoe of columns

    hdul_2 = fits.open(fpath + 'UPenn_PhotDec_Models_rband.fits')
    data_2 = hdul_2[1].data

    keys_2 = ['m_tot', 'BT', 'r_tot', 'ba_tot']
    dts_2 = ['f4','f4','f4','f4']

    hdul_3 = fits.open(fpath + 'UPenn_PhotDec_nonParam_rband.fits')
    data_3 = hdul_3[1].data

    keys_3 = []
    dts_3 = []


    # build table to store catalog
    N = len(data_1)
    colnames = keys_1 + keys_2 + keys_3
    dts = dts_1 + dts_2 + dts_3
    # first build columns to store data
    cols = []
    for i, colname in enumerate(colnames):
        dt = dts[i]
        col = Column(name=colname, dtype=dt, length=N)
        cols.append(col)
    # then build empty table
    catalog_table = Table()
    for col in cols:
        catalog_table.add_column(col)

    # fill table
    for key in keys_1:
        catalog_table[key]=data_1[key]

    for key in keys_2:
        catalog_table[key]=data_2[key]

    for key in keys_3:
        catalog_table[key]=data_3[key]

    # save catalog
    fpath = './'
    fname = 'meert_vagc.hdf5'
    catalog_table.write(fname, path='data', overwrite=True)


if __name__ == '__main__':
    main()