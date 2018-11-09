"""
script to make a value added galaxy catalogfor an SDSS main sample
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time


def main():

	# load sample selection
    fpath = './'
    fname = 'post_catalog.dr72bbright0.fits'
    t_1 = Table.read(fpath + fname)

    keys_1 = []  # columns to copy into vagc
    dts_1 = []  # dtyoe of columns
    
    idx = t_1['OBJECT_POSITION']

    # load lss file
    fpath = './'
    fname = 'lss_index.dr72.fits'
    t_2 = Table.read(fpath + fname)
    
    keys_2 = ['RA', 'DEC', 'Z', 'ZTYPE', 'FGOTMAIN']  # columns to copy into vagc
    dts_2 = ['f4','f4','f4','i8','f4']  # dtyoe of columns

    # load shape table
    fpath = './'
    fname = 'shapes_r.dr72.dat'
    colnames = ['STAR_LNL','EXP_LNL','DEV_LNL','ISO_A','ISO_B','AB_DEV','AB_EXP','PHI_ISO_DEG','PHI_DEV_DEG','PHI_EXP_DEG']
    t_3 = Table.read(fpath + fname, format='ascii', names=colnames)

    keys_3 = ['AB_DEV','AB_EXP','AB_ISO','PHI_ISO_DEG','PHI_DEV_DEG','PHI_EXP_DEG']   # columns to copy into vagc
    dts_3 = ['f4','f4','f4','f4','f4','f4']  # dtyoe of columns

    # caclulate isophotal B/A ratio
    # note this is not in the table already
    # so I add it into the table
    AB_ISO = t_3['ISO_B']/t_3['ISO_A']
    t_3['AB_ISO'] = AB_ISO 

    # load table
    fpath = './'
    fname = 'kcorrect.nearest.model.z0.10.fits'
    t_4 = Table.read(fpath + fname)

    keys_4 = ['ABSMAG_u0.1','ABSMAG_g0.1','ABSMAG_r0.1','ABSMAG_i0.1','ABSMAG_z0.1']  # columns to copy into vagc
    dts_4 = ['f4','f4','f4','f4','f4']  # dtyoe of columns

    # these quantities are stored as arrays in columns
    # rename and out each in their own column
    ABSMAG_u = t_4['ABSMAG'][:,0]
    ABSMAG_g = t_4['ABSMAG'][:,1]
    ABSMAG_r = t_4['ABSMAG'][:,2]
    ABSMAG_i = t_4['ABSMAG'][:,3]
    ABSMAG_z = t_4['ABSMAG'][:,4]
    t_4['ABSMAG_u0.1'] = ABSMAG_u
    t_4['ABSMAG_g0.1'] = ABSMAG_g
    t_4['ABSMAG_r0.1'] = ABSMAG_r
    t_4['ABSMAG_i0.1'] = ABSMAG_i
    t_4['ABSMAG_z0.1'] = ABSMAG_z


    # build table to store catalog
    N = len(idx)
    colnames = keys_1 + keys_2 + keys_3 + keys_4
    dts = dts_1 + dts_2 + dts_3 + dts_4
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
        catalog_table[key]=t_1[key]

    for key in keys_2:
        catalog_table[key]=t_2[key][idx]

    for key in keys_3:
        catalog_table[key]=t_3[key][idx]

    for key in keys_4:
        catalog_table[key]=t_4[key][idx]


    # save catalog
    fpath = './'
    fname = 'sdss_vagc.hdf5'
    catalog_table.write(fname, path='data', overwrite=True)


if __name__ == '__main__':
    main()

