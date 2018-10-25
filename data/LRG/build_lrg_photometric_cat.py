"""
script to build a catalog of photometric quantities for an LRG sample
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table, Column
import time


def main():

    
    # load example calibObj file
    fpath = './parameters/'
    fname = 'calibObj-004184-2.fits'
    t_0 = Table.read(fpath + fname)

    # load selected columns
    fpath = './'
    fname = 'selected_photo_props.txt'
    with open(fpath + fname) as f:
        colnames = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    colnames = [x.strip() for x in colnames] 

    # load object_sdss_imaging table
    fpath = './'
    fname = 'object_sdss_imaging.fits'
    t_1 = Table.read(fpath + fname)

    # load lrg indices
    fpath = './'
    fname = 'lrg_sample_indices.dat'
    t_2 = Table.read(fpath + fname, format='ascii')
    lrg_inds = np.array(t_2['index'])

    # build table to store photometric catalog
    N = len(lrg_inds)
    # first build columns to store data
    cols = []
    for colname in colnames:
        dt = t_0[colname].dtype
        try:
            s = np.shape(t_0[colname])[1]
        except IndexError:
            s = 1
        col = Column(name=colname, dtype=dt, length=N, shape=(s))
        cols.append(col)
    # then build empty table
    photo_table = Table()
    for col in cols:
        photo_table.add_column(col)
    
    start = time.time()

    # populate table using calibObj files
    fpath = './parameters/'
    # loop through each galaxy in the selection
    N = len(lrg_inds)
    for i in range(N):
        index = lrg_inds[i]
        # open calibObj for that galaxy
        run = t_1['RUN'][index]
        camcol = t_1['CAMCOL'][index]
        fname = 'calibObj-'+str(run).zfill(6)+'-'+str(camcol)+'.fits'
        row = t_1['CALIBOBJ_POSITION'][index]
        # loop through selected quantities and add to table
        t = Table.read(fpath + fname)
        for colname in colnames:
            photo_table[colname][i] = t[colname][row]
    
    print('time to build catalog: {0} min.'.format((time.time()-start)/60.0))

    # save catalog
    fpath = './'
    fname = 'lrg_photo_cat.hdf5'
    photo_table.write(fname, path='data', overwrite=True)



if __name__ == '__main__':
    main()