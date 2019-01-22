"""
"""

from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.table import Table
from grid import Grid
import matplotlib.pyplot as plt

def main():

    # open random catalog
    fpath = '../data/SDSS_MAIN/'
    fname = 'random-0.dr72bbright.fits'
    hdulist = fits.open(fpath+fname)
    randoms = Table(hdulist[1].data)
    print(randoms)
    
    # define some regions
    # northern galactic cap
    mask_1 = ((randoms['RA']>100) & (randoms['RA']<300))
    # equitorial stripe
    mask_2 = ((randoms['RA']<100) | (randoms['RA']>300)) & ((randoms['DEC']>-2.5) & (randoms['DEC']<2.5))
    # nothern stripe
    mask_3 = ((randoms['RA']<100) | (randoms['RA']>300)) & ((randoms['DEC']>5))
    # southern stripe
    mask_4 = ((randoms['RA']<100) | (randoms['RA']>300)) & ((randoms['DEC']<-2.5))

    # plot regions
    #fig = plt.figure(figsize=(3.3*2,3.3*2))
    #plt.scatter(randoms['RA'][mask_1], randoms['DEC'][mask_1], s=0.1)
    #plt.scatter(randoms['RA'][mask_2], randoms['DEC'][mask_2], s=0.1)
    #plt.scatter(randoms['RA'][mask_3], randoms['DEC'][mask_3], s=0.1)
    #plt.scatter(randoms['RA'][mask_4], randoms['DEC'][mask_4], s=0.1)
    #plt.show()

    # define jackknife regions for northern cap
    Ncols = 7
    Nrows = 7
    Nran = np.sum(mask_1)
    x = randoms['RA'][mask_1]
    y = randoms['DEC'][mask_1]
    min_x = np.min(x)-0.001
    max_x = np.max(x)+0.001
    min_y = np.min(y)-0.001
    max_y = np.max(y)+0.001

    # define grid
    jk_regions = Grid(min_x, max_x,
                      min_y, max_y,
                      Ncols, Nrows)
    labels = jk_regions.region_id(x, y)
    
    #number of randoms per region
    Nsub = Nran/jk_regions.Nud
    
    #set up row parameters
    row_params = np.array([min_y]*(Nrows+1))
    row_params[-1] = max_y
    
    #set up column parameters
    col_params = np.zeros((Nrows,Ncols+1))
    col_params[:,:] = min_x
    col_params[:,-1] = max_x
    
    #loop over rows
    print("looping over rows...")
    row_epsilon = 0.01
    for i in range(1,Nrows):
        row_params[i] = row_params[i-1]
        print(i)
        N=0
        while N<Nsub:
            row_params[i] += row_epsilon
            mask = (y>row_params[i-1]) & (y<row_params[i]) 
            N = np.sum(mask)
        #randomly choose upper or lower boundary
        if np.random.random(1)>0.5:
            row_params[i] -= row_epsilon
    
    #loop over columns in rows
    print("looping over columns...")
    col_epsilon = 0.01
    for i in range(0,Nrows):
        row_mask = (y>row_params[i]) & (y<row_params[i+1])
        for j in range(1,Ncols):
            col_params[i,j] = col_params[i,j-1]
            print(i,j)
            N=0
            while N<Nsub/Ncols:
                col_params[i,j] += col_epsilon
                col_mask = (x>col_params[i,j-1]) & (x<col_params[i,j]) 
                mask = (col_mask & row_mask)
                N = np.sum(mask)
            #randomly choose upper or lower boundary
            if np.random.random(1)>0.5:
                col_params[i,j] -= col_epsilon
    
    jk_regions.define_regions(row_params[1:-1], col_params[:,1:-1])
    labels = jk_regions.region_id(x, y)

    # plot regions
    fig = plt.figure(figsize=(3.3*2,3.3*2))
    plt.scatter(randoms['RA'][mask_1], randoms['DEC'][mask_1], s=0.1, color=labels)
    plt.show()


if __name__ == '__main__':
    main()