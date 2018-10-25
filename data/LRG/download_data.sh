#!/bin/bash

# large scale structure catalog
# http://sdss.physics.nyu.edu/lss/dr72/
wget http://sdss.physics.nyu.edu/lss/dr72/lss_index.dr72.fits

# object catalogs
# http://sdss.physics.nyu.edu/vagc-dr7/vagc2
wget http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_catalog.fits

# tiling information
# http://sdss.physics.nyu.edu/vagc-dr7/vagc2/
wget http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_sdss_tiling.fits

# imaging information
# http://sdss.physics.nyu.edu/vagc-dr7/vagc2/
wget http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_sdss_imaging.fits

# spectral inforamtion
# http://sdss.physics.nyu.edu/vagc-dr7/vagc2/
wget http://sdss.physics.nyu.edu/vagc-dr7/vagc2/object_sdss_spectro.fits

# lrg catalog
# http://cosmo.nyu.edu/~eak306/SDSS-LRG.html
wget http://cosmo.nyu.edu/~eak306/files/DR7-Full.ascii

