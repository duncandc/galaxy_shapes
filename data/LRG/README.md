# LRG Data

This directory creates an LRG sample using the NYU value added galaxy catalog (VAGC).  The VAGC is described [here](http://sdss.physics.nyu.edu/vagc/).


## Required Data

The files needed are:

*   `lss_index.dr72.fits`
*   `object_sdss_tiling.fits`
*   `object_sdss_imaging.fits`

and a file from Eyal Kazin available [here](http://cosmo.nyu.edu/~eak306/SDSS-LRG.html):

*    `DR7-Full.ascii`

and a large number "parameter" files (~2500, 6.3G in total) of the form:

*    `calibObj-[RUN]-[CAMCOL].fits`

The first four files can be downloaded using the `./download_data.sh` script.  The parameter files can be downloaded using the `./parameters/download_all_files.sh` script.


## LRG Selection

The LRG sample is built closely following the directions from Eyal Kazin [here](http://cosmo.nyu.edu/~eak306/SDSS-LRG.html).  The steps he takes are reproduced in the `build_lrg_indices.py` script, except for the "sector completeness" cut which uses an approximation for completeness which takes some work to reproduce.  I do not reproduce that here, and instead I only keep galaxies in sectors which are present in his catalog, given in `DR7-Full.ascii`.    

Indices into the NYU VAGC that select LRGs are produced by the `build_lrg_indices.py` script.  This script creates an ascii table of 0-indexed indices, that when applied to the NYU VAGC files, produce an LRG sample.  This file is saved as `lrg_sample_indices.dat`


## Value Added LRG catalog

This project uses a variety of photometric shape measurements that are not provided in the standard VAGC file, `object_sdss_imaging.fits`.  These values must be grabbed from a large number of parameter files available [here](http://sdss.physics.nyu.edu/vagc-dr7/vagc2/sdss/parameters/) that store all of the photometric properties prodced by the standard SDSS pipeline.

A description of the columns in these files is available [here](http://photo.astro.princeton.edu), whith more information from SDSS [here](https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/CAMCOL/photoObj.html), although it seems not all of the parameters retain the same name between the two.  

  