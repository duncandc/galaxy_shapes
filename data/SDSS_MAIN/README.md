# SDSS MAIN Data

This directory contains scripts to create a flux-limited sample for strudying galaxy shapes using the NYU value added galaxy catalog (VAGC).  The VAGC is described [here](http://sdss.physics.nyu.edu/vagc/).


## Required Data

The primary files needed are:

*   `lss_index.dr72.fits`
*   `object_sdss_imaging.fits`
*   `object_sdss_spectro.fits`

and files storing additional parameters:

*    `kcorrect.nearest.model.z0.10.fits`
*    [`shapes_r.dr72.dat`](http://sdss.physics.nyu.edu/vagc/flatfiles/shapes_r.dat.html)
*    [`sersic_catalog.fits`](http://cosmo.nyu.edu/blanton/vagc/sersic.html)

The file storing the sample selection:

*   `post_catalog.dr72bbright0.fits`

The files storing a random sample:

* `random-0.dr72bbright.fits.gz`


These files can be downloaded using the `./download_data.sh` script.

## Sample Selection

For the 1-point statistics relavent for this work, we use the 'bbright' LSS sample, `[letter]=bbright`.  The LSS samples are defined [here](http://sdss.physics.nyu.edu/vagc/lss.html).  We use the full sample, "post-redshift" selection: `[post]=0`, described [here](http://sdss.physics.nyu.edu/vagc/data/sdss/lss_post.par).

The `post_catalog.dr72bbright0.fits` file contains 0-based indices into all other files for the galaxies which pass the 'bbright' selection criteria.  


## Value Added LRG catalog


## Notes

the "fracDEV" paramater in `object_sdss_imaging.fits` is misleadingly termed "fracPSF".  This is mentioned [here](http://classic.sdss.org/dr7/algorithms/photometry.html).

The columns in `shapes_r.dr72.dat` are listed [here](http://sdss.physics.nyu.edu/vagc/flatfiles/shapes_r.dat.html), and reproduced below:

* STAR_LNL
* EXP_LNL
* DEV_LNL
* ISO_A
* ISO_B
* AB_DEV
* AB_EXP
* PHI_ISO_DEG
* PHI_DEV_DEG
* PHI_EXP_DEG


  