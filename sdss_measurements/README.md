# SDSS shape measurements

This directory contains code to calculate the shape distribution of galaxies in SDSS.

The galaxy catalogs are compiled and defined for this project [here](https://github.com/duncandc/galaxy_shapes/tree/master/data/SDSS_MAIN).


## Galaxy Samples

We define multiple galaxy 'samples'.  These samples are defined in bins of r-band absolute magnitudes k-corrected to z=0.1.

* `sample_1` where -17 > M_r0.1 > -18
* `sample_2` where -18 > M_r0.1 > -19
* `sample_3` where -19 > M_r0.1 > -20
* `sample_4` where -20 > M_r0.1 > -21
* `sample_5` where -21 > M_r0.1 > -22
* `sample_6` where -22 > M_r0.1 > -23

We also split each sample up into disk and elliptical sub-samples.  This is done by splitting galaxies on `FRACPSF` value in the catalogs which is a misnomer for the `fracDEV` parameter.  

* `disks` where `FRACPSF` < 0.8
* `ellipticals` where `FRACPSF` >= 0.8 

## SDSS Measurements

We measure the distribution of galaxy shapes and estimate the disk fraction for each galaxy sample (defined above) using the following scripts: 

* `measure_shape_distributions.py`
* `estimate_disk_fraction.py`

Measurements for each sample are stored in the `./data` directory.

All of the measurememnts can be run using the `run_all_measurements.sh` script.

Tools for estimating galaxy completeness, e.g. vmax as a function of M_r0.1, are in `sdss_utils.py`.

## Notebooks

Figures are created in the `sdss_measurements.ipynb` ipython notebook and stored in the `./figures` directory.   
