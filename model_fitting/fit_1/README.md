# Fit \#1

Here we fit a two component model (disk and elliptical-like populations) for the 3D shapes of galaxies in luminosity bins, each fit indpendently.

Some assumptions are made for these fits.  The dust extinction law for disk galaxies is assumed to be th esame for each luminosity bin.  

The intrinsic luminosity function is fixed, and not allowed to vary.  The opbserved luminosity function is not used in the fits.

When fitting a luminosoity bin, it is assumed that all more luminious galaxies use the same shape model and disk fraction.  This is not quite right, as luminious disk galaxies can "leak" into a less lumious bin via dust extiction.  

## Technical Details

The `like.py` and `prob.py	` files store the liklihood and probability functions needed to run the MCMC.  `like.py` uses the scripts in `make_mock.py` in order to create a mock galaxy sample.  The `priors.py` file stores the priors set on the parameters of the fit.

## Running The Fits

The fitting is done using the [emcee](https://emcee.readthedocs.io/en/stable/) python package.  We use the deveoper version (v3.0 or greater).  This can be installed from the github page [here](https://github.com/dfm/emcee) with instructions [here](https://emcee.readthedocs.io/en/latest/tutorials/quickstart/).

The parameters for running the fits are stored in the `[sample]_fitting_params.py` files.

The fits are performed using the `fit_sample.py` script.  This script takes the sample name as a positional argument, and it uses the paramaters found the in the corresponding parameter file.  For example, executing the command: `python fit_sample.py sample_4` will run the fitting routine for sample 4 using the parameters stored in `sample_4_fitting_params.py`

All the fits can be performed in serial using the `run_all_fits.sh` script. This will take a few hours.

The MCMC chains are stored in the `./chains/` directory.

The results/progress of the fits can be checked using the follwoing scripts:

*  `plot_chains.py`
*  `plot_corner_plot.py`
*  `plot_observables.py`

Each of these scripts are run with a positional argument specifying the sample, e.g. `python plot_chains.py sample_4`.

## Analysis

The `./notebooks/` directory contains some ipython notebooks useful for setting up the fits and analysing the results. 
