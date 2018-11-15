# Fitting Shape Model

## Model

Here we fit a model for galaxy shapes for each sample seperatly.  I assume that there is a a fixed diusk fraction in each magnitude bin.  

The fitting parameters and initial estimates for the parameters for each sample are specified in the `sample_[#]_fitting_params.py` files.

Priors on the model parameters are set in `priors.py`.

## Requirements

The model fits are performed using the MCMC ensemble sampler, [emcee](http://dfm.io/emcee/current/).
