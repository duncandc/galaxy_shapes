# Galaxy Shape Models

This directory stores Halotools-style component models used to model the 3D and 2D shapes of galaxies.


## Shape Models

Galaxy shapes are modelled as 3D ellipsoids.  The `shape_model_components.py` file contains various models for the probability a galaxy of a certain type has a set of axis ratios.


## Morphology Models

We assume the morpholoigically, galaxies can be split into two classes, disks and ellipticals.  The `morphology_model_components.py` file contains various models for the probability a galaxy is of a certain type.
 

## Extinction Models

We assume disk galaxies suffer inclication dependent dust extinction.  The `extinction_model_components.py` file contains various models for the amount of dust extinction a galaxy is subject to.  


## Size Models

Under Construction.


## Demos

We provde a set of ipython notebooks that demonstrate (and test) the various model components here.  These include:

*  `extinction_demo.ipynb`
*  `shape_demo.ipynb`