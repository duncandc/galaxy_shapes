"""
function that returns a mock glalaxy sample
"""

from __future__ import print_function, division, absolute_import
import numpy as np
from astropy.table import Table
from astro_utils.schechter_functions import MagSchechter

# model for fraction of disk and elliptical galaxies
from galaxy_shapes.shape_models.morphology_model_components import Morphology_2
morpholopgy_model = Morphology_2()

# model for 3D disk and elliptical galaxy shapes
from galaxy_shapes.shape_models.shape_model_components import EllipticalGalaxyShapes, DiskGalaxyShapes
elliptical_shape_model = EllipticalGalaxyShapes()
disk_shape_model       = DiskGalaxyShapes()

# model for galaxy alignments
from intrinsic_alignments.ia_models.ia_model_components import RandomAlignment
orientation_model = RandomAlignment()

# model for projected galaxy shapes
from galaxy_shapes.shape_models.shape_model_components import ProjectedShapes
proj_shapes_model = ProjectedShapes()

# model for disk galaxuy dust extinction
from galaxy_shapes.shape_models.extinction_model_components import Shao07DustExtinction
extinction_model = Shao07DustExtinction()

__all__ = ['make_galaxy_sample']


def make_galaxy_sample(mag_lim=-18, size=10**5, **kwargs):
    """
    Parameters
    ----------
    mag_lim : float
        minimum magnitude to sample from schechter function

    size : int
	    number of galaxies to sample from schechter function

    kwargs : dictionary
        dictionary of parameters for the galaxy model
    
    Returns
    -------
    galaxy_table : astropy.table
        mock galaxy sample
    """
    
    # set parameters
    for key in kwargs:
        try:
            morpholopgy_model.param_dict[key] = kwargs[key]
        except KeyError:
            pass
        try:
            elliptical_shape_model.param_dict[key] = kwargs[key]
        except KeyError:
            pass
        try:
            disk_shape_model.param_dict[key] = kwargs[key]
        except KeyError:
            pass
        try:
            extinction_model.param_dict[key] = kwargs[key]
        except KeyError:
            pass
        try:
            orientation_model.param_dict[key] = kwargs[key]
        except KeyError:
            pass
    
    # pick intrinsic luminosity function
    lum_func = MagSchechter(1.49 * 10**(-2), -20.44, -1.05)

    galaxy_table = Table()
    galaxy_table['gal_type'] = ['centrals']*size
    galaxy_table['Mag_r'] = lum_func.rvs(m_max=mag_lim, size=size)
    
    if 'f_disk' in kwargs:
        f_disk = kwargs['f_disk']
        ran_num = np.random.random(size)
        galaxy_table['disk'] = False
        galaxy_table['elliptical'] = False
        galaxy_table['disk'][ran_num<f_disk] = True
        galaxy_table['elliptical'][ran_num>=f_disk] = True
    else:
        galaxy_table = morpholopgy_model.assign_morphology(table=galaxy_table)
    
    galaxy_table = orientation_model.assign_orientation(table=galaxy_table)
    
    galaxy_table = elliptical_shape_model.assign_elliptical_b_to_a(table=galaxy_table)
    galaxy_table = elliptical_shape_model.assign_elliptical_c_to_a(table=galaxy_table)
    
    galaxy_table = disk_shape_model.assign_disk_b_to_a(table=galaxy_table)
    galaxy_table = disk_shape_model.assign_disk_c_to_a(table=galaxy_table)
    
    galaxy_table = proj_shapes_model.assign_projected_b_to_a(table=galaxy_table)
    
    galaxy_table = extinction_model.assign_extinction(table=galaxy_table)
    galaxy_table['obs_Mag_r'] = galaxy_table['Mag_r'] + galaxy_table['deltaMag_r']
    
    return galaxy_table

