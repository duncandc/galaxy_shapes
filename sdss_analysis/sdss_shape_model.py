"""
model for SDSS galaxy shapes
"""

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
from galaxy_shapes.shape_models.extinction_model_components import PS08DustExtinction, Shao07DustExtinction
extinction_model = Shao07DustExtinction()

# these are only needed in order to build a composite model
from halotools.empirical_models import TrivialPhaseSpace
prof_model = TrivialPhaseSpace()
from halotools.empirical_models import Zheng07Cens
occupation_model =  Zheng07Cens()

# combine model componenets
from halotools.empirical_models import HodModelFactory
model_instance = HodModelFactory(centrals_occupation_model = occupation_model,
	                              centrals_prof_model = prof_model,
	                              centrals_morphology = morpholopgy_model,
                                 centrals_elliptical_galaxy_shape = elliptical_shape_model,
                                 centrals_disk_galaxy_shape = disk_shape_model,
                                 centrals_orientation = orientation_model,
                                 centrals_proj_shapes = proj_shapes_model,
                                 centrals_extinction_model = extinction_model,
                                 model_feature_calling_sequence = (
                                 'centrals_occupation_model',
                                 'centrals_prof_model',
                                 'centrals_morphology',
                                 'centrals_elliptical_galaxy_shape',
                                 'centrals_disk_galaxy_shape',
                                 'centrals_orientation',
                                 'centrals_proj_shapes',
                                 'centrals_extinction_model'
                                 )
                                )