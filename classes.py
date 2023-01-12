from .operators.install_dependencies import InstallDependencies
from .operators.dmt_meshes import DMTMesh, ReleaseGenerator, CancelGenerator
from .ui.panels import dmt_meshes
from .preferences import OpenContributors, DMTMeshesPreferences, PointCloud
from .property_groups.dmt_prompt import DMTPrompt

CLASSES = (
    DMTMesh,
    ReleaseGenerator,
    CancelGenerator,
    dmt_meshes.SCENE_UL_pc_list,
    *dmt_meshes.dmt_meshes_panels()
)
PREFERENCE_CLASSES = (
                      PointCloud,
                      InstallDependencies,
                      OpenContributors,
                      DMTPrompt,
                      DMTMeshesPreferences
)