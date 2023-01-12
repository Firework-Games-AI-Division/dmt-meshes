bl_info = {
    "name": "DMT Meshes",
    "author": "DMT Meshes contributors",
    "description": "Use OpenAI Point-e to generate meshes from images or texts.",
    "blender": (3, 0, 0),
    "version": (0, 0, 1),
    "location": "Image Editor -> Sidebar -> DMT",
    "category": "Paint"
}

from multiprocessing import current_process

if current_process().name != "__actor__":
    from .absolute_path import absolute_path
    import sys
    sys.path.insert(0, absolute_path(".python_dependencies"))
    import bpy
    from bpy.props import IntProperty, PointerProperty, \
                          EnumProperty, StringProperty
    import os
    from .operators.dmt_meshes import DMTMesh, kill_generator
    from .preferences import set_pc_list
    from .generator_process.actions.utils import get_point_cloud

    module_name = os.path.basename(os.path.dirname(__file__))
    def clear_modules():
        for name in list(sys.modules.keys()):
            if name.startswith(module_name) and name != module_name:
                del sys.modules[name]
    clear_modules() # keep before all addon imports

    from .classes import CLASSES, PREFERENCE_CLASSES
    from .property_groups.dmt_prompt import DMTPrompt

    requirements_path_items = (
        (os.path.join('requirements', 'win-linux-cuda.txt'), 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
    )

    def register():
        dt_op = bpy.ops
        for name in DMTMesh.bl_idname.split("."):
            dt_op = getattr(dt_op, name)
        if hasattr(bpy.types, dt_op.idname()):
            raise RuntimeError('Another instance of DMT Meshes is already running.')

        bpy.types.Scene.dmt_meshes_requirements_path = EnumProperty(name="Platform",
                                                                      items=requirements_path_items,
                                                                      description="Specifies which set of dependencies to install",
                                                                      default=os.path.join('requirements', 'win-linux-cuda.txt'))

        for cls in PREFERENCE_CLASSES:
            bpy.utils.register_class(cls)

        bpy.types.Scene.dmt_meshes_prompt = PointerProperty(type=DMTPrompt)
        bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
        bpy.types.Scene.dmt_meshes_progress = IntProperty(name="", default=0, min=0, max=0)
        bpy.types.Scene.dmt_meshes_info = StringProperty(name="Info")
        bpy.types.Scene.point_cloud_results_selection = IntProperty(default=1)

        for cls in CLASSES:
            bpy.utils.register_class(cls)
        
        set_pc_list('point_cloud_results', get_point_cloud(''))


    def unregister():
        for cls in PREFERENCE_CLASSES:
            bpy.utils.unregister_class(cls)
        for cls in CLASSES:
            bpy.utils.unregister_class(cls)
            
        kill_generator()

