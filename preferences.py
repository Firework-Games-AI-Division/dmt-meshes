import bpy
import os
import webbrowser

from .absolute_path import absolute_path
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .operators.install_dependencies import InstallDependencies

class PointCloud(bpy.types.PropertyGroup):
    bl_label = "Point Cloud"
    bl_idname = "dmt_meshes.point_cloud"

    point_cloud: bpy.props.StringProperty(name="point_cloud")

class OpenContributors(bpy.types.Operator):
    bl_idname = "dmt_meshes.open_contributors"
    bl_label = "See All Contributors"

    def execute(self, context):
        webbrowser.open("https://github.com/Firework-Games-AI-Division/dmt-meshes/graphs/contributors")
        return {"FINISHED"}

class DMTMeshesPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    point_cloud_results: CollectionProperty(type=PointCloud)

    def draw(self, context):
        layout = self.layout
        has_dependencies = len(os.listdir(absolute_path(".python_dependencies"))) > 2

        if context.preferences.view.show_developer_ui: # If 'Developer Extras' is enabled, show addon development tools
            developer_box = layout.box()
            developer_box.label(text="Development Tools", icon="CONSOLE")
            developer_box.label(text="This section is for addon development only. You are seeing this because you have 'Developer Extras' enabled.")
            developer_box.label(text="Do not use any operators in this section unless you are setting up a development environment.")
            if has_dependencies:
                warn_box = developer_box.box()
                warn_box.label(text="Dependencies already installed. Only install below if you developing the addon", icon="CHECKMARK")
            developer_box.prop(context.scene, 'dmt_meshes_requirements_path')
            developer_box.operator(InstallDependencies.bl_idname, icon="CONSOLE")

@staticmethod
def set_pc_list(l: str, point_clouds: list):
    getattr(bpy.context.preferences.addons[__package__].preferences, l).clear()
    for point_cloud in point_clouds:
        m = getattr(bpy.context.preferences.addons[__package__].preferences, l).add()
        m.point_cloud = point_cloud