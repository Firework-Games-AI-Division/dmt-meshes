import bpy
from bpy.types import Panel
from bpy_extras.io_utils import ImportHelper
from ..space_types import SPACE_TYPES
from ...operators.dmt_meshes import DMTMesh, ReleaseGenerator, CancelGenerator
from ...preferences import DMTMeshesPreferences
import webbrowser
import os
import shutil


def dmt_meshes_panels():
    for space_type in SPACE_TYPES:
        class DMTMeshPanel(Panel):
            bl_label = "DMT Mesh"
            bl_idname = f"DMT_PT_mesh_panel_{space_type}"
            bl_category = "DMT"
            bl_space_type = space_type
            bl_region_type = "UI"
        
            @classmethod
            def poll(self, context):
                if self.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False


        DMTMeshPanel.__name__ = f"DMT_PT_mesh_panel_{space_type}"
        yield DMTMeshPanel
        def get_prompt(context):
            return context.scene.dmt_meshes_prompt
        yield from create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, prompt_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, init_image_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, point_cloud_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, advanced_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, run_mode_panel, get_prompt)
        yield create_panel(space_type, 'UI', DMTMeshPanel.bl_idname, actions_panel, get_prompt)


def create_panel(space_type, region_type, parent_id, ctor, get_prompt, use_property_decorate=False):
    class BasePanel(bpy.types.Panel):
        bl_category = "DMT"
        bl_space_type = space_type
        bl_region_type = region_type

    class SubPanel(BasePanel):
        bl_category = "DMT"
        bl_space_type = space_type
        bl_region_type = region_type
        bl_parent_id = parent_id

        def draw(self, context):
            self.layout.use_property_decorate = use_property_decorate
    
    return ctor(SubPanel, space_type, get_prompt)


def prompt_panel(sub_panel, space_type, get_prompt):
    class PromptPanel(sub_panel):
        """Create a subpanel for prompt input"""
        bl_label = "Prompt"
        bl_idname = f'DMT_PT_mesh_panel_prompt_{space_type}'

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.prop(get_prompt(context), "prompt", text="")
    yield PromptPanel

def init_image_panel(sub_panel, space_type, get_prompt):
    class InitImagePanel(sub_panel):
        """Create a subpanel for init image options"""
        bl_idname = f"DMT_PT_mesh_panel_init_image_{space_type}"
        bl_label = "Source Image"
        bl_options = {'DEFAULT_CLOSED'}

        def draw_header(self, context):
            self.layout.prop(get_prompt(context), "use_init_img", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            prompt = get_prompt(context)
            layout.enabled = prompt.use_init_img
            layout.prop(prompt, "init_img_src", expand=True)
            if prompt.init_img_src == 'file':
                layout.template_ID(context.scene, "init_img", open="image.open")
            layout.use_property_split = True
    yield InitImagePanel


def point_cloud_panel(sub_panel, space_type, get_prompt):
    class PointCloudPanel(sub_panel):
        """Create a subpanel for advanced options"""
        bl_idname = f"DMT_PT_mesh_panel_point_cloud_{space_type}"
        bl_label = "Point Cloud"
        # bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.template_list("SCENE_UL_pc_list", "dmt_meshes_point_cloud_results",
                                 context.preferences.addons[DMTMeshesPreferences.bl_idname].preferences,
                                 "point_cloud_results", context.scene, "point_cloud_results_selection")
    yield PointCloudPanel


class SCENE_UL_pc_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=item.point_cloud, translate=False, icon_value=icon)
        # 'GRID' layout type should be as compact as possible (typically a single icon!).
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.point_cloud, icon_value=icon)


def advanced_panel(sub_panel, space_type, get_prompt):
    class AdvancedPanel(sub_panel):
        """Create a subpanel for advanced options"""
        bl_idname = f"DMT_PT_mesh_panel_advanced_{space_type}"
        bl_label = "Advanced"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.prop(get_prompt(context), "image_to_pc_model")
            layout.prop(get_prompt(context), "pc_to_mesh_method")

            layout.prop(get_prompt(context), "steps")
            layout.prop(get_prompt(context), "view_every")
            layout.prop(get_prompt(context), "step_preview_mode")
            layout.prop(get_prompt(context), "lr")
            layout.prop(get_prompt(context), "laplacian_weight")
            layout.prop(get_prompt(context), "gridres")
            layout.prop(get_prompt(context), "multires")
    yield AdvancedPanel

def run_mode_panel(sub_panel, space_type, get_prompt):
    class RunModePanel(sub_panel):
        """Create a subpanel for advanced options"""
        bl_idname = f"DMT_PT_mesh_panel_run_mode_{space_type}"
        bl_label = "Run Mode"

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.prop(get_prompt(context), "run_mode")
    yield RunModePanel

def actions_panel(sub_panel, space_type, get_prompt):
    class ActionsPanel(sub_panel):
        """Create a subpanel for actions"""
        bl_idname = f"DMT_PT_mesh_panel_actions_{space_type}"
        bl_label = "Advanced"
        bl_options = {'HIDE_HEADER'}

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            
            row = layout.row()
            row.scale_y = 1.5
            if context.scene.dmt_meshes_progress <= 0:
                if context.scene.dmt_meshes_info != "":
                    row.label(text=context.scene.dmt_meshes_info, icon="INFO")
                else:
                    row.operator(DMTMesh.bl_idname, icon="PLAY", text="Generate")
            else:
                disabled_row = row.row()
                disabled_row.use_property_split = True
                disabled_row.prop(context.scene, 'dmt_meshes_progress', slider=True)
                disabled_row.enabled = False
            if CancelGenerator.poll(context):
                row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
            row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
    return ActionsPanel